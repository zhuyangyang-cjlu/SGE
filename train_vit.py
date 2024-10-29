import os
import argparse
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import visformer_vis
from data.dataloader import EpisodeSampler, RepeatSampler
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC
from utils import mean_confidence_interval
from data.dataset import DatasetWithTextLabel, aug_DatasetWithTextLabel
from data.dataloader import EpisodeSampler, MultiTrans,TESTEpisodeSampler
# 定义主函数，参数为args
def main(args):
    # checkpoint and tensorboard dir
    # 检查tensorboard和checkpoint目录是否存在，若不存在则创建
    args.tensorboard_dir = 'tensorboard/'+args.dataset+'/'+args.model+'/'+args.exp + '/'
    args.checkpoint_dir = 'checkpoint/'+args.dataset+'/'+args.model+'/'+args.exp + '/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

    # prepare training and testing dataloader
    # 准备训练和测试dataloader
   # 定义归一化函数
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    # 定义训练时使用的变换
    train_aug = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    norm])
    # 定义训练时使用增强变换
    if args.aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        norm])
    # 定义训练时使用随机增强变换
    if args.rand_aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        RandAugmentMC(2, 10, args.image_size),
                                        transforms.ToTensor(),
                                        norm])
    # 定义测试时使用的变换
    test_aug = transforms.Compose([transforms.Resize(int(args.image_size * 1.1)),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   norm])

    # 加载训练集
    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    print(f"train_dataset:{len(train_dataset)}")
    # 如果需要重复增强，则使用重复采样器
    if args.repeat_aug:
        repeat_sampler = RepeatSampler(train_dataset, batch_size=128, repeat=4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=repeat_sampler, num_workers=8)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=8)
    # 获取训练集类别数
    num_classes = len(train_dataset.dataset.classes)
    print(num_classes)
    args.num_classes = num_classes

    # 加载测试集
    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split=args.split)
    # 加载测试集采样器
    episode_sampler = TESTEpisodeSampler(test_dataset.dataset.targets, 400, args.way, args.shot + 15)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=6)

    # build model

   # 加载模型
    if args.model == 'visformer-t':
        student = visformer_vis.visformer_tiny(num_classes=num_classes)
    elif args.model == 'visformer-t-84':
        student = visformer_vis.visformer_tiny_84(num_classes=num_classes)
    else:
        raise ValueError(f'unknown model: {args.model}')

    # 将模型移动到GPU上
    student = student.cuda(args.gpu)

    # 加载优化器 
    if args.optim == 'adam':
        optim = torch.optim.Adam(student.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=5e-2)
    else:
        raise ValueError(f'unknown optim: {args.optim}')

    # 加载学习率调度器
    scheduler = None
    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.annealing_period)

    # 加载模型参数
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        student.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    # 进行测试
    if args.test:
        test(student, test_loader, start_epoch, args)
        return 0

    # 开始训练
    print('start training...')
    for epoch in range(start_epoch, args.epochs):
        print(1)
        train(student, train_loader, optim, scheduler, epoch, args)

        if (epoch+1) % 1 == 0:
            test(student, test_loader, epoch, args)
            print('测试完成')

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': student.state_dict(),
                'optimizer': optim.state_dict(),
            }
            try:
                torch.save(checkpoint, args.checkpoint_dir+f'checkpoint_epoch_{epoch+1:03d}.pth')
                print("保存成功")
            except Exception as e:
                print(f"保存失败: {e}")

    # 根据学习率调度器调整学习率
        if args.cosine_annealing and (epoch + 1) % args.annealing_period == 0:
            lr = args.lr * args.gamma**int((epoch + 1)/args.annealing_period)
            if args.optim == 'adam':
                optim = torch.optim.Adam(student.parameters(), lr=lr)
            elif args.optim == 'adamw':
                optim = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=5e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.annealing_period)



# 定义训练函数，用于训练学生模型
def train(student, train_loader, optim, scheduler, epoch, args):
    # 设置学生模型为训练模式
    student.train()
    # 初始化损失和准确率
    losses = 0.
    accs = 0.
    # 遍历训练数据集
    for idx, episode in enumerate(train_loader):
        # 将训练数据集转换为cuda格式
        image = episode[0].cuda(args.gpu)  # way * (shot+15)
        labels = episode[1].cuda(args.gpu)

        # 计算学生模型的输出和特征
        logit, features = student(image)
        # 计算损失
        loss = F.cross_entropy(logit, labels)
        # 累加损失
        losses += loss.item()
        # 计算预测结果
        _, pred = logit.max(-1)
        # 累加准确率
        accs += labels.eq(pred).sum().float().item() / labels.shape[0]

        # 梯度归零
        optim.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optim.step()

        # 每print_step个batch打印一次训练信息
        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            print_string = f'Train epoch: {epoch}, step: {idx:3d}, loss: {losses/(idx+1):.4f}, acc: {accs*100/(idx+1):.2f}'
            print(print_string)
    # 添加损失和准确率到日志中
    args.logger.add_scalar('train/loss', losses/len(train_loader), epoch)
    args.logger.add_scalar('train/acc', accs/len(train_loader), epoch)

    # 如果使用学习率衰减，则添加学习率到日志中
    if scheduler is not None:
        args.logger.add_scalar('train/lr', float(scheduler.get_last_lr()[0]), epoch)
        scheduler.step()


# 定义一个函数test，用于测试学生模型的准确率
# 参数：student：学生模型；test_loader：测试数据集；epoch：当前epoch；args：参数
def test(student, test_loader, epoch, args):
    # 将学生模型设置为评估模式
    student.eval()
    # 初始化一个空列表，用于存放每次测试的准确率
    accs = []
    # 使用torch.no_grad()禁止梯度计算
    with torch.no_grad():
        # 遍历测试数据集
        for episode in test_loader:
            # 将图像数据和标签数据放入cuda中
            image = episode[0].cuda(args.gpu)  # way * (shot+15)
            labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

            # 计算学生模型的特征向量
            _, im_features = student(image)
            # 将特征向量按way和shot进行分割
            im_features = im_features.view(args.way, args.shot + 15, -1)
            # 将训练集图像特征和测试集图像特征分割开
            sup_im_features, que_im_features = im_features[:, :args.shot], im_features[:, args.shot:]

            # 对训练集图像特征进行平均
            sup_im_features = sup_im_features.mean(dim=1)
            # 将测试集图像特征转换为一维
            que_im_features = que_im_features.contiguous().view(args.way * 15, -1)

            # 计算测试集图像特征和训练集图像特征的余弦相似度
            sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
            # 获取余弦相似度中最大值对应的索引
            _, pred = sim.max(-1)
            # 计算准确率
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            # 将准确率添加到accs列表中
            accs.append(acc)

    # 计算accs列表中准确率的均值和置信区间
    m, h = mean_confidence_interval(accs)
    # 打印测试结果
    print(f'Test epoch: {epoch}, test acc: {m*100:.2f}+-{h*100:.2f}')
    # 将测试结果添加到日志中
    args.logger.add_scalar('test/acc', m*100, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--repeat_aug', action='store_true')
    parser.add_argument('--model', type=str, default='visformer-t', choices=['visformer-t', 'visformer-t-84'])
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--cosine_annealing', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--annealing_period', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)


    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)