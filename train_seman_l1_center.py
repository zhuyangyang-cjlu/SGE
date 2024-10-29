import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import argparse
import os.path
from data.dataset import DatasetWithTextLabel, aug_Dataset_view_WithTextLabel,aug_DatasetWithTextLabel
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from fusion import ImageFusion
import visformer_vis
from torchvision.datasets import ImageFolder
from logger import loggers
from utils import Cosine_classifier, count_95acc, count_kacc, transform_train_224_cifar, transform_val_224_cifar, \
    transform_val_224, transform_train_224,mean_confidence_interval
import torchvision.transforms as transforms
import clip
from data.randaugment import RandAugmentMC
from data.dataloader import EpisodeSampler, MultiTrans,TESTEpisodeSampler,view_EpisodeSampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])

    parser.add_argument('--feat_size', type=str, default='3,224,224')

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--drop', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--step-size', type=int, default=2)

    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true', default=True)
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--aug_support', type=int, default=1)
    parser.add_argument('--model', type=str, default='visformer-t', choices=['visformer-t', 'visformer-t-84']) 
    parser.add_argument('--nlp_model', type=str, default='clip', choices=['clip', 'glove', 'mpnet'])
    parser.add_argument('--prompt_mode', type=str, default='spatial+channel', choices=['spatial', 'channel', 'spatial+channel'])
    parser.add_argument('--no_template', action='store_true')
    parser.add_argument('--eqnorm', action='store_true', default=True)
    parser.add_argument('--stage', type=float, default=3.2, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
    parser.add_argument('--projector', type=str, default='linear', choices=['linear', 'mlp', 'mlp3'])
    parser.add_argument('--avg', type=str, default='all', choices=['all', 'patch', 'head'])
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'adamw'])

    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--encoder_lr', type=float, default=1e-6)
    parser.add_argument('--init', type=str, default='checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth')
    parser.add_argument('--resume', type=str, default='checkpoint/miniImageNet/visformer-t/test/checkpoint_epoch_003_better_exp1.pth')
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--train_way', type=int, default=-1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--test_classifier', type=str, default='prototype', choices=['prototype', 'fc'])
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10)
    args = parser.parse_args()
    args.model_path = '../checkpoint/Swin-Tiny-{}.pth'.format(args.dataset)

    args.work_dir = 'Vit_{}_{}'.format(args.dataset,args.center)

    if os.path.exists(args.work_dir) is False:
        os.mkdir(args.work_dir)

    log = loggers(os.path.join(args.work_dir, 'train'))
    log.info(vars(args))

    # checkpoint and tensorboard dir
    # 检查点目录和tensorboard目录
    args.tensorboard_dir = 'tensorboard/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    args.checkpoint_dir = 'checkpoint/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

    # prepare training and testing dataloader
    # 准备训练和测试数据加载器
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_aug = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.CenterCrop((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    norm])
    if args.aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        norm])
    if args.rand_aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        RandAugmentMC(2, 10, args.image_size),
                                        transforms.ToTensor(),
                                        norm])

    test_aug = transforms.Compose([transforms.Resize(int(args.image_size * 1.1)),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   norm])
   # 如果args.aug_support大于1，则使用transforms.Compose()函数创建一个数据增强的转换，并将其赋值给aug，
    # 同时将MultiTrans函数传入aug，将aug的参数设置为args.aug_support-1，norm，
    # 并将aug赋值给test_aug，
    if args.aug_support > 1:
        aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                  # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  norm])
        test_aug = MultiTrans([test_aug] + [aug]*(args.aug_support-1))


    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    n_episodes = args.train_episodes
    args.train_way = args.way if args.train_way == -1 else args.train_way
    if n_episodes == -1:
        n_episodes = int(len(train_dataset) / (args.train_way * (args.shot + 15)))
    episode_sampler = EpisodeSampler(train_dataset.dataset.targets,
                                     n_episodes,
                                     args.train_way,
                                     args.shot + 15, fix_seed=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=8)
    
    num_classes = len(train_dataset.dataset.classes)

    aug_train_dataset = aug_Dataset_view_WithTextLabel(args.dataset, train_aug, split='aug_train')
    

    episode_sampler = EpisodeSampler(aug_train_dataset.dataset.targets,
                                     300,
                                     args.train_way,
                                     args.shot + 15, fix_seed=False)
    aug_train_loader = torch.utils.data.DataLoader(aug_train_dataset, batch_sampler=episode_sampler, num_workers=6)
    
    


    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split=args.split)
    episode_sampler = EpisodeSampler(test_dataset.dataset.targets, args.episodes, args.way, args.shot + 15)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=6)

    aug_test_dataset = aug_Dataset_view_WithTextLabel(args.dataset, test_aug, split=args.split)
    episode_sampler = EpisodeSampler(test_dataset.dataset.targets, args.episodes, args.way, args.shot + 15)
    aug_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=6)

    proto_center = torch.load('center_{}_vit_{}.pth'.format(args.dataset,args.center))[args.center]

#加载模型

    teacher, _ = clip.load("ViT-B/32", device='cuda:1')
    text_dim = 512
    teacher.context_length = args.text_length
    teacher.positional_embedding.data = teacher.positional_embedding.data[:args.text_length]
    for layer in teacher.transformer.resblocks:
        layer.attn_mask.data = layer.attn_mask.data[:args.text_length, :args.text_length]

    def test(text, student,H,test_loader,aug_test_loader, epoch,args):
        student.eval()
        H.eval()
        accs = []
        # 使用torch.no_grad()函数，不计算梯度
        with torch.no_grad():
            for episode,aug_episode in zip(test_loader,aug_test_loader):
                
                if args.aug_support == 1:
                    image = episode[0].cuda(args.gpu)
                    aug_image = aug_episode[0].cuda(args.gpu)
                    
                    glabels = episode[1].cuda(args.gpu)
                    labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)
                    image = image.view(args.way, args.shot + 15, *image.shape[1:])  #5，16张
                    aug_image = aug_image.view(args.way,args.shot + 15 , *aug_image.shape[1:])
                    #拼接
                    sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
                    # print(sup.shape)
                    #aug_sup, aug_que = aug_image[:, :args.shot].contiguous(), aug_image[:, args.shot:].contiguous() #aug_sup 1张
                    aug_sup, aug_que = aug_image[:, :args.shot+4].contiguous(), aug_image[:, args.shot:].contiguous() #aug_sup 15张
                    # print(aug_sup.shape)
                    sup = torch.cat([sup, aug_sup], dim=1)

                    sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])
                    # print(sup.shape)
                    # aug_sup, aug_que = aug_sup.view(-1, *aug_sup.shape[2:]), aug_que.view(-1, *aug_que.shape[2:])
                    # print(aug_sup.shape)
            
                    # sup = torch.cat([sup, aug_sup], dim=1)
                    # sup = sup.view(-1,3,224,224)

                    glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot]
                    glabels = glabels.contiguous().view(-1)
                    text_features = text[glabels]

                    # 创建一个包含重复张量的列表
                    repeated_features = [text_features for _ in range(6)]

                    # 使用torch.stack来拼接这些张量，dim=1表示在第2维上进行拼接
                    text_features = torch.stack(repeated_features, dim=1)
                    # text_features = torch.stack((text_features,text_features,text_features,text_features,text_features,text_features,text_features,
                    #                             text_features,text_features, text_features, text_features,text_features,text_features,text_features,
                    #                             text_features,text_features,text_features), dim=1)#拼接
                    text_features = text_features.view(-1, 512)
                    sup_list=[]
                    image = torch.cat([sup, aug_sup], dim=1) # 5,6,3,224,224
                    for i in range(args.train_way):
                        im=image[i,:]
                        im_h = H(im)
                        sup_list.append(im_h)
                    sup = torch.stack(sup_list, dim=0)



                    if args.prompt_mode == 'spatial':
                        text_features = student.t2i(text_features)
                        _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                    else:
                        _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args)
                    _, que_im_features = student(que)

                    if args.test_classifier == 'prototype':


                        # sup_fusion_list=[]
                        # im_features = sup_im_features.view(args.train_way, args.shot+5, -1)#文本加图像  5，6 ，feature
                        # for i in range(args.train_way):
                        #     im_feature=im_features[i,:]
                        #     # fusion = H(im_feature)
                        #     sup_fusion_list.append(im_feature)
                        # sup_im_features = torch.stack(sup_fusion_list, dim=0)

                        sup_im_features = sup_im_features.view(args.way, args.shot, -1).mean(dim=1)#要除以sup数量
                        # 计算测试图像特征和训练图像特征的余弦相似度
                        sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                        _, pred = sim.max(-1)
                    elif args.test_classifier == 'fc':
                        x_train = F.normalize(sup_im_features, dim=-1).cpu().numpy()
                        y_train = torch.arange(args.way).unsqueeze(-1).repeat(1, args.shot).view(-1).numpy()
                        # x_test = F.normalize(que_im_features, dim=-1).cpu().numpy()
                        x_test = que_im_features.cpu().numpy()
                        # 使用sklearn的LogisticRegression模型进行分类
                        from sklearn.linear_model import LogisticRegression
                        clf = LogisticRegression(penalty='l2',
                                                random_state=0,
                                                C=1,
                                                solver='lbfgs',
                                                max_iter=1000,
                                                multi_class='multinomial')
                        clf.fit(x_train, y_train)
                        # 预测测试图像特征的类别
                        pred = clf.predict(x_test)
                        pred = torch.tensor(pred).cuda(args.gpu)

                elif args.aug_support > 1:
                    image = torch.cat(episode[0]).cuda(args.gpu) 
                    glabels = episode[1].cuda(args.gpu)
                    labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)
                    image = image.view(args.aug_support, args.way, args.shot + 15, *image.shape[1:])
                    sup = image[:, :, :args.shot].contiguous().view(-1, *image.shape[3:])
                    que = image[0, :, args.shot:].contiguous().view(-1, *image.shape[3:])

                    glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot]
                    glabels = glabels.unsqueeze(0).repeat(args.aug_support, 1, 1).contiguous().view(-1)
                    text_features = text[glabels]
                    # text_features = student.t2i(text_features)
                    # _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                    if args.prompt_mode == 'spatial':
                        text_features = student.t2i(text_features)
                        _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                    else:
                        _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args)

                    _, que_im_features = student(que)

                    if args.test_classifier == 'prototype':
                        sup_im_features = sup_im_features.view(args.aug_support, args.way, args.shot, -1).mean(dim=0).mean(dim=1)
                        sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                        _, pred = sim.max(-1)
                    elif args.test_classifier == 'fc':
                        # 计算支持集图像特征的归一化
                        x_train = F.normalize(sup_im_features, dim=-1).cpu().numpy()
                        # 计算支持集图像特征的标签
                        y_train = torch.arange(args.way).unsqueeze(0).unsqueeze(-1).repeat(args.aug_support, 1, args.shot).view(-1).numpy()
                        # 计算Query图像特征的归一化
                        x_test = F.normalize(que_im_features, dim=-1).cpu().numpy()
                        # 使用逻辑回归模型进行分类
                        from sklearn.linear_model import LogisticRegression
                        clf = LogisticRegression(penalty='l2',
                                                random_state=0,
                                                C=1.0,
                                                solver='lbfgs',
                                                max_iter=1000,
                                                multi_class='multinomial')
                        clf.fit(x_train, y_train)
                        # 预测Query图像特征的标签
                        pred = clf.predict(x_test)
                        # 将预测结果转换为tensor
                        pred = torch.tensor(pred).cuda(args.gpu)

                # 计算准确率
                acc = labels.eq(pred).sum().float().item() / labels.shape[0]
                # 将准确率添加到accs列表中
                accs.append(acc)

        # 计算准确率的均值和置信区间
        m, h = mean_confidence_interval(accs)
        print(f'[Test epoch: {epoch}] [test acc: {m * 100:.2f} +- {h * 100:.2f}]')

        # 将准确率的均值和置信区间添加到日志中
        log.info(
            "[Test epoch: %d] [test acc: %f +- %f]"
            % (epoch, m * 100, h * 100)
        )

        # 返回准确率
        return m
        
    def get_text_feature(teacher, dataset, args):
        # 获取类别索引
        class_idx = dataset.dataset.classes
        # 获取索引到文本的映射
        idx2text = dataset.idx2text
        # 判断是否使用模板
        if args.no_template:
            # 若不使用模板，则文本为索引到文本的映射
            text = [idx2text[idx] for idx in class_idx]
        else:
            # 若使用模板，则文本为模板加上索引到文本的映射
            text = ['A photo of ' + idx2text[idx] for idx in class_idx]

        # 设置模型为评估模式
        teacher.eval()
        # 判断使用哪个模型
        if args.nlp_model == 'clip':
            # 若使用clip模型，则对文本进行tokenize，并转换为cuda设备
            text_token = clip.tokenize(text).cuda(args.gpu)
            # 判断文本长度是否为-1
            if args.text_length != -1:
                # 若不为-1，则截取文本长度
                text_token = text_token[:, :args.text_length]
            # 对文本进行编码，并转换为float类型
            with torch.no_grad():
                text_feature = teacher.encode_text(text_token)
                text_feature = text_feature.float()
        else:
            # 若使用其他模型，则对文本进行编码，并转换为cuda设备
            with torch.no_grad():
                text_feature = teacher.encode(text)
                text_feature = torch.tensor(text_feature).cuda(args.gpu)

        # 返回文本特征
        return text_feature
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_aug = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    norm])

    train_text = get_text_feature(teacher, train_dataset, args)
    test_text = get_text_feature(teacher, test_dataset, args)
    
    # 如果使用eqnorm，则对文本特征进行归一化
    if args.eqnorm:
        if args.nlp_model in ['mpnet', 'glove']:
            # the bert features have been normalized to unit length. use the avg norm of clip text features
            # bert特征已经归一化，使用clip文本特征的平均长度
            avg_length = 9.
        else:
            # clip文本特征的平均长度
            avg_length = (train_text ** 2).sum(-1).sqrt().mean().item()
        # 对训练集文本特征进行归一化
        train_text = F.normalize(train_text, dim=-1) * avg_length
        # 对测试集文本特征进行归一化
        # aug_train_text = F.normalize(aug_train_text, dim=-1) * avg_length
        test_text = F.normalize(test_text, dim=-1) * avg_length


    num_classes = len(train_dataset.dataset.classes)
    # 加载student模型
    student = visformer_vis.visformer_tiny(num_classes=num_classes)
    feature_dim = 384
    if 2 <= args.stage < 3:
        feature_dim = 192
    # 判断args.projector的值，并设置student.t2i
    if args.projector == 'linear':
        student.t2i = torch.nn.Linear(text_dim, feature_dim, bias=False)
    elif args.projector == 'mlp':
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, feature_dim, bias=False))
    elif args.projector == 'mlp3':
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, feature_dim, bias=False))

    # 判断args.prompt_mode的值，并设置student.t2i2和student.se_block
    if 'channel' in args.prompt_mode:
        student.t2i2 = torch.nn.Linear(text_dim, feature_dim, bias=False)
        student.se_block = torch.nn.Sequential(torch.nn.Linear(feature_dim*2, feature_dim, bias=True),
                                               torch.nn.Sigmoid(),
                                               torch.nn.Linear(feature_dim, feature_dim),
                                               torch.nn.Sigmoid(),)

    # 将student移动到cuda上
    # 将student模型从cpu转换到指定gpu上
    student = student.cuda(args.gpu)

    checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')

    student.load_state_dict(checkpoint['state_dict'])
    student.eval()
    feat_size=1
    feat_size_list = list(map(int, args.feat_size.split(',')))
    feature_size = 1
    for dim in feat_size_list:
        feat_size *= dim
    # H = ImageFusion(feat_size,num=args.shot+5, hidden_size=8192, drop_rate=args.drop).cuda(args.gpu)
    H = ImageFusion(feat_size,num=args.shot+5,attention_type='multi-head').cuda(args.gpu)
    # H = ImageFusion(args.feat_size,num=args.shot+5).cuda(args.gpu)
    optimizer = torch.optim.Adam(H.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    #标签索引
    if args.dataset == 'miniImageNet':
        args.val = 'dataset/miniImageNet/val'
        args.train = 'dataset/miniImageNet/base'
        train_dataset = ImageFolder(args.train, transform=transform_train_224)
        val_dataset = ImageFolder(args.val, transform=transform_val_224)
    elif args.dataset == 'FC100':
        args.val = 'dataset\FC100\\val'
        args.train = 'dataset\FC100\\base'
        train_dataset = ImageFolder(args.train, transform=transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        args.val = 'dataset\CIFAR-FS\cifar100\\val'
        args.train = 'dataset\CIFAR-FS\cifar100\\base'
        train_dataset = ImageFolder(args.train, transform=transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=transform_val_224_cifar)
    elif args.dataset == 'TieredImageNet':
        args.val = 'dataset\\tiered_imagenet\\val'
        args.train = 'dataset\\tiered_imagenet\\base'
        train_dataset = ImageFolder(args.train, transform=transform_train_224)
        val_dataset = ImageFolder(args.val, transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')
    idx_to_class = train_dataset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}


    best_acc = 0.

    for epoch in range(args.max_epoch):
        H.train()

        for idx, (episode, aug_episode) in enumerate(tqdm(zip(train_loader,aug_train_loader))):
            image = episode[0].cuda(args.gpu)  # way * (shot+15)
            image = image.view(args.train_way, args.shot + 15, *image.shape[1:])
            glabels = episode[1].cuda(args.gpu)
            sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()

            aug_image = aug_episode[0].cuda(args.gpu)
            # a_glabels = aug_episode[1].cuda(args.gpu)
            a_glabels = aug_episode[1]
            aug_image = aug_image.view(args.train_way, args.shot + 15, 3,224,224)
            aug_sup, aug_que = aug_image[:, :args.shot+4].contiguous(), image[:, args.shot+4:].contiguous()

            glabels = glabels.view(args.train_way, args.shot + 15)[:, :args.shot]
            glabels = glabels.contiguous().view(-1)
            # a_glabels = a_glabels.view(args.train_way, args.shot + 15)[:, :args.shot]
            # a_glabels = a_glabels.contiguous().view(-1)

            text = aug_episode[2]
            sup_list=[]
            image = torch.cat([sup, aug_sup], dim=1) # 5,6,3,224,224
            for i in range(args.train_way):
                im=image[i,:]
                im_h = H(im)
                sup_list.append(im_h)
            image = torch.stack(sup_list, dim=0)

            protos = torch.tensor(np.array([proto_center[idx_to_class[l.item()]] for l in glabels])).cuda(args.gpu)

            text_features = train_text[glabels]
            # 创建一个包含重复张量的列表
            repeated_features = [text_features for _ in range(6)]

            # 使用torch.stack来拼接这些张量，dim=1表示在第2维上进行拼接
            text_features = torch.stack(repeated_features, dim=1)

            image = image.view(-1,*image.shape[2:])

            text_features =text_features.view(-1,512)
            if args.prompt_mode == 'spatial':
                text_features = student.t2i(text_features)
                
                _, im_features = student.forward_with_semantic_prompt(image, text_features, args)
            else:
                _, im_features = student.forward_with_semantic_prompt_channel(image, text_features, args)

            im_features = im_features.view(args.train_way, args.shot+5, -1)#文本加图像  5，6 ，feature

            optimizer.zero_grad()  
            # 初始化总损失变量，用于计算平均损失
            total_loss = 0

            # 循环遍历批次中的每个元素
            for i in range(args.train_way):
                im_feature=im_features[i,:]
                proto=protos[i:]
                # fusion = H(im_feature)
                # 计算损失
                recon_loss = F.l1_loss(im_feature, proto)
                # 累加损失
                total_loss += recon_loss
                # 反向传播
            total_loss.backward()

            optimizer.step()

            # 计算并记录平均损失
            avg_loss = total_loss.item() / args.train_way
            log.info("[Epoch %d/%d] [avg recon loss: %f]" % (epoch, args.max_epoch, avg_loss))


        lr_scheduler.step()
        
        if (epoch + 1) % args.test_freq == 0:
            acc = test(test_text, student, H, test_loader,aug_test_loader, epoch, args)

        # 保存模型
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': H.state_dict(),
        }
        torch.save(checkpoint, args.checkpoint_dir + f'H_epoch_latest.pth')
        # 保存模型
        if (epoch + 1) % args.save_freq == 0:
            torch.save(checkpoint, args.checkpoint_dir + f'H_epoch_{epoch + 1:03d}.pth')
        # 保存最佳模型
        if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, args.checkpoint_dir + f'H_epoch_{epoch+1:03d}_best.pth')


# conda activate da-fusion 
# python train_seman_l1_center.py --exp=fusion
