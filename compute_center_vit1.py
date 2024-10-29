import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.utils.data
import os
import visformer_vis
import clip
from utils import cluster, transform_val_224_cifar, transform_val_224
from data.dataset import DatasetWithTextLabel, aug_DatasetWithTextLabel
import torchvision.transforms as transforms
def main():
    # 加载clip模型
    save_path = 'checkpoint/{}/{}/center_vit.pth'.format(args.dataset, args.exp)
    dir_path = os.path.dirname(save_path)

    # 检查文件夹是否存在
    if not os.path.exists(dir_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(dir_path)
    teacher, _ = clip.load("ViT-B/32", device='cuda:1')
    text_dim = 512
    teacher.context_length = args.text_length
    teacher.positional_embedding.data = teacher.positional_embedding.data[:args.text_length]
    for layer in teacher.transformer.resblocks:
        layer.attn_mask.data = layer.attn_mask.data[:args.text_length, :args.text_length]



    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_aug = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    norm])
    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    train_text = get_text_feature(teacher, train_dataset, args)
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

    data = {}
    batch_size = 128
    shuffle = True
    # train
    if args.dataset == 'miniImageNet':
        trainset = ImageFolder('dataset/miniImageNet/base', transform=transform_val_224)
    elif args.dataset == 'FC100':
        trainset = ImageFolder('dataset/FC100/base', transform=transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        trainset = ImageFolder('CIFAR-FS/cifar100/base', transform=transform_val_224_cifar)
    elif args.dataset == 'tieredImageNet':
        trainset = ImageFolder('../da-fusion/dataset/tieredimagenet/train', transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')
    
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8,
                              pin_memory=True)

    # 建立类别索引映射
    idx_to_class = trainset.class_to_idx

    idx_to_class = {k: v for v, k in idx_to_class.items()}

    # 遍历数据加载器获取图像和标签
    for images, labels in tqdm(train_loader):
        images = images.cuda(args.gpu)
        glabels = [idx_to_class[l.item()] for l in labels]  # 获取全局标签

        class_to_idx = {v: k for k, v in idx_to_class.items()}
        class_glabels = [class_to_idx[glabel] for glabel in glabels]

        text_features = train_text[class_glabels] # 根据标签获取文本特征


        with torch.no_grad():
            # 假设 forward_with_semantic_prompt_channel 需要一个标签列表和对应的文本特征
            _ , im_features = student.forward_with_semantic_prompt_channel(images, text_features, args)

        # 将特征存储到数据字典中
        for i, glabel in enumerate(glabels):
            print(glabel)
            if glabel in data:
                data[glabel].append(im_features[i].detach().cpu().numpy())
            else:
                data[glabel] = [im_features[i].detach().cpu().numpy()]


    center_mean = {}
    for k, v in data.items():
        center_mean[k] = np.array(v).mean(0)

    if args.dataset == 'TieredImageNet':
        data = {k: v[:700] for k, v in data.items()}
        center_cluster = cluster(data, len(data), 700)
    else:
        center_cluster = cluster(data, len(data), 600)

    torch.save({
        'mean': center_mean,
        'cluster': center_cluster,
    }, save_path)


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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--center', default='cluster',
                        choices=['mean', 'cluster'])
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
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
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--encoder_lr', type=float, default=1e-6)
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--resume', type=str, default='checkpoint/miniImageNet/visformer-t/test/checkpoint_epoch_003_better_exp1.pth')
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--test_classifier', type=str, default='prototype', choices=['prototype', 'fc'])
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=20)
    args = parser.parse_args()
    print(vars(args))
    main()
