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
from data.dataloader import EpisodeSampler, MultiTrans,TESTEpisodeSampler,view_EpisodeSampler,SharedClassSampler

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])
    parser.add_argument('--center_exp', default='center',
                        choices=['center', 'center_5shot'])         

    parser.add_argument('--feat_size', type=int, default=384)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--drop', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
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
    parser.add_argument('--resume', type=str, default='checkpoint/miniImageNet/visformer-t/test/checkpoint_epoch_003_better_exp1.pth')
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--aug_shot', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--test_classifier', type=str, default='prototype', choices=['prototype', 'fc'])
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--resume_H', type=str, default='None')
    args = parser.parse_args()

    H = ImageFusion(args.feat_size,args.shot,args.aug_shot,args.aug_support).cuda(args.gpu)
    optimizer = torch.optim.Adam(H.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    if args.resume_H is not None and os.path.isfile(args.resume_H):
        checkpoint = torch.load(args.resume_H, map_location=f'cuda:{args.gpu}')
        H.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'load H checkpoint at epoch {start_epoch}')
    print("H module weights:", H.weights)