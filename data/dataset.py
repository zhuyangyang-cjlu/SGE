import torchvision
import os

# train_dataset_path = {
#         'miniImageNet': 'dataset/miniImageNet/base',
#         'tieredImageNet': 'dataset/tieredImageNet/base',
#         'CIFAR-FS': 'dataset/cifar100/base',
#         'FC100': 'dataset/FC100_hd/base',
#     }
#
# val_dataset_path = {
#         'miniImageNet': 'dataset/miniImageNet/val',
#         'tieredImageNet': 'dataset/tieredImageNet/val',
#         'CIFAR-FS': 'dataset/cifar100/val',
#         'FC100': 'dataset/FC100_hd/val',
#     }
#
# test_dataset_path = {
#         'miniImageNet': 'dataset/miniImageNet/novel',
#         'tieredImageNet': 'dataset/tieredImageNet/novel',
#         'CIFAR-FS': 'dataset/cifar100/novel',
#         'FC100': 'dataset/FC100_hd/novel',
#     }
#
# dataset_path = {
#         'miniImageNet': ['dataset/miniImageNet/base', 'dataset/miniImageNet/val', 'dataset/miniImageNet/novel'],
#         'tieredImageNet': ['dataset/tieredImageNet/base', 'dataset/tieredImageNet/val', 'dataset/tieredImageNet/novel'],
#         'CIFAR-FS': ['dataset/cifar100/base', 'dataset/cifar100/val', 'dataset/cifar100/novel'],
#         'FC100': ['dataset/FC100_hd/base', 'dataset/FC100_hd/val', 'dataset/FC100_hd/novel'],
# }

train_dataset_path = {
        'miniImageNet': '/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/base',
        'tieredImageNet': '/home/RAID-5/datasets/Tieredimagenet/train',
        'CUB': '/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/base',
        'CIFAR-FS': '/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/base',
        'FC100': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/base',
        'FC100_hd': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100_hd/base',
        'CropDisease': '/home/RAID-5/ZYY/Semanticprompt/dataset/CD-FSL/CropDisease/base'
    }

aug_train_dataset_path = {
        'miniImageNet': '/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/aug_base',
        'tieredImageNet': '/home/RAID-5/ZYY/da-fusion/dataset/tieredimagenet/aug_train',
        'CUB': '/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/base',
        'CIFAR-FS': '/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/aug_base',
        'FC100': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/aug_base',
        'FC100_hd': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100_hd/aug_base',
        'CropDisease': '/home/RAID-5/ZYY/Semanticprompt/dataset/CD-FSL/CropDisease/base'
    }

val_dataset_path = {
        'miniImageNet': '/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/val',
        'tieredImageNet': '/home/RAID-5/datasets/Tieredimagenet/val',
        'CUB': '/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/val',
        'CIFAR-FS': '/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/val',
        'FC100': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/val',
        'CropDisease': '/home/RAID-5/ZYY/Semanticprompt/dataset/CD-FSL/CropDisease/val'
    }
    
aug_val_dataset_path = {
        'miniImageNet': '/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/aug_val',
        'tieredImageNet': '/home/RAID-5/ZYY/Semanticprompt/dataset/tieredImageNet/aug_val',
        'CUB': '/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/aug_val',
        'CIFAR-FS': '/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/aug_val',
        'FC100': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/val',
        'CropDisease': '/home/RAID-5/ZYY/Semanticprompt/dataset/CD-FSL/CropDisease/val'
    }

test_dataset_path = {
        'miniImageNet': '/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/novel',
        'tieredImageNet': '/home/RAID-5/datasets/Tieredimagenet/test',
        'CUB': '/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/dataset/test',
        'CIFAR-FS': '/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/novel',
        'FC100': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/novel',
        'FC100_hd': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100_hd/novel',
        'EuroSAT': '/home/RAID-5/ZYY/Semanticprompt/dataset/CD-FSL/EuroSAT/2750',
        'CropDisease': '/home/RAID-5/ZYY/Semanticprompt/dataset/CD-FSL/CropDisease/novel',
        'cars': '/home/RAID-5/ZYY/Semanticprompt/dataset/cars/test',
    }
aug_test_dataset_path = {
        'miniImageNet': '/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/aug_novel',
        'tieredImageNet': '/home/RAID-5/ZYY/da-fusion/dataset/tieredimagenet/aug_novel',
        'CUB': '/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/aug_novel',
        'CIFAR-FS': '/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/aug_novel',
        'FC100': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/aug_novel',
        'FC100_hd': '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100_hd/novel',
        'EuroSAT': '/home/RAID-5/ZYY/Semanticprompt/dataset/CD-FSL/EuroSAT/2750',
        'CropDisease': '/home/RAID-5/ZYY/Semanticprompt/dataset/CD-FSL/CropDisease/novel'
    }
    
dataset_path = {
        'miniImageNet': ['/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/base', '/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/val', '/home/RAID-5/ZYY/Semanticprompt/dataset/miniImageNet/novel'],
        'tieredImageNet': ['/home/RAID-5/ZYY/Semanticprompt/dataset/tieredImageNet/base', '/home/RAID-5/ZYY/Semanticprompt/dataset/tieredImageNet/val', '/home/RAID-5/ZYY/Semanticprompt/dataset/tieredImageNet/novel'],
        'CUB': ['/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/base', '/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/val', '/home/RAID-5/ZYY/Semanticprompt/dataset/CUB/novel'],
        'CIFAR-FS': ['/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/base', '/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/val', '/home/RAID-5/ZYY/Semanticprompt/dataset/cifar100/novel'],
        'FC100': ['/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/base', '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/val', '/home/RAID-5/ZYY/Semanticprompt/dataset/FC100/novel'],
        'CropDisease': ['../dataset/CD-FSL/CropDisease/train'],
        'EuroSAT': ['../dataset/CD-FSL/EuroSAT/2750']
}



class DatasetWithTextLabel(object):
    def __init__(self, dataset_name, aug, split='test'):
        self.dataset_name = dataset_name
        if split == 'train':
            dataset_path = train_dataset_path[dataset_name]
        elif split == 'val':
            dataset_path = val_dataset_path[dataset_name]
        elif split == 'test':
            dataset_path = test_dataset_path[dataset_name]
        self.dataset = torchvision.datasets.ImageFolder(dataset_path, aug)
        self.idx2text = {}
        if dataset_name == 'miniImageNet' or dataset_name == 'tieredImageNet':
            with open('data/ImageNet_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, _, text = line.strip().split()
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        elif dataset_name == 'FC100':
            with open('data/cifar100_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, text = line.strip().split()
                    idx = idx.strip(':')
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        elif dataset_name == 'CIFAR-FS':
            for idx in self.dataset.classes:
                text = idx.replace('_', ' ')
                self.idx2text[idx] = text
        elif dataset_name == 'cars':
            for idx in self.dataset.classes:
                text = idx.replace('_', ' ')
                self.idx2text[idx] = text
        elif dataset_name == 'CUB':
            with open('data/classes.txt', 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        idx_str, text = parts
                        l = text

                        text = text.replace('_', ' ').strip('.')
                        text = text.strip('0123456789.')
                        self.idx2text[l] = text
				


    def __getitem__(self, i):
        image, label = self.dataset[i]
        
        text = self.dataset.classes[label]
        
        text = self.idx2text[text]
        # text prompt: A photo of a {label}
        text = 'A photo of a ' + text
        return image, label, text

    def __len__(self):
        return len(self.dataset)
    


from torchvision.datasets.folder import default_loader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image


class aug_DatasetWithTextLabel(object):
    def __init__(self, dataset_name, aug, split='test'):
        self.dataset_name = dataset_name
        self.idx2text = {}
        self.samples = []

        # 根据split参数确定数据集路径
        if split == 'aug_train':
            dataset_path = aug_train_dataset_path[dataset_name]
        elif split == 'val':
            dataset_path = aug_val_dataset_path[dataset_name]
        elif split == 'test':
            dataset_path = aug_test_dataset_path[dataset_name]

        # 创建ImageFolder实例
        self.dataset = self._create_dataset_with_views(dataset_path, aug)
        self.samples = self.dataset.samples

        if dataset_name == 'miniImageNet' or dataset_name == 'tieredImageNet':
            with open('data/ImageNet_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, _, text = line.strip().split()
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        elif dataset_name == 'FC100':
            with open('data/cifar100_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, text = line.strip().split()
                    idx = idx.strip(':')
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        elif dataset_name == 'CIFAR-FS':
            for idx in self.dataset.classes:
                text = idx.replace('_', ' ')
                self.idx2text[idx] = text

    def __getitem__(self, i):
        image, label_index = self.samples[i]
        text_label = self.idx2text[label_index]
        #view_text = f'A photo of a {text_label} from the {view_name}'
        return image, label_index

    def __len__(self):
        return len(self.samples)
    
    def _create_dataset_with_views(self, dataset_path, aug):
        # 创建一个空的ImageFolder实例
        dataset = ImageFolder(root=dataset_path, transform=aug)
        dataset.samples = []  # 清空samples列表以准备添加新的样本

        # 遍历每个类别和视角，整合样本
        for class_name in sorted(os.listdir(dataset_path)):
            class_path = os.path.join(dataset_path, class_name)
            for view_name in sorted(os.listdir(class_path)):
                view_path = os.path.join(class_path, view_name)
                if os.path.isdir(view_path):
                    for fname in sorted(os.listdir(view_path)):
                        file_path = os.path.join(view_path, fname)
                        if os.path.isfile(file_path):  # 使用os.path.isfile来检查文件是否存在
                            # 添加样本到dataset.samples
                            dataset.samples.append((file_path, dataset.class_to_idx[class_name]))
                            # 更新idx2text字典
                            self.idx2text[dataset.class_to_idx[class_name]] = class_name.replace('_', ' ')
        transformed_samples = []
        for (file_path, class_index) in dataset.samples:
            image = Image.open(file_path).convert('RGB')  # 加载图像
            image = aug(image)  # 应用aug
            transformed_samples.append((image, class_index))

        # 更新dataset.samples为已转换的样本
        dataset.samples = transformed_samples

        return dataset


import os
from torchvision.datasets import ImageFolder

class aug_Dataset_view_WithTextLabel(object):
    def __init__(self, dataset_name, aug, split='test'):
        self.dataset_name = dataset_name
        self.idx2text = {}
        self.samples = []

        # 根据split参数确定数据集路径
        if split == 'aug_train':
            dataset_path = aug_train_dataset_path[dataset_name]
        elif split == 'val':
            dataset_path = aug_val_dataset_path[dataset_name]
        elif split == 'test':
            dataset_path = aug_test_dataset_path[dataset_name]

        # 创建ImageFolder实例
        self.dataset = self._create_dataset_with_views(dataset_path, aug)
        self.samples = self.dataset.samples

        # 动态生成 idx2text 字典
        for class_name, class_idx in self.dataset.class_to_idx.items():
            text = class_name.replace('_', ' ')
            self.idx2text[class_idx] = text

    def __getitem__(self, i):
        image, label_index = self.samples[i]
        if label_index not in self.idx2text:
            raise KeyError(f"Label index {label_index} not found in idx2text.")
        text_label = self.idx2text[label_index]
        view_text = f'A photo of a {text_label} '
        view_text = f'{text_label}'
        return image, label_index

    def __len__(self):
        return len(self.samples)
    
    def _create_dataset_with_views(self, dataset_path, aug):
        dataset = ImageFolder(root=dataset_path, transform=aug)
        dataset.samples = []

        # 获取所有类别的所有视角路径
        all_view_paths = []
        for class_name in sorted(os.listdir(dataset_path)):
            class_path = os.path.join(dataset_path, class_name)
            for view_name in sorted(os.listdir(class_path)):
                view_path = os.path.join(class_path, view_name)
                if os.path.isdir(view_path):
                    all_view_paths.append(view_path)

        # 遍历所有视角的所有图像
        for view_path in all_view_paths:
            view_name = os.path.basename(view_path)
            file_names = sorted(os.listdir(view_path))
            for file_name in file_names:
                file_path = os.path.join(view_path, file_name)
                if os.path.isfile(file_path):
                    class_name = os.path.basename(os.path.dirname(view_path))
                    dataset.samples.append((file_path, dataset.class_to_idx[class_name]))

        # 确保图像路径被正确加载为张量
        for i, (file_path, class_idx) in enumerate(dataset.samples):
            image = Image.open(file_path).convert('RGB')
            if aug:
                image = aug(image)
            dataset.samples[i] = (image, class_idx)

        return dataset




