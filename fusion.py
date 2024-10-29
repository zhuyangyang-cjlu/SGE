import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageFusion(nn.Module):
    def __init__(self, feature_size, shot, aug_shot, aug_support):
        super(ImageFusion, self).__init__()
        self.feature_size = feature_size
        self.shot = shot
        self.aug_shot = aug_shot
        self.aug_support = aug_support
        # 初始化权重
        original_weights = torch.full((shot,), 1 / shot)

        augmented_weights = torch.full((aug_shot,), 1/ aug_shot)
        self.weights = nn.Parameter(torch.cat((original_weights, augmented_weights)))

    def forward(self, im_feature):
        # 计算总的支持样本数
        total_support = self.aug_support * (self.shot + self.aug_shot)

        # 归一化权重，使其和为1
        normalized_weights = F.softmax(self.weights, dim=0)

        # 获取原始特征和增强特征
        original_features = im_feature[:self.shot, :]
        augmented_features = im_feature[self.shot:, :]

        # 计算加权特征
        weighted_original_features = original_features * normalized_weights[:self.shot].unsqueeze(1)
        weighted_augmented_features = augmented_features * normalized_weights[self.shot:].unsqueeze(1)

        # 将所有加权特征相加
        fused_features = torch.cat([weighted_original_features, weighted_augmented_features], dim=0).sum(dim=0)

        # 扩大到 aug_support * (shot + aug_shot)
        expanded_fused_features = fused_features.repeat(self.aug_support, 1)

        return expanded_fused_features

