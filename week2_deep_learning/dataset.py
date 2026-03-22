"""
Week 2 - 数据加载器
把 numpy 图像数组转换为 PyTorch 可训练的格式
并添加数据增强（Data Augmentation）
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset：把图像+标签打包成可迭代的数据集

    数据增强的数学原理：
    - 随机旋转：模拟不同拍摄角度
    - 随机翻转：乳腺左右对称
    - 颜色抖动：模拟不同成像设备的亮度差异
    这些操作人工扩大了训练集，提高模型泛化能力
    """

    def __init__(self, images, labels, transform=None):
        self.images = images.astype(np.uint8)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 转换为 PIL Image（transforms 需要这个格式）
        image = Image.fromarray(image, mode='L')

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(np.array(image)).unsqueeze(0) / 255.0

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms():
    """
    训练集：有数据增强
    验证/测试集：无增强（要公平评估）
    """

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),           # 随机旋转 ±15°
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 小幅平移
        transforms.ColorJitter(brightness=0.2, contrast=0.2),      # 亮度/对比度抖动
        transforms.ToTensor(),                   # 转为 [0,1] 张量
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1,1]
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_transform