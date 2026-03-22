"""
Week 2 - CNN 模型架构
MedicalCNN：专为医学图像设计的卷积神经网络

架构哲学：
  Block 1：检测基础特征（边缘、角点）
  Block 2：组合成纹理和形状
  Block 3：识别复杂医学结构
  Block 4：高层语义特征
  分类头：做出最终诊断决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalCNN(nn.Module):

    def __init__(self, num_classes=2):
        super(MedicalCNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1：基础特征检测（1 → 32 通道）
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # Conv2d原理：用32个3×3滤波器在图像上卷积
            # 每个滤波器学会检测一种模式（横边、竖边、斜边等）
            nn.BatchNorm2d(32),   # 批归一化：稳定训练，防止梯度消失
            nn.ReLU(inplace=True),  # 激活函数：f(x) = max(0,x)，引入非线性
            nn.MaxPool2d(2, 2),   # 最大池化：224→112，保留最显著特征

            # Block 2：纹理和形状识别（32 → 64 通道）
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 112→56

            # Block 3：复杂模式整合（64 → 128 通道）
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 56→28

            # Block 4：高层医学特征（128 → 256 通道）
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化：28×28 → 1×1
        )

        # 分类头：把 256 个特征映射到 2 个类别
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),       # 随机丢弃 50% 神经元，防止过拟合
            nn.Linear(256, 128),   # 全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # 最终输出：[正常概率, 癌症概率]
        )

    def forward(self, x):
        """前向传播：数据从输入流向输出"""
        x = self.features(x)        # 特征提取
        x = x.view(x.size(0), -1)  # 展平：(batch, 256, 1, 1) → (batch, 256)
        x = self.classifier(x)      # 分类
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MedicalCNN(num_classes=2)
    print(f"模型参数量：{model.count_parameters():,}")

    # 测试前向传播
    test_input = torch.randn(4, 1, 224, 224)  # batch=4张图
    output = model(test_input)
    print(f"输入形状：{test_input.shape}")
    print(f"输出形状：{output.shape}")  # 应该是 (4, 2)
    print("模型结构测试通过！")