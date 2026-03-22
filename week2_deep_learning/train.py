"""
Week 2 - 训练流程
反向传播 + 梯度下降 + 早停机制
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from week2_deep_learning.cnn_model import MedicalCNN
from week2_deep_learning.dataset import MedicalImageDataset, get_transforms
from week1_traditional_cv.create_synthetic_data import create_synthetic_mammograms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")


def train_medical_cnn(images, labels, num_epochs=30):

    train_transform, val_transform = get_transforms()

    # 划分 训练60% / 验证20% / 测试20%
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)

    print(f"训练:{len(X_train)} | 验证:{len(X_val)} | 测试:{len(X_test)}")

    train_loader = DataLoader(
        MedicalImageDataset(X_train, y_train, train_transform),
        batch_size=16, shuffle=True)
    val_loader = DataLoader(
        MedicalImageDataset(X_val, y_val, val_transform),
        batch_size=16, shuffle=False)
    test_loader = DataLoader(
        MedicalImageDataset(X_test, y_test, val_transform),
        batch_size=16, shuffle=False)

    # 初始化模型
    model = MedicalCNN(num_classes=2).to(device)
    print(f"模型参数量：{model.count_parameters():,}")

    # 损失函数：交叉熵 L = -Σ y·log(p)
    criterion = nn.CrossEntropyLoss()

    # 优化器：Adam（自适应学习率）
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 学习率调度：验证损失不降则减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    print(f"\n{'Epoch':>5} | {'训练损失':>8} | {'训练准确':>8} | {'验证损失':>8} | {'验证准确':>8}")
    print("-" * 55)

    for epoch in range(num_epochs):

        # === 训练阶段 ===
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()           # 清空梯度
            output = model(data)            # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()                 # 反向传播（计算梯度）
            optimizer.step()               # 更新参数

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        # === 验证阶段 ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():  # 验证时不计算梯度，节省内存
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        scheduler.step(avg_val_loss)

        print(f"{epoch+1:5d} | {avg_train_loss:8.4f} | {train_acc:7.1f}% | "
              f"{avg_val_loss:8.4f} | {val_acc:7.1f}%")

        # 早停：验证准确率不提升则计数，达到上限则停止
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_medical_cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n早停触发：第 {epoch+1} 轮")
                break

    # === 测试阶段 ===
    model.load_state_dict(torch.load('best_medical_cnn.pth'))
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy    = accuracy_score(all_labels, all_preds)
    precision   = precision_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds)
    f1          = f1_score(all_labels, all_preds)
    auc         = roc_auc_score(all_labels, all_probs)
    tn = np.sum((np.array(all_preds)==0) & (np.array(all_labels)==0))
    fp = np.sum((np.array(all_preds)==1) & (np.array(all_labels)==0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n{'='*50}")
    print(f"         Week 2 CNN 测试结果")
    print(f"{'='*50}")
    print(f"准确率    : {accuracy*100:.1f}%")
    print(f"敏感性    : {sensitivity*100:.1f}%  ← 医学最重要！")
    print(f"特异性    : {specificity*100:.1f}%")
    print(f"F1 分数   : {f1:.3f}")
    print(f"AUC-ROC  : {auc:.3f}")
    print(f"最佳验证准确率: {best_val_acc:.1f}%")
    print(f"{'='*50}")

    return model


if __name__ == "__main__":
    images, labels = create_synthetic_mammograms(500)
    model = train_medical_cnn(images, labels, num_epochs=30)