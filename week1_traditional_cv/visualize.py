"""
Week 1 + Week 2 完整可视化
包含：样本图像 / 特征分布 / 训练曲线 / 混淆矩阵
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120


# ─────────────────────────────────────────
# 1. 样本图像可视化
# ─────────────────────────────────────────
def plot_sample_images(images, labels, n=5):
    fig, axes = plt.subplots(2, n, figsize=(15, 6))
    fig.suptitle('Synthetic Mammogram Samples', fontsize=16, fontweight='bold', y=1.02)

    normal_idx = np.where(labels == 0)[0][:n]
    cancer_idx = np.where(labels == 1)[0][:n]

    for i, idx in enumerate(normal_idx):
        axes[0, i].imshow(images[idx], cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'Normal #{idx}', color='#1D9E75', fontsize=10)
        axes[0, i].axis('off')
        # 绿色边框
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor('#1D9E75')
            spine.set_linewidth(2)

    for i, idx in enumerate(cancer_idx):
        axes[1, i].imshow(images[idx], cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f'Cancer #{idx}', color='#E24B4A', fontsize=10)
        axes[1, i].axis('off')
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor('#E24B4A')
            spine.set_linewidth(2)

    axes[0, 0].set_ylabel('Normal', fontsize=12, color='#1D9E75', fontweight='bold')
    axes[1, 0].set_ylabel('Cancer', fontsize=12, color='#E24B4A', fontweight='bold')

    plt.tight_layout()
    plt.savefig('viz_1_sample_images.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("已保存：viz_1_sample_images.png")


# ─────────────────────────────────────────
# 2. 特征分布对比
# ─────────────────────────────────────────
def plot_feature_distributions(features_array, labels):
    feature_names = [
        'Mean Intensity', 'Std Intensity', 'Intensity Range',
        'Skewness', 'Kurtosis', 'Median Intensity', 'IQR',
        'Edge Density', 'Edge Strength', 'Edge Variance',
        'Texture Uniformity', 'Texture Entropy',
        'Circularity', 'Aspect Ratio', 'Solidity',
        'FFT Mean', 'FFT Std', 'Low Freq Energy', 'High Freq Energy'
    ]

    normal_features = features_array[labels == 0]
    cancer_features = features_array[labels == 1]

    # 选最有区分度的 9 个特征展示
    top_features = [0, 1, 7, 8, 11, 12, 14, 17, 18]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle('Feature Distribution: Normal vs Cancer', fontsize=16, fontweight='bold')

    for plot_idx, feat_idx in enumerate(top_features):
        ax = axes[plot_idx // 3][plot_idx % 3]

        normal_vals = normal_features[:, feat_idx]
        cancer_vals = cancer_features[:, feat_idx]

        ax.hist(normal_vals, bins=30, alpha=0.6, color='#1D9E75',
                label=f'Normal (n={len(normal_vals)})', density=True)
        ax.hist(cancer_vals, bins=30, alpha=0.6, color='#E24B4A',
                label=f'Cancer (n={len(cancer_vals)})', density=True)

        ax.set_title(feature_names[feat_idx], fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 标注均值
        ax.axvline(np.mean(normal_vals), color='#1D9E75',
                   linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(np.mean(cancer_vals), color='#E24B4A',
                   linestyle='--', linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig('viz_2_feature_distributions.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("已保存：viz_2_feature_distributions.png")


# ─────────────────────────────────────────
# 3. 训练曲线（模拟 RF 交叉验证学习曲线）
# ─────────────────────────────────────────
def plot_training_curves(features_array, labels):
    from sklearn.model_selection import learning_curve

    print("计算学习曲线（需要约30秒）...")

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight='balanced'
    )

    train_sizes, train_scores, val_scores = learning_curve(
        clf, features_array, labels,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Random Forest Learning Curves', fontsize=16, fontweight='bold')

    # 左图：AUC 学习曲线
    ax = axes[0]
    ax.plot(train_sizes, train_mean, 'o-', color='#185FA5',
            label='Training AUC', linewidth=2, markersize=6)
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color='#185FA5')
    ax.plot(train_sizes, val_mean, 's-', color='#E24B4A',
            label='Validation AUC', linewidth=2, markersize=6)
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color='#E24B4A')

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('AUC-ROC Score', fontsize=12)
    ax.set_title('AUC vs Training Size', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

    # 右图：特征重要性 Top 10
    ax2 = axes[1]
    clf_full = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight='balanced'
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_array)
    clf_full.fit(X_scaled, labels)

    feature_names = [
        'Mean', 'Std', 'Range', 'Skewness', 'Kurtosis',
        'Median', 'IQR', 'Edge Density', 'Edge Strength',
        'Edge Variance', 'Tex Uniformity', 'Tex Entropy',
        'Circularity', 'Aspect Ratio', 'Solidity',
        'FFT Mean', 'FFT Std', 'Low Freq', 'High Freq'
    ]

    importances = clf_full.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_names = [feature_names[i] for i in top_idx]
    top_imp   = importances[top_idx]

    colors = ['#185FA5' if i < 3 else '#5DCAA5' for i in range(10)]
    bars = ax2.barh(range(10), top_imp[::-1], color=colors[::-1], alpha=0.85)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(top_names[::-1], fontsize=10)
    ax2.set_xlabel('Feature Importance', fontsize=12)
    ax2.set_title('Top 10 Feature Importances', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, top_imp[::-1]):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('viz_3_training_curves.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("已保存：viz_3_training_curves.png")


# ─────────────────────────────────────────
# 4. 混淆矩阵 + ROC 曲线
# ─────────────────────────────────────────
def plot_confusion_and_roc(features_array, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, labels,
        test_size=0.2, random_state=42, stratify=labels
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight='balanced'
    )
    clf.fit(X_train_s, y_train)
    y_pred  = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Model Evaluation', fontsize=16, fontweight='bold')

    # 左图：混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Cancer'],
                yticklabels=['Normal', 'Cancer'],
                ax=ax, cbar=False,
                annot_kws={'size': 18, 'weight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # 标注指标
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ax.text(2.15, 0.5,
            f'Sensitivity: {sensitivity:.1%}\nSpecificity: {specificity:.1%}\n'
            f'TP={tp}  FN={fn}\nFP={fp}  TN={tn}',
            fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='#E6F1FB', alpha=0.8))

    # 右图：ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    ax2 = axes[1]
    ax2.plot(fpr, tpr, color='#185FA5', lw=2.5,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.fill_between(fpr, tpr, alpha=0.08, color='#185FA5')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5,
             alpha=0.5, label='Random Classifier')

    # 标注最优阈值点
    optimal_idx = np.argmax(tpr - fpr)
    ax2.scatter(fpr[optimal_idx], tpr[optimal_idx],
                color='#E24B4A', s=100, zorder=5,
                label=f'Optimal threshold = {thresholds[optimal_idx]:.2f}')

    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])

    plt.tight_layout()
    plt.savefig('viz_4_confusion_roc.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("已保存：viz_4_confusion_roc.png")


# ─────────────────────────────────────────
# 主函数：一键运行所有可视化
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from week1_traditional_cv.create_synthetic_data import create_synthetic_mammograms
    from week1_traditional_cv.feature_extraction import extract_all

    print("生成数据集...")
    images, labels = create_synthetic_mammograms(500)

    print("提取特征...")
    features_array = extract_all(images)

    print("\n[1/4] 绘制样本图像...")
    plot_sample_images(images, labels)

    print("\n[2/4] 绘制特征分布...")
    plot_feature_distributions(features_array, labels)

    print("\n[3/4] 绘制训练曲线...")
    plot_training_curves(features_array, labels)

    print("\n[4/4] 绘制混淆矩阵 + ROC...")
    plot_confusion_and_roc(features_array, labels)

    print("\n全部完成！共生成 4 张图片")