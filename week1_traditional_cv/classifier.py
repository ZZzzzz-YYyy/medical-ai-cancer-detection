"""
Week 1 - Step 3: 随机森林分类器
数学原理：信息熵 + 决策树集成投票
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)


def build_and_evaluate(features, labels):
    # 1. 划分数据集（80% 训练 / 20% 测试）
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels       # 保持癌症比例一致
    )
    print(f"训练集：{len(X_train)} 张 | 测试集：{len(X_test)} 张")

    # 2. 标准化：z = (x - μ) / σ
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("特征已标准化（零均值，单位方差）")

    # 3. 训练随机森林（100 棵决策树）
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'  # 处理类别不平衡
    )
    print("\n正在训练随机森林...")
    clf.fit(X_train_scaled, y_train)

    # 4. 预测
    y_pred  = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # 5. 计算所有评估指标
    accuracy    = accuracy_score(y_test, y_pred)
    precision   = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_proba)

    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "="*50)
    print("           第一周实验结果")
    print("="*50)
    print(f"准确率      : {accuracy*100:.1f}%")
    print(f"精确率      : {precision*100:.1f}%")
    print(f"敏感性(召回): {sensitivity*100:.1f}%  ← 医学最重要！")
    print(f"特异性      : {specificity*100:.1f}%")
    print(f"F1 分数     : {f1:.3f}")
    print(f"AUC-ROC    : {auc:.3f}")
    print("="*50)

    # 6. 交叉验证（检测是否过拟合）
    print("\n正在运行 5 折交叉验证...")
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        features, labels,
        cv=5,
        scoring='roc_auc'
    )
    print(f"各折 AUC: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"平均 AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 7. 特征重要性
    feature_names = [
        '均值亮度', '标准差', '亮度范围', '偏度', '峰度',
        '中位亮度', 'IQR', '边缘密度', '边缘强度', '边缘方差',
        '纹理均匀度', '纹理熵', '圆度', '长宽比',
        '凸度', 'FFT均值', 'FFT标准差', '低频能量', '高频能量'
    ]
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]

    print("\n🔍 Top 5 最重要特征：")
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1}. {feature_names[idx]:12s}: {importances[idx]:.4f}")

    return clf, scaler


if __name__ == "__main__":
    from create_synthetic_data import create_synthetic_mammograms
    from feature_extraction import extract_all

    images, labels = create_synthetic_mammograms(500)
    features_array = extract_all(images)
    clf, scaler = build_and_evaluate(features_array, labels)