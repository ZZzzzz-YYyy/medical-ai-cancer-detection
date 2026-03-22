"""
Week 1 - Step 2: 数学特征提取
特征类别：统计 | 边缘(Sobel) | 纹理(LBP) | 形态 | 频率(FFT)
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern


def extract_comprehensive_features(image):
    features = {}

    # === 1. 统计特征（概率论）===
    features['mean_intensity']   = np.mean(image)
    features['std_intensity']    = np.std(image)
    features['intensity_range']  = np.max(image) - np.min(image)

    norm = (image - features['mean_intensity']) / (features['std_intensity'] + 1e-7)
    features['skewness'] = np.mean(norm**3)   # 三阶矩：分布对称性
    features['kurtosis'] = np.mean(norm**4)   # 四阶矩：分布尖锐程度

    features['median_intensity'] = np.median(image)
    features['iqr'] = np.percentile(image, 75) - np.percentile(image, 25)

    # === 2. 边缘特征（Sobel 算子 / 离散梯度）===
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    features['edge_density']  = np.mean(edge_magnitude > 50)
    features['edge_strength'] = np.mean(edge_magnitude)
    features['edge_variance'] = np.var(edge_magnitude)

    # === 3. 纹理特征（局部二值模式 LBP）===
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 9))
    lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)

    features['texture_uniformity'] = np.sum(lbp_hist**2)
    features['texture_entropy']    = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))

    # === 4. 形态特征（形状分析）===
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest  = max(contours, key=cv2.contourArea)
        area     = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        # 圆度公式：4πA/P²（完美圆 = 1.0）
        features['circularity'] = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

        x, y, w, h = cv2.boundingRect(largest)
        features['aspect_ratio'] = w / h if h > 0 else 0

        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = area / hull_area if hull_area > 0 else 0
    else:
        features.update({'circularity': 0, 'aspect_ratio': 0, 'solidity': 0})

    # === 5. 频率特征（2D 傅里叶变换）===
    fft       = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    features['fft_mean'] = np.mean(magnitude)
    features['fft_std']  = np.std(magnitude)

    cy, cx = magnitude.shape[0]//2, magnitude.shape[1]//2
    y_c, x_c = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]

    low_r = min(cx, cy) // 4
    features['low_freq_energy']  = np.sum(magnitude[(x_c-cx)**2 + (y_c-cy)**2 <= low_r**2])

    high_r = min(cx, cy) // 2
    features['high_freq_energy'] = np.sum(magnitude[(x_c-cx)**2 + (y_c-cy)**2 >= high_r**2])

    return list(features.values())


def extract_all(images):
    """对所有图像批量提取特征"""
    print("开始提取特征...")
    features_list = []

    for i, img in enumerate(images):
        features_list.append(extract_comprehensive_features(img))
        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1}/{len(images)} 张图像")

    features_array = np.array(features_list)
    print(f"\n✅ 特征提取完成！")
    print(f"特征矩阵形状：{features_array.shape}")
    print(f"→ {features_array.shape[0]} 张图像，每张 {features_array.shape[1]} 个特征")
    return features_array


if __name__ == "__main__":
    from create_synthetic_data import create_synthetic_mammograms
    images, labels = create_synthetic_mammograms(500)
    features_array = extract_all(images)