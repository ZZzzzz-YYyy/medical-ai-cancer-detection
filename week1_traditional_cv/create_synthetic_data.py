"""
Week 1 - Step 1: 合成医学图像生成
数学原理：高斯分布 N(120,25) + 三角函数纹理 + 不规则肿块
"""

import numpy as np
import matplotlib.pyplot as plt

def create_synthetic_mammograms(num_samples=500):
    print(f"正在创建 {num_samples} 张合成医学图像...")

    np.random.seed(42)
    images, labels = [], []
    image_size = 256
    y, x = np.ogrid[:image_size, :image_size]

    for i in range(num_samples):
        # 正态分布模拟乳腺组织 N(120, 25)
        image = np.random.normal(120, 25, (image_size, image_size))

        # 三角函数模拟纤维纹理
        tissue_pattern = 20 * np.sin(x/30) * np.cos(y/25)
        image += tissue_pattern

        # 30% 概率生成癌症图像
        has_cancer = np.random.random() < 0.3

        if has_cancer:
            center_x = np.random.randint(40, image_size-40)
            center_y = np.random.randint(40, image_size-40)
            radius = np.random.randint(15, 25)

            for dy in range(-radius, radius):
                for dx in range(-radius, radius):
                    if dx**2 + dy**2 < radius**2:
                        irregularity = np.random.normal(0, 5)
                        if (0 <= center_y+dy < image_size and
                            0 <= center_x+dx < image_size):
                            image[center_y+dy, center_x+dx] += 60 + irregularity
            labels.append(1)
        else:
            if np.random.random() < 0.2:
                center_x = np.random.randint(30, image_size-30)
                center_y = np.random.randint(30, image_size-30)
                benign_radius = np.random.randint(5, 12)
                benign_mask = ((x - center_x)**2 + (y - center_y)**2) < benign_radius**2
                image[benign_mask] += 25
            labels.append(0)

        # 添加 X 光量子噪声
        noise = np.random.normal(0, 10, image.shape)
        image += noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        images.append(image)

        if (i + 1) % 100 == 0:
            print(f"已生成 {i+1}/{num_samples} 张图像")

    images = np.array(images)
    labels = np.array(labels)

    cancer_count = np.sum(labels)
    print(f"\n📊 数据集统计：")
    print(f"  总图像数：{len(images)}")
    print(f"  癌症图像：{cancer_count} ({cancer_count/len(labels)*100:.1f}%)")
    print(f"  正常图像：{len(labels) - cancer_count}")

    return images, labels


def visualize_samples(images, labels, n=4):
    """可视化正常和癌症样本"""
    fig, axes = plt.subplots(2, n, figsize=(12, 6))

    normal_idx = np.where(labels == 0)[0][:n]
    cancer_idx = np.where(labels == 1)[0][:n]

    for i, idx in enumerate(normal_idx):
        axes[0, i].imshow(images[idx], cmap='gray')
        axes[0, i].set_title('正常', color='green')
        axes[0, i].axis('off')

    for i, idx in enumerate(cancer_idx):
        axes[1, i].imshow(images[idx], cmap='gray')
        axes[1, i].set_title('癌症', color='red')
        axes[1, i].axis('off')

    plt.suptitle('合成乳腺 X 光图像样本', fontsize=14)
    plt.tight_layout()
    plt.savefig('week1_samples.png', dpi=150)
    plt.show()
    print("图像已保存为 week1_samples.png")


if __name__ == "__main__":
    images, labels = create_synthetic_mammograms(500)
    visualize_samples(images, labels)
