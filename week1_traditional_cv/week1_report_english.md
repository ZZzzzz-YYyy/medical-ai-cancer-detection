# Week 1 Report: Traditional Computer Vision for Cancer Detection

**Course:** FURP 2026 — Cancer Detection Using Artificial Intelligence  
**Instructor:** Prof. Elio Espejo  
**Date:** March 2026  

---

## 1. Introduction

Cancer remains one of the leading causes of death worldwide. Early detection is critical — survival rates for many cancers exceed 90% when caught at an early stage. However, radiologists face significant challenges: fatigue, high image volume, and the subtle nature of early-stage malignancies all contribute to a reported miss rate of 20–30% for early cancers.

This week, we explored **traditional computer vision** as a baseline approach to automated cancer detection in medical images. Rather than using modern deep learning, we manually engineered mathematical features from synthetic mammograms and trained a classical machine learning classifier. This exercise establishes both a performance baseline and a conceptual foundation for the deep learning methods introduced in Week 2.

---

## 2. Background: What Makes Medical Images Challenging

A medical image is mathematically a 2D discrete function:

$$I: \mathbb{Z}^2 \rightarrow [0, 255]$$

Each pixel coordinate $(x, y)$ maps to an 8-bit integer representing brightness. A standard 256×256 grayscale image contains 65,536 individual values.

The core challenge is that **cancer tissue and dense normal tissue can appear visually similar**. Differences are often subtle variations in brightness, texture, or shape that are difficult to detect consistently — especially across hundreds of images per day.

---

## 3. Methodology

### 3.1 Synthetic Data Generation

Since real clinical data requires ethics approval and is difficult to obtain, we generated **500 synthetic mammograms** using mathematical models of tissue structure.

**Normal tissue** was modelled using a Gaussian distribution plus a trigonometric texture pattern:

$$I(x, y) = \mathcal{N}(\mu=120,\ \sigma=25) + 20\sin\!\left(\frac{x}{30}\right)\cos\!\left(\frac{y}{25}\right) + \mathcal{N}(0, 10)$$

The three terms represent:
- Background tissue intensity (Gaussian, mean 120)
- Fibrous tissue structure (trigonometric oscillation)
- Quantum noise inherent to X-ray imaging

**Cancerous masses** were added to 30% of images using an irregular circular region:

$$\text{if } dx^2 + dy^2 < r^2: \quad I[c_y + dy,\ c_x + dx] \mathrel{+}= 60 + \mathcal{N}(0, 5)$$

The $+60$ brightness increment simulates the higher density of malignant tissue, and the Gaussian perturbation $\mathcal{N}(0,5)$ creates irregular boundaries — a key radiological marker of malignancy.

**Dataset statistics:**
| Class | Count | Percentage |
|-------|-------|------------|
| Normal | ~350 | ~70% |
| Cancer | ~150 | ~30% |
| **Total** | **500** | **100%** |

---

### 3.2 Feature Extraction

Raw pixel values contain too much redundant information for classical classifiers. We extracted **19 mathematical features** per image, grouped into five categories.

#### 3.2.1 Statistical Features (Probability Theory)

These capture the overall distribution of pixel intensities:

| Feature | Formula | Intuition |
|---------|---------|-----------|
| Mean | $\mu = \frac{1}{n}\sum x_i$ | Average brightness |
| Std Dev | $\sigma = \sqrt{\frac{\sum(x_i-\mu)^2}{n}}$ | Brightness variability |
| Skewness | $\frac{1}{n}\sum\left(\frac{x_i-\mu}{\sigma}\right)^3$ | Distribution asymmetry |
| Kurtosis | $\frac{1}{n}\sum\left(\frac{x_i-\mu}{\sigma}\right)^4$ | Distribution tail weight |
| IQR | $Q_{75} - Q_{25}$ | Robust spread measure |

Cancer images tend to have **higher mean, higher standard deviation**, and **higher kurtosis** due to the bright dense mass against relatively uniform background.

#### 3.2.2 Edge Features (Discrete Calculus)

Edges correspond to tissue boundaries. We used the **Sobel operator** to approximate image gradients:

$$G_x \approx \frac{\partial I}{\partial x} = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I$$

$$G_y \approx \frac{\partial I}{\partial y} = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$

$$G = \sqrt{G_x^2 + G_y^2}, \qquad \theta = \arctan\!\left(\frac{G_y}{G_x}\right)$$

The operator slides across the entire image (256×256 = 65,536 convolution operations), producing an **edge magnitude map**. From this we derived edge density, mean edge strength, and edge variance.

#### 3.2.3 Texture Features (Local Binary Patterns)

The **Local Binary Pattern (LBP)** operator encodes local texture by comparing each pixel to its 8 neighbours:

$$\text{LBP}(x, y) = \sum_{n=0}^{7} s(g_n - g_c) \cdot 2^n, \qquad s(u) = \begin{cases} 1 & u \geq 0 \\ 0 & u < 0 \end{cases}$$

where $g_c$ is the centre pixel and $g_n$ are the surrounding neighbours. The resulting 8-bit code describes the local texture pattern. A **normalised histogram** of LBP codes over the entire image yields two summary features:

- **Texture Uniformity:** $U = \sum_k h_k^2$ (high = regular texture)
- **Texture Entropy:** $E = -\sum_k h_k \log_2 h_k$ (high = complex/irregular texture)

Cancer regions exhibit **higher entropy** due to their disordered cellular structure.

#### 3.2.4 Morphological Features (Shape Analysis)

After Otsu thresholding to binarise the image, we extracted shape descriptors from the largest contour:

$$\text{Circularity} = \frac{4\pi A}{P^2}$$

where $A$ is the contour area and $P$ is its perimeter. A perfect circle gives circularity = 1.0; irregular shapes give values below 1.0. Malignant masses typically have **lower circularity** (0.4–0.7) compared to benign lesions (0.8–0.95).

Additional shape features: **aspect ratio** ($w/h$) and **solidity** ($A / A_\text{hull}$).

#### 3.2.5 Frequency Features (Fourier Analysis)

The **2D Discrete Fourier Transform** decomposes the image into spatial frequency components:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) \cdot e^{-2\pi i\left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

After applying `fftshift` to centre the zero-frequency component, we computed:
- **Low-frequency energy** (within radius $r/4$): overall structure and shape
- **High-frequency energy** (beyond radius $r/2$): fine texture and sharp edges

Cancer images tend to have **higher high-frequency energy** due to the sharp, irregular boundaries of malignant masses.

---

### 3.3 Classification

#### Feature Standardisation

Before training, all features were standardised using z-score normalisation:

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

This ensures features with vastly different scales (e.g., FFT energy $\sim 10^7$ vs. circularity $\sim 0$–$1$) contribute equally to the classifier.

#### Random Forest

We trained a **Random Forest** with 100 decision trees. Each tree is built by:

1. Sampling a random subset of training data (bootstrap)
2. At each node, selecting the feature $f$ and threshold $t$ that maximises **information gain**:

$$\text{IG}(S, f, t) = H(S) - \frac{|S_L|}{|S|}H(S_L) - \frac{|S_R|}{|S|}H(S_R)$$

where entropy $H(S) = -\sum_c p_c \log_2 p_c$.

The final prediction is determined by **majority voting** across all 100 trees, reducing variance and improving robustness compared to a single decision tree.

**Hyperparameters:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 100 | Balance between performance and speed |
| `max_depth` | 10 | Prevent overfitting |
| `min_samples_split` | 5 | Minimum node size |
| `class_weight` | `'balanced'` | Compensate for class imbalance |

---

## 4. Results

### 4.1 Classification Performance

| Metric | Value | Clinical Meaning |
|--------|-------|-----------------|
| Accuracy | 100% | Overall correct predictions |
| Sensitivity | 100% | Cancer cases correctly identified |
| Specificity | 100% | Normal cases correctly identified |
| F1 Score | 1.000 | Harmonic mean of precision and recall |
| AUC-ROC | 1.000 | Overall discriminative ability |

### 4.2 Top 5 Most Important Features

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | Mean Intensity | ~0.25 |
| 2 | Standard Deviation | ~0.18 |
| 3 | High Frequency Energy | ~0.14 |
| 4 | Edge Strength | ~0.12 |
| 5 | Texture Entropy | ~0.10 |

---

## 5. Critical Analysis

### 5.1 Why 100% Accuracy is a Warning Sign

Perfect accuracy on synthetic data does **not** imply real-world performance. This result reflects a fundamental limitation of our experimental design: the synthetic cancer masses are highly distinctive — a bright circular blob against relatively uniform tissue. The classifier essentially needs to detect "is there an unusually bright region?", which is trivial given the features extracted.

In real clinical mammography:
- Cancers can be tiny, diffuse, or obscured by dense tissue
- Normal tissue can be highly heterogeneous
- Different imaging equipment introduces systematic variations
- Patient anatomy differs substantially between individuals

### 5.2 The Overfitting Risk

Cross-validation AUC close to 1.0 on synthetic data indicates the **data generation process is too simple**, not that the model is exceptionally powerful. Real datasets would likely reduce performance to AUC 0.75–0.90 for this approach.

### 5.3 Limitations of Hand-Crafted Features

Traditional computer vision requires human experts to anticipate which features are diagnostically relevant. Key limitations:

- Cannot discover unanticipated patterns
- Feature engineering is time-consuming and domain-specific
- May miss subtle, high-dimensional patterns that exist in pixel space but not in any single engineered feature

This motivates the transition to **convolutional neural networks** in Week 2, which learn feature representations automatically from data.

---

## 6. Conclusion

This week established a complete traditional computer vision pipeline for medical image analysis:

1. **Mathematical image generation** using Gaussian distributions and trigonometric functions to simulate realistic mammographic tissue structure
2. **Multi-domain feature extraction** capturing statistical, edge, texture, morphological, and frequency characteristics — compressing 65,536 pixel values into 19 discriminative numbers
3. **Random Forest classification** using information-theoretic tree construction and ensemble voting

While the pipeline achieved perfect accuracy on synthetic data, critical analysis reveals this reflects data simplicity rather than true clinical capability. This baseline nonetheless provides an important conceptual foundation: we now understand exactly which mathematical properties distinguish cancer from normal tissue, and why automated detection is both possible and challenging.

The key insight motivating deep learning (Week 2) is that **hand-crafted features are limited by human knowledge** — neural networks can automatically discover more powerful, task-specific representations directly from raw pixel data.

---

## 7. References

- Gonzalez, R. C., & Woods, R. E. (2017). *Digital Image Processing* (4th ed.). Pearson.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE TPAMI*, 24(7), 971–987.
- Litjens, G., et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60–88.

---

*Repository: `medical-ai-cancer-detection/week1_traditional_cv/`*  
*Environment: Python 3.10 | NumPy | OpenCV | scikit-learn | scikit-image*
