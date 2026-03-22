# ResearchProgram
# Cancer Detection Using Artificial Intelligence
**FURP 2026 | Applied Mathematics | Prof. Elio Espejo**

## 项目概述
用 AI 技术实现癌症早期检测，4周从传统计算机视觉到临床级AI系统。

## 进度
- [x] Week 1: 传统计算机视觉（特征提取 + 随机森林）
- [ ] Week 2: 深度学习（CNN）
- [ ] Week 3: 高级技术（迁移学习 + 不确定性量化）
- [ ] Week 4: 临床部署

## Weekly report
-- [Week1 CH-ZN](./week1_traditional_cv/week1_report_chinese.md) 
-- [Week1 EN](./week1_traditional_cv/week1_report_english.md)

## 环境安装
```bash
pip install -r requirements.txt
```

## 使用方法
```bash
cd week1_traditional_cv
python create_synthetic_data.py
python feature_extraction.py
python classifier.py
```

## 技术栈
Python | NumPy | OpenCV | scikit-learn | PyTorch