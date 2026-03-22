"""
Week 1: Traditional Computer Vision Pipeline
课程：FURP 2026 - Cancer Detection Using AI
内容：合成数据生成 + 数学特征提取 + 随机森林分类
"""

from .create_synthetic_data import create_synthetic_mammograms, visualize_samples
from .feature_extraction import extract_comprehensive_features, extract_all
from .classifier import build_and_evaluate

__all__ = [
    'create_synthetic_mammograms',
    'visualize_samples',
    'extract_comprehensive_features',
    'extract_all',
    'build_and_evaluate'
]