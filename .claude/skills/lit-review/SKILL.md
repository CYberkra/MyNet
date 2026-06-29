---
name: lit-review
description: GPR 领域文献调研——搜索 IEEE/MDPI/arXiv，提取对比方法，生成结构化综述表。Use when user says "文献调研", "literature review", "相关工作", "找论文".
---
# lit-review: 文献调研

## 使用方式
```
/lit-review "unsupervised domain adaptation for GPR clutter suppression"
/lit-review "physics-guided neural network ground penetrating radar"
/lit-review --recent --year 2023-2026
```

## 搜索范围
1. **IEEE Xplore** — GPR 信号处理、杂波抑制、深度学习
2. **MDPI Remote Sensing** — GPR + DL 应用
3. **arXiv** — 预印本（physics-informed ML, domain adaptation）
4. **Google Scholar** — 综合搜索

## 提取信息
对每篇论文提取：
- 标题 / 作者 / 年份 / 期刊
- 方法类型（SVD/wavelet/RPCA/CNN/GAN/Physics-guided/DA）
- 数据类型（仿真/实测/混合）
- 关键指标（MAE/Pick Rate/SNR Improvement）
- 与本项目的对比点

## 输出格式
### 结构化文献表
| # | 论文 | 年份 | 方法 | 数据 | 指标 | vs PGDA-CSNet |
|---|------|------|------|------|------|:---:|
| 1 | Author et al. | 2024 | SVD+DL | 实测 | MAE=3.5 | PGDA更优 |
| 2 | ... | ... | ... | ... | ... | ... |

### 分类总结
- **传统方法**（SVD/wavelet/RPCA）: 优缺点
- **纯 DL 方法**（CNN/GAN）: 优缺点
- **Physics-guided 方法**: 优缺点
- **Domain Adaptation 方法**: 优缺点

### Gap 分析
指出当前文献中的空白，说明 PGDA-CSNet 的创新点。

## 使用 deep-research 底层
调用 `/deep-research` skill 做多源搜索，然后在此基础上结构化提取。
