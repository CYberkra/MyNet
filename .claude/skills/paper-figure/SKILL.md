---
name: paper-figure
description: 论文图表生成——自动从实验输出生成对比图、训练曲线、B-scan overlay。Use when user says "论文图", "论文图表", "figure", "画图".
---
# paper-figure: 论文图表生成

## 使用方式
```
/paper-figure --type comparison
/paper-figure --type training_curve --run outputs/run_gpu_xxx
/paper-figure --type bscan_overlay --line Line9
/paper-figure --type architecture
```

## 图表类型

### comparison（实验对比柱状图）
从多个实验的 metrics CSV 提取指标，生成：
- MAE 对比柱状图（含 error bar 如有多种子）
- Pick Rate 对比
- IoU 对比
- 输出: PNG 300dpi + 矢量 SVG

### training_curve（训练收敛曲线）
从 history.json 读取 loss 数据：
- train_loss vs epoch（实线）
- val_loss vs epoch（虚线，如有）
- 标注最佳 epoch
- 输出: PNG 300dpi

### bscan_overlay（B-scan 叠加对比）
从评估输出的 preview 图像中提取：
- 原始 B-scan
- 预测 clean B-scan
- 杂波图
- Ground truth overlay
- 三列并排布局

### architecture（网络架构图）
生成 PGDA-CSNet 架构示意图（需 matplotlib + matplotlib-venn 或手动绘制）

## 格式规范
- 分辨率: 300 DPI
- 字体: Times New Roman / 宋体（中英文双版）
- 尺寸: 单栏 85mm, 双栏 170mm
- 配色: 色盲友好（viridis/cividis）
- 标注: 中英文双语标题

## 输出目录
`reports/figures/<实验名>/`
