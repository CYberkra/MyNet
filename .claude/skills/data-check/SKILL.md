---
name: data-check
description: 数据集完整性检查——index CSV、样本文件、标签维度、随机可视化。Use when user says "检查数据", "data integrity", "数据完整性".
---
# data-check: 数据集完整性检查

## 使用方式
```
/data-check data/simulation_pretrain_v2
/data-check data/measured/yingshan_v15
```

## 检查项目

### 1. Index CSV 完整性
读取 `window_index.csv`（或等效索引文件）：
- 无空行、无重复行
- 所有引用的文件路径存在
- line 列值在预期范围内
- trace_start/trace_end 合理

### 2. 样本文件检查
对每个样本验证：
- B-scan 文件 (.out.h5 或 .npy) 存在且大小合理（>1KB）
- Label 文件存在且维度匹配
- 随机抽 5 个样本加载并验证 shape

### 3. 数据分布统计
- 每条测线的样本数
- 正/负样本比例（如有标注）
- 数值范围（min/max/mean/std）

### 4. 异常检测
- 全零样本
- NaN/Inf 值
- 尺寸异常的样本
- 标签为空的样本

## 输出格式
```
📁 数据集检查: simulation_pretrain_v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Index CSV: 60 行, 0 重复
✅ 样本文件: 60/60 存在
✅ 维度: (128, 256) × 60
⚠️ 警告: case_000015 标签全零
📊 数值范围: [-0.12, 0.34], mean=0.002
✅ 无 NaN/Inf
```
