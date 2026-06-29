---
name: exp-compare
description: 实验结果对比——从多个 metrics CSV 提取指标、与 baseline 对比、输出 Markdown 表格。Use when user says "对比结果", "compare experiments", "实验对比".
---
# exp-compare: 实验结果对比

## 使用方式
```
/exp-compare run_gpu_v3_pilot_mixed loo_Line9
/exp-compare run_gpu_v3_pilot_mixed --all-folds
```

## 参数
- 第一个参数: 实验名称前缀（匹配 outputs/ 下的目录）
- 第二个参数: 具体 fold（可选，默认显示所有匹配目录）
- `--all-folds`: 自动收集所有 fold 的结果

## 流程

### Step 1: 收集 metrics 文件
在 `outputs/` 下搜索匹配 `*<实验名>*` 的目录，找 `*_full_metrics.csv` 或 `history.json`。

### Step 2: 提取指标
从每个匹配目录提取：
- **训练指标** (from history.json): 最终 train_loss, 最佳 epoch, 收敛趋势
- **评估指标** (from *_full_metrics.csv): DP Center MAE, Pick Rate, Mean Center MAE, IoU@0.2/0.3/0.5

### Step 3: 基线对比
已知 baselines:
| Baseline | MAE | Pick Rate | 来源 |
|----------|-----|-----------|------|
| P0-3 Center Fusion | 3.268 | 0.562 | v3_pilot_mixed, 20% robust + 80% ensemble |
| SVD Traditional | ~4.5 | ~0.3 | 文献参考值 |
| CR-Net | ~3.8 | ~0.4 | 文献参考值 |

### Step 4: 输出 Markdown 表格
```
📊 实验对比: loo_Line9 (LOLO-CV)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
| 指标 | P0-3 Baseline | LOLO Line9 | 增益 |
|------|:---:|:---:|:---:|
| DP Center MAE | 3.268 | X.XXX | ↓ XX% |
| Pick Rate | 0.562 | X.XXX | ↑ XX% |
| IoU@0.3 | - | X.XXX | - |

训练收敛: epoch 1 (loss X.XX) → epoch 80 (loss 0.XX)
最佳 epoch: XX (val_loss=X.XX)
```

### Step 5: 智能建议
根据结果给出建议：
- MAE 改善 > 10%: "显著优于 baseline，可以考虑提交"
- MAE 改善 < 5%: "改善有限，建议检查数据质量或调整超参"
- MAE 恶化: "性能下降，需要排查训练过程"
