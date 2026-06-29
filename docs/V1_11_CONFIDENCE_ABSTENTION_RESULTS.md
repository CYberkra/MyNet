# PGDA-CSNet v1.11 — Confidence Abstention 首轮结果

**日期**: 2026-06-27  
**状态**: 阶段稳定，可以作为 v1 配套方法继续推进  
**模型**: frozen v1.9D MambaVision hybrid seed-1902  
**数据**: corrected v1.4 terrain direction, 5 lines

---

## 1. 实验设计

在 frozen v1.9D 基础上，不修改模型，只在推理后增加 confidence scoring + abstention：

### 提取的特征（per-trace, 25 个）
- `path_prob`: DP 路径概率
- `presence_prob`: 界面存在概率
- `mask_center_agreement`: mask 与 center head 的一致性
- `local_contrast`: DP pick 位置的局部对比度
- `segment_len`: 连续 pick 段长度
- `jump_ns`: 相邻 trace 间跳变
- `curvature_ns`: 曲率
- `p050_consistent`: P=0.50 阈值下的一致性
- `robust_agree_ns`: robust norm 视图的一致性
- 等

### 5 个候选分数
1. `score_path` — path_prob 本身
2. `score_presence` — presence_prob 本身
3. `score_p050_consistent` — P=0.50 一致性
4. `score_composite` — 6 因子复合分
5. `score_crossview` — composite × cross-view 一致性

### 评估协议
- **Leave-One-Line-Out**: 在 4 条源线上调阈值，在 1 条目标线上测试
- 多个 `min_source_coverage` 阈值扫描（0.20–0.90）
- 报告 MAE-coverage Pareto

---

## 2. 核心结果

### 2.1 LOO 汇总（min_source_coverage = 0.35, Phase 1 结果）

| Score | Baseline MAE | Abstained MAE | Δ MAE | Baseline Cov | Abstained Cov |
|-------|:-----------:|:-------------:|:-----:|:-----------:|:-------------:|
| score_composite | 1.083 | 0.735 | **-0.349** | 0.878 | 0.388 |
| score_crossview | 1.083 | 0.751 | **-0.332** | 0.878 | 0.380 |
| score_path | 1.083 | 0.813 | **-0.270** | 0.878 | 0.455 |
| score_presence | 1.083 | 0.890 | **-0.193** | 0.878 | 0.480 |
| score_p050_consistent | 1.083 | 1.020 | -0.064 | 0.878 | 0.856 |

### 2.2 Pareto 最优点（Phase 2 扫描结果）

#### 覆盖率 ≥ 50% 的最佳点
| Score | MAE | Coverage | Δ MAE | min_cov |
|-------|:---:|:--------:|:-----:|:-------:|
| **score_path** | **0.809** | **0.525** | **-0.274** | 0.50 |
| score_composite | 0.905 | 0.588 | -0.178 | 0.50 |
| score_crossview | 0.914 | 0.598 | -0.169 | 0.50 |

#### 覆盖率 ≥ 70% 的最佳点
| Score | MAE | Coverage | Δ MAE | min_cov |
|-------|:---:|:--------:|:-----:|:-------:|
| **score_composite** | **0.950** | **0.794** | **-0.133** | 0.80 |
| score_path | 0.950 | 0.789 | -0.133 | 0.80 |

#### 加权最优（MAE × (1.5 - coverage)）
| Score | MAE | Coverage | min_cov |
|-------|:---:|:--------:|:-------:|
| score_composite | 0.950 | 0.794 | 0.80 |

---

## 3. 关键发现

### ✅ 有效的
1. **Confidence scoring 确实能改善 MAE** — 所有分数在多数配置下都降低了 MAE
2. **score_path 在 50% 覆盖率下最优** — MAE 从 1.083 降到 0.809（-25.3%）
3. **score_composite 在高覆盖率下最优** — 79.4% 覆盖率下 MAE 仍降 12.3%
4. **不需要重新训练** — 在 frozen v1.9D 上直接应用，工程上零成本
5. **source-tuned 阈值迁移有效** — 在源线调的阈值能迁移到目标线

### ⚠️ 需要注意的
1. **MAE 改善幅度中等** — 最好 -0.274 ns（-25%），不是数量级突破
2. **高覆盖率下改善有限** — 要保持 80% 覆盖率，MAE 只能降 ~13%
3. **Line9 holdout 上效果需单独验证** — LOO 平均 ≠ Line9 单线
4. **当前阈值是 heuristic** — 没有用 learnable abstention

---

## 4. 产出文件

所有结果在 `outputs/v11_frozen_v19d_confidence/`：

| 文件 | 内容 |
|------|------|
| `trace_confidence_features.csv` | 9456 traces × 25 特征 |
| `score_threshold_pareto.csv` | 阈值扫描全量 |
| `source_tuned_leave_one_line_abstention.csv` | 逐线 LOO |
| `pareto_sweep_results.csv` | 多 min_cov 扫描 |
| `pareto_sweep_lolo.png` | MAE-Coverage Pareto 图 |

eval 输出在 `outputs/eval_v11_frozen_*` 目录。

---

## 5. 与 frozen baseline 的关系

| 方案 | MAE (LOO avg) | Coverage | 需要重训练 | 备注 |
|------|:-----------:|:--------:|:---------:|------|
| frozen v1.9D baseline | 1.083 | 0.878 | 否 | 当前锁定基线 |
| + robust norm (P0.10) | ~1.221 | ~0.894 | 否 | 五线平均退化 |
| + score_path (50% cov) | 0.809 | 0.525 | 否 | **最佳 MAE** |
| + score_composite (80% cov) | 0.950 | 0.794 | 否 | **最佳平衡** |

---

## 6. 阶段结论

**v1.11 confidence abstention 已验证为有效方向**：

- 证据：在 frozen v1.9D 上，无需重训练，LOO MAE 平均降低 13–25%
- 方向确认：confidence scoring 是正确的下一步
- 工程就绪：特征提取 + 阈值调优 + Pareto 分析 pipeline 已完成
- 下一步建议：
  1. 可视化逐线 pick/reject 结果
  2. 考虑 learnable abstention（比固定阈值更灵活）
  3. 在 Line9 holdout 上单独验证最优配置
  4. 如果效果稳定，作为 v1 论文的方法贡献之一
