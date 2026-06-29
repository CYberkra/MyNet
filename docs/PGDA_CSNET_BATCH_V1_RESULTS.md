# PGDA-CSNet 批量仿真实验总结

## 批次结果

**仿真构建**: 24/24 全部成功 ✅
**产出的窗口**: 186 个仿真 + 78 个实测 = 264 窗口

| 模型 | Line9 MAE | Line9 Pick Rate |
|------|:---------:|:---------------:|
| v1.9D 冻结 | **3.768** | 0.513 |
| + 186 sim 混合训练 | 5.975 | **0.962** |

## 规律

仿真数据越多 → pick rate 越高 → 但 MAE 也略增。
本质原因：仿真和实测的域差仍然存在（~0.4 系数差距），模型学到的特征不完全适用于 holdout。

## 当前产出

| 内容 | 位置 |
|------|------|
| 6 个场景 × 4 变体 = 24 个仿真 | `uavgpr_simlab/workspace/pgda_batch_v1_3060/models/` |
| 186 个训练窗口 | `data/simulation_pretrain_v1/windows/` |
| 混合训练 best checkpoint | `outputs/run_gpu_v1_16_batch186_mixed/checkpoint_best.pt` |
| 实验总结 | `docs/PGDA_CSNET_V1_13_EXPERIMENT_SUMMARY.md` |
| 所有对比图 | `C:\Users\17844\Desktop\PGDA_v1_results\` |

## 现状

v1.9D frozen baseline 仍然是当前最好的单模型。**仿真数据可以提升 pick rate（从 51% 到 96%），但还无法降低 MAE。** confidence abstention 是把二者结合起来最现实的方案。
