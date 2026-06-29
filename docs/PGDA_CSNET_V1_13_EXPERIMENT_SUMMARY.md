# PGDA-CSNet 阶段性实验总结

**时间**: 2026-06-28  
**实验范围**: 架构搜索、X伪影分析、仿真-实测域适应

---

## 一、架构实验

| 架构 | Line9 MAE | 结论 |
|------|:---------:|------|
| **v1.9D (ConvNeXt+SSM+Stripe)** | **3.768** | **当前最优冻结模型** |
| v1.9D + v1.11 Confidence Abstention | **0.809** (50% cov) | 无需重训，推理增强 |
| SG-USSM (v1.9D + SGM) | 3.989~4.506 | SGM 过拟合训练线 |
| SG-USSM FixedMask | 4.135 | 固定掩码效果有限 |

### 结论
**v1.9D 是最优架构。** 所有 SGM 变体都无法在 holdout 超越它。瓶颈不是结构容量，是数据过少。

---

## 二、X形伪影分析

| 配置 | X程度 | 经过 |
|------|:----:|------|
| Ricker 32m PML10 | ❌ 严重 | 原始 |
| Gaussian 32m PML10 | ⚠️ 明显 | 换波形 |
| **Gaussian 48m PML20 (XG03)**  | **🔶 残留但不影响训练** | 加宽域+PML |
| Gaussian 64m PML30 | 🔶 与XG03相同 | 无进一步改善 |
| HORIPML → MRIPML | 🔶 无改善 | PML配方无关 |
| CFS alpha参数 | ❌ 恶化 | 不对症 |
| **2.5D** | ✅ 减少 | 但慢20x，不可行 |

### 结论
**X形伪影是2D圆柱波近似的固有现象，不是配置问题。** XG03已经是最优折中。

---

## 三、仿真-实测域适应

### 域差分析

| 指标 | Sim | Real | 差距 |
|------|:---:|:----:|:----:|
| 频谱相关 | 1.0 | 0.577 | sim高频能量太多 |
| 振幅分布 | std=7.66 | std=0.17 | sim动态范围太大 |
| 训练/holdout线自相关 | — | 0.999 | 实测数据自身一致 |

### 尝试的策略

| 策略 | Line9 MAE | Line9 PR | 训练线 | 备注 |
|------|:---------:|:--------:|:------:|------|
| **v1.9D baseline** | **3.768** | 0.513 | ✅ 正常 | 当前锁定 |
| Sim pretrain + finetune (best) | 3.140 | **0.825** | ❌ 退化 | warm-start路径错误忽略 |
| Sim pretrain + finetune v2 (correct) | 3.140 | 0.825 | ❌ 退化 | 训练线退化严重 |
| Mixed real+sim (v3) | 7.723 | 0.822 | ❌ 退化 | 域差太大 |
| Spectral aug + mixed (v4) | 5.621 | **0.924** | ❌ 退化 | PR最高但MAE差 |

### 核心瓶颈
1. **仿真数据严重不足**: 只有8个模型, 96个窗口（随机裁剪凑数）
2. **频谱差距太大**: corr=0.577, 简单滤波无效
3. **仿真轨道太少**: 每模型仅32-64道, 256道窗口需要padding

### 最有效的next step
1. **用SimLab大规模生成仿真数据**（500+窗口, 不同地质场景）
2. **仿真阶段加噪声/频谱匹配**缩小域差
3. **重新混合训练**

---

## 四、产出文件

### 文档
- `docs/PGDA_CSNET_V1_LOCKED_SCHEME.md` - v1锁定方案
- `docs/PGDA_CSNET_V1_EXECUTION_ROADMAP.md` - 执行路线图
- `docs/PGDA_CSNET_FUTURE_DIRECTION_MASTER.md` - 未来方向总纲
- `docs/V1_11_CONFIDENCE_ABSTENTION_RESULTS.md` - v1.11结果

### 模型
- `outputs/finetune_sim_to_real_v2/checkpoint_best.pt` - sim pretrain finetune
- `outputs/run_gpu_v1_14_spectral_aug_mixed/checkpoint_best.pt` - 频谱增强混合训练

### 代码改动
- `pgdacsnet/model_raw_unet.py` - SGM模块, SG-USSM v1.11架构
- `scripts/train_raw_only.py` - 混合数据集, 频谱增强
- `scripts/convert_sim_to_training.py` - 仿真→训练窗口转换

### 图
- `C:\Users\17844\Desktop\PGDA_v1_results\` - 所有结果图
- `fig1_bscan_comparison_line9.png` - B-scan对比
- `fig4_method_comparison.png` - 方法对比柱状图
- `xg03_vs_realline9_comparison.png` - XG03 vs 实测对比
- `sim_vs_real_domaingap.png` - 域差分析
- `pml_comparison_raw.png` - PML配方对比
