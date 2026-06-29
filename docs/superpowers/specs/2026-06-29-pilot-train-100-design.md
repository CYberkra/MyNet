# PGDA-CSNet Pilot-Train 100 场景仿真设计方案

日期：2026-06-29  
状态：设计审阅  
相关：run_plan_3060_pilot_train_v1.yaml, scene_world_generator.py, convert_pilot_to_training.py

---

## 1. 目标

解决当前训练数据严重不足（169 样本 → ~49 有效样本/fold）导致的过拟合瓶颈，生成 **100 场景 × 多窗口 → ~500 有效训练样本**。

## 2. 仿真配置（锁定）

### 2.1 FDTD 参数

| 参数 | 值 | 依据 |
|------|:--:|------|
| 网格 dx/dy/dz | 0.05 m | 19 cells/λ (eps_r=10 时)，远超 10 格要求 |
| 时窗 | 700 ns | 与实测数据一致，覆盖 0~22m 深度 |
| 时间步 | 5937 | dt=0.118ns × 5937 = 700ns |
| 中心频率 | 100 MHz Ricker | 与实测天线一致 |
| 飞行高度 | 12 m | 与实测 UAV 作业高度一致 |
| 道间距 | 0.5 m | 与实测数据一致 |
| PML | 10 层 | gprMax 2D 默认，经验证有效 |

### 2.2 几何与地质参数

| 参数 | 值 | 说明 |
|------|:--:|------|
| 扫描长度 | 64 m | 128 道 × 0.5m |
| 道数 | 128 | 支持多窗口滑动 |
| 基岩界面深度 | 5.0 ~ 22.0 m | 覆盖浅层和深层 |
| 地形起伏 | 0.3 ~ 3.5 m（典型），8~30 m（cross_slope）| 中低起伏为主，含高起伏变体 |
| 坡度 | 0 ~ 25° | 覆盖缓坡到中坡 |
| 地表覆盖层 | 粉质黏土/含砾黏土 | eps_r=8~28, σ=0.002~0.01 S/m |
| 基岩 | 砂岩 | eps_r=4~10, σ=0.0003~0.008 S/m |
| 风化泥岩互层 | 2~5 层 | 模拟营山 ZK07/ZK09 典型层序 |

### 2.3 地层材质（锁定，不改）

```
材质              eps_r              σ (S/m)         
─────────────────────────────────────────────────
air               1.0               0               
silty_clay        8 ~ 28            0.002~0.010     
gravelly_silty    6 ~ 22            0.001~0.008     
weathered_mud     6 ~ 16            0.002~0.015     
sandstone_bed     4 ~ 10            0.0003~0.008    
saturated_zone   18 ~ 32            0.005~0.030     
```

每场景随机在范围内采样，保留地质多样性。

### 2.4 层界面处理

**保持突变界面（sharp boundaries）。** 理由：
- gprMax 原生 `#box` 命令，学术界标准做法
- 100MHz 波长 ~1m（土壤中），突变界面的 FDTD 数值色散已模拟部分渐变效果
- 多层薄 box 模拟渐变的实现复杂度高，对当前问题增益有限

### 2.5 变体（只跑需要的）

```
raw (A+S+G)           → 训练输入
target_only (A+S+G)   → 与 raw 相同（无外部杂波），仅作冗余备份
background_only (A+S) → 后续物理分支可能用
```

不跑 `air_only`（无外部杂波下 target_only 等价于 raw，air_only 暂时用不到）。

### 2.6 外部杂波

**本轮不做。** 电线/树木/建筑的几何设置和 FDTD 验证复杂度高，留作后续迭代。当前 5 个地形家族 + 域随机化（boulder/fracture/moisture/bedrock_step）已提供足够的地质多样性。

## 3. 地形家族分布（100 场景）

| 家族 | 比例 | 场景数 | 关键特征 |
|:----|:----:|:------:|----------|
| gentle_interbed | 25% | 25 | 缓坡+砂泥互层，基础样本，含随机 boulder/fracture |
| terrace_paddy | 25% | 25 | 阶地+水田+饱和带，water_zones 保证生成 |
| wire_tree_endpoint | 25% | 25 | 含随机 boulder/fracture（外部杂波暂关闭） |
| deep_anomaly_21m | 17% | 17 | 深部异常体 ~21m，anomaly_objects 保证生成 |
| cross_slope_high_relief | 8% | 8 | 高起伏地形 8~30m，跨坡大落差 |

循环方案：PILOT_FAMILY_CYCLE（12 场景循环，code 已有）。

## 4. 数据转换管线（改进点）

### 4.1 窗口策略

128 道仿真输出（501 × 128）比窗口宽度（256）窄，所以：

```
核心策略：128 道 → 填充到 256（不滑动）
补充策略：对覆盖层材料不同的列区域分别切窗口
         → 每场景 1~2 个窗口 → 100~200 训练样本
```

这样已经比当前 20 窗口多 5~10 倍。更密集的滑动窗口需要增加仿真道数到 256+，这会使 GPU 时间翻倍，留作后续优化。

### 4.2 软 Mask 保存

**不再二值化。** 保留 interface mask 为连续概率 (0~1 float)，让模型学习到界面位置的不确定性。训练时使用 `core_threshold=0.55` 做决策，但保留底层 soft target。

### 4.3 统一 P99 归一化

不再对每个 case 独立 P99 归一化。改为：
1. 收集全部 100 场景的联合幅值分布
2. 使用全局 P99 作为统一参考值
3. 所有场景统一除以该参考值，保持 case 间幅值相对关系

### 4.4 status_code 改进

当前阈值 `peak_per_trace > 0.05` 可能过低。改为：
- absent: `max(mask) < 0.1`
- present: `max(mask) > 0.5`
- weak: 中间值

## 5. 训练配置调整

### 5.1 数据量变化

| 指标 | 当前 | 新方案 |
|:----|:---:|:-----:|
| 实测窗口 | 78 | 78（不变）|
| 仿真窗口 | 20 | **~500** |
| 有效每 fold | ~49 | **~350** |
| sim_batch_ratio | 0.3 | 0.5~0.7（仿真实例更多）|

### 5.2 Hyperparameter 初步建议

| 参数 | 当前 | 建议 | 理由 |
|:----|:---:|:----:|------|
| epochs | 80 | **60** | 数据增多后收敛更快 |
| batch_size | 2 | **4** | 更大 batch = 更稳定梯度 |
| lr | 5e-4 | 5e-4（不变）| 先不动，看 loss 曲线 |
| lr_scheduler | 无 | **余弦退火** | 更大数据量需要更精细的 LR 调度 |

### 5.3 预期效果

```
训练前 (169 样本)             训练后 (~600 样本)
───────────────              ────────────────
best epoch 6~24 (严重过拟合)   best epoch 30~50 (正常)
train-val gap 0.1→1.0          train-val gap <0.3
val_loss >0.93                  val_loss 预期 <0.6
任何正则化→更差                 正则化开始有效
```

## 6. 执行步骤

### Phase 1: 生成 100 场景并仿真（~56h GPU）
1. 验证 `run_plan_3060_pilot_train_v1.yaml` 配置
2. 运行 `scripts/run_batch_safe_3060.py` 批量启动
3. 每场景跑 raw + target_only + background_only
4. 监控进度（~34min/场景 × 3 变体 ≈ 100min/场景 ÷ 可并行 2 变体 ≈ 50min/场景 × 100 = 50~60h GPU）

### Phase 2: 数据转换（~15min）
1. 批量 merged.out → .npy（已有 batch_postprocess_pilot.py）
2. 运行新版本 convert_pilot_to_training.py（滑动窗口 + soft mask + 统一 P99）
3. 生成 ~500 个 NPZ 训练窗口

### Phase 3: 重新训练 LOLO-CV
1. 用新数据跑 Line9 holdout（3-seed）
2. 对比性能：val_loss、DP MAE、Pick Rate
3. 如果效果提升明显，跑全部 5 折

## 7. 验证标准

- 新仿真 NPZ 的 x_raw 幅值范围一致（统一 P99）
- status_code 分布合理（absent/present/weak 各占比例不极端）
- 滑动窗口覆盖场景所有核心道
- 训练后 val_loss 比当前基线降低 (>0.2 降幅即为显著改善)
