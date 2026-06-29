# PGDA-CSNet Pilot-Mini → Pilot-Train 数据集生产计划

**版本**: v2 (2026-06-12)
**状态**: 待执行

---

## 1. 总体路线

```
Pilot-Mini (20场景, 40道) → QC通过 → Pilot-Train (100场景, 128-256道)
         ↓                              ↓
   验证链路/标签/频谱              训练第一版网络
```

---

## 2. 信号模型（锁定定义）

```
A = 直达波 / 天线间空气耦合
S = 地表参考反射
G = 地下有效地质信号（基覆界面、互层、深部异常）
E = 外部杂波（电线、树木、建筑等人工散射体）

Y_full   = A + S + G + E          (raw 变体)
Y_target = A + S + G              (target_only 变体)
Y_air    = A                      (air_only 变体，全局模板)

X_clean = Y_target - Y_air = S + G     (干净B-scan，保留地表反射)
C_gt    = Y_full - X_clean  = A + E    (操作性杂波标签)
```

**关键说明**：
- X_clean **保留地表反射**。地表反射在 UavGPR 中承担零时参考、地形参考和后续 RTM/FWI 成像作用，第一阶段不强制网络删除。
- C_gt 是**操作性杂波标签**（operational clutter label），基于同源仿真变体构造，用于监督网络学习直达波与外部杂波抑制。由于电磁波传播中存在多次散射和非线性相互作用，该标签**不等同于严格物理意义上的独立杂波场分解**。作为 ML 监督标签可用，论文或报告中应声明此限制。
- C_gt 包含直达波（A）和外部杂波（E），不包含地表反射（S）。

---

## 3. FDTD 仿真参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 网格 | 773×965×1 (75万单元) | dx=0.05m, 域~39m×48m |
| 时窗 | **700 ns** | 覆盖 εr=25 时 22m 双程走时 |
| native 时间步 | **由 gprMax 根据 CFL 自动确定**（约 5938 步） | 不手写死 |
| ML 重采样 | **501 samples** | 对齐营山实测格式 |
| 中心频率 | 100 MHz Ricker | 匹配实测峰值 104.1 MHz |
| TX-RX 间距 | 1.4m 共偏移 | CDUT-UavGPR UG10 |
| 飞行高度 | 12m 定高 | 匹配实测 L3 线 |
| 道间距 | 0.75m | Pilot-Mini |
| 道数 | Pilot-Mini: 40 / Pilot-Train: 128-256 | |

---

## 4. 材料参数

| 材料 | εr | σ (S/m) |
|------|-----|---------|
| silty_clay | 8-28 | 0.002-0.010 |
| gravelly_silty_clay | 6-22 | 0.001-0.008 |
| weathered_mudstone | 6-16 | 0.002-0.015 |
| sandstone_bedrock | 4-10 | 0.0003-0.008 |
| saturated_zone | 18-32 | 0.005-0.030 |
| surface_water | 24-32 | 0.001-0.020 |

每个 SceneWorld 从范围内随机采样一次，同一场景下所有变体共享同一组材料。

---

## 5. 五族场景定义

| 族 | 特征 | 强制生成 |
|----|------|---------|
| gentle_interbed | 常规基覆界面 + 互层 | 无 |
| terrace_paddy | 梯田/水田 | water_zones + saturated_zones |
| wire_tree_endpoint | 强外部杂波 | wires + trees + buildings |
| deep_anomaly_21m | 深部异常体 | anomaly_objects (18-23m) |
| cross_slope_high_relief | 高起伏地形 | ground_relief 8-30m |

杂波概率已按族独立控制（SimLab `yingshan_families.py`）。

---

## 6. 数据集规格

### Pilot-Mini

| 参数 | 值 |
|------|-----|
| 场景数 | 20 |
| 分布 | gentle(4) / terrace(4) / wire(4) / anomaly(4) / slope(4) |
| 变体 | raw / target_only / background_only |
| air_only | **1 全局模板**（天线/高度/网格固定） |
| 道数/场景 | 40 |
| manifest 行数 | 20 × 3 + 1 = **61** |
| **目的** | 验证链路、标签、频谱、稳定性 |

### Pilot-Train

| 参数 | 值 |
|------|-----|
| 场景数 | 100 |
| 分布 | **根据 Pilot-Mini 通过率决定**（见下） |
| 变体 | raw / target_only / background_only |
| air_only | 固定几何复用模板；若高度/测线长度/域尺寸变化则改为 by_geometry |
| 道数/场景 | **128-256** |
| ML 样本尺寸 | 501 × patch_width |
| **目的** | 训练第一版 PGDA-CSNet |

**cross_slope 比例由 mini 通过率决定**：

| Mini 通过率 | Train 建议比例 |
|------------|--------------|
| ≥80% | 8-10% |
| 50-80% | 5-8% |
| <50% | 暂停扩量，先修正几何生成器 |

### 频谱匹配分级标准

| 阶段 | 要求 |
|------|------|
| Pilot-Mini | corr ≥ 0.55 可接受 |
| Pilot-Train | corr ≥ 0.60 可进入训练，但需记录 |
| 正式论文数据集 | 尽量 ≥0.70；若达不到，必须解释源子波/系统响应差异 |
| 拿到 UG10 实测波形后 | 重新校准，目标 ≥0.70-0.80 |

---

## 7. air_only 模板复用条件

以下参数在 Pilot-Mini 中完全固定，air_only 跑一次全局复用：

1. TX/RX 高度一致（12m）
2. 源子波一致（100 MHz Ricker）
3. trace 位置一致（固定起始点，固定步长）
4. time window 一致（700ns）
5. domain 尺寸一致（39m × 48m）
6. PML 和网格一致（10 层，dx=0.05m）
7. TX-RX offset 一致（1.4m）

若 Pilot-Train 阶段上述任何参数变化，air_only 需改为 by_geometry 模板。

---

## 8. 质量筛选

每场景仿真完成后计算：

| flag | 含义 | 阈值 |
|------|------|------|
| target_visible | 基覆界面响应可检测 | 界面区域 SNR ≥ 3 dB |
| interface_in_window | 界面在时间窗内 | 双程走时 ≤ 650ns |
| has_nan | 无 NaN | False |
| has_inf | 无 Inf | False |
| amplitude_valid | 振幅范围合理 | max ∈ [1, 1000] V/m |
| clutter_valid | C_gt 非零 | C_gt RMS ≥ 0.001 |

训练前排除 `target_visible = false` 的场景。

---

## 9. 训练标签

| 标签 | 公式 | 用途 |
|------|------|------|
| X_clean | target_only - air_only | 干净 B-scan（监督目标）|
| C_gt | raw - X_clean | 杂波图（监督标签）|
| interface_gt | scene_world 界面深度 | L_interface 损失 |
| layer_gt | 材料柱状图 | 辅助标签 |

---

## 10. background_only 的用途

1. 约束网络预测的 clutter 不应吞掉基覆界面
2. 检查 raw - background 是否能突出深部目标
3. 作为 SVD/RPCA/传统背景扣除 baseline 的参考

训练中默认不加入 L_bg 损失，后续版本再评估。

---

## 11. 验收标准 (Pilot-Mini)

- [ ] 所有 raw / target_only / background_only B-scan 非 NaN、非 Inf
- [ ] C_gt 可计算（air_only 模板对齐）
- [ ] interface_gt 能映射到 0–700ns 时间窗
- [ ] 每族至少 3 个 target_visible = true
- [ ] 频谱相关 > 0.55（目标后续版本 > 0.70）
- [ ] 20 张预览图人工检查无明显标签错位
- [ ] 传统基线 (dewow/SVD/f-k) 产出完整
- [ ] manifest CSV 行数与实际一致

---

## 12. 资源估算

| | 3060 (6GB) | 4090 (16GB) |
|---|---|---|
| 每道 (700ns) | ~8-12s | ~4-6s |
| Pilot-Mini (20×3×40=2400道) | **5-8小时** | 3-4小时 |
| Pilot-Train (100×3×200=60000道) | 5-8天 | 2.5-4天 |
| GPU 显存 | 1.2GB (20%) | 1.2GB (8%) |
