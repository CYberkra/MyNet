# PGDA-CSNet v1 最终锁定方案

**版本**: v1.0 (2026-06-27)  
**状态**: 锁定 / 作为当前主线执行  
**适用范围**: 当前实测主线、Line9 locked holdout、measured-data baseline

---

## 1. 方案定位

**PGDA-CSNet v1 最终定位：**

一个面向**低频 UAV-GPR 实测 B-scan 的 raw-only 弱监督界面拾取基线系统**，目标是在强杂波、少标签、单场地条件下，稳定输出**基覆/强地质界面概率图与中心线拾取结果**。

### 它是

- raw-only
- weakly supervised
- interface-oriented
- confidence-aware / abstention-compatible
- measured-data anchored

### 它不是

- 完整 clean B-scan 重建系统
- 严格物理分量分解系统
- 已解决 zero-shot cross-line 泛化的系统
- 大而全的 physics-guided clutter suppression 总系统

---

## 2. 为什么锁定为这个 v1

当前最需要解决的核心问题不是“构建一个完整物理引导大网络”，而是：

> 在只有一个实测区、标签存在不确定性、跨线域偏移明显的条件下，能否稳定、可复现地拾取基覆/界面，并在高风险区域避免自信乱报。

因此，v1 必须优先追求：

1. 标签语义自洽
2. split 协议严格
3. raw-only 可复现
4. 单模型可交付
5. 结果表述科学诚实
6. 为后续 confidence / target adaptation 路线留足接口

---

## 3. 最终 v1 锁定方案

### 3.1 任务定义

v1 主任务定义为：

> **从 raw B-scan 中预测基覆/强地质界面带，并提取最终中心线。**

更准确的表述是：

> **在原始低频 UAV-GPR B-scan 上进行弱监督界面增强与界面拾取。**

### 3.2 主输出

v1 主输出锁定为：

1. 界面概率图 / soft mask
2. presence（该段是否应拾取）
3. 中心线 / 脊线提示
4. 最终 DP 中心线结果
5. 可留空 / 拒判区间

### 3.3 输入：锁定为 raw-only

只允许：

- `raw_full_normalized`
- 对应窗口字段：`x_raw`

明确禁止作为模型输入：

- BG501
- AGC
- gained view
- processed view
- background-suppressed view
- 任何人工增强后图像

这些只能用于可视化、QC、对照，不可作为主模型输入。

### 3.4 主数据集：锁定为 `data_corrected_v1_4_terrain_direction`

这是 v1 的**唯一主训练 / 主评估标签集**。

原因：

- 当前 frozen baseline 与其标签约定一致
- 它是目前最保守、最统一、最不容易自相矛盾的一版
- V16 / V17 都只能作为实验 / 审计标签，不适合升为主线

### 3.5 主监督字段：锁定为四类

1. `soft_mask_train`
2. `status_code`
3. `label_weight`
4. `ignore_mask`

这四个构成 v1 的核心 supervision。

#### 含义

- `soft_mask_train`：界面带弱监督，不假装是精确单像素真值
- `status_code`：区分强可见 / 弱可见 / no-pick
- `label_weight`：表达 trace 级别监督可信度
- `ignore_mask`：显式承认某些区域不适合强监督

### 3.6 Line9 标签约束：锁死

- Line9 基覆 / 界面主带：约 `12–16 m`
- **不把 `20–23 m` 深部异常标为 basal / interface**
- 这个规则在 v1 中不得再改

### 3.7 LineX1 规则：锁死

LineX1：

- 只作 review-only
- 不进入训练
- 不进入验证
- 不进入 ranking
- 不进入平均指标
- 不作为模型升降级依据

### 3.8 仿真标签的定位

保留仿真物理分解概念，但**不纳入 v1 主监督叙事**。

可保留用途：

- 仿真预训练候选
- QC / 机制验证
- 数据构建工具链
- 后续 v2 / v3 物理增强方向

特别说明：

- `C_gt = A + E` 只能称为 **操作性 clutter surrogate label**
- 不称为严格物理真值

### 3.9 网络结构：锁定为现有冻结主线

v1 主网络锁定为：

**`PGDA-CSNet v1.9D MambaVision-style hybrid` 单模型 seed-1902**

锁定 checkpoint：

`outputs/run_gpu_paper_v1_9d_mambavision_hybrid_final_seed1902_line9holdout/checkpoint_final.pt`

SHA256：

`ADC107BFC89345BADEB37D80636D803DA0D97B16EA7AE61F3CBFB9DF3A308042`

### 3.10 v1 结构原则

保留：

- raw-only 输入
- 主干网络提取局部 + 中长程上下文
- mask head
- presence head
- centerline / refined center head
- 后处理融合

v1 不再要求必须有：

- f-k 分支
- SVD 分支
- 显式 clutter head
- clean reconstruction head
- FiLM 几何调制
- terrain metadata 分支
- 大规模物理条件分支

### 3.11 后处理：锁定

v1 的方法**包含后处理**，不把后处理视为附带技巧。

默认后处理流程：

1. mask probability
2. center head
3. `mask + center fusion`
4. `breakable DP`
5. 最终中心线输出
6. 允许低置信断裂，不强行整线连续

锁定默认参数：

- `center_fusion_weight = 0.5`
- `presence_thr = 0.45`
- `path_prob_thr = 0.50`
- `dp_max_jump = 6`
- `dp_smooth_weight = 0.16`
- `dp_min_segment = 16`

锁定 search-window 约束：

- `320–560 ns`

注意：这只是最终提取约束，不是网络输入。

### 3.12 训练协议

v1 不再继续“主线换模型训练”，而是：

> **冻结当前 v1.9D 单模型作为 v1 baseline 主模型。**

锁定 split 协议：

#### Line9

- train: `0–1407`
- guard: `1408–1663`
- holdout: `1664–2377`

这个协议不得再动。

#### 主 valid lines

- Line3
- Line6
- Line7
- Line9
- LineL1

#### 排除

- LineX1

#### 训练纪律

- 按测线切分，不能随机 patch 混切
- 同一测线不同功率档必须在同一 split
- 不允许通过训练线结果升级模型
- 不允许把单条线最优表现冒充整体进步

### 3.13 对 V16 / V17 的锁定结论

#### V16

- 只作审计 / 对照
- 不进入主线
- 不替代 v1.4

#### V17

- 只作实验标签
- 不进入主线
- 不继续扩展其 line-balanced / oversampling 路线

### 3.14 评估协议

v1 必须把结果拆成三层：

#### A. valid-line 平均

这是 measured-data 主体表现。

#### B. Line9 locked holdout

这是 v1 的关键 held-out 结果。

#### C. strict zero-material leave-one-line-out

这是当前弱项，必须报告，但不作为 v1 主成功标准。

锁定主指标：

#### frozen v1.9D

- valid-line average MAE：`0.868 ns`
- valid-line average pick rate：`0.940`
- Line9 holdout MAE：`3.765 ns`
- Line9 holdout pick rate：`0.514`

这是 v1 当前锁定参考值。

### 3.15 robust-normalization 的定位

它不是默认方案，只是 ablation：

- Line9 holdout MAE 可改善到 `3.542 ns`
- pick rate 可到 `0.634`
- 但五线平均退化到 `1.221 ns`

因此：

> **不升为 v1 默认推理配置。**

### 3.16 strict LOO 的定位

当前 strict zero-material LOO 很差：

- 最好约 `45.079 ns`
- coverage / pick 约 `0.668`

因此必须明确写：

> **v1 不解决无资料跨线泛化。**

### 3.17 v1 成功标准

v1 的成功，不是“全解决”，而是：

> 建立一个可信、可复现、保守表述、在固定协议下有效的实测 UAV-GPR 界面拾取基线。

建议锁定的 pass 条件：

- valid-line average MAE ≤ `0.90 ns`
- valid-line average pick rate ≥ `0.93`
- Line9 holdout MAE ≤ `3.80 ns`
- Line9 holdout pick rate ≥ `0.50`

不作为主 pass / fail 的内容：

- strict LOO
- LineX1
- ensemble upper bound
- 单条训练线最优值

---

## 4. 明确不纳入 v1 的内容

### 4.1 不纳入 v1 架构

- f-k spectral branch
- SVD low-rank branch
- FiLM 几何调制
- terrain / flight / relative-height / surface proxy 主输入
- clutter map 主监督头
- clean reconstruction 主头
- 完整 dual clean/clutter 重建框架
- 更大 backbone 搜索
- 三种子 ensemble 作为主模型

### 4.2 不纳入 v1 训练主线

- V16 替代主标签
- V17 替代主标签
- Line9 oversampling / line-balanced 路线
- robust norm 作为默认部署方案
- borehole weak supervision 作为主流程核心
- adversarial domain adaptation 作为主结果

### 4.3 不纳入 v1 论文主 claim

- “已经实现物理真值级杂波分解”
- “已经实现 clean B-scan 重建”
- “已经解决 zero-shot cross-line 泛化”
- “可推广到所有 UAV-GPR 场景”

---

## 5. 论文表述建议

### 5.1 推荐标题 / 方法表述方向

建议把 v1 说成：

#### 中文

**面向低频 UAV-GPR 实测 B-scan 的 raw-only 弱监督基覆界面拾取方法**

#### 英文思路

- raw-only weakly supervised interface picking
- confidence-aware bedrock interface extraction
- measured-data anchored baseline for low-frequency UAV-GPR

### 5.2 建议避免的说法

不要写：

- physics-ground-truth clutter decomposition
- clean radargram reconstruction
- robust cross-line zero-shot generalization achieved
- universally deployable interface detector

### 5.3 建议主动承认的限制

必须主动说明：

1. 只有一个实测区
2. 标签是弱监督，不是严格真值
3. 当前跨线无资料泛化仍弱
4. 当前主系统是 measured-data anchored baseline
5. confidence / abstention 是下一步重点，而不是已完全解决

---

## 6. v1 完成标准

### 6.1 工程完成

- frozen checkpoint 可复现
- smoke 可复现
- full Line9 holdout eval 可复现
- config / dataset / postprocess 都锁定
- SHA256 锁定
- 结果文件路径锁定

### 6.2 科学完成

- 主标签语义统一
- split 协议统一
- 主指标统一
- 负结果（V16 / V17 / robust norm / strict LOO）如实报告
- 不过度宣称

### 6.3 论文完成

- 方法定义稳定
- 主图 / 主表稳定
- baseline 与 ablation 关系清晰
- v1 / v2 / v3 边界清晰

---

## 7. v2 / v3 预留方向

### v2

- confidence / abstention-first
- MAE-coverage Pareto
- frozen v1.9D 上的 error control
- target-line self-supervised adaptation
- limited borehole-assisted enhancement

### v3

- physics-guided auxiliary branches
- f-k / SVD / geometric conditioning
- sim-to-real pretraining refinement
- interface-oriented enhancement stage + picking stage two-stage system
- 更完整的 suppression / reconstruction 路线

---

## 8. 最终一句话锁定

> **PGDA-CSNet v1 最终锁定为：一个基于 raw-only 输入、使用 corrected v1.4 弱监督标签、以 frozen v1.9D 单模型为核心、通过 mask+center fusion 与 breakable-DP 完成界面拾取的 measured-data 基线系统。**

并且：

> **v1 不再追求完整物理杂波分解与 clean 重建，重点转为“稳定界面拾取 + 保守评估 + 为后续 confidence / target adaptation 打基础”。**
