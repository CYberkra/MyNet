# 营山测线航向、B-scan 与工程剖面对应关系全面审计

## 结论

- Canonical 数据必须保持原始 CSV 采集顺序，不允许为了匹配图件而物理翻转训练数组。
- Line3、Line6、Line9、LineX1 的工程/报告剖面显示需要相对采集顺序翻转；Line7、LineL1 不翻转。
- 8 个应有交叉点的 GNSS 最近距离均小于 0.05 m，测线身份、起止顺序和交叉位置映射可靠。
- 工程剖面里程与 GNSS 累计轨迹长度是不同坐标轴，现已分别保存。
- 标签在交叉点并非全部一致：Line3-Line9 为 critical，Line6-Line9 与 LineL1-LineX1 为 high-risk；正式训练继续冻结。

## 航向与显示方向

| 测线 | CSV 采集航向 | 工程剖面 | 显示翻转 | 剖面左→右 | 置信度 |
|---|---:|---|---|---|---|
| Line3 | 173.926° S | 3-3′ | True | 3 / ZK07 / south → 3′ / ZK08 / north | high |
| Line6 | 175.654° S | 6-6′ | True | 6 / ZK09 / south → 6′ / ZK10 / north | high |
| Line7 | 86.183° E | 7-7′ | False | 7 / west / ZK09 side → 7′ / east / Line3 crossing side | high |
| Line9 | 266.83° W | 9-9′ | True | 9 / west → 9′ / east / ZK08 side | high |
| LineL1 | 265.408° W | report-only L1 section | False | east / Line3 crossing side → west / Line6 crossing side | medium |
| LineX1 | 175.619° S | report-only X1 section | True | south / L1 crossing side → north / Line9 crossing side | medium |

## 交叉点核验

| 交叉 | GNSS 最近距离/m | 两线剖面里程/m | 空气程校正后标签时间差/ns | QC |
|---|---:|---|---:|---|
| Line3-Line9 | 0.0261 | 140.434 / 186.679 | 63.413 | critical_mismatch_weak_label |
| Line3-LineL1 | 0.0431 | 104.237 / 5.906 | 10.474 | review_weak_label |
| Line3-Line7 | 0.0354 | 30.989 / 121.42 | 0.17 | pass |
| Line6-Line9 | 0.0182 | 106.275 / 89.657 | 33.651 | high_risk_weak_label |
| Line6-LineL1 | 0.0287 | 65.709 / 106.691 | 15.662 | review |
| Line6-Line7 | 0.042 | 2.617 / 19.985 | 2.012 | pass_but_weak_label |
| Line9-LineX1 | 0.0105 | 138.577 / 69.109 | 16.057 | review_weak_label |
| LineL1-LineX1 | 0.0411 | 54.108 / 25.962 | 21.471 | high_risk_weak_label |

这里的时间差仅用于同一空间交叉点的标签一致性 QC，不用于直接换算地质深度。

## PDF/图件内部问题

1. Line6 页面中的 9/L1 橙色交叉标记沿采集顺序排列，但地形剖面本身采用反向显示，不能把橙色标记作为精确里程。
2. LineL1 没有独立工程剖面，报告中的航拍箭头存在冲突；以 GNSS 采集顺序和主迁移图为准，置信度为 medium。
3. LineX1 没有独立工程剖面，保持 review-only；报告主迁移图支持反向显示。
4. 图件中的“无人机真高 8 m”是成像/展示参数，不能覆盖原始 CSV 逐道飞高。

## 数据使用规则

- 训练、指标、窗口索引：始终为 acquisition_csv。
- 地图轨迹：使用 gnss_cumulative_distance_m。
- 与工程剖面对照：使用 profile_chainage_m，并按 profile_display_flip 只在显示层翻转。
- 横向镜像增强后必须对 terrain_slope_z 与 trace_position 取反；该错误已修复并测试。


## 审计证据与可追溯性

- 工程图源：`data_corrected_v1_4_terrain_direction/source/ying_shan_profiles_and_boreholes.zip`
- 工程图源 SHA256：`0e802c7d4ff7d64600fc3e97d6bdad394c13dabc22c09f60d9ecea5dc3f82a2a`
- 内容：总平面图、3/6/7/9 号工程地质剖面、8 份钻孔柱状图。
- L1 与 X1 没有独立工程剖面 PDF，因此它们的“剖面显示方向”只能由 GNSS 采集顺序、报告主迁移图和交叉点相对位置联合确定，置信度保持 medium，不能伪装成 high。
- 原始 B-scan 仍以 501 点、0–700 ns 和 CSV trace-major 顺序为 canonical；剖面显示方向只是派生视图。

## 两套距离轴审计

| 测线 | GNSS累计轨迹/m | 名义剖面里程/m | 差值/m | 相对差 |
|---|---:|---:|---:|---:|
| Line3 | 171.332 | 154.690 | +16.642 | +10.76% |
| Line6 | 153.578 | 155.160 | -1.582 | -1.02% |
| Line7 | 155.876 | 155.828 | +0.048 | +0.03% |
| Line9 | 224.375 | 216.141 | +8.234 | +3.81% |
| LineL1 | 179.430 | 185.662 | -6.232 | -3.36% |
| LineX1 | 80.702 | 87.506 | -6.804 | -7.78% |

解释：GNSS 累计轨迹会把横向抖动和局部弯折累计进去；工程剖面里程来自道间距。地图定位、交叉点搜索使用 GNSS；与工程剖面横坐标对照使用名义剖面里程。Line3 和 X1 的两轴差异最明显，禁止混用。

## 钻孔—成像深度锚点

| 测线 | 钻孔 | 钻孔基岩面/m | 报告成像深度/m | 绝对误差/m |
|---|---|---:|---:|---:|
| Line3 | ZK07 | 16.5 | 16.7 | 0.2 |
| Line3 | ZK08 | 14.0 | 14.4 | 0.4 |
| Line6 | ZK09 | 11.2 | 11.5 | 0.3 |
| Line7 | ZK09 | 11.2 | 11.5 | 0.3 |
| Line9 | ZK08 | 14.0 | 14.4 | 0.4 |

这些锚点来自报告的迁移/成像解释，证明剖面图与钻孔位置的总体关系成立。但当前网络标签是时间域 visible-phase 标签，不能在未完成零时、飞高和速度模型统一前直接把 `ns` 与上述 `m` 一一换算。

## 交叉点标签一致性解释

- **通过**：Line3–Line7（0.17 ns）、Line6–Line7（2.01 ns，弱标签）。
- **需复核**：Line3–L1、Line6–L1、Line9–X1。
- **高风险**：Line6–Line9（33.65 ns）、L1–X1（21.47 ns）。
- **严重不一致**：Line3–Line9（63.41 ns）。

交叉点最近距离全部小于 0.05 m，因此这些差异不是“交叉点找错”造成的。更可能来自弱标签选错相位、不同测线零时偏移、或标签指向不同反射层。Line3–Line9 和 Line6–Line9 在解决前不得用于跨线一致性监督或正式论文结果。

## 已完成的代码修复

1. 新增 `pgdacsnet/spatial_orientation.py`，集中维护采集航向和剖面显示合同。
2. Canonical NPZ 写入 `acquisition_bearing_deg`、`profile_chainage_m`、`profile_display_flip`、剖面左右端及证据等级。
3. `eval_full_line.py` 新增 acquisition/profile 显示选项；指标和原始预测始终保持采集顺序。
4. `window_index.csv` 写入采集航向、剖面翻转和名义剖面里程。
5. 横向增强后对 `terrain_slope_z` 与 `trace_position` 取反，修复方向性元数据语义错误。
6. 原始 CSV 导入器每次都从 78 个重叠窗口重建标签/校验波形，避免重复运行时“canonical 文件与自身比较”的循环校验漏洞。
7. 数据与项目校验器现在强制检查方向注册表、剖面里程、显示合同和工程图源哈希。
8. 正式训练门禁新增交叉点标签一致性阻断项。

## 最终放行判断

- **航向、起止点、测线身份和交叉位置：通过。**
- **3/6/7/9 工程剖面显示方向：通过。**
- **L1/X1 显示方向：可用但仅 medium confidence；X1 继续 review-only。**
- **B-scan 与剖面横向对齐机制：已修复。**
- **钻孔与报告成像关系：总体通过。**
- **时间域标签跨线一致性：不通过，仍是正式训练阻断项。**
