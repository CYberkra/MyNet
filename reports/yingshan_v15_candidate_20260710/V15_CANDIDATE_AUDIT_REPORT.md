# 营山实测标签 V15 候选版：零时校准与交叉点审计报告

审计日期：2026-07-10  
候选版本：`YINGSHAN_V15_CANDIDATE_20260710`  
基线数据：`data_corrected_v1_4_terrain_direction`  
性质：**审阅候选版，不是正式训练标签发布版**。

## 结论

六条测线的直达波主导相位均落在 **14.0 ns**，没有证据支持对某一整条测线进行几十纳秒的统一零时平移。
早期地表响应的高度配准相位在不同测线间有明显差异，但该差异可能包含地表材料、极性和波形相位变化，因此只作为交叉点 QC 参考，不能冒充仪器绝对零时。
V15 候选版没有自动移动任何 V14 标签中心；只对四类未解决的高风险交叉邻域设置 `ignore_mask`、清零训练 mask/weight，并保留原标签为 `soft_mask_review_v15`。

## 零时审计

| 测线 | 直达波主相位/ns | bootstrap 标准差/ns | 地表高度配准相位偏移/ns | 解释 |
|---|---:|---:|---:|---|
| Line3 | 14.0 | 0.697 | -22.0 | 直达波用于相对零时；地表相位仅用于 QC |
| Line6 | 14.0 | 0.000 | -21.4 | 直达波用于相对零时；地表相位仅用于 QC |
| Line7 | 14.0 | 0.000 | -17.4 | 直达波用于相对零时；地表相位仅用于 QC |
| Line9 | 14.0 | 0.000 | -15.4 | 直达波用于相对零时；地表相位仅用于 QC |
| LineL1 | 14.0 | 0.000 | -8.2 | 直达波用于相对零时；地表相位仅用于 QC |
| LineX1 | 14.0 | 0.000 | -9.2 | 直达波用于相对零时；地表相位仅用于 QC |

## 八个交叉点决策

| 交叉点 | 状态 A/B | 直达波+空气程差/ns | 地表参考差/ns | V15 候选决策 | 处理 |
|---|---|---:|---:|---|---|
| Line3-Line9 | 2/1 | 63.41 | 56.81 | `IGNORE_WEAK_SIDE_PENDING_REVIEW` | 屏蔽 Line3 的 ±6 m 邻域 |
| Line3-LineL1 | 2/2 | 10.47 | 24.27 | `REVIEW_KEEP_EXISTING_WEAK_LABELS` | 保留 |
| Line3-Line7 | 1/1 | 0.17 | 4.43 | `PASS` | 保留 |
| Line6-Line9 | 1/2 | 33.65 | 27.65 | `IGNORE_WEAK_SIDE_PENDING_REVIEW` | 屏蔽 Line9 的 ±6 m 邻域 |
| Line6-LineL1 | 1/1 | 15.66 | 2.46 | `REVIEW_PHASE_REFERENCE_DISAGREEMENT_KEEP` | 保留 |
| Line6-Line7 | 2/2 | 2.01 | 1.99 | `PASS` | 保留 |
| Line9-LineX1 | 2/2 | 16.06 | 22.26 | `IGNORE_BOTH_WEAK_SIDES_PENDING_REVIEW` | 屏蔽 Line9;LineX1 的 ±6 m 邻域 |
| LineL1-LineX1 | 1/2 | 21.47 | 22.47 | `IGNORE_WEAK_SIDE_PENDING_REVIEW` | 屏蔽 LineX1 的 ±6 m 邻域 |

### 自动候选相位的信号支持

跨线候选时间仅在一侧为强标签、另一侧为弱标签时生成，而且**没有自动写入标签**：
- **Line3-Line9 → Line3**：候选 453.01 ns；候选/当前局部包络支持比 0.677；`weak_signal_support_review_only`。
- **Line6-Line9 → Line9**：候选 403.55 ns；候选/当前局部包络支持比 0.423；`poor_signal_support_do_not_apply`。
- **LineL1-LineX1 → LineX1**：候选 327.53 ns；候选/当前局部包络支持比 0.985；`signal_supported_for_manual_review`。

Line3–Line9 和 Line6–Line9 的跨线候选在局部信号上弱于现有标签，因此不能机械替换；L1–X1 的 X1 候选约 327.53 ns 与当前局部信号强度接近，可作为下一轮人工确认的首选。

## V15 候选数据处理

- `soft_mask_review_v15`：完整保留 V14 几何，供人工复核。
- `soft_mask_train`：仅在未解决的高风险交叉区清零。
- `ignore_mask`：对相应整列置 1，从 mask、curve、centerline 等损失中排除。
- 受影响 trace 的 `status_code=2`、`label_weight=0`，不会训练 presence/no-target。
- acquisition trace 顺序、GNSS 距离、剖面方向和所有原始空间字段保持不变。

| 测线 | 总道数 | 活跃强标签 | 活跃弱标签 | 屏蔽道数 | 中心几何是否自动改动 |
|---|---:|---:|---:|---:|---|
| Line3 | 1813 | 949 | 738 | 126 | 否 |
| Line6 | 1661 | 1078 | 583 | 0 | 否 |
| Line7 | 1654 | 515 | 1139 | 0 | 否 |
| Line9 | 2378 | 937 | 1187 | 254 | 否 |
| LineL1 | 1950 | 1406 | 544 | 0 | 否 |
| LineX1 | 938 | 300 | 360 | 278 | 否 |

## 下一步人工判定顺序

1. L1–X1：优先检查 X1 的 327.53 ns 候选是否对应与 L1 同一连续反射。
2. Line3–Line9：检查 Line3 当前约 396 ns 与跨线候选约 453 ns 哪一条才是基覆界面；当前候选信号支持不足，不能自动替换。
3. Line6–Line9：检查 Line9 当前约 431 ns 与候选约 404 ns；候选局部支持较差。
4. Line9–X1：两侧均为弱标签，缺少可靠强锚点，应保持 ignore，等待剖面/地质证据。
5. Line3–L1、Line6–L1：保留现状但列入人工复核；前者两侧弱标签，后者两套时间参考给出不同判断。

## 验证

- V15 专项校验：通过。
- 数据治理/模型接口/评估回归：`103 passed`。
- Python 编译检查：通过。
- 正式训练状态：`formal_training_allowed=false`。

## 尚未解除的全局阻断项

- 交叉区人工最终确认未完成。
- 实测真负样本仍为 0。
- 正式获批、非 Line9 条件化仿真仍为 0。
- 因此本候选集只能用于复核和代码联调，不能作为论文正式训练标签。
