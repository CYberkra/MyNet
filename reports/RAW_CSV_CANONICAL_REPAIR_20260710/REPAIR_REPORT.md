# 营山原始 CSV Canonical 数据修复报告

日期：2026-07-10
分支：`audit-fixes-20260710`

## 修复依据

项目源数据格式说明已确认五列含义：

1. 经度；
2. 纬度；
3. 地表高程（m）；
4. 雷达反射波数据；
5. 飞行高度（m，AGL）。

原始 ZIP 已复制到：

`data_corrected_v1_4_terrain_direction/source/ying_shan_measurement_lines_original.zip`

SHA256：

`a147ea2e5b47da80dfafbea68ed6728823df45e0b00709846ef4144bc3ea04ad`

## 已完成

- 新增 `scripts/import_yingshan_raw_csv.py`，直接从原始 ZIP 构建 6 条 canonical 全测线；
- 原始波形由 CSV 第 4 列读取，按整条测线绝对幅值 P99 归一化；
- 经度、纬度、地表高程、飞行高度、天线绝对高程、GNSS 累计距离全部写入全线 NPZ；
- 标签、状态和权重继续使用已经通过重叠一致性检查的窗口缓存；
- 原始 CSV 与旧窗口缓存波形相关系数全部大于 `0.99999998`；
- `window_index.csv` 增加原始 CSV member/hash、源 trace、窗口飞高统计、GNSS 距离范围和高度质量标志；
- 训练数据集现在输出逐 trace 实测飞高，而不是固定 2.4 m 或单一窗口常量；
- arrival prior 已支持 `(B, W)` 逐 trace 高度；
- 旧窗口反拼接脚本默认禁止覆盖原始 CSV canonical 全线；
- terrain features 改为直接使用原始 CSV 元数据和固定物理尺度，不再使用 Line9 或其他 split 的统计量；
- 全线评估图和中心线 CSV 优先使用 GNSS 累计距离轴；
- 数据合同和项目校验器现在验证原始 ZIP、CSV、NPZ 哈希及高程关系。

## 逐测线导入结果

| 测线 | traces | GNSS 长度(m) | 声明长度(m) | 飞高最小/中位/最大(m) | 旧缓存相关系数 |
|---|---:|---:|---:|---:|---:|
| Line3 | 1813 | 171.332 | 154.690 | 8.357 / 11.154 / 12.532 | 0.999999987 |
| Line6 | 1661 | 153.578 | 155.160 | 9.564 / 10.900 / 12.624 | 0.999999991 |
| Line7 | 1654 | 155.876 | 155.828 | 8.679 / 10.396 / 26.085 | 0.999999989 |
| Line9 | 2378 | 224.375 | 216.141 | 8.163 / 9.050 / 12.652 | 0.999999990 |
| LineL1 | 1950 | 179.430 | 185.662 | 7.477 / 8.512 / 13.910 | 0.999999990 |
| LineX1 | 938 | 80.702 | 87.506 | 7.654 / 8.982 / 13.695 | 0.999999991 |

Line7 有 234 道实测飞高超过规划的 20 m。它们是合法的实测值，不再被当作缺失数据；数据合同保留 `measured_outside_planned_range_review` 标志用于作业/QC 复核。

## 验证

- 原始 CSV 导入、数据合同、arrival prior、评估语义和模型接口相关测试：`44 passed`；
- GprMambaSep、分解评估和断点恢复测试：`30 passed`；
- 合计：`74 passed`；
- `check_dataset.py` 普通结构检查：通过；
- `validate_project_contracts.py` 普通治理检查：通过；
- formal-ready 检查：按预期失败。

## 仍然阻断正式训练

1. 实测窗口没有 `status_code=0` 真负类；
2. 当前 33 个仿真 case 均为 Line9-conditioned 或未获人工训练批准；
3. V4 visible-phase 标签尚未重建；
4. Batch 3 尚未逐 case 完成人工决定；
5. Line7、X1、L1 标签仍需按第二轮审计优先级复核。

当前结论：**原始测线、空间坐标和飞高数据链已修复；正式训练冻结的剩余原因已转为负样本、独立仿真和标签质量，而不是数据文件缺失或飞高字段错误。**
