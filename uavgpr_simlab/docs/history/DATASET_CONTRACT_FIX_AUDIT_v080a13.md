# v0.8.0-alpha.13 数据集契约修复审计

## 背景

v080a12 的 25-run smoke 骨架方向正确，但在接入后续训练和 4090 pilot 前存在几个数据契约问题：`clutter_gt_bscan.npy` placeholder 缺失、B-scan 输出命名不统一、constant-level 轨迹中 TX/RX 高度不一致、高起伏 case 最小离地高度过低、distance axis 语义不明确、`layer_gt.npy` 缺少模型网格坐标轴、quick smoke 状态没有充分声明 not-ready-to-train。

## 修复内容

1. 每个 case 的 `outputs/` 现在包含 `clutter_gt_bscan.npy` NaN placeholder，shape 与 B-scan 合同一致。
2. constant-level 轨迹统一为同一平台高度：`source_y_m == receiver_y_m == max_ground_y + nominal_flight_height_m`。
3. 高起伏 case 最小离地高度不低于 nominal flight height；默认 smoke 为 8 m。
4. 新增 `tx_x_axis_m.npy`、`rx_x_axis_m.npy`、`midpoint_x_axis_m.npy`，并声明 `distance_axis_role = midpoint_x`。
5. B-scan canonical 文件名统一为：`raw_bscan.npy`、`target_only_bscan.npy`、`background_only_bscan.npy`、`clutter_only_bscan.npy`、`air_only_bscan.npy`、`clutter_gt_bscan.npy`。
6. `layer_gt.npy` 增加 `layer_gt_x_axis_m.npy` 和 `layer_gt_y_axis_m.npy`，并在 metadata 中标记 `layer_gt_coordinate_system = model_grid`。
7. `dataset_summary.json` 标记 quick smoke: `dataset_grade = quick_smoke`、`ready_to_run = true`、`not_ready_to_train = true`、`training_ready = false`。
8. smoke workspace 增加 `README_SMOKE.md`。
9. `clutter_level` 按 family 区分，便于后续筛选和数据集分析。

## 未改变内容

- 未扩大 case 数。
- 未把 quick smoke 当作训练数据。
- 未改变 25-run smoke 的链路验证定位。
- 未改变 pilot/formal 的严格 QC 原则。

## 验证

- 静态编译通过：`python -m compileall -q src scripts`
- `check-sceneworld-case-package` 对 ultra tiny 和 smoke 均通过。
- smoke manifest 无绝对路径。
- 每个 smoke case 的 `clutter_gt_bscan.npy` placeholder 存在且 shape 为 `(501, 72)`。
- `case_000005` high-relief 的 source/rx 高度相等，最小离地高度为 8 m。
