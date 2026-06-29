# SceneWorld alpha.2 数据集结构审计

## 目标

本轮将 `v0.8.0-alpha.1` 的 SceneWorld 标签 smoke 骨架提升为 PGDA-CSNet 仿真数据集 case 包骨架。

## 已落实

1. 每个 case 文件夹保留完整输入与元数据：`raw.in`、`target_only.in`、`background_only.in`、`clutter_only.in`、`air_only.in`、`scene_world.json`、`metadata_summary.json`、`model_preview.png`、`variant_preview.png`。
2. 每个 case 额外写出 B-scan 对齐占位数组：`outputs/raw_bscan.npy`、`outputs/target_bscan.npy`、`outputs/background_bscan.npy`、`outputs/clutter_bscan.npy`、`outputs/air_bscan.npy`。这些数组为 NaN placeholder，等待真实 gprMax 输出合并替换。
3. manifest 中所有路径字段改为相对 workspace 根目录，不再写入 `E:\...` 或 sandbox 绝对路径。
4. 新增正式 pilot 配置 `configs/run_plan_yingshan_sceneworld_pilot.yaml`，使用 `samples=501` 与 `time_window_ns=700`。`run_plan_yingshan_sceneworld_smoke.yaml` 仍保留 450 ns 用于 quick smoke。
5. smoke 配置至少覆盖五类：`gentle_interbed`、`terrace_paddy`、`wire_tree_endpoint`、`deep_anomaly_21m`、`cross_slope_high_relief`。
6. `deep_anomaly_21m` 会生成 `anomaly_objects`，并保存 `center_depth_m`，范围 18–23 m。
7. `wire_tree_endpoint` 会生成 `external_clutter_objects`，至少包含 wire/tree，通常包含 building。
8. `terrace_paddy` 会生成 `water_zones` / saturated-zone surrogate。
9. `cross_slope_high_relief` 的 `ground_relief_m` 目标为 8–30 m。
10. `labels/interface_mask_bscan.npy` 与 `labels/layer_mask_bscan.npy` 和 `outputs/*_bscan.npy` 使用相同 `samples x traces` shape，并共享 `outputs/time_axis_ns.npy`、`outputs/distance_axis_m.npy`。
11. flight mode 仍明确为 `constant_level`，不声明为真实仿地飞行。

## 边界

- placeholder B-scan 不是真实仿真波形。真实训练数据集仍需运行 gprMax 并用后处理替换 placeholder。
- 本轮未改变正式 gprMax 运行命令、fingerprint、marker 或 B-scan 后处理语义。
