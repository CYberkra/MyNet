# SCENEWORLD_ALPHA_AUDIT - v0.8.0-alpha.1

## 目标

本轮将模型生成器从 v0.7 的展示/烟测图库，升级为 SceneWorld 同源仿真数据集生成骨架。重点是保证同一个 `case_id` 下的 `raw / target_only / background_only / clutter_only / air_only` 共享同一个地质世界，而不是在各 variant 中重新随机地表、基覆界面、材料、水体或杂波。

## 新增架构

```text
src/uavgpr_simlab/core/scene_world.py
src/uavgpr_simlab/simulation/yingshan_families.py
src/uavgpr_simlab/simulation/scene_world_generator.py
src/uavgpr_simlab/simulation/scene_variant_writer.py
configs/run_plan_yingshan_sceneworld_smoke.yaml
```

### SceneWorld 边界

`SceneWorld` 是单个 case 的唯一随机源，包含：

- ground profile；
- bedrock interface；
- cover / bedrock material map；
- sandstone / mudstone interbed lines；
- water zones；
- external clutter objects；
- constant-level flight trajectory；
- metadata and quality flags。

variant writer 只能根据 `include_in` 规则保留或屏蔽对象，不允许重新随机。

## 已实现场景族

v0.8.0-alpha.1 仅实现两个低风险场景族：

```text
gentle_interbed
wire_tree_endpoint
```

暂未实现：

```text
terrace_paddy
deep_anomaly_21m
cross_slope_high_relief
```

## 输出变化

每个 case 现在输出：

```text
models/<case_id>/scene_world.json
models/<case_id>/metadata_summary.json
models/<case_id>/labels/interface_gt.npy
models/<case_id>/labels/layer_gt.npy
models/<case_id>/previews/model_preview.png
models/<case_id>/previews/raw_preview.png
models/<case_id>/previews/target_only_preview.png
models/<case_id>/previews/background_only_preview.png
models/<case_id>/previews/clutter_only_preview.png
models/<case_id>/previews/air_only_preview.png
```

manifest 新增字段包括：

```text
family
random_seed
model_length_config_m
model_length_actual_m
domain_x_m
scan_start_x_m
scan_end_x_m
flight_height_mode
scene_world_json
metadata_summary_json
interface_gt_npy
layer_gt_npy
model_preview_png
variant_preview_png
is_ml_pair_valid
```

`model_length_m` 现在写入实际仿真域长度，与 `model_length_actual_m` 一致；配置值另存为 `model_length_config_m`。

## 飞行轨迹修正

新增 metadata：

```yaml
trajectory:
  mode: constant_level
  actual_height_profile_available: false
  note: FDTD source/rx path is constant level, not true terrain following.
```

模型预览中的飞行路径也改为 constant-level flight path，不再将固定高度 FDTD 路径误表达为真实仿地飞行。

## 未改变内容

- 不改变正式 gprMax 调用语义。
- 不改变 batch runner、fingerprint、marker、B-scan 后处理。
- 不把 PGDA-CSNet 训练塞进 GUI。
- 不直接生成大规模 paper/main 数据集。
- 不声明 water surrogate 是真实水体高保真模拟。

## 验证

本轮自测新增 SceneWorld 同源性断言：

- 同一 case 的 5 个 variant 指向同一个 `scene_world.json`；
- 同一 case 的 5 个 variant 共享同一个 random seed；
- `flight_height_mode == constant_level`；
- `scene_world.json / metadata_summary.json / interface_gt.npy / layer_gt.npy / preview_png` 均存在；
- `model_length_m == model_length_actual_m`。
