# SMOKE25_GUI_VALIDATOR_AUDIT_v080a10

## 目标

在 v0.8.0-alpha.9 已经通过目标机 ultra tiny 验证的基础上，新增 GUI 内置 25-run smoke 验证入口，用于验证五类营山 SceneWorld 场景族的 gprMax 全链路。

## 验证范围

- gentle_interbed
- terrace_paddy
- wire_tree_endpoint
- deep_anomaly_21m
- cross_slope_high_relief

每类 1 个 case，每个 case 5 个 variants：raw、target_only、background_only、clutter_only、air_only，共 25 次 gprMax 运行。

## 关键策略

- ultra tiny：允许显式 resample，仅用于最小链路验证。
- 25-run smoke：严格 QC，不允许 resample。若 gprMax 输出 shape 与 manifest 期望不一致，标记 failed。
- pilot：仍不在本轮生成真实数据，需 smoke 25-run 通过后再推进。

## 修改点

- 设置页新增 `运行 25-run smoke 验证` 按钮。
- GUI worker 泛化为 SceneWorldFullChainWorker。
- 新增 `workspace/yingshan_sceneworld_smoke_v080a10`。
- 新增 `configs/run_plan_yingshan_sceneworld_smoke_v080a10.yaml`。
- 新增 `workspace/yingshan_sceneworld_ultra_tiny_v080a10` 作为最新 ultra tiny 骨架。

## 风险

- 25-run smoke 仍需目标 Windows / gprMax 实机验证。
- 若 strict QC 因 gprMax 原始输出 shape 不匹配而失败，不能直接改为静默 resample；应先记录真实 shape、时间轴和采样策略，再决定训练数据合同。
