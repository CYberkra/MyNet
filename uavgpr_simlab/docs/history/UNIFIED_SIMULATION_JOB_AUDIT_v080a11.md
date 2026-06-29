# UavGPR-SimLab v0.8.0-alpha.11 统一仿真任务系统审计

## 背景

v0.8.0-alpha.10 已经在 GUI 中提供 ultra tiny 与 25-run smoke 两个专用按钮。用户明确要求：不能随着任务类型增加继续堆按钮，也不能每个任务单独配置实时结果预览与历史联动。

## 本轮目标

将以下任务收敛到同一套批量仿真入口：

- ultra tiny 最小链路验证；
- 25-run smoke 五类场景验证；
- 后续 4090 pilot / formal 数据集；
- 用户手工选择的自定义 SceneWorld manifest。

设置页只保留环境检查、最小 CPU 测试和最小链路验证；所有正式/半正式仿真由批量仿真页的“运行配置 + manifest + 统一 runner”驱动。

## 主要修改

1. 新增 `services/simulation_job_service.py`，定义可复用 `SimulationRunProfile`：
   - `ultra_tiny_check`：1 case × 5 variants，允许显式重采样，仅用于链路验证；
   - `smoke_25run`：5 families × 5 variants，严格 QC；
   - `pilot_4090`：4090 pilot 预留，严格 QC，需用户选择 manifest；
   - `custom`：使用当前 manifest 和参数。
2. 设置页移除 25-run 专用按钮，保留“运行最小链路验证”。
3. 批量仿真页新增“运行配置”下拉框和“应用运行配置”。
4. `run_pending_batch()` 自动判断 manifest 是否为 SceneWorld：
   - SceneWorld manifest 走 `SceneWorldFullChainWorker`；
   - 普通 manifest 仍走原 `LiveQueueWorker`。
5. `sceneworld_bscan_service.py` 增加统一事件：
   - `job_started`
   - `case_started`
   - `variant_started`
   - `variant_output_ready`
   - `variant_done`
   - `case_qc_done`
   - `job_done`
6. 批量页接收事件后实时刷新：
   - 当前 case / variant 状态；
   - 最新 B-scan 画布；
   - 批量表格状态与进度；
   - case QC 完成后刷新历史页。
7. 历史页新增 SceneWorld manifest / QC 适配器。SceneWorld 数据集不依赖旧 job marker，也能显示已完成和失败 variant。
8. `sceneworld_bscan_service.py` 在每个 case 完成 QC 后立即写回 manifest 与 dataset_summary，使长任务运行中也能被历史页读取。

## 未改变内容

- 不改变 `.in` 语义；
- 不改变 gprMax 调用方式；
- 不改变 ultra tiny 的“允许重采样，仅用于链路验证”定位；
- 不改变 smoke/pilot/formal 的严格 QC 原则；
- 不把 pilot / formal 固化成新按钮。

## 风险

- P1：真实 25-run smoke 仍需用户本机验证。
- P2：SceneWorld 统一任务当前不支持强制中断正在运行的 gprMax 子进程，只能提示等待当前子进程结束。
- P2：4090 pilot profile 只是严格 QC 预留入口，正式 50–100 case 数据集仍需在 smoke 全通过后生成。
- P3：批量表格的进度目前以 variant 完成事件为单位，不显示单个 gprMax trace 的内部百分比。

## 验证

- `python -m compileall -q src scripts`：通过。
- `PYTHONPATH=src python scripts/run_all_gprmax.py --help`：通过。
- `services/simulation_job_service.py` profile / manifest 识别：通过。
- SceneWorld 历史适配器在模拟 success manifest 下可返回 5 条 history entries：通过。
- 完整 GUI 启动和真实 25-run gprMax：需目标机验证。
