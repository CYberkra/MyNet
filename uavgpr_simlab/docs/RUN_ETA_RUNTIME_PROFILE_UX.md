# 运行 ETA 与机器环境看板

目标：让操作者在批量仿真页同时知道“还要跑多久”和“当前到底在用哪套 GPU Runtime”。

## 本轮新增

- `core/run_dashboard.py` 从 `reports/sceneworld_bscan_run_report.json` 读取已完成 variant 的 `elapsed_sec`，计算平均耗时。
- 预计剩余时间按 `平均 variant 耗时 × 待运行/运行中任务数` 粗略估算。
- 批量页增加当前运行环境摘要，包括：
  - `UAVGPR_MACHINE_PROFILE`
  - GPU 开关与 GPU ID
  - `nvidia-smi` 查询到的 GPU 名称 / 显存 / 驱动
  - OpenMP 线程
  - Python 解释器
  - gprMax 源码目录
- `run-dashboard` JSON 输出新增 `runtime_profile`、`average_variant_seconds`、`estimated_remaining_seconds`。

## 设计原则

ETA 只作为运行辅助，不作为 QC 依据。若任务规模、模型尺寸或 GPU 负载变化较大，ETA 会随实际完成记录逐步变准。

机器环境摘要只显示必要信息，避免把设置页内容全部堆到前端；正式修改仍在“设置与帮助”页完成。
