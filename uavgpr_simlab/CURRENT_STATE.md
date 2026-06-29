# CURRENT_STATE - UavGPR-SimLab v0.8.0-alpha.38

- 版本：`0.8.0a38` / 显示版本：`v0.8-alpha.38`
- 当前重点：把批量页从工程看板收敛为面向单一专家用户的日常操作界面。
- gprMax：外部长期目录，不随软件包发送。
- GPU Runtime：两台电脑各自用统一脚本重建 `D:\UavGPR_Runtime\conda_envs\uavgpr_gprmax_py310_gpu`。

## a38 状态

本轮完成前台简化。批量页默认只显示日常需要的主流程按钮、状态卡片、任务表和 B-scan 结果；运行配置、manifest 路径、variant 标签、任务上限、运行队列、失败聚合、runtime profile 和原始日志被收进“运行细节/高级诊断”。

## 当前未完成

- 3060 / 4090 GPU smoke 仍需目标机实测。
- 实机 PySide6 GUI 截图仍需在 Windows 目标机检查，重点确认高级诊断折叠/展开正常。
- 历史页后续可继续增强“只重跑当前 case / 当前 variant”。
