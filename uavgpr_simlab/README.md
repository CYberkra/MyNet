# UavGPR-SimLab v0.8.0-alpha.38

UavGPR-SimLab 是面向无人机探地雷达仿真、数据集骨架导入、gprMax 批量运行、B-scan 后处理和历史结果复盘的软件。

当前推荐工作流：

```text
设计数据集骨架 → 导入数据集骨架 → 合同检查 → 迁移/修复路径 → 一键 GPU 仿真 → 实时看结果 → 历史复盘 / failed 重跑
```

## v0.8.0-alpha.38 变更

- 批量页改成面向单一专家用户的前台模式：默认只显示导入骨架、修复路径、预检、一键开始、停止、任务状态和 B-scan 结果。
- 运行配置、manifest 路径、variant 标签、任务上限、failed-only / force-rerun、运行队列、失败聚合和原始日志全部保留，但默认收进“运行细节/高级诊断”。
- 不删除诊断能力，只降低日常使用时的界面噪声。
- 保持 gprMax、Miniconda、PyCUDA 环境在外部 RuntimeRoot，不随软件包发送。

## 两台电脑 GPU Runtime

两台电脑都用同一个脚本各自创建环境：

```powershell
.\setup_uavgpr_gpu_runtime_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "你的gprMax源码目录"
.\scripts\Verify_Current_GPU_Runtime.bat
.\run_gui.bat
```

不要复制 conda 环境；3060 和 4090 各自在本机编译 gprMax 扩展和 PyCUDA。

## 主要文档

- 前台简化规则：`docs/OPERATOR_FOCUSED_FRONTEND.md`
- 骨架合同说明：`docs/DATASET_SKELETON_CONTRACT.md`
- 路径迁移说明：`docs/WORKSPACE_RELOCATION.md`
- 运行队列和失败聚合说明：`docs/RUN_QUEUE_FAILURE_PANEL_UX.md`
- ETA 和机器环境看板说明：`docs/RUN_ETA_RUNTIME_PROFILE_UX.md`
- 历史复盘说明：`docs/HISTORY_RUN_MONITOR_UX.md`
