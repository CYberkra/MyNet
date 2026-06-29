# DEV_HANDOFF - UavGPR-SimLab v0.8.0-alpha.38

当前目标软件是 UavGPR-SimLab。不要混淆 MyGPR。

## 当前主线

```text
设计骨架 → 导入骨架 → 检查合同 → 迁移/修复路径 → 一键运行 → 实时预览 → 运行队列/失败聚合 → 历史复盘
```

## 当前关键能力

- `src/uavgpr_simlab/core/dataset_contract.py`：数据集骨架合同检查。
- `src/uavgpr_simlab/core/run_dashboard.py`：manifest-first 运行看板、ETA 与 runtime profile 汇总。
- `src/uavgpr_simlab/core/workspace_relocator.py`：两台电脑之间迁移 workspace 时修复旧绝对路径。
- `src/uavgpr_simlab/gui/controllers/batch_queue_panel.py`：批量页运行队列树和失败原因聚合面板。
- `src/uavgpr_simlab/gui/controllers/batch_recent_preview.py`：最近完成 B-scan 对比条。
- 历史页支持 pending/running/done/failed 树状浏览、右键打开 case/QC、复制失败原因、只重跑 failed。

## 开发注意

- 不要把 gprMax 或 PyCUDA 放回软件包。
- 新增 Windows 运行入口必须走 `scripts/windows_runtime_bootstrap.bat` 和 `%PY_RUN%`。
- 新增数据集骨架必须通过 `check-dataset-skeleton`、`run-dashboard` 和 `relocate-workspace` 基础检查。
- 批量页体验逻辑应继续拆到 controller helper 或 service，避免 `batch_controller.py` 膨胀。
- 3060 / 4090 的 conda env 必须各自在本机创建，不要跨机器复制。

## 后续优先级

1. 两台真机分别验证 GPU Runtime：`scripts\Verify_Current_GPU_Runtime.bat`。
2. 在目标机做 PySide6 GUI 截图检查。
3. 历史页增强只重跑当前 case / variant。
4. ETA 后续可细化到 family / variant 类型分组。


## a38 handoff note

Batch page is now operator-focused. Do not expose new engineering diagnostics on the front page unless they are needed in normal daily operation. Put them under `运行细节/高级诊断` or in Settings/History. See `docs/OPERATOR_FOCUSED_FRONTEND.md`.
