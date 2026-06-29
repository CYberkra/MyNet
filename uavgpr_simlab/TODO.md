# TODO - UavGPR-SimLab

## P1

- 在本地 3060 和 4090 上分别运行 `scripts\Verify_Current_GPU_Runtime.bat`，确认 PyCUDA / gprMax GPU smoke 通过。
- 在目标机执行 GUI 截图检查，确认批量页默认前台模式和“运行细节/高级诊断”展开模式均可操作。

## P2

- 历史页可继续增强“只重跑当前 case / 当前 variant”。
- ETA 后续可按 family / variant 类型分组估算，提高不同复杂度模型下的准确性。
- 完善 workspace 迁移差异预览的 GUI 展示。

## P3

- 历史页可增加按 family / variant / status 的快速筛选按钮。
- 批量页最近 B-scan 对比条可增加固定 raw / target / clutter_gt 三联视图。
