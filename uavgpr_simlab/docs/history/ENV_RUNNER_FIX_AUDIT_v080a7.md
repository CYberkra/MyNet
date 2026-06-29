# UavGPR-SimLab v0.8.0-alpha.7 环境与 SceneWorld runner 修复审计

## 修复目标

基于 v0.8.0-alpha.6 深度审计结果，alpha.7 聚焦实机验证前的 P1 风险治理：Windows Python 路径、GUI 默认 conda 策略、gprMax stale `.out`、B-scan strict QC、保留数据集配置一致性和发布包清理。

## 修复项

### P1-1 Windows python.exe 路径

- 移除 SceneWorld runner 中对 `python_executable` 的 `shlex.split()`。
- 普通 Python 模式按单个 executable 处理，例如 `D:\Miniconda3\python.exe` 不会被反斜杠转义破坏。
- conda 模式由 `scripts/run_all_gprmax.py` 显式传入 list：`['conda', 'run', '-n', env, 'python']`。

### P1-2 GUI 默认运行策略

- `EasyEnvironmentSettings` 默认改为 source-tree 模式：`conda_env=''`、`use_conda_run=False`、`omp_threads=1`。
- `load_easy_environment_settings()` 会读取 `GPRMAX_SOURCE_DIR` / `UAVGPR_GPRMAX_ROOT`。
- `windows_runtime_bootstrap.bat` 检测到 gprMax 源码目录后，会设置 `UAVGPR_GPRMAX_ROOT` 和 `UAVGPR_USE_CONDA_RUN=0`。

### P1-3 stale `.out` 清理

- SceneWorld runner 在 `--force` 运行每个 variant 前，删除该 input stem 对应旧 `.out` 文件。
- run report 中记录 `stale_out_files_deleted`。

### P1-4 B-scan shape 严格检查

- 默认不再静默 resample gprMax 输出。
- gprMax 合并输出 shape 必须等于 manifest 期望的 `samples × traces`。
- 不一致时 variant 标记 failed，并写入 `shape_mismatch_without_resample`。

### P1-5 保留骨架配置可移植性

- `yingshan_framework_quick_v080a4` 与 `yingshan_sceneworld_smoke_v080a3` 的 `generated_config.yaml` 已统一为：
  - `conda_env_gprmax: ''`
  - `gprmax_source_dir: ''`
  - `use_conda_run: false`
  - `omp_threads: 1`

## 其他治理

- 新增 `yingshan_sceneworld_ultra_tiny_v080a7`，使用 `wire_tree_endpoint`，包含 wire/tree/building 外部杂波对象，便于验证 `raw - target_only` 的非零 clutter 语义。
- 所有保留 manifest 均包含 `clutter_gt_bscan_npy`、`bscan_qc_report_json`、`bscan_status`、`bscan_error`。
- 发布包只保留三类 workspace：
  - `yingshan_sceneworld_ultra_tiny_v080a7`
  - `yingshan_framework_quick_v080a4`
  - `yingshan_sceneworld_smoke_v080a3`
- 删除 `workspace/self_test_runtime`、`__pycache__`、`.pyc` 和历史乱码文件。

## 验证

- `python -m compileall -q src scripts`：通过。
- `PYTHONPATH=src python scripts/run_all_gprmax.py --help`：通过。
- 三个保留 workspace 的 case package 检查：通过。
- manifest 绝对路径扫描：通过，未发现 Windows 盘符或 Unix 绝对路径。
- `_python_cmd(r'D:\Miniconda3\python.exe')`：返回单元素 list，路径未被破坏。

## 未验证

- Windows GUI 实机启动：需要用户目标机验证。
- ultra-tiny 真实 5 variant gprMax 求解：需要用户目标机验证。
- GPU 模式：本轮仍默认 CPU；GPU 需另行验证。

## 风险

- P0：暂无。
- P1：真实 gprMax 求解仍需目标机验证。
- P2：环境诊断页面的 readiness 分层仍可继续细化，但不阻塞 alpha.7 的 ultra-tiny 实机验证。
