# CHANGELOG

## v0.8.0a38

- Simplified the batch page into an operator-focused front mode.
- Kept only daily-run controls visible by default: import skeleton, relocate/fix paths, precheck, one-click start and stop.
- Moved runtime profile, manifest path, variant tags, task limit, skip/failed/force switches, runtime summary, run queue, failure aggregation and raw logs into a hidden "运行细节/高级诊断" panel.
- Added `docs/OPERATOR_FOCUSED_FRONTEND.md` to document the UI rule: keep routine actions in front and move engineering diagnostics behind an explicit details control.

# Changelog

## v0.8.0-alpha.37 - 运行 ETA 与机器环境看板

- 批量页运行状态增加预计剩余时间和平均 variant 耗时。
- 批量页新增当前运行环境摘要：machine profile、GPU 开关与设备、Python、gprMax、OMP。
- `run-dashboard` 报告新增 `average_variant_seconds`、`estimated_remaining_seconds`、`runtime_profile`。
- SceneWorld 运行事件携带 `elapsed_sec`，便于实时刷新 ETA。
- Easy UI 合同检查覆盖 `batch_runtime_summary_label` 和 `_update_batch_eta_cells`。


## v0.8.0-alpha.36 - 运行队列与失败聚合体验增强

- 批量页新增运行队列树，按 case / variant 展示 running、failed、pending、done。
- 批量页新增失败原因聚合面板，按错误摘要聚合 failed 任务，方便 failed-only 重跑前定位问题。
- 运行队列支持双击跳转历史页；已有 marker 的记录会自动选中对应 case / variant，pending 任务会给出说明。
- 新增 `batch_queue_panel.py`，将运行队列和失败聚合逻辑从主批量 controller 中拆出，避免主窗口继续膨胀。
- Easy UI 合同检查加入 batch queue / failure panel / history jump 防回归项。


## v0.8.0-alpha.35 - 历史复盘与运行中预览体验优化

- 历史页新增右键菜单：打开 case 文件夹、打开 QC JSON、复制失败原因、复制路径摘要、失败定位、只重跑 failed。
- 批量页运行中 B-scan 新增最近完成 variant 对比条，最多保留 5 个结果。
- 历史树 pending / running / done / failed 展示逻辑进一步修正，pending case 不再误显示为 running。
- `check_easy_ui_contract.py` 和 `check_release_integrity.py` 增加对应防回归守卫。
- 新增 `docs/HISTORY_RUN_REVIEW_UX.md`。

# CHANGELOG

## v0.8.0-alpha.34

- 新增 workspace 迁移 / 路径重定位核心：`src/uavgpr_simlab/core/workspace_relocator.py`。
- 新增 CLI：`relocate-workspace`，支持 dry-run、写入、旧根目录替换、相对路径化、备份和迁移后验证。
- 新增脚本：`scripts/check_workspace_relocation.py`。
- 批量仿真页新增“迁移/修复路径”按钮，导入骨架后可直接修复旧电脑绝对路径。
- 发布完整性检查新增 workspace relocator 守卫，防止示例骨架夹带开发机绝对路径。
- 新增 `docs/WORKSPACE_RELOCATION.md`。

## v0.8.0-alpha.33

- 批量页新增“导入数据集骨架”入口。
- 批量页新增运行看板，统一显示即将运行、正在运行、历史完成、失败待处理。
- 历史页新增 pending 状态展示，导入骨架后也能看到即将要跑的模型。
- 新增 `core/run_dashboard.py` 与 CLI `run-dashboard`。
- 修复 `__display_version__` 缺失导致 GUI 实机启动潜在失败的问题。
- 新增 `docs/UX_DATASET_RUN_WORKFLOW.md`。

## v0.8.0-alpha.32

- 新增数据集骨架导入合同检查：`src/uavgpr_simlab/core/dataset_contract.py`。
- 新增 `scripts/check_dataset_skeleton.py` 与 CLI `check-dataset-skeleton`，用于在导入/批量运行前验证 manifest、五变体、相对路径、输入文件和运行 BAT 合同。
- 批量仿真页在预检和启动 SceneWorld 任务前执行 dataset contract，错误会阻止运行，避免骨架不完整时进入 GPU 批量。
- 修复直接 CLI `run-sceneworld-bscan` 不能传 GPU 的问题，新增 `--gpu-ids` / `--no-gpu`。
- 发布守卫现在会验证包内示例 skeleton 的 dataset contract。


## v0.8.0-alpha.31

- 深度逐步审计 Windows GPU Runtime setup 脚本。
- 新增核心 Windows 命令检查：`cmd.exe`、`where.exe`、`chcp.com`、`powershell.exe`。
- 新增目录 root / 盘符存在性检查，避免不存在盘符导致 PowerShell 异常。
- Visual Studio Build Tools 安装参数显式包含 Windows 10/11 SDK 组件。
- build_ext 和 PyCUDA 前均重新加载 MSVC Developer Environment。
- 新增 CUDA `cuda.h` 检查。
- 新增 conda env 构建模块导入检查。
- 新增 gprMax `.pyd` 编译结果数量检查。
- 增强 `check_windows_script_contract.py` 防回归项。

## v0.8.0-alpha.30

- Fixed Windows GPU runtime setup for gprMax Cython extension compilation failures caused by incomplete MSVC/Windows SDK environment initialization.
- Added Visual Studio Developer Environment import through `VsDevCmd.bat` or `vcvars64.bat` before running `setup.py build_ext --inplace`.
- Added pre-build checks for `cl.exe`, `io.h`, and `windows.h`, so missing Windows SDK/UCRT headers fail early with a clear message instead of failing later inside `pyconfig.h`.
- Set `DISTUTILS_USE_SDK=1` and `MSSdk=1` after loading the VS developer environment to keep setuptools/distutils on the initialized toolchain.
- Extended `check_windows_script_contract.py` to guard against regression in MSVC/SDK initialization.

## v0.8.0-alpha.28

- Performed a deeper Windows/runtime audit after a27.
- Hardened the top-level setup launcher to call WindowsPowerShell via explicit SystemRoot path before falling back to powershell.exe.
- Routed Preview_Example_CSV, Run_Full_Pipeline_Example and Setup_GUI_Only through windows_runtime_bootstrap.bat and %PY_RUN%.
- Hardened generic generated BAT commands: quoted pushd, shared runtime bootstrap, and Windows-safe subprocess.list2cmdline quoting.
- Extended check_windows_script_contract.py to guard these launcher and generated-BAT contracts.


## v0.8.0-alpha.27

- Hardened Windows GPU setup against a26-class quoting and PATH propagation failures.
- Replaced setup PowerShell `Start-Process -ArgumentList` single-string execution with native argument-array invocation (`& $exe @argv`) to preserve `python -c` code blocks and quoted paths.
- Refactored `scripts/windows_runtime_bootstrap.bat` Python selection to a label-based flow, avoiding stale `%PY_RUN%` expansion inside parenthesized blocks.
- Simplified `run_gui.bat` to use bootstrap-selected `%PY_RUN%` consistently instead of branching on `UAVGPR_CONDA_ENV`.
- Moved `Generate_3060_Quick_Dataset.bat` onto the shared runtime bootstrap.
- Strengthened `check_windows_script_contract.py` to guard against reintroducing Start-Process quoting, fragile `%PY_RUN%` block expansion, and launcher branch drift.

## v0.8.0a26

- Fixed Windows GPU runtime setup quoting for `python -c` smoke checks in PowerShell/Start-Process.
- Hardened PATH restoration so `chcp`, `powershell`, `cmd`, `where`, and System32 tools remain available inside setup and conda subprocesses.
- Switched build/install/PyCUDA/verify steps to the explicit conda-prefix `python.exe` after environment creation, avoiding fragile `conda run` invocation during setup.
- Runtime profiles now default to `UAVGPR_USE_CONDA_RUN=0` and use `UAVGPR_PYTHON_EXE` / `UAVGPR_CONDA_ENV_PREFIX\python.exe` directly.


## v0.8.0a26

- Added unified multi-machine GPU runtime setup for local RTX 3060 and RTX 4090 laptops.
- Added `setup_uavgpr_gpu_runtime_windows.bat`, `setup_local_3060_gpu_runtime.bat`, and `setup_laptop_4090_gpu_runtime.bat`.
- Added `scripts/Verify_Current_GPU_Runtime.bat`.
- Default GPU conda environment changed to `uavgpr_gprmax_py310_gpu` under RuntimeRoot.
- Runtime env now records `UAVGPR_MACHINE_PROFILE`, `UAVGPR_GPU_RUNTIME_ENV`, and `UAVGPR_RUN_SCALE`.
- gprMax remains external and persistent; no gprMax zip is bundled.

## v0.8.0-alpha.24

- Fixed local Windows diagnostics for source-tree gprMax mode: when conda run is disabled, missing conda is now optional instead of a failing requirement.
- Added source-tree gprMax import validation by injecting `PYTHONPATH=<gprMax root>` before running the current Python diagnostic. This matches the actual runner behavior and supports valid compiled gprMax source trees without `pip install gprMax`.
- Fixed `scripts/windows_runtime_bootstrap.bat` so it no longer invents `UAVGPR_CONDA_ENV=gprMax` when no RuntimeRoot conda environment exists; this prevents the GUI from defaulting to conda run on local machines without conda.
- Added `scripts/Configure_Local_CPU_GprMax.bat` to write a local `.simlab_env` for CPU/source-tree validation using a user-provided gprMax root and Python executable.
- Added `docs/LOCAL_SOURCE_TREE_RUNTIME_v080a24.md`.

## v0.8.0-alpha.23

- Fixed PowerShell setup path probing on machines without an `E:` drive by adding quiet path checks and safe gprMax source validation.
- Hardened Miniconda bootstrap: the installer download now tries TLS 1.2, `Invoke-WebRequest`, `curl.exe`, and BITS. If the download still fails, the script can fall back to an existing conda executable as a controller while keeping the actual gprMax/PyCUDA environment under `RuntimeRoot\conda_envs\gprMax`.
- Added `-NoExternalCondaFallback` for strict users who want the script to fail instead of reusing an existing conda controller.
- Kept gprMax as a persistent external runtime asset; release packages still do not bundle `gprMax-v.3.1.7.zip`.

## v0.8.0-alpha.22

- Updated version metadata to `0.8.0a22`.
- Removed bundled `gprMax-v.3.1.7.zip` from the release package. gprMax is now treated as a persistent external runtime asset.
- Updated Windows setup scripts to prefer `-GprMaxDir`, `RuntimeRoot\uavgpr_runtime.env`, and stable RuntimeRoot locations for gprMax source discovery.
- Disabled automatic gprMax cloning by default; cloning now requires explicit `-AllowCloneGprMax`.
- Updated release and Windows-script guards to enforce the external-gprMax contract and to fail if a gprMax zip is bundled again.

# CHANGELOG - UavGPR-SimLab

## v0.8.0-alpha.21

- Updated version metadata to `0.8.0a21`.
- Added centralized `-RuntimeRoot` setup for Windows 4090 machines: Miniconda, conda env prefix, gprMax source, downloads and setup logs are kept under one root, defaulting to `E:\UavGPR_Runtime` when available.
- Bundled `gprMax-v.3.1.7.zip` in the release package and made the one-click setup prefer the local zip before cloning.
- Added `.simlab_env` keys `UAVGPR_RUNTIME_ROOT`, `UAVGPR_MINICONDA_DIR`, `UAVGPR_CONDA_EXE`, and `UAVGPR_CONDA_ENV_PREFIX`; bootstrap now supports `conda run -p <env-prefix>` without requiring conda on PATH.
- Added runtime preflight for SceneWorld batch execution: selected Python, gprMax import/help, and PyCUDA/CUDA driver check when GPU is enabled.
- Added fatal gprMax startup error classification for missing PyCUDA, CUDA driver initialization failure, and missing gprMax import; SceneWorld runner now fails fast instead of repeating the same environment failure across all variants.
- Removed duplicate SceneWorld batch log lines for case/run/failed/status events.
- Hardened Windows 4090 setup scripts: PyCUDA install, PyCUDA driver import, and final GPU smoke verification now fail the setup with non-zero exit codes when GPU mode is requested.

## v0.8.0-alpha.18

- Updated version metadata to `0.8.0a18`.
- Added `scripts/check_release_integrity.py` for release-level version, entrypoint, run-plan, `.simlab_env`, workspace skeleton and docs-root checks.
- Added `scripts/check_easy_ui_contract.py` for EasyMainWindow/controller/page-widget behavior-contract checks.
- Added `scripts/check_windows_script_contract.py` for Windows BAT, PowerShell setup, runtime bootstrap and generated gprMax BAT static checks.
- Fixed `scripts/windows_runtime_bootstrap.bat` so persisted GPU IDs, GPU flags and OpenMP thread settings are loaded from `.simlab_env`.
- Added `scripts/audit_yingshan_real_data_package.py`, `docs/real_data/YINGSHAN_REAL_DATA_AUDIT_v080a18.md`, `docs/real_data/YINGSHAN_REAL_DATA_AUDIT_v080a18.json`, and `configs/yingshan_real_data_inventory.yaml` for real Yingshan data package pre-audit.

## v0.8.0-alpha.17

- Updated version metadata to `0.8.0a17`.
- Split `src/uavgpr_simlab/gui/easy_window.py` into a compact window shell and page-behavior controller mixins under `src/uavgpr_simlab/gui/controllers/`.
- Added `scripts/check_architecture_guard.py` to prevent `easy_window.py` and root `docs/` from regressing.
- Moved historical audits, old UI reports, and phase repair notes from `docs/` into `docs/history/`; added `docs/history/README.md`.
- Added `docs/CURRENT_ARCHITECTURE_v080a17.md` to document current architecture boundaries.
- No change to gprMax/CUDA/PyCUDA/4090 GPU runtime semantics.

## v0.8.0-alpha.16

- Added RTX 4090 Windows gprMax environment file `configs/environment_gprmax_4090_windows.yml` with a pinned Python 3.10 conda stack.
- Reworked Windows gprMax setup scripts to prefer local `gprMax-v.3.1.7.zip`, build gprMax extensions, install PyCUDA, persist `.simlab_env`, and run a CPU/GPU smoke verifier.
- Added `scripts/check_4090_gprmax_gpu.py` and `scripts/Verify_4090_GPRMAX_GPU.bat`.
- Updated `scripts/windows_runtime_bootstrap.bat` so generated dataset BAT files, GUI launchers and verification scripts share `.simlab_env`.
- Updated SceneWorld GUI full-chain runner to honor conda environment and GPU settings from the settings page.
- Updated 4090 formal and validation run plans to current SceneWorld five-variant schema.
- Improved environment diagnostics subprocess handling to avoid unbounded hangs in self-test / GUI diagnostics.

## v0.8.0-alpha.15

- Fixed the user-reported batch launch crash when the manifest field was empty or pointed to a directory. `Path("")` previously resolved to `.` and reached `manifest_csv.open(...)`, causing Windows `PermissionError: [Errno 13] Permission denied: '.'`.
- Added `require_manifest_file()` in `easy_batch_service.py` and reused it from the GUI batch launch path and pending-task builder.
- Added `is_file()` guards for model-manifest loading, batch precheck, `tasks_from_manifest()`, and `manifest_input_files()`. Invalid manifest paths now produce actionable UI warnings instead of tracebacks.
- Updated version metadata to `0.8.0a15` and added `docs/history/MANIFEST_PATH_GUARD_AUDIT_v080a15.md`.

## v0.8.0-alpha.14

- 批量仿真页新增统一运行配置，下拉选择 ultra tiny、25-run smoke、4090 pilot 或 custom。
- 设置页移除 25-run 专用按钮，只保留最小链路验证。
- SceneWorld runner 新增 job/case/variant/QC 事件。
- 批量页实时显示最新 B-scan，case QC 完成后刷新历史页。
- 历史页新增 SceneWorld manifest / QC 结果适配，不再只依赖旧 job marker。
- 新增 `docs/history/UNIFIED_SIMULATION_JOB_AUDIT_v080a11.md`。

## v0.8.0-alpha.10

- Added GUI 25-run smoke validator: 5 Yingshan SceneWorld families × 1 case per family × 5 variants.
- Regenerated `workspace/yingshan_sceneworld_smoke_v080a14` and `workspace/yingshan_sceneworld_ultra_tiny_v080a14`.
- Kept ultra tiny GUI validator for fast target-machine checks; smoke validator runs strict QC without resampling.
- Added Settings page button `运行 25-run smoke 验证` with live gprMax log streaming and per-case summary.

## v0.8.0-alpha.10

- Fixed the user-reported `clutter_only.in` gprMax geometry failure: a 0.05 m surface proxy was thinner than the 0.25 m FDTD grid and was rejected as an invalid `#box`.
- Added SceneWorld `#box` safe generation and validation in `scene_variant_writer.py`; generated boxes are clipped to domain bounds and widened to at least one grid cell where required.
- Regenerated `yingshan_sceneworld_ultra_tiny_v080a14` and made the GUI ultra-tiny verifier prefer it.
- Kept ultra-tiny explicit resampling for chain validation only; formal smoke/pilot workflows remain strict-QC by default.
- Did not change gprMax source bootstrap, SceneWorld semantics, or `constant_level` trajectory metadata.

## v0.8.0-alpha.8

- Added GUI button “运行 ultra tiny 全链路验证” on 设置与帮助页. It runs the SceneWorld ultra-tiny 1 case × 5 variant chain in a background worker and streams progress/log messages into the GUI.
- Added `SceneWorldUltraTinyWorker` for GUI-side full-chain validation without requiring the user to open a separate BAT window.
- Added `--allow-resample` to `scripts/run_all_gprmax.py`. This is enabled only for ultra-tiny chain validation so `.out` extraction can be aligned to the tiny expected shape while recording the original extracted shape.
- Preserved strict shape QC by default for smoke/pilot datasets. Formal training/pilot workflows must not silently resample.
- Improved gprMax run diagnostics: non-zero return codes now record explicit `gprMax_returncode_*`, stdout tail, merged shape, output files, extracted raw B-scan path and resampling state.
- Regenerated ultra tiny skeleton as `yingshan_sceneworld_ultra_tiny_v080a8`.
- Updated documentation and audit notes for the user-reported `nan_or_inf_present` failure, which was caused by placeholders not being replaced.

## v0.8.0-alpha.7

- Fixed Windows Python executable handling in SceneWorld gprMax runner; full paths like `D:\Miniconda3\python.exe` are no longer parsed with `shlex.split`.
- Switched easy GUI defaults to source-tree mode: current Python + `GPRMAX_SOURCE_DIR` / `PYTHONPATH`, no implicit `conda run -n gprMax`.
- Added stale `.out` cleanup before forced SceneWorld reruns to avoid mixing previous gprMax outputs.
- Made SceneWorld B-scan QC strict by default: gprMax output shape must match expected `samples × traces`; silent resampling is removed from the default runner.
- Regenerated ultra-tiny verification skeleton as `yingshan_sceneworld_ultra_tiny_v080a7` using `wire_tree_endpoint` so `raw - target_only` has non-zero clutter semantics.
- Normalised retained dataset `generated_config.yaml` files to portable source-tree defaults.
- Unified retained manifest schema with `clutter_gt_bscan_npy`, `bscan_status`, and `bscan_error`.
- Cleaned release package outputs: removed self-test runtime artifacts, `__pycache__`, `.pyc`, and obsolete mojibake historical document.


## v0.8.0-alpha.7

- 新增 `core/scene_world.py`，定义 SceneWorld、TrajectoryModel、InterbedLayer 和 SceneObject。
- 新增 `simulation/scene_world_generator.py`、`scene_variant_writer.py` 和 `yingshan_families.py`，开始支持营山同源仿真数据集骨架。
- `generate_cases()` 改为每个 case 先生成一个 SceneWorld，再派生 `raw / target_only / background_only / clutter_only / air_only`，避免 variant 间重新随机。
- 当前 alpha 实现 `gentle_interbed` 和 `wire_tree_endpoint` 两个场景族。
- manifest 新增 `family`、`random_seed`、`scene_world_json`、`metadata_summary_json`、`interface_gt_npy`、`layer_gt_npy`、`flight_height_mode`、`model_length_actual_m` 等字段。
- `model_length_m` 改为实际仿真域长度；配置值保存在 `model_length_config_m`。
- 模型预览与 metadata 明确标注 `constant_level`，不再把固定高度 FDTD 路径误称为真实仿地飞行。
- 新增 `configs/run_plan_yingshan_sceneworld_smoke.yaml`。
- 新增 `docs/history/SCENEWORLD_ALPHA_AUDIT.md`。
- 本轮不改变正式 gprMax 调用、fingerprint、marker 或 B-scan 后处理。


## v0.7.26

- 模型预览页新增“3D 视图 / 2D 剖面”点击切换，默认显示 3D。
- 2D 剖面复用现有 `render_model_preview()` 生成的模型剖面图，不改变 manifest、模型生成或 gprMax 仿真语义。
- “打开 2D 剖面图路径”会自动切换到 2D 视图，并显示当前模型剖面图路径。
- 新增 `docs/history/MODEL_PREVIEW_2D3D_TOGGLE_AUDIT.md` 记录本轮 UI 边界。
- 本轮不改变模型生成、批量仿真、fingerprint、marker 或 B-scan 后处理。

## v0.7.25

- 项目管理页新增“模型配置”下拉框，支持一键生成模型前选择 `configs/run_plan*.yaml`。
- 新增 `ModelPlanPreset` 和 `discover_model_plan_presets()`，只暴露真正的模型生成 run-plan，不混入材料表、ML 配置或 pipeline 配置。
- 选择模型配置后自动回填“仿真计划”路径、清空旧 manifest、刷新计划预览，并在状态栏显示当前配置。
- “生成一批模型”现在会在完成提示中显示所使用的模型配置文件名。
- 新增 `docs/history/MODEL_PLAN_SELECTION_AUDIT.md` 记录本轮功能边界。
- 本轮不改变模型生成语义、manifest 结构、gprMax 调用、fingerprint、marker 或 B-scan 后处理。

## v0.7.24

- 修复易用界面“生成一批模型”在 Windows 下触发的 `TypeError: 'WindowsPath' object is not iterable`。
- `services/project_service.generate_model_batch()` 现在正确处理 `core.scenario.generate_cases()` 返回的 `(models_dir, manifest_path)`，并按 manifest 唯一 `case_id` 统计模型数量。
- 新增 `find_latest_manifest()`，支持“加载模型图库”在 manifest 输入为空时自动发现 `<workspace>/datasets/` 或 `<workspace>/*/datasets/` 下的最新模型清单。
- “加载模型图库”不再静默返回；找不到清单或清单为空时会给出明确中文提示。
- 修复 `run_gui.bat` 与 `启动_UavGPR-SimLab.bat` 的 Windows 双击启动稳定性，使用 quoted SET 处理路径和 `PYTHONPATH`。
- 新增 `docs/history/MODEL_GALLERY_BUGFIX_AUDIT.md` 记录本轮问题、原因和边界。
- 本轮不改变模型生成语义、manifest 结构、gprMax 调用、fingerprint、marker 或 B-scan 后处理。

## v0.7.23

- 一次性完成高级工程界面剩余 UI 页签拆分：工作台、仿真计划、3D 预览、预检去重、结果/QC、PGDA/训练预留。
- 新增 `src/uavgpr_simlab/gui/advanced_pages/dashboard_tab.py`、`generation_tab.py`、`model_preview_tab.py`、`preflight_tab.py`、`qc_tab.py`、`train_tab.py`。
- `main_window.py` 不再直接构建高级界面 10 个页签的主体 UI；仍负责跨页状态、服务层调用、worker 生命周期、表格数据填充、画布刷新、错误弹窗和状态栏。
- 自测新增高级工作台、仿真计划、3D 预览、预检去重、QC、训练页控件断言。
- 新增 `docs/history/ADVANCED_REMAINING_TABS_AUDIT.md`，记录本轮高级页签拆分边界。
- 新增 `docs/UAVGPR_APPLICATION_CONTEXT_FROM_PDF.md`，归档用户上传 UAV-GPR 汇报 PDF 对软件需求的业务背景参考。
- 本轮不改变正式 gprMax 命令、fingerprint、marker、B-scan 后处理、CSV 解析、模型生成或历史管理语义。

## v0.7.22

- 新增 `src/uavgpr_simlab/services/real_csv_service.py`，承接高级工程界面真实 CSV 预览数据准备和 QC 导出协调。
- `main_window.py` 不再直接调用 `read_uavgpr_csv()`、`subtract_mean_background()` 或 `convert_real_csv()`；仍负责真实 CSV 页签回调、画布刷新、f-k 显示触发和后台 worker 调度。
- 自测新增真实 CSV 服务层断言，覆盖 `load_real_csv_preview()` 和 `export_real_csv_qc()`。
- 新增 `docs/history/REAL_CSV_SERVICE_AUDIT.md`，记录真实 CSV 服务层边界。
- 本轮不改变 `core.real_data` 的 CSV 解析、背景扣除、robust normalize、指数增益、SNR 或 NPZ/PNG/JSON 导出语义。

## v0.7.21

- 新增 `src/uavgpr_simlab/gui/advanced_pages/real_csv_tab.py`，承接高级工程界面的“7 实测/弱监督”页签 UI 构建。
- `main_window.py` 不再直接构造真实 CSV 页签中的路径输入、最大道数、加载预览、f-k 和 QC 导出控件；仍保留 CSV 读取、预览、f-k 和 QC 导出回调。
- 自测新增高级真实 CSV 页签控件断言。
- 新增 `docs/history/ADVANCED_REAL_CSV_TAB_AUDIT.md`，记录高级真实 CSV 页签拆分边界。
- 本轮不改变真实 CSV 解析、背景扣除 / robust normalize、f-k 预览、NPZ / PNG 质控导出、gprMax 调用或 history marker。

## v0.7.20

- 新增 `src/uavgpr_simlab/gui/advanced_workers.py`，承接高级工程界面的 `GenericWorker` 和 `LiveQueueWorker`。
- `main_window.py` 不再直接定义通用后台任务和实时队列运行 worker；仍负责 worker 创建、信号连接、按钮状态恢复、实时预览刷新和页面状态协调。
- `LiveQueueWorker` 的正式 gprMax 进程调用、conda run/GPU/OpenMP 参数、job fingerprint、history marker、done/failed/running 状态和 B-scan 后处理语义保持不变。
- 自测新增高级 worker 类挂接断言，确认高级界面仍能离屏启动并识别 `GenericWorker` / `LiveQueueWorker`。
- 新增 `docs/history/ADVANCED_WORKERS_AUDIT.md`，记录高级 worker 拆分边界和后续风险。

## v0.7.19

- 新增 `src/uavgpr_simlab/services/advanced_queue_service.py`，承接高级工程界面队列页签的 manifest 预览、BAT 生成、选中行转任务、批量任务构造和任务摘要。
- `main_window.py` 不再直接读取队列 manifest CSV、拼接队列列表显示文本、生成队列 BAT 或把 UI 行数据转换为 `GprMaxTask`；仍保留 `LiveQueueWorker`、实时预览、进度更新、fingerprint、marker 和正式 gprMax 运行回调。
- 自测新增高级队列服务断言，确认 manifest 预览截断、选中行转任务和批量任务摘要正常。
- 新增 `docs/history/ADVANCED_QUEUE_SERVICE_AUDIT.md`，记录高级队列服务层拆分边界和后续队列 worker/运行语义治理建议。
- 本轮不改变正式 gprMax 调用语义、conda run、GPU/OpenMP 参数、任务 fingerprint、history marker、队列运行或 B-scan 后处理。

## v0.7.18

- 新增 `src/uavgpr_simlab/gui/advanced_widgets/` 高级工程界面复用组件包。
- 新增 `src/uavgpr_simlab/gui/advanced_widgets/canvases.py`，承接 `MplCanvas` 和 `Model3DCanvas`。
- `main_window.py` 不再直接定义高级 B-scan/f-k 画布和 3D/2.5D 模型预览画布；仍保留数据来源协调、历史详情刷新、队列实时预览和运行回调。
- 自测新增高级画布类挂接断言，确认队列画布、历史 B-scan 画布和模型 3D 画布均来自新组件模块。
- 新增 `docs/history/ADVANCED_CANVASES_AUDIT.md`，记录高级画布拆分边界和后续高级队列服务治理建议。
- 本轮不改变正式 gprMax 调用语义、conda run、GPU/OpenMP 参数、任务 fingerprint、history marker、队列运行或 B-scan 后处理。

## v0.7.17

- 新增 `src/uavgpr_simlab/gui/advanced_pages/queue_tab.py`，承接高级工程界面的“5 批量运行”页签 UI 构建。
- `main_window.py` 不再直接构造高级队列页签中的 manifest 输入、variant、批量数量、geometry-only/跳过/强制重跑开关、BAT/运行/停止按钮、任务列表、进度条、日志区和实时 B-scan 画布；仍保留 manifest 读取、BAT 生成、任务转换、LiveQueueWorker、实时预览和进度回调。
- 自测新增高级队列页签控件断言，确认高级界面仍有 10 个页签、队列默认参数、日志只读和预览提示正常挂接。
- 新增 `docs/history/ADVANCED_QUEUE_TAB_AUDIT.md`，记录高级队列页签拆分边界和后续高级画布/队列服务治理建议。
- 本轮不改变正式 gprMax 调用语义、conda run、GPU/OpenMP 参数、任务 fingerprint、history marker、队列运行或 B-scan 后处理。

## v0.7.16

- 新增 `src/uavgpr_simlab/gui/advanced_pages/history_tab.py`，承接高级工程界面的“6 历史记录”页签 UI 构建。
- `main_window.py` 不再直接构造高级历史页签中的 workspace 输入、筛选、缩略图开关、刷新/导出/删除按钮、历史表格、模型画布、B-scan 画布和详情区；仍保留历史扫描、缩略图填充、详情预览、导出、删除和 B-scan 加载逻辑。
- 自测新增高级历史页签控件断言，确认高级界面仍有 10 个页签、历史筛选 7 项、历史表格 13 列、日志和详情框只读。
- 新增 `docs/history/ADVANCED_HISTORY_TAB_AUDIT.md`，记录高级历史页签拆分边界和后续高级界面治理顺序。
- 本轮不改变历史扫描语义、历史导出/删除、B-scan 加载、gprMax 调用、任务 fingerprint 或 history marker。

## v0.7.15

- 新增 `src/uavgpr_simlab/gui/advanced_pages/` 高级工程界面页面组件包。
- 新增 `src/uavgpr_simlab/gui/advanced_pages/env_tab.py`，承接高级工程界面的“1 环境检查”页签 UI 构建。
- `main_window.py` 不再直接构造高级环境页签中的工作目录、gprMax 源码目录、conda 环境、GPU/OpenMP 选项、操作按钮和日志区；仍保留保存、检查、smoke 命令和安装脚本回调。
- 自测新增高级环境页签控件断言，确认高级界面仍有 10 个页签、环境日志只读、smoke 命令按钮正常挂接。
- 新增 `docs/history/ADVANCED_ENV_TAB_AUDIT.md`，记录高级环境页签拆分边界和后续高级界面治理顺序。
- 本轮不改变正式 gprMax 调用语义、conda run、GPU/OpenMP 参数、任务 fingerprint、history marker、队列运行或 B-scan 后处理。

## v0.7.14

- 新增 `src/uavgpr_simlab/gui/easy_workers.py`，承接易用界面的后台诊断 worker。
- 设置页“最小 CPU 测试”从同步执行改为 `QThread` 后台执行，避免 gprMax 源码 smoke test 期间阻塞 GUI 主线程。
- `easy_window.py` 仅负责启动 worker、接收完成/失败信号、刷新日志和恢复按钮状态。
- 新增 `docs/history/SETTINGS_SMOKE_WORKER_AUDIT.md`，记录后台 worker 边界、验证方式和后续风险。
- 本轮不改变正式批量仿真运行语义、gprMax GPU 调用、conda run、任务 fingerprint、history marker 或 B-scan 后处理。

## v0.7.13

- 新增 `src/uavgpr_simlab/services/gprmax_smoke_service.py`，把 gprMax 源码最小 CPU smoke test 从 CLI 脚本沉到服务层，供脚本和 GUI 复用。
- `scripts/smoke_gprmax_source.py` 改为服务层包装脚本，减少重复实现。
- 设置页新增“最小 CPU 测试”按钮，可从 GUI 调用本地 gprMax 源码极小 A-scan CPU 测试，并在日志区显示中文摘要和原始 JSON。
- 自测新增设置页 smoke 按钮挂接和 smoke 报告格式化断言。
- 已再次使用用户提供的 gprMax 源码执行最小 CPU smoke test，输出记录在 `workspace/gprmax_source_smoke_v0713/`。
- 本轮不改变正式批量仿真运行语义、gprMax GPU 调用、任务 fingerprint、history marker 或 B-scan 后处理。

## v0.7.12

- 新增 `scripts/smoke_gprmax_source.py`，可对本地 gprMax 源码树执行最小 CPU smoke test。
- `services/environment_service.py` 的 gprMax 源码诊断新增 Cython/OpenMP 编译扩展数量检查。
- 已在当前 sandbox 对用户提供的 gprMax 源码完成编译扩展检测与极小 CPU A-scan 运行，生成 HDF5 `.out`，结果记录在 `docs/history/GPRMAX_LOCAL_SOURCE_SMOKE_RESULT.md`。
- 新增 `docs/history/GPRMAX_SOURCE_SMOKE_SCRIPT_AUDIT.md`，明确 smoke 脚本边界。
- 本轮不改变 GUI 页面流程、gprMax 调用语义、任务 fingerprint、history marker 或 B-scan 后处理。

## v0.7.11

- 新增 `docs/GPRMAX_SMOKE_TEST_TEMPLATE.md`，用于 Windows + conda + CUDA / CPU 目标机记录真实 gprMax 验收结果。
- 新增 `docs/history/MAIN_WINDOW_AUDIT.md`，完成高级工程界面 `main_window.py` 快速审计，明确后续高级页签拆分顺序。
- 新增 `docs/history/ENV_DIAGNOSTICS_AUDIT.md`，记录设置页环境诊断中文摘要增强。
- `services/environment_service.py` 新增 `format_easy_environment_report()`，设置页环境检查现在先显示中文摘要，再保留原始 JSON。
- 自测新增环境诊断格式化输出断言。
- 本轮不改变 gprMax 调用语义、任务 fingerprint、history marker、B-scan 后处理、模型生成或批量运行逻辑。


## v0.7.10

- 新增 `src/uavgpr_simlab/gui/pages/home_page.py`，承接产品化易用界面的首页 UI 构建。
- `easy_window.py` 不再直接构造首页状态卡、最近 B-scan 预览区、下一步提示区和首页流程步骤条，只保留首页统计刷新、B-scan 画布刷新和跨页面状态协调。
- 自测新增首页控件断言，确认首页 B-scan 画布、下一步提示和项目指标卡数值标签正常挂接。
- 新增 `docs/history/HOME_PAGE_AUDIT.md` 记录首页拆分边界、验证方式和后续真实环境验证建议。
- 本轮不改变首页统计来源、历史扫描、模型生成、批量预检、gprMax 调用语义、任务 fingerprint 或 history marker。

## v0.7.9

- 新增 `src/uavgpr_simlab/gui/pages/model_preview_page.py`，承接产品化易用界面的模型预览页 UI 构建。
- `easy_window.py` 不再直接构造模型图库、3D 预览画布和模型信息卡，只保留 manifest 加载、预览刷新、批量同步和相关状态协调。
- 自测新增模型预览页控件断言，确认模型页输入框、列表、信息标签和 3D 画布正常挂接。
- 新增 `docs/history/MODEL_PREVIEW_PAGE_AUDIT.md` 记录模型预览页拆分边界、验证方式和后续首页拆分建议。
- 本轮不改变模型生成服务、manifest 结构、3D label 解析、预览图生成、gprMax 调用语义、任务 fingerprint 或 history marker。

## v0.7.8

- 新增 `src/uavgpr_simlab/gui/pages/history_page.py`，承接产品化易用界面的历史与结果页 UI 构建。
- `easy_window.py` 不再直接构造历史页中的状态筛选、刷新/导出/重跑/删除按钮、历史记录列表、模型画布、B-scan 画布和详情标签。
- 自测补充历史页控件断言，确认离屏启动后历史筛选、列表间距和详情标签挂接正常。
- 新增 `docs/history/HISTORY_PAGE_AUDIT.md` 记录历史页拆分边界、验证方式和后续页面拆分建议。
- 本轮不改变历史扫描、B-scan 加载、历史导出、删除记录、gprMax 调用语义、任务 fingerprint 或 history marker。

## v0.7.7

- 新增 `src/uavgpr_simlab/gui/pages/batch_page.py`，承接产品化易用界面的批量仿真页 UI 构建。
- `easy_window.py` 不再直接构造批量页中的 manifest 输入、variant 输入、任务数限制、跳过开关、统计卡片、流程步骤、任务表、实时 B-scan 画布和日志区。
- 自测补充批量页控件断言，确认离屏启动后批量页控件挂接、任务表列数和日志只读状态正常。
- 新增 `docs/history/BATCH_PAGE_AUDIT.md` 记录批量页拆分边界、验证方式和后续页面拆分建议。
- 本轮不改变 gprMax 调用语义、批量预检、待运行任务构造、任务 fingerprint、history marker 或 B-scan 后处理。

## v0.7.6

- 新增 `src/uavgpr_simlab/gui/pages/project_page.py`，承接产品化易用界面的项目管理页 UI 构建。
- `easy_window.py` 不再直接构造项目管理页中的工作目录、仿真计划、模型数量和计划预览控件，只保留跨页面状态、回调绑定和服务调用。
- 新增 `docs/history/PROJECT_PAGE_AUDIT.md` 记录项目页拆分边界、验证方式和后续页面拆分建议。
- 本轮不改变模型生成服务、计划解析、manifest 结构、gprMax 调用语义、任务 fingerprint、history marker 或 B-scan 后处理。

## v0.7.5

- 新增 `src/uavgpr_simlab/gui/pages/` 页面组件包。
- 新增 `src/uavgpr_simlab/gui/pages/settings_page.py`，承接产品化易用界面的设置与帮助页 UI 构建。
- `easy_window.py` 不再直接构造设置页中的环境表单、帮助步骤和日志区，只保留回调绑定、状态读取和服务调用。
- 新增 `docs/history/SETTINGS_PAGE_AUDIT.md` 记录设置页拆分边界、验证方式和后续页面拆分建议。
- 本轮不改变 gprMax 调用语义、环境诊断服务、任务 fingerprint、history marker 或 B-scan 后处理。

## v0.7.4

- 新增 `src/uavgpr_simlab/services/environment_service.py`，承接易用界面的环境设置读取/保存、运行配置组装和 gprMax 源码目录结构诊断。
- 新增 `src/uavgpr_simlab/services/project_service.py`，承接项目计划预览和模型批次生成，继续降低 `easy_window.py` 对 core/cli 的直接依赖。
- 设置页环境检查现在会输出包含 `gprmax_source` 的扩展报告，能区分“目录存在”和“是否像 gprMax 源码根目录”。
- 自测新增 Easy 项目/环境服务检查。
- 已只读审视用户提供的 `gprMax-v.3.1.7.zip`：其源码树结构有效；压缩包目录名为 v.3.1.7，但内部 `_version.py` 标识为 `3.1.6`。
- 本轮不修改 gprMax 源码、不改变 gprMax 命令语义、任务 fingerprint、history marker 或 B-scan 后处理。

## v0.7.3

- 新增 `src/uavgpr_simlab/gui/easy_cards.py`，承接页面标题、指标卡、流程步骤、模型信息行、历史记录卡和帮助步骤等 GUI 小控件。
- `easy_window.py` 继续收敛为页面组合与事件响应，避免卡片/历史列表渲染逻辑继续堆在主窗口。
- 新增 `docs/history/EASY_WIDGETS_AUDIT.md`，记录 GUI 小控件拆分边界和后续拆分建议。
- 本轮不改变 gprMax 调用、任务 fingerprint、history marker、B-scan 后处理或模型生成语义。

## v0.7.2

- 新增 `src/uavgpr_simlab/services/` 服务层包。
- 新增 `easy_batch_service.py`，承接 manifest 读取、variant 解析、批量预检、job plan 写入、待运行任务构造和批量缩略图准备。
- 新增 `easy_history_service.py`，承接首页状态统计、历史记录扫描、marker 读取、历史预览、B-scan 加载、导出和删除。
- `easy_window.py` 不再直接调用任务 registry、runner、history 扫描/导出/删除等 core API，GUI 职责进一步收敛到展示与触发。
- 新增 `docs/history/EASY_SERVICE_LAYER_AUDIT.md` 记录服务层边界和后续拆分建议。
- 本轮不改变 gprMax、任务 fingerprint、history marker、B-scan 后处理或模型生成语义。

## v0.7.1

- 新增 `src/uavgpr_simlab/gui/easy_ui.py`，把 v0.7 易用界面的样式、状态标签、缩略图、标题和表格辅助函数从主窗口拆出。
- 新增 `docs/history/EASY_WINDOW_REFACTOR_AUDIT.md`，记录 `easy_window.py` 的拆分边界、风险和后续顺序。
- 更新 `DEV_HANDOFF.md`、`CURRENT_STATE.md`、`TODO.md` 和 `README.md`，把本轮结构整理纳入后续交接。
- 本轮不改变 gprMax、任务 registry、历史记录或 B-scan 后处理语义。

## v0.7.0

当前产品化易用 UI 版本。

### Changed

- 默认启动产品化易用界面。
- 保持“看模型 → 跑批量 → 看结果”的用户主线。
- 高级工程界面改为显式高级入口。
- 统一项目版本标识为 `0.7.0` / `v0.7`。
- 整理 README，使当前版本说明与历史版本资料分离。
- 补齐 `CURRENT_STATE.md`、`DEV_HANDOFF.md`、`TODO.md`。
- 修复 v0.7 截图总览图中文字体渲染问题。

### Verified

- `python -m compileall -q src scripts`
- `PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/self_test.py`
- `PYTHONPATH=src QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/make_v070_product_screenshots.py`

## v0.6

易用化界面重构版本。默认 GUI 从工程参数页签式界面改为更直观的工作台。详细说明见：

```text
docs/history/V0_6_EASY_UI_REDESIGN.md
```

## v0.5.5

最终审计修复版。保留任务去重、历史记录、B-scan 预览和批量运行链路。详细说明见：

```text
docs/history/V0_5_5_FINAL_AUDIT.md
```

## v0.5.4 及更早

历史审计、自动化 pipeline、安全批量运行和论文数据生产流程说明保留在 `docs/` 中。

## v0.8.0-alpha.3

- Added SceneWorld B-scan runner service and scripts for replacing NaN placeholders with real gprMax outputs.
- Added per-case `bscan_qc_report.json` generation and QC update logic.
- Added `run_plan_yingshan_sceneworld_smoke_v080a3.yaml` for one-case-per-family B-scan smoke execution.
- Added `run_plan_yingshan_sceneworld_pilot_v080b1.yaml` with 501 samples, 700 ns, 300 traces, 60 cases and ~8.3% high-relief cases.
- Added SceneWorld case package integrity checker.

## v0.8.0-alpha.7

- Added `scripts/run_all_gprmax.py`, a cross-platform SceneWorld runner that executes all requested variants and replaces B-scan placeholders.
- Generated SceneWorld `logs/run_all_gprmax.bat` now runs `raw,target_only,background_only,clutter_only,air_only`, not raw only.
- B-scan QC reports now use `success` / `failed` and record NaN/Inf, min/max/mean/std, shape and clutter ground-truth generation.
- Manifest rows now include `bscan_status` and `bscan_error`; dataset summary propagates run success/failure.
- Added `run_plan_yingshan_framework_quick_v080a4.yaml` and generated a small framework-validation skeleton.
- Cleaned packaged workspace to retain only the latest v080a3 ready-to-run skeleton and the quick framework skeleton.


## v0.8.0-alpha.7

- Added ultra-tiny SceneWorld full-chain verification skeleton `yingshan_sceneworld_ultra_tiny_v080a7`.
- Added `configs/run_plan_yingshan_sceneworld_ultra_tiny_v080a7.yaml`.
- Hardened generated SceneWorld `run_all_gprmax.bat` to quote paths and set `PYTHONPATH`.
