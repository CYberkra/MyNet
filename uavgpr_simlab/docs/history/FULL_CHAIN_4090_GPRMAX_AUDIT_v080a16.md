# UavGPR-SimLab v0.8.0-alpha.16 4090 / gprMax 全链路审计

## 审计范围

本轮审计目标是确保当前版本在目标 Windows RTX 4090 笔记本上具备以下闭环：

```text
一键配置 gprMax/CUDA/PyCUDA/GUI 环境
  ↓
写入 .simlab_env
  ↓
run_gui.bat / GUI 设置页 / 批量仿真页读取同一套环境
  ↓
生成 SceneWorld 4090 formal / validation 数据集
  ↓
logs/run_all_gprmax.bat 调用统一 run_all_gprmax.py
  ↓
gprMax CPU/GPU 运行五变体
  ↓
输出 raw / target_only / background_only / clutter_only / air_only / clutter_gt
  ↓
QC、历史页、导出与重跑
```

## 已发现并修复的问题

### P1-1：一键脚本写入的环境没有被运行脚本完整继承

- 修复前：`scripts/install_gprmax_windows.ps1` 会写 `.simlab_env`，但 `scripts/windows_runtime_bootstrap.bat` 没有完整读取该文件。
- 风险：用户安装成功后双击 `run_gui.bat` 或数据集 `logs/run_all_gprmax.bat` 时可能回退到系统 Python，绕过 gprMax conda 环境。
- 修复：`windows_runtime_bootstrap.bat` 读取 `UAVGPR_GPRMAX_ROOT`、`GPRMAX_SOURCE_DIR`、`UAVGPR_CONDA_ENV`、`UAVGPR_PYTHON_EXE`、`UAVGPR_USE_CONDA_RUN`。

### P1-2：GUI 批量全链路任务未遵守 conda / GPU 设置

- 修复前：批量页 SceneWorld full-chain worker 固定使用 `sys.executable`。
- 风险：即使设置页配置了 gprMax conda 环境，GUI 后台任务也可能用错误 Python 运行，导致 gprMax 或 PyCUDA 不可用。
- 修复：`run_sceneworld_profile_from_batch()` 根据设置页的 `use_conda_run`、`conda_env`、`use_gpu`、`gpu_ids` 生成运行命令，并传入 worker。

### P1-3：4090 formal / validation 计划仍是旧命名，生成后不会进入 SceneWorld 统一运行 BAT 分支

- 修复前：`plan_name` 为 `4090_formal_dataset` / `4090_validation_hifi`，不包含 `sceneworld`。
- 风险：`uavgpr_simlab.cli generate` 生成旧式 gprMax BAT，不走当前五变体全链路服务。
- 修复：改为 `yingshan_sceneworld_4090_formal` 与 `yingshan_sceneworld_4090_validation_hifi`，使用五变体 schema。

### P1-4：gprMax Windows 环境使用上游未固定 conda_env.yml 风险较高

- 修复前：一键配置脚本主要依赖 gprMax 源码中的环境文件。
- 风险：新机器可能拉到过新的 Python / Cython / numpy，影响 gprMax Cython 扩展和 PyCUDA 编译。
- 修复：新增 `configs/environment_gprmax_4090_windows.yml`，固定 Python 3.10 和主要科学计算栈版本范围。

### P2-1：环境诊断子进程存在长时间挂起风险

- 修复前：GUI/self-test 环境诊断直接等待 import 检查，PySide/matplotlib 等二进制库在不兼容环境下可能长时间挂起。
- 修复：`core/environment.py` 使用 Popen + 文件输出 + hard timeout，降低 GUI 和自测卡死风险。

## 新增验收脚本

### `scripts/check_4090_gprmax_gpu.py`

目标机上执行以下检查：

1. gprMax 源码结构检查；
2. `nvidia-smi`；
3. `nvcc --version`；
4. Python 核心依赖导入；
5. PySide6 导入；
6. gprMax import/help；
7. PyCUDA CUDA driver 检查；
8. gprMax tiny CPU smoke；
9. gprMax tiny GPU smoke：`python -m gprMax tiny_gpu_Ascan_2D.in -n 1 -gpu 0`；
10. HDF5 输出读取。

输出：

```text
logs/check_4090_gprmax_gpu_report.json
```

### `scripts/Verify_4090_GPRMAX_GPU.bat`

双击后会调用 `windows_runtime_bootstrap.bat`，读取 `.simlab_env`，然后执行上述 Python 验证脚本。

## 目标机验收步骤

```bat
setup_gprmax_4090_windows.bat
scripts\Verify_4090_GPRMAX_GPU.bat
scripts\Generate_4090_Validation_Dataset.bat
workspace\yingshan_sceneworld_4090_validation_hifi\logs\run_all_gprmax.bat
run_gui.bat
```

最低通过标准：

- `logs/check_4090_gprmax_gpu_report.json` 的 `ok=true`；
- validation 数据集至少 1 case × 5 variant 可完成；
- 每个 case 的 `outputs/clutter_gt_bscan.npy` 存在且 finite；
- `models/<case_id>/bscan_qc_report.json` 为 success；
- GUI 历史页可以查看 raw / target_only / clutter_gt 对比。

## 当前 sandbox 验证边界

当前 Linux sandbox 没有 Windows、RTX 4090、CUDA Toolkit、PyCUDA 和 PySide6。因此本轮只能确认：

- 静态编译通过；
- CLI 入口可解析；
- 4090 plan 可生成 SceneWorld 数据集；
- 生成的 `run_all_gprmax.bat` 走统一服务；
- GPU 验证脚本能 dry-run 并生成失败报告；
- 真实 `-gpu` smoke 必须在目标机完成。
