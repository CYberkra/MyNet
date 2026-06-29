# UavGPR-SimLab v0.8.0-alpha.20 RuntimeRoot / 4090 环境说明

## 目标

本版将 4090 笔记本上的主要运行环境集中到一个目录，避免 Miniconda、gprMax、PyCUDA、日志和下载缓存散落在多个位置。

推荐：

```text
E:\UavGPR_Runtime
```

目录结构：

```text
E:\UavGPR_Runtime\
├─ miniconda3\                     # 独立 Miniconda
├─ conda_envs\
│  └─ gprMax\                      # Python 3.10 + gprMax + PyCUDA + GUI 依赖
├─ gprMax\
│  └─ gprMax-v.3.1.7\              # gprMax 源码树
├─ downloads\                      # Miniconda / 安装包缓存
└─ logs\                           # 一键配置日志
```

CUDA Toolkit 和 Visual Studio Build Tools 是 Windows 系统级组件，仍安装到系统标准目录，例如：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\...
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\...
```

## 目标机执行顺序

建议先把本软件解压到短路径：

```text
E:\UavGPR-SimLab_a20
```

然后在项目根目录运行：

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "E:\UavGPR_Runtime"
scripts\Verify_4090_GPRMAX_GPU.bat
run_gui.bat
```

本发布包根目录已内置：

```text
gprMax-v.3.1.7.zip
```

一键脚本会优先解压这个本地包，不需要手动找 gprMax 源码。

## 写入的 .simlab_env

配置完成后，项目根目录会生成：

```text
.simlab_env
```

关键字段：

```text
UAVGPR_RUNTIME_ROOT=E:\UavGPR_Runtime
UAVGPR_MINICONDA_DIR=E:\UavGPR_Runtime\miniconda3
UAVGPR_CONDA_EXE=E:\UavGPR_Runtime\miniconda3\Scripts\conda.exe
UAVGPR_CONDA_ENV=gprMax
UAVGPR_CONDA_ENV_PREFIX=E:\UavGPR_Runtime\conda_envs\gprMax
UAVGPR_PYTHON_EXE=E:\UavGPR_Runtime\conda_envs\gprMax\python.exe
UAVGPR_GPRMAX_ROOT=E:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7
GPRMAX_SOURCE_DIR=E:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7
UAVGPR_USE_CONDA_RUN=1
UAVGPR_USE_GPU=1
UAVGPR_GPU_IDS=0
UAVGPR_OMP_THREADS=8
```

`run_gui.bat`、`scripts\Verify_4090_GPRMAX_GPU.bat` 和数据集内的 `logs\run_all_gprmax.bat` 都读取该文件。目标是避免再次调用 `E:\python\python.exe` 这类没有 PyCUDA 的系统 Python。

## 验收标准

先看：

```text
logs\check_4090_gprmax_gpu_report.json
```

必须满足：

```json
"ok": true
```

再进入 GUI 批量仿真页运行：

```text
1 case × 5 variants
```

通过后再运行：

```text
25-run smoke（5 cases × 5 variants）
```

## 常见失败解释

- `ModuleNotFoundError: No module named 'pycuda'`：实际 Python 不是 RuntimeRoot 下的 `conda_envs\gprMax\python.exe`，或 PyCUDA 安装失败。
- `nvcc not found`：CUDA Toolkit 没安装或 PATH 未刷新。
- `nvidia-smi not found`：NVIDIA 驱动未正确安装。
- `gprMax import/help failed`：gprMax 源码目录或 editable install 未完成。

