# 本地 gprMax 源码树运行说明

> 本文件为当前本地源码树运行稳定说明；旧版版本化说明已归档。

## 适用场景

用户本地 Windows 电脑已经有可用的 gprMax 源码目录，例如：

```text
E:\gprMax\gprMax-v.3.1.7
```

且该目录结构有效、Cython 扩展已编译，但当前 Python 没有通过 `pip install` 安装 `gprMax` 包。

此时不需要为了 CPU 验证强制安装 conda。软件应通过 `PYTHONPATH=<gprMax源码目录>` 让 `python -m gprMax` 可运行。

## 快速配置

在项目根目录运行：

```bat
scripts\Configure_Local_CPU_GprMax.bat "E:\gprMax\gprMax-v.3.1.7" "E:\python\python.exe"
run_gui.bat
```

脚本会写入：

```text
.simlab_env
```

关键值：

```text
UAVGPR_GPRMAX_ROOT=E:\gprMax\gprMax-v.3.1.7
GPRMAX_SOURCE_DIR=E:\gprMax\gprMax-v.3.1.7
UAVGPR_USE_CONDA_RUN=0
UAVGPR_PYTHON_EXE=E:\python\python.exe
UAVGPR_USE_GPU=0
```

## 诊断预期

设置页环境检查中：

```text
Conda command: not found; optional because conda run is disabled
gprMax import via source tree: OK
```

只要 GUI 依赖、gprMax 源码目录和编译扩展均通过，本地 CPU 验证可以继续。

## 不适用场景

GPU 仿真仍需要 PyCUDA 与 CUDA driver 可用。若开启 GPU，运行前检查必须通过：

```text
PyCUDA GPU driver
python -m gprMax ... -gpu 0
```

本地 CPU/source-tree 模式不等于 GPU 环境已配置完成。
