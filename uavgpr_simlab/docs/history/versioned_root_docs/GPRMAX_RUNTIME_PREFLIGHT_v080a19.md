# UavGPR-SimLab v0.8.0-alpha.19 gprMax / GPU 运行前检查

## 背景

目标机日志显示批量仿真页使用：

```text
python=E:\python\python.exe
gpu=on 0
```

随后 gprMax 在 GPU 模式下导入 `pycuda.driver` 失败：

```text
ModuleNotFoundError: No module named 'pycuda'
ImportError: To use gprMax in GPU mode the pycuda package must be installed
```

这说明问题不在模型清单或 SceneWorld 五变体，而在“启用 GPU 的运行环境不具备 PyCUDA”。GPU 模式下继续执行 25-run 只会重复同一错误。

## a19 修复

1. 批量仿真页在启动 SceneWorld 统一任务前执行 runtime preflight。
2. 检查项包括：
   - gprMax 源码目录结构；
   - 当前任务实际使用的 Python 命令；
   - `python -m gprMax --help`；
   - GPU 模式下的 `pycuda.driver` 导入、CUDA driver 初始化和设备数量。
3. 检查失败时，GUI 不再启动批量任务，而是在日志区和弹窗中给出处理建议。
4. SceneWorld runner 同时增加 fail-fast 保护：如果绕过 GUI 直接调用 runner，首个 `gpu_pycuda_missing` / CUDA driver 初始化失败 / gprMax import 缺失会中止后续 case/variant。
5. Windows 一键配置脚本把 PyCUDA 安装和 GPU smoke 验证升级为硬验收，失败时返回非零错误码。

## 目标机处理顺序

推荐重新运行：

```bat
setup_gprmax_4090_windows.bat
scripts\Verify_4090_GPRMAX_GPU.bat
```

通过后，设置页应满足：

```text
gprMax 源码目录 = E:\gprMax\gprMax-v.3.1.7
conda 环境名 = gprMax
使用 conda run 调用 gprMax = 勾选
使用 GPU = 勾选
GPU ID = 0
```

不要让 GPU 任务继续使用没有 PyCUDA 的系统 Python，例如 `E:\python\python.exe`。如需 CPU smoke，可关闭“使用 GPU”后运行。
