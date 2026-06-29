# Multi-machine GPU Runtime

UavGPR-SimLab 使用轻量软件包 + 每台机器独立 RuntimeRoot 的长期方案。

## 统一环境合同

两台电脑都使用同一套路径和环境名，但各自独立创建：

```text
RuntimeRoot = D:\UavGPR_Runtime
conda env   = D:\UavGPR_Runtime\conda_envs\uavgpr_gprmax_py310_gpu
Python      = 3.10
gprMax      = 外部长期源码目录
GPU ID      = 0
```

不要在两台电脑之间复制 conda env，也不要把 gprMax zip 打回软件发布包。

## 推荐命令

本地 RTX 3060：

```powershell
.\setup_uavgpr_gpu_runtime_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "E:\gprMax\gprMax-v.3.1.7" -ForceRecreateEnv
.\scripts\Verify_Current_GPU_Runtime.bat
.\run_gui.bat
```

RTX 4090：

```powershell
.\setup_uavgpr_gpu_runtime_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7" -ForceRecreateEnv
.\scripts\Verify_Current_GPU_Runtime.bat
.\run_gui.bat
```

## a30 Windows 编译链加固

gprMax 的 Cython 扩展需要 MSVC 和 Windows SDK / UCRT headers。典型失败为：

```text
pyconfig.h(59): fatal error C1083: 无法打开包括文件: “io.h”: No such file or directory
```

这表示 `cl.exe` 被找到，但 Windows SDK / UCRT include 路径没有进入 `INCLUDE`。a30 setup 脚本已在 `build_ext` 前执行：

1. 查找 Visual Studio Build Tools；
2. 加载 `VsDevCmd.bat -arch=x64 -host_arch=x64`；
3. 若不可用，fallback 到 `vcvars64.bat`；
4. 导入 `cmd /c ... && set` 输出的环境变量；
5. 检查 `cl.exe`、`io.h`、`windows.h`；
6. 设置 `DISTUTILS_USE_SDK=1`、`MSSdk=1`。

如果第 5 步提示 Windows SDK / UCRT headers 不可用，需要打开 Visual Studio Installer，修改 Build Tools，安装：

```text
Desktop development with C++
Windows 10 SDK 或 Windows 11 SDK
MSVC v143 x64/x86 build tools
```

## 验收顺序

```text
1. setup_uavgpr_gpu_runtime_windows.bat
2. Verify_Current_GPU_Runtime.bat
3. geometry-only / ultra tiny
4. 1 case × 5 variants
5. 25-run smoke
6. validation / formal
```

验证通过前不要直接跑大批量 GPU 仿真。

## a31 逐步配置检查加固

a31 进一步将 Windows GPU Runtime 配置流程拆成可验证检查点：

- Windows 核心命令：`cmd.exe`、`where.exe`、`chcp.com`、`powershell.exe`；
- RuntimeRoot / conda env / gprMax 目录所在盘符；
- MSVC Developer Environment 与 Windows SDK / UCRT；
- CUDA `nvcc` 与 `cuda.h`；
- conda env 中的 `numpy`、`Cython`、`setuptools`；
- gprMax `.pyd` 编译结果数量；
- PyCUDA 构建前的 MSVC / CUDA 环境。

这些检查会在真正进入批量仿真前失败前移，避免等到 gprMax 运行阶段才发现环境问题。
