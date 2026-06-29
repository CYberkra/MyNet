# UavGPR-SimLab v0.8.0a26 多电脑统一 GPU Runtime 方案

本版本将本地 RTX 3060 与 4090 笔记本的 GPU 仿真环境统一为同一套 Python/conda 依赖合同。

## 核心原则

- 软件版本可以频繁替换。
- gprMax 源码、Miniconda、conda 环境、PyCUDA 与日志放在长期 RuntimeRoot。
- 两台电脑各自创建环境，但环境名、Python 版本、依赖文件一致。
- 不复制 conda 环境，不复制已编译扩展；每台电脑各自编译 gprMax 和 PyCUDA。

## 推荐目录

```text
D:\UavGPR_Runtime\
├─ miniconda3\
├─ conda_envs\
│  └─ uavgpr_gprmax_py310_gpu\
├─ gprMax\
│  └─ gprMax-v.3.1.7\
├─ runtime_profiles\
├─ downloads\
├─ logs\
└─ uavgpr_runtime.env
```

## 一键配置

通用入口：

```bat
setup_uavgpr_gpu_runtime_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7" -ForceRecreateEnv
```

本地 RTX 3060：

```bat
setup_local_3060_gpu_runtime.bat -GprMaxDir "你的gprMax源码目录" -ForceRecreateEnv
```

4090 笔记本：

```bat
setup_laptop_4090_gpu_runtime.bat -GprMaxDir "你的gprMax源码目录" -ForceRecreateEnv
```

## 验证

```bat
scripts\Verify_Current_GPU_Runtime.bat
```

通过后再运行：

```bat
run_gui.bat
```

## 持久配置

配置写入：

```text
D:\UavGPR_Runtime\uavgpr_runtime.env
D:\UavGPR_Runtime\runtime_profiles\<machine_profile>.env
当前软件目录\.simlab_env
```

软件启动时优先读取 RuntimeRoot 的长期配置。
