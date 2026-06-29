# v0.8.0-alpha.21：4090 RuntimeRoot 持久环境说明

## 目标

把 gprMax、Miniconda、conda 环境、PyCUDA、日志和运行配置固定到一个长期目录，避免每个 UavGPR-SimLab 版本都重复放置 gprMax 或误用系统 Python。

推荐目录：

```text
D:\UavGPR_Runtime\
├─ miniconda3\
├─ conda_envs\gprMax\
├─ gprMax\gprMax-v.3.1.7\
├─ downloads\
├─ logs\
└─ uavgpr_runtime.env
```

## 推荐命令

在新的软件包根目录运行：

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -ForceRecreateEnv
```

完成后运行：

```bat
scripts\Verify_4090_GPRMAX_GPU.bat
run_gui.bat
```

## 与 v0.8.0-alpha.20 的差异

- 默认不再复用 `C:\Users\...\miniconda3`。
- 默认安装/使用 `D:\UavGPR_Runtime\miniconda3`。
- 如果 `D:\UavGPR_Runtime\conda_envs\gprMax` 是不完整环境，会自动删除并重建。
- 新增 `-ForceRecreateEnv`，可强制重建 gprMax conda 环境。
- 新增 `D:\UavGPR_Runtime\uavgpr_runtime.env`，未来版本可以直接复用。
- `windows_runtime_bootstrap.bat` 会读取 RuntimeRoot 持久配置，不需要每个版本重新配置 gprMax。

## 参数说明

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "D:\UavGPR_Runtime"
```

普通集中安装。

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -ForceRecreateEnv
```

删除并重建 `D:\UavGPR_Runtime\conda_envs\gprMax`。当第 8 步 conda 环境失败、环境半创建或包冲突时优先使用。

```bat
setup_gprmax_4090_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -UseExistingConda
```

显式允许复用系统已有 conda。长期管理不推荐，只有在 RuntimeRoot Miniconda 安装受限时使用。

## 验收标准

`logs\check_4090_gprmax_gpu_report.json` 中应为：

```json
"ok": true
```

批量运行日志中不应再出现：

```text
python=E:\python\python.exe
```

应指向 RuntimeRoot 环境，例如：

```text
D:\UavGPR_Runtime\conda_envs\gprMax\python.exe
```

或：

```text
D:\UavGPR_Runtime\miniconda3\Scripts\conda.exe run -p D:\UavGPR_Runtime\conda_envs\gprMax python
```
