# v0.8.0a26 多电脑 GPU Runtime 修复说明

本版本修复 v0.8.0a25 在 Windows PowerShell / conda base 环境下配置失败的问题。

## 修复点

1. `python -c` 参数在 PowerShell `Start-Process` 中会丢失引号，导致目标机日志出现 `SyntaxError: import`。a26 已对所有 subprocess 参数做 Windows quoting。
2. 部分 conda/base PowerShell 会收窄 PATH，导致 `conda run` 内部找不到 `chcp`、`powershell`、`python`。a26 在 setup 与 runtime bootstrap 中强制恢复 System32、Windows、Wbem、WindowsPowerShell、OpenSSH 路径。
3. 环境创建完成后，gprMax 编译、pip 安装、PyCUDA 安装、GPU 验证均直接使用固定环境解释器：

```text
D:\UavGPR_Runtime\conda_envs\uavgpr_gprmax_py310_gpu\python.exe
```

不再依赖 fragile 的 `conda run`。

## 推荐命令

PowerShell 下必须使用 `./` 或 `.\`：

```powershell
.\setup_uavgpr_gpu_runtime_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "E:\gprMax\gprMax-v.3.1.7" -ForceRecreateEnv
.\scripts\Verify_Current_GPU_Runtime.bat
.\run_gui.bat
```

4090 端只需要把 `-GprMaxDir` 改成该机器实际路径。
