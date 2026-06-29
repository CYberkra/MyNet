# v0.8.0-alpha.27 多电脑 GPU Runtime 二次审计

本轮针对 a26 后仍可能出现的 Windows 脚本问题做防回归修复。

## 修复点

1. PowerShell setup 脚本不再使用 `Start-Process -ArgumentList` 单字符串运行子进程，改为 `& $exe @argv` 参数数组调用。
2. `windows_runtime_bootstrap.bat` 不再在同一括号块内 `set PY_RUN` 后立即用 `%PY_RUN%` 判断，避免 CMD 预展开导致错误。
3. `run_gui.bat` 统一使用 bootstrap 输出的 `%PY_RUN%`，不再根据 `UAVGPR_CONDA_ENV` 另行分支。
4. `Generate_3060_Quick_Dataset.bat` 走共享 bootstrap，避免本地 3060 仍误用系统 Python。
5. `check_windows_script_contract.py` 已加入以上静态守卫。

## 推荐命令

PowerShell 当前目录运行：

```powershell
.\setup_uavgpr_gpu_runtime_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "你的gprMax源码目录" -ForceRecreateEnv
.\scripts\Verify_Current_GPU_Runtime.bat
.\run_gui.bat
```

