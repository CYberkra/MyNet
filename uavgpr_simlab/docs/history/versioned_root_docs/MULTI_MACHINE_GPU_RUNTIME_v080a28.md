# UavGPR-SimLab v0.8.0-alpha.28 Windows runtime 深度审计

本轮是在 v0.8.0-alpha.27 基础上继续逐项审计 Windows 3060/4090 GPU Runtime 链路，目标是降低启动器、示例脚本、生成 BAT 和命令行 quoting 的同类复发风险。

## 修复点

1. `setup_uavgpr_gpu_runtime_windows.bat` 不再依赖 PATH 中存在 `powershell`，优先调用：

   ```bat
   %SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe
   ```

2. 以下脚本统一走 `scripts\windows_runtime_bootstrap.bat` 和 `%PY_RUN%`：

   - `scripts\Preview_Example_CSV.bat`
   - `scripts\Run_Full_Pipeline_Example.bat`
   - `scripts\Setup_GUI_Only.bat`

3. 通用 `write_manifest_commands_bat()` 生成的 BAT 改为：

   - quote `pushd "%~dp0\.."`；
   - 调用 `windows_runtime_bootstrap.bat`；
   - 使用 `%PY_RUN%` 执行 Python；
   - 不再在生成脚本里假定系统 `python` 可用。

4. `runner.command_to_string()` 改为 `subprocess.list2cmdline()`，用于 Windows BAT 命令行参数引用，避免路径含空格、引号或特殊字符时生成脆弱命令。

5. `scripts/check_windows_script_contract.py` 新增防回归检查，覆盖顶层 setup PowerShell 调用、示例脚本 bootstrap、通用生成 BAT、Windows-safe command quoting。

## 仍需真机验证

本轮为静态与自动化可执行审计，不能替代 Windows RTX 3060 / RTX 4090 真机上的 PyCUDA 与 gprMax `-gpu` smoke。两台机器仍需分别执行：

```powershell
.\setup_uavgpr_gpu_runtime_windows.bat -RuntimeRoot "D:\UavGPR_Runtime" -GprMaxDir "你的gprMax源码目录" -ForceRecreateEnv
.\scripts\Verify_Current_GPU_Runtime.bat
```

