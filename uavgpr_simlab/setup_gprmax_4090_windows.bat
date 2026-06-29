@echo off
setlocal EnableExtensions
cd /d "%~dp0"
REM Backward-compatible wrapper. New generic entry: setup_uavgpr_gpu_runtime_windows.bat
call "%CD%\setup_uavgpr_gpu_runtime_windows.bat" -MachineProfile "laptop_4090" -RuntimeRoot "D:\UavGPR_Runtime" -CondaEnv "uavgpr_gprmax_py310_gpu" -GpuIds "0" -OmpThreads 12 %*
exit /b %ERRORLEVEL%
