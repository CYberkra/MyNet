@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
REM Backward-compatible one-click entry. New generic entry: setup_uavgpr_gpu_runtime_windows.bat
call "%CD%\setup_uavgpr_gpu_runtime_windows.bat" %*
exit /b %ERRORLEVEL%
