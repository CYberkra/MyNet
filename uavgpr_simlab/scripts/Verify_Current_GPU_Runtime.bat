@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
call "%CD%\scripts\windows_runtime_bootstrap.bat"
if errorlevel 1 exit /b 1
if "%GPRMAX_SOURCE_DIR%"=="" (
  echo [ERROR] GPRMAX_SOURCE_DIR is empty. Run setup_uavgpr_gpu_runtime_windows.bat first or set it manually.
  pause
  exit /b 1
)
set "GPU_IDS=%UAVGPR_GPU_IDS%"
if "%GPU_IDS%"=="" set "GPU_IDS=0"
set "REPORT=%CD%\logs\check_current_gpu_runtime_report.json"
%PY_RUN% "%CD%\scripts\check_4090_gprmax_gpu.py" --gprmax-root "%GPRMAX_SOURCE_DIR%" --python-executable "%PY_EXE%" --gpu-ids "%GPU_IDS%" --out "%REPORT%"
set RC=%ERRORLEVEL%
echo.
echo Report: %REPORT%
if "%RC%"=="0" (echo [OK] Current GPU runtime smoke passed.) else (echo [FAILED] Current GPU runtime smoke failed.)
pause
exit /b %RC%
