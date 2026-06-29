@echo off
setlocal EnableExtensions

REM Configure UavGPR-SimLab to use a local gprMax source tree with the current/system Python.
REM This is intended for local CPU validation when conda is not installed.
REM Usage:
REM   scripts\Configure_Local_CPU_GprMax.bat [gprMaxRoot] [pythonExe]
REM Example:
REM   scripts\Configure_Local_CPU_GprMax.bat "E:\gprMax\gprMax-v.3.1.7" "E:\python\python.exe"

pushd "%~dp0\.."
set "PROJECT_ROOT=%CD%"

set "GPRMAX_DIR=%~1"
if "%GPRMAX_DIR%"=="" set "GPRMAX_DIR=E:\gprMax\gprMax-v.3.1.7"

set "PY_EXE=%~2"
if "%PY_EXE%"=="" set "PY_EXE=E:\python\python.exe"
if not exist "%PY_EXE%" (
  where python >nul 2>nul
  if not errorlevel 1 set "PY_EXE=python"
)

if not exist "%GPRMAX_DIR%\gprMax\__main__.py" (
  echo [ERROR] Invalid gprMax source directory:
  echo   %GPRMAX_DIR%
  echo It must contain gprMax\__main__.py
  popd
  pause
  exit /b 2
)

if not "%PY_EXE%"=="python" if not exist "%PY_EXE%" (
  echo [ERROR] Python executable not found:
  echo   %PY_EXE%
  popd
  pause
  exit /b 3
)

set "PYTHONPATH=%GPRMAX_DIR%;%PROJECT_ROOT%\src;%PYTHONPATH%"

%PY_EXE% -c "import sys, gprMax; print(sys.executable); print('gprMax import OK'); print(getattr(gprMax, '__file__', ''))"
if errorlevel 1 (
  echo.
  echo [ERROR] Selected Python cannot import gprMax even after PYTHONPATH injection.
  echo Try checking the gprMax directory and Python version.
  popd
  pause
  exit /b 4
)

(
  echo UAVGPR_RUNTIME_ROOT=
  echo UAVGPR_GPRMAX_ROOT=%GPRMAX_DIR%
  echo GPRMAX_SOURCE_DIR=%GPRMAX_DIR%
  echo UAVGPR_CONDA_ENV=
  echo UAVGPR_CONDA_ENV_PREFIX=
  echo UAVGPR_CONDA_EXE=
  echo UAVGPR_USE_CONDA_RUN=0
  echo UAVGPR_PYTHON_EXE=%PY_EXE%
  echo UAVGPR_USE_GPU=0
  echo UAVGPR_GPU_IDS=0
  echo UAVGPR_OMP_THREADS=8
) > "%PROJECT_ROOT%\.simlab_env"

echo.
echo [OK] Local CPU/source-tree runtime has been configured.
echo Project env file:
echo   %PROJECT_ROOT%\.simlab_env
echo gprMax:
echo   %GPRMAX_DIR%
echo Python:
echo   %PY_EXE%
echo.
echo Next:
echo   run_gui.bat

echo.
pause
popd
exit /b 0
