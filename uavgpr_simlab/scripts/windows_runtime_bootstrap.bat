@echo off
REM Shared Windows runtime bootstrap for UavGPR-SimLab.
REM It reuses a persistent RuntimeRoot across software versions.
REM Expected persistent file: D:\UavGPR_Runtime\uavgpr_runtime.env or E:\UavGPR_Runtime\uavgpr_runtime.env

if "%PROJECT_ROOT%"=="" (
  pushd "%~dp0\.." >nul
  set "PROJECT_ROOT=%CD%"
  popd >nul
)

REM Ensure Windows core command locations are available even when launched from
REM a narrowed conda/base PowerShell. Conda run and build tools require chcp,
REM cmd, powershell, where, and other System32 tools to be on PATH.
if not "%SystemRoot%"=="" set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%SystemRoot%\System32\OpenSSH;%PATH%"

REM Default RuntimeRoot: prefer D: for the 4090 workstation; then E:; then project-local fallback.
if "%UAVGPR_RUNTIME_ROOT%"=="" if exist "D:\" set "UAVGPR_RUNTIME_ROOT=D:\UavGPR_Runtime"
if "%UAVGPR_RUNTIME_ROOT%"=="" if exist "E:\" set "UAVGPR_RUNTIME_ROOT=E:\UavGPR_Runtime"
if "%UAVGPR_RUNTIME_ROOT%"=="" set "UAVGPR_RUNTIME_ROOT=%PROJECT_ROOT%\UavGPR_Runtime"

REM Load persistent RuntimeRoot settings first. Project-local .simlab_env can override or complement it.
if exist "%UAVGPR_RUNTIME_ROOT%\uavgpr_runtime.env" call :LoadEnvFile "%UAVGPR_RUNTIME_ROOT%\uavgpr_runtime.env"
if exist "%PROJECT_ROOT%\.simlab_env" call :LoadEnvFile "%PROJECT_ROOT%\.simlab_env"

REM If a project .simlab_env changed RuntimeRoot, load that persistent file as well.
if exist "%UAVGPR_RUNTIME_ROOT%\uavgpr_runtime.env" call :LoadEnvFile "%UAVGPR_RUNTIME_ROOT%\uavgpr_runtime.env"

if "%UAVGPR_MINICONDA_DIR%"=="" set "UAVGPR_MINICONDA_DIR=%UAVGPR_RUNTIME_ROOT%\miniconda3"
REM Do not invent a conda env when none was configured. For local CPU/source-tree
REM mode this keeps the GUI from defaulting to conda run and reporting false failures.
if "%UAVGPR_CONDA_ENV_PREFIX%"=="" if exist "%UAVGPR_RUNTIME_ROOT%\conda_envs\uavgpr_gprmax_py310_gpu\python.exe" (
  set "UAVGPR_CONDA_ENV=uavgpr_gprmax_py310_gpu"
  set "UAVGPR_CONDA_ENV_PREFIX=%UAVGPR_RUNTIME_ROOT%\conda_envs\uavgpr_gprmax_py310_gpu"
)
if "%UAVGPR_CONDA_ENV_PREFIX%"=="" if exist "%UAVGPR_RUNTIME_ROOT%\conda_envs\gprMax\python.exe" (
  set "UAVGPR_CONDA_ENV=gprMax"
  set "UAVGPR_CONDA_ENV_PREFIX=%UAVGPR_RUNTIME_ROOT%\conda_envs\gprMax"
)
if not "%UAVGPR_CONDA_ENV%"=="" if "%UAVGPR_CONDA_ENV_PREFIX%"=="" set "UAVGPR_CONDA_ENV_PREFIX=%UAVGPR_RUNTIME_ROOT%\conda_envs\%UAVGPR_CONDA_ENV%"
if "%GPRMAX_SOURCE_DIR%"=="" if not "%UAVGPR_GPRMAX_ROOT%"=="" set "GPRMAX_SOURCE_DIR=%UAVGPR_GPRMAX_ROOT%"
if "%GPRMAX_SOURCE_DIR%"=="" set "GPRMAX_SOURCE_DIR=%UAVGPR_RUNTIME_ROOT%\gprMax\gprMax-v.3.1.7"

if not "%UAVGPR_GPU_IDS%"=="" set "GPU_IDS=%UAVGPR_GPU_IDS%"
if not "%UAVGPR_OMP_THREADS%"=="" set "OMP_NUM_THREADS=%UAVGPR_OMP_THREADS%"

REM Auto-detect persistent gprMax source when settings are missing or stale.
if not exist "%GPRMAX_SOURCE_DIR%\gprMax\__main__.py" (
  if exist "%UAVGPR_RUNTIME_ROOT%\gprMax\gprMax-v.3.1.7\gprMax\__main__.py" set "GPRMAX_SOURCE_DIR=%UAVGPR_RUNTIME_ROOT%\gprMax\gprMax-v.3.1.7"
)
if not exist "%GPRMAX_SOURCE_DIR%\gprMax\__main__.py" (
  if exist "D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7\gprMax\__main__.py" set "GPRMAX_SOURCE_DIR=D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7"
)
if not exist "%GPRMAX_SOURCE_DIR%\gprMax\__main__.py" (
  if exist "E:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7\gprMax\__main__.py" set "GPRMAX_SOURCE_DIR=E:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7"
)

if not "%GPRMAX_SOURCE_DIR%"=="" (
  set "PYTHONPATH=%GPRMAX_SOURCE_DIR%;%PROJECT_ROOT%\src;%PYTHONPATH%"
  set "UAVGPR_GPRMAX_ROOT=%GPRMAX_SOURCE_DIR%"
) else (
  set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"
)

REM Choose Python. Prefer explicit prefix python under RuntimeRoot; conda run is optional.
REM Keep this section label-based rather than parenthesized so variables set in
REM one branch are visible to following checks without delayed expansion.
set "PY_RUN="
set "PY_EXE="

if "%UAVGPR_CONDA_EXE%"=="" if exist "%UAVGPR_MINICONDA_DIR%\Scripts\conda.exe" set "UAVGPR_CONDA_EXE=%UAVGPR_MINICONDA_DIR%\Scripts\conda.exe"
if "%UAVGPR_PYTHON_EXE%"=="" if exist "%UAVGPR_CONDA_ENV_PREFIX%\python.exe" set "UAVGPR_PYTHON_EXE=%UAVGPR_CONDA_ENV_PREFIX%\python.exe"

if exist "%UAVGPR_CONDA_ENV_PREFIX%\python.exe" goto USE_PREFIX_PYTHON
if not "%UAVGPR_PYTHON_EXE%"=="" if exist "%UAVGPR_PYTHON_EXE%" goto USE_EXPLICIT_PYTHON
if not "%UAVGPR_CONDA_ENV%"=="" if exist "%UAVGPR_CONDA_EXE%" goto TRY_NAMED_CONDA_ENV
goto TRY_SYSTEM_PYTHON

:USE_PREFIX_PYTHON
set "PY_EXE=%UAVGPR_CONDA_ENV_PREFIX%\python.exe"
if "%UAVGPR_USE_CONDA_RUN%"=="1" if exist "%UAVGPR_CONDA_EXE%" set "PY_RUN="%UAVGPR_CONDA_EXE%" run -p "%UAVGPR_CONDA_ENV_PREFIX%" python"
if not defined PY_RUN set "PY_RUN="%PY_EXE%""
goto PYTHON_SELECTED

:USE_EXPLICIT_PYTHON
set "PY_EXE=%UAVGPR_PYTHON_EXE%"
set "PY_RUN="%PY_EXE%""
goto PYTHON_SELECTED

:TRY_NAMED_CONDA_ENV
"%UAVGPR_CONDA_EXE%" run -n "%UAVGPR_CONDA_ENV%" python -c "import sys; print(sys.executable)" >nul 2>nul
if not errorlevel 1 (
  set "PY_RUN="%UAVGPR_CONDA_EXE%" run -n "%UAVGPR_CONDA_ENV%" python"
  set "PY_EXE=python"
  goto PYTHON_SELECTED
)

:TRY_SYSTEM_PYTHON
where python >nul 2>nul
if not errorlevel 1 (
  set "PY_RUN=python"
  set "PY_EXE=python"
  goto PYTHON_SELECTED
)

echo [ERROR] Python was not found. Run setup_uavgpr_gpu_runtime_windows.bat first, or set UAVGPR_PYTHON_EXE to python.exe.
exit /b 1

:PYTHON_SELECTED
exit /b 0

:LoadEnvFile
for /f "usebackq eol=# tokens=1,* delims==" %%A in (%~1) do (
  if /I "%%A"=="UAVGPR_RUNTIME_ROOT" set "UAVGPR_RUNTIME_ROOT=%%B"
  if /I "%%A"=="UAVGPR_MINICONDA_DIR" set "UAVGPR_MINICONDA_DIR=%%B"
  if /I "%%A"=="UAVGPR_CONDA_EXE" set "UAVGPR_CONDA_EXE=%%B"
  if /I "%%A"=="UAVGPR_CONDA_ENV_PREFIX" set "UAVGPR_CONDA_ENV_PREFIX=%%B"
  if /I "%%A"=="UAVGPR_GPRMAX_ROOT" set "UAVGPR_GPRMAX_ROOT=%%B"
  if /I "%%A"=="GPRMAX_SOURCE_DIR" set "GPRMAX_SOURCE_DIR=%%B"
  if /I "%%A"=="UAVGPR_CONDA_ENV" set "UAVGPR_CONDA_ENV=%%B"
  if /I "%%A"=="UAVGPR_PYTHON_EXE" set "UAVGPR_PYTHON_EXE=%%B"
  if /I "%%A"=="UAVGPR_USE_CONDA_RUN" set "UAVGPR_USE_CONDA_RUN=%%B"
  if /I "%%A"=="UAVGPR_GPU_IDS" set "UAVGPR_GPU_IDS=%%B"
  if /I "%%A"=="UAVGPR_USE_GPU" set "UAVGPR_USE_GPU=%%B"
  if /I "%%A"=="UAVGPR_GPU_ENABLED" set "UAVGPR_GPU_ENABLED=%%B"
  if /I "%%A"=="UAVGPR_OMP_THREADS" set "UAVGPR_OMP_THREADS=%%B"
  if /I "%%A"=="UAVGPR_MACHINE_PROFILE" set "UAVGPR_MACHINE_PROFILE=%%B"
  if /I "%%A"=="UAVGPR_GPU_RUNTIME_ENV" set "UAVGPR_GPU_RUNTIME_ENV=%%B"
  if /I "%%A"=="UAVGPR_RUN_SCALE" set "UAVGPR_RUN_SCALE=%%B"
)
exit /b 0
