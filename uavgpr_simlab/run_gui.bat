@echo off
setlocal EnableExtensions

REM UavGPR-SimLab robust Windows launcher.
REM This launcher supports using a downloaded gprMax source tree directly;
REM gprMax does not need to be pip-installed when GPRMAX_SOURCE_DIR is valid.

pushd "%~dp0"
set "PROJECT_ROOT=%CD%"

call "%PROJECT_ROOT%\scripts\windows_runtime_bootstrap.bat"
if errorlevel 1 (
  echo.
  echo [ERROR] Runtime bootstrap failed.
  pause
  popd
  exit /b 1
)

echo.
echo [UavGPR-SimLab] Project root:
echo   %PROJECT_ROOT%
echo [UavGPR-SimLab] Python command:
echo   %PY_RUN%
echo [UavGPR-SimLab] gprMax source:
if "%GPRMAX_SOURCE_DIR%"=="" (
  echo   [not set]
) else (
  echo   %GPRMAX_SOURCE_DIR%
)
echo.

%PY_RUN% -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Selected Python cannot run.
  echo Set UAVGPR_PYTHON_EXE to a valid python.exe path and retry.
  pause
  popd
  exit /b 1
)

%PY_RUN% -c "import PySide6, matplotlib, numpy, pandas, h5py, yaml" >nul 2>nul
if errorlevel 1 (
  echo [UavGPR-SimLab] GUI dependencies are missing in selected Python.
  echo [UavGPR-SimLab] Installing requirements_gui.txt ...
  %PY_RUN% -m pip install -r requirements_gui.txt
  if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install GUI dependencies.
    echo Try manually:
    echo   python -m pip install -r requirements_gui.txt
    pause
    popd
    exit /b 1
  )
)

if not "%GPRMAX_SOURCE_DIR%"=="" (
  %PY_RUN% -m gprMax --help >nul 2>nul
  if errorlevel 1 (
    echo [WARN] gprMax source was found but python -m gprMax --help failed.
    echo        You can still open the GUI, but simulation runs may fail until gprMax is fixed.
    echo.
  ) else (
    echo [UavGPR-SimLab] gprMax source-tree import check passed.
  )
)

echo [UavGPR-SimLab] Launching GUI...
%PY_RUN% -m uavgpr_simlab.app
set RC=%ERRORLEVEL%

echo.
echo [UavGPR-SimLab] GUI closed. Exit code: %RC%
pause
popd
exit /b %RC%
