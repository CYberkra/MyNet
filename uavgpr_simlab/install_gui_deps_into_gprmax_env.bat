@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
call "%PROJECT_ROOT%\scripts\windows_runtime_bootstrap.bat"
if errorlevel 1 (
  echo [ERROR] Runtime bootstrap failed. Run setup_gprmax_4090_windows.bat first.
  pause
  exit /b 1
)
%PY_RUN% -m pip install --upgrade pip
if errorlevel 1 goto fail
%PY_RUN% -m pip install -r requirements_gui.txt
if errorlevel 1 goto fail
%PY_RUN% -m pip install -e .
if errorlevel 1 goto fail
echo [OK] GUI dependencies installed into selected runtime Python:
echo   %PY_RUN%
pause
exit /b 0
:fail
echo [FAILED] Install failed.
pause
exit /b 1
