@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
call "%CD%\scripts\windows_runtime_bootstrap.bat"
if errorlevel 1 (
  echo [WARN] Runtime bootstrap failed; falling back to system python.
  set "PY_RUN=python"
)
%PY_RUN% -m pip install --upgrade pip
if errorlevel 1 goto fail
%PY_RUN% -m pip install -r requirements_gui.txt
if errorlevel 1 goto fail
%PY_RUN% -m pip install -e .
if errorlevel 1 goto fail
echo [OK] GUI dependencies installed into selected Python:
echo   %PY_RUN%
pause
exit /b 0
:fail
echo [FAILED] GUI dependency install failed.
pause
exit /b 1
