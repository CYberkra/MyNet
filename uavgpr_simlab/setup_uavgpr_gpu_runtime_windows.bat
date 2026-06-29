@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_EXE%" set "PS_EXE=powershell.exe"
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\install_gprmax_windows.ps1" %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo [FAILED] UavGPR-SimLab GPU runtime setup failed with code %RC%.
if "%RC%"=="0" echo [OK] UavGPR-SimLab GPU runtime setup completed.
pause
exit /b %RC%
