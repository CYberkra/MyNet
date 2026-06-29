@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
call "%CD%\scripts\windows_runtime_bootstrap.bat"
if errorlevel 1 exit /b 1
%PY_RUN% -m uavgpr_simlab.cli generate --plan configs\run_plan_3060_quick.yaml --workspace workspace
set RC=%ERRORLEVEL%
pause
exit /b %RC%
