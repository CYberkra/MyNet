@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
call "%CD%\scripts\windows_runtime_bootstrap.bat"
if errorlevel 1 exit /b 1
%PY_RUN% -m uavgpr_simlab.cli pipeline --config configs\pipeline_automation_template.yaml
set RC=%ERRORLEVEL%
pause
exit /b %RC%
