@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
call "%CD%\scripts\windows_runtime_bootstrap.bat"
if errorlevel 1 exit /b 1
%PY_RUN% -m uavgpr_simlab.cli preview-csv sample_data\Line9origin36_first16traces.csv --max-traces 16 --out outputs\example_line9 --convert
set RC=%ERRORLEVEL%
pause
exit /b %RC%
