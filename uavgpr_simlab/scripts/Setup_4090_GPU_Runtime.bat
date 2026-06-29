@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
call "%CD%\setup_laptop_4090_gpu_runtime.bat" %*
exit /b %ERRORLEVEL%
