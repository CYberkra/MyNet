@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
call "%CD%\setup_local_3060_gpu_runtime.bat" %*
exit /b %ERRORLEVEL%
