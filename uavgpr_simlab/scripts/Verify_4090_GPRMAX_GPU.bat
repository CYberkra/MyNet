@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
REM Backward-compatible wrapper. New generic entry: scripts\Verify_Current_GPU_Runtime.bat
call "%CD%\scripts\Verify_Current_GPU_Runtime.bat" %*
exit /b %ERRORLEVEL%
