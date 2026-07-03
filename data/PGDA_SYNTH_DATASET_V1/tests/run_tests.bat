@echo off
REM Run all PGDA_SYNTH_DATASET_V1 tool tests
cd /d "%~dp0.."
"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe" -m pytest tests/ -v %*
