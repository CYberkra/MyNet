@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_PATH=F:\codex\envs\psgn-csnet\Library"
set "PATH=F:\codex\envs\psgn-csnet\Library\bin;%PATH%"
set "PYTHONPATH=F:\codex\PSGN-CSNet\gprMax-master;%PYTHONPATH%"
set "ROOT=F:\codex\PSGN-CSNet\MyNet\outputs\simulation_v2_controls\official_audited_20260711"
set "PY=F:\codex\envs\psgn-csnet\python.exe"
set "GPRMAX=F:\codex\PSGN-CSNet\gprMax-master"

%PY% scripts\run_physical_sim_v2_controls.py --root "%ROOT%" --case-id CTRL02_FLAT_DEEP_MODERATE_POS --gpu 0 --execute --python-executable "%PY%" --gprmax-root "%GPRMAX%" --plan-output reports\simulation_v2_control_stage_20260711\CTRL02_gpu_full_run_plan.json
if errorlevel 1 exit /b %errorlevel%
%PY% scripts\run_physical_sim_v2_controls.py --root "%ROOT%" --case-id CTRL03_SMOOTH_INTERFACE_POS --gpu 0 --execute --python-executable "%PY%" --gprmax-root "%GPRMAX%" --plan-output reports\simulation_v2_control_stage_20260711\CTRL03_gpu_full_run_plan.json
if errorlevel 1 exit /b %errorlevel%
%PY% scripts\run_physical_sim_v2_controls.py --root "%ROOT%" --case-id CTRL04_MATCHED_BACKGROUND_NEG --gpu 0 --execute --python-executable "%PY%" --gprmax-root "%GPRMAX%" --plan-output reports\simulation_v2_control_stage_20260711\CTRL04_gpu_full_run_plan.json
exit /b %errorlevel%
