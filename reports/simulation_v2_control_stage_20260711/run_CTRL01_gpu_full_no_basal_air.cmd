@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_PATH=F:\codex\envs\psgn-csnet\Library"
set "PATH=F:\codex\envs\psgn-csnet\Library\bin;%PATH%"
set "PYTHONPATH=F:\codex\PSGN-CSNet\gprMax-master;%PYTHONPATH%"
"F:\codex\envs\psgn-csnet\python.exe" scripts\run_physical_sim_v2_controls.py --root "F:\codex\PSGN-CSNet\MyNet\data\PGDA_SYNTH_DATASET_V2\01_solver_runs\official_audited_20260711" --case-id CTRL01_FLAT_SHALLOW_LOWLOSS_POS --gpu 0 --execute --python-executable "F:\codex\envs\psgn-csnet\python.exe" --gprmax-root "F:\codex\PSGN-CSNet\gprMax-master" --plan-output "reports\simulation_v2_control_stage_20260711\CTRL01_gpu_full_run_plan.json"
exit /b %errorlevel%
