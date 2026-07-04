@echo off
title GprMambaSep Stage 1 Pretrain
echo ============================================================
echo  GprMambaSep Stage 1 — Simulation-Only Pretrain
echo  Config: configs/gpu_pretrain_v2_gprmambasep.json
echo  Data:   simulation_pretrain_v1 (191 windows)
echo  GPU:    RTX 3060 6GB
echo  ETA:    ~6-8 hours (80 epochs)
echo ============================================================
echo.

set PYTHON=E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe
set CONFIG=configs/gpu_pretrain_v2_gprmambasep.json
set LOG=outputs/run_gprmambasep_pretrain_v2/training.log

:: Ensure log dir exists
if not exist "outputs/run_gprmambasep_pretrain_v2" mkdir "outputs/run_gprmambasep_pretrain_v2"

:: Check CUDA available
"%PYTHON%" -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

:: Launch training
echo.
echo Starting training at %DATE% %TIME% ...
echo Log: %LOG%
echo.

"%PYTHON%" -u scripts/train_raw_only.py %CONFIG% 2>&1 | tee "%LOG%"

:: Check result
if errorlevel 1 (
    echo.
    echo Training FAILED. Check log: %LOG%
    exit /b 1
) else (
    echo.
    echo Training completed successfully!
    echo Check outputs at outputs/run_gprmambasep_pretrain_v2/
    pause
)
