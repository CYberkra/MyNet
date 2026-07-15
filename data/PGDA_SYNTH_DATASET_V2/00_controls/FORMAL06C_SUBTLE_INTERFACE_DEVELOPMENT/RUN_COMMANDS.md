# FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT

This case is development-only and blocked from training.

```powershell
$case = "data/PGDA_SYNTH_DATASET_V2/00_controls/FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT"
$cudaArgs = @()
if ($env:PGDA_CUDA_BIN) { $cudaArgs = @("--cuda-bin", $env:PGDA_CUDA_BIN) }

python scripts/run_native_256_release_pilot.py $case --run-id formal06c_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal06c_smoke1 --trace-count 1 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal06c_blind8_full --trace-count 8 --skip-air-reference --full-scene-only @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal06c_distributed32_full --trace-count 32 --trace-stride 8 --skip-air-reference --full-scene-only @cudaArgs --execute
```

Do not start the distributed 32-trace full scene until the one-trace causal
audit and blind local 8-trace amplitude review pass. Do not start a matched
32-trace pair until the distributed full scene passes morphology review. The
unlabelled full-scene preview is reviewed before any target path is overlaid.
