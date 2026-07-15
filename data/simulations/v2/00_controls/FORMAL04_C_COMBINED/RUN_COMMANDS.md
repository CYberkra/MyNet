# FORMAL04_C_COMBINED

This case is blocked from training. Run from the repository root.

```powershell
$case = "data/simulations/v2/00_controls/FORMAL04_C_COMBINED"
$cudaArgs = @()
if ($env:PGDA_CUDA_BIN) { $cudaArgs = @("--cuda-bin", $env:PGDA_CUDA_BIN) }

python scripts/run_native_256_release_pilot.py $case --run-id formal04_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal04_smoke1 --trace-count 1 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal04_distributed8_stride36 --trace-count 8 --trace-stride 36 --skip-air-reference @cudaArgs --execute
```

Do not run the sparse pair until the one-trace factorial is audited. Do not
run 256 traces until the selected geology passes a human morphology review.
