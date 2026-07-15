# FORMAL03_CORRELATED_COVER_RICKER65

This source-ablation case is blocked from training. Run from the repository
root and stop at the first failed gate.

```powershell
$case = "data/simulations/v2/00_controls/FORMAL03_CORRELATED_COVER_RICKER65"
$cudaArgs = @()
if ($env:PGDA_CUDA_BIN) { $cudaArgs = @("--cuda-bin", $env:PGDA_CUDA_BIN) }

python scripts/run_native_256_release_pilot.py $case --run-id formal03_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal03_smoke1 --trace-count 1 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal03_distributed24_stride11 --trace-count 24 --trace-stride 11 --skip-air-reference @cudaArgs --execute
```

Do not run 256 traces until a source variant passes the 24-trace causal and
human morphology gates. The supervision-valid interval ends at
500 ns; later samples are diagnostics only.
