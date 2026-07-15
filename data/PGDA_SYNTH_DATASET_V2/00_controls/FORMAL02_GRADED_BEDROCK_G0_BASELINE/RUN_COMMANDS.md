# FORMAL02_GRADED_BEDROCK_G0_BASELINE

This pre-solver baseline is not trainable. Run from the repository root and
stop at the first failed gate. The runner stages a disposable solver copy;
never execute gprMax inside this versioned source deck.

```powershell
$case = "data/PGDA_SYNTH_DATASET_V2/00_controls/FORMAL02_GRADED_BEDROCK_G0_BASELINE"
$cudaArgs = @()
if ($env:PGDA_CUDA_BIN) { $cudaArgs = @("--cuda-bin", $env:PGDA_CUDA_BIN) }

python scripts/run_native_256_release_pilot.py $case --run-id formal02_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal02_smoke1 --trace-count 1 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal02_distributed32_stride8 --trace-count 32 --trace-stride 8 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal02_full256_pair --trace-count 256 --skip-air-reference @cudaArgs --execute
```

Do not run all 256 traces until the one-trace pair and a
distributed 32-trace pair pass. Run `air_reference.in` only after the source
proxy is accepted. The declared supervision-valid interval ends at
500 ns; 500-650 ns is boundary diagnostics only.
