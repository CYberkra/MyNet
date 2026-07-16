# FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE

This case is development-only and blocked from training.

```powershell
$case = "data/simulations/v2/00_controls/FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE"
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p2_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p2_smoke1 --trace-count 1 --skip-air-reference --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p2_native64_full --trace-count 64 --skip-air-reference --full-scene-only --execute
```

Compare the native 64 full scene against the exact FORMAL06C native 64 run.
Do not start a matched 64-trace control until the blind morphology passes.
