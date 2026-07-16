# FORMAL09C_P1_DENSE_PHYSICAL_FINITE_LAMINAE

This case is development-only and blocked from training.

```powershell
$case = "data/simulations/v2/00_controls/FORMAL09C_P1_DENSE_PHYSICAL_FINITE_LAMINAE"
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p1_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p1_smoke1 --trace-count 1 --skip-air-reference --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p1_native64_full --trace-count 64 --skip-air-reference --full-scene-only --execute
```

Do not use a stride-8 subset to judge finite-event topology. Run the exact
matched pair only after the native 64-trace blind image passes.
