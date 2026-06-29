# UavGPR-SimLab v0.8.0-alpha.3 SceneWorld B-scan Runner Audit

## Scope

This iteration preserves the v0.8.0-alpha.2 SceneWorld case package structure and adds the software-side runner needed to replace NaN B-scan placeholders with real gprMax outputs.

## Implemented

- Added `services/sceneworld_bscan_service.py`.
- Added CLI command `run-sceneworld-bscan`.
- Added `scripts/run_sceneworld_bscan_outputs.py`.
- Added `scripts/check_sceneworld_case_package.py`.
- Added `bscan_qc_report.json` to every generated case.
- Added `configs/run_plan_yingshan_sceneworld_smoke_v080a3.yaml`.
- Added `configs/run_plan_yingshan_sceneworld_pilot_v080b1.yaml`.
- Updated pilot configuration to 501 samples, 700 ns, 300 traces, 60 cases.
- Pilot family mix uses a 12-case cycle with one `cross_slope_high_relief` case, i.e. about 8.3% high-relief.

## B-scan QC contract

Each case contains `bscan_qc_report.json`. Before gprMax execution the status is `not_run`. After successful merging it records:

- per-variant success/failure;
- NaN/Inf state;
- min/max amplitude;
- shape match against expected `samples × trace_count`;
- whether `raw - target_only` can be computed;
- `clutter_gt_bscan.npy` status.

## Important runtime note

Full v080a3 acceptance means running all five variants for one case per family. For the smoke configuration this is 5 families × 5 variants × 72 traces. This is a long offline solver job and is intentionally exposed as a command/script instead of being hidden inside GUI self-test.

The current sandbox verified source-level gprMax availability and generated ready-to-run case packages, but did not complete the full 25-task SceneWorld B-scan run within the tool timeout. Target workstation execution is required for full v080a3 B-scan replacement.

## Recommended target-machine command

```bat
cd /d <UavGPR-SimLab project root>
conda activate gprMax
set "PYTHONPATH=%CD%\src"
python scripts\run_sceneworld_bscan_outputs.py ^
  --manifest workspace\yingshan_sceneworld_smoke_v080a3\datasets\yingshan_sceneworld_smoke_v080a3_manifest.csv ^
  --gprmax-root <your gprMax source root> ^
  --one-case-per-family ^
  --variants raw,target_only,background_only,clutter_only,air_only ^
  --omp-threads 1 ^
  --timeout 3600 ^
  --force
```

Then run:

```bat
python scripts\check_sceneworld_case_package.py ^
  --workspace workspace\yingshan_sceneworld_smoke_v080a3
```

## Risk

- The runner changes dataset post-processing only. It does not alter formal gprMax command semantics used by batch simulation.
- The pilot dataset remains a target-machine workload and should not be generated implicitly during normal GUI startup or self-test.
