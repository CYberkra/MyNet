# SceneWorld v0.8.0-alpha.4 Runner Audit

## Scope

This release keeps the v080a3 ready-to-run smoke dataset skeleton and adds a small framework-validation skeleton. It does not claim that real training B-scans have already been computed inside the sandbox.

## Accepted dataset skeletons

- `workspace/yingshan_sceneworld_smoke_v080a3`: five Yingshan families, one case per family, five variants per case, 25 gprMax runs required.
- `workspace/yingshan_framework_quick_v080a4`: one minimal case with five variants for checking whether local gprMax execution and post-merge code are wired correctly.

## Runner changes

- Added `scripts/run_all_gprmax.py` with `--workspace`, `--gprmax-source-dir`, `--conda-env`, `--gpu-ids`, and `--variants`.
- Generated `logs/run_all_gprmax.bat` now delegates to `scripts/run_all_gprmax.py` and runs all five variants instead of raw only.
- Scripts are relative to the dataset workspace and project root. They do not embed generation-machine paths.

## QC changes

`bscan_qc_report.json` now uses status `success` / `failed` and records per-variant `has_nan`, `has_inf`, `min`, `max`, `mean`, `std`, and `shape`.

If `raw_bscan.npy` and `target_bscan.npy` are valid and aligned, `outputs/clutter_gt_bscan.npy = raw_bscan.npy - target_bscan.npy` is generated.

Failures are propagated into both manifest columns `bscan_status` / `bscan_error` and `reports/dataset_summary.json`.

## Non-goals

- No full v080b1 pilot dataset is computed in this release.
- No Windows/CUDA/GPU long-run result is claimed without target-machine execution.
