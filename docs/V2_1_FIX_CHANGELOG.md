# v2.1-fix Changelog

## Added

- `v2_1_curvegassist_lite` / `curvegassist_lite` / `g_assisted_curvemamba` model aliases.
- Optional G-assisted task path in `GprMambaSep`:
  - shared full-resolution decoder from bottleneck;
  - raw-local projection;
  - fusion of shared task feature + G feature + raw-local feature.
- Optional heads:
  - `curve_logits` for trace-wise `P(t|trace)`;
  - `global_no_target_logits` for line/window-level abstention;
  - `uncertainty_logits` reserved for later uncertainty calibration.
- Curve losses:
  - curve CE/KL-like trace-wise distribution loss;
  - softargmax center loss;
  - first-order smoothness;
  - second-order curvature;
  - shallow suppression.
- Global no-target BCE loss.
- `eval_full_line.py` support for curve logits. By default, if available, `softmax(curve_logits, dim=time)` is used for DP path probability.
- `scripts/audit_component_arrays.py`.
- `scripts/plot_gprmambasep_diagnostics.py`.
- 6G/12G configs and review-val config.
- Smoke test for the new model and loss.

## Changed

- `GprMambaSepOutput` now keeps legacy 6-value tuple unpacking, while exposing v2.1 fields via attributes/dict-style access.
- `build_model()` now maps `v2_1_curvegassist_lite` to `build_gprmambasep()` with route-2 defaults.

## Not changed

- Existing `v2_1_gprmambasep_lite` behavior remains legacy-compatible unless new config flags are enabled.
- A/S/G component outputs remain available.
- Legacy `mask_logits`, `presence_logits`, `center_logits` remain available.

## Second audit fixes (2026-07-08)

- Implemented `grad_accum_steps` in `scripts/train_raw_only.py`; 6G/12G configs no longer silently ignore gradient accumulation.
- `eval_full_line.py` now records `curve_source='curve_logits_dp'` when curve logits drive DP.
- `eval_full_line.py` now always writes `*_path_softmask.npy`, so diagnostic plots use the exact DP path probability image used for metrics.
- `plot_gprmambasep_diagnostics.py` now recognises `soft_mask_train` line-level labels.
- Reduced default shallow-suppression cutoff from 300 ns to 260 ns in CurveGAssist configs to avoid penalising medium-depth positives around 296–324 ns.
- Re-ran project-wide `py_compile` over `pgdacsnet/`, `scripts/`, and `tests/`; no syntax errors.
- Re-ran `tests/test_curvegassist_smoke.py`; passed.

## Third audit fixes (2026-07-08)

- Addressed reviewer note: `enable_uncertainty_head=true` in the 12G config now has a real low-weight training objective instead of producing an untrained diagnostic output.
- Added `uncertainty_nll_loss()` in `scripts/losses_gprmambasep.py`:
  - derives target/predicted trace-wise centers from `curve_logits` and the existing soft mask;
  - collapses `uncertainty_logits` to trace-wise log-sigma under `P(t|trace)`;
  - applies a heteroscedastic NLL with clamped log-sigma for stable calibration.
- Added `loss.uncertainty_weight=0.02` to the 12G CurveGAssist config.
- Extended the smoke test to enable `uncertainty_logits` and verify `uncertainty_nll` is present.
- Re-ran project-wide `py_compile` over `pgdacsnet/`, `scripts/`, and `tests/`; no syntax errors.
- Re-ran `tests/test_curvegassist_smoke.py`; passed.

## Fourth audit fixes (2026-07-08)

- Fixed diagnostic-path inconsistency: `plot_gprmambasep_diagnostics.py` now prefers `*_pred_centerline.csv` for exact decoded `mean/DP/GT` centerlines instead of approximating the DP path by center-of-mass over `*_path_softmask.npy`.
- Added a minimal diagnostics smoke test with fake line/eval files during audit; figure generation passed.
- Confirmed gradient flow for curve head, uncertainty head, global no-target head, and fused task path in a small backward smoke.
