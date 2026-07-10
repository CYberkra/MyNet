# YingShan V15 Final Label Validation

Date: 2026-07-10

## Release status

- Label release: **complete**
- Version: `YINGSHAN_V15_FINAL_20260710`
- Formal training release: **not allowed**
- Canonical trace order: original CSV acquisition order
- Line9: unchanged, test-only, primary crossing anchor
- LineX1: review-only/excluded

## Final decisions

| Crossing | Final action | V14 air-corrected mismatch | V15 air-corrected mismatch | Supervision state |
|---|---|---:|---:|---|
| Line3-Line9 | Relabel Line3 weakly to the accepted ~453 ns ridge | 63.413 ns | 6.698 ns | active weak; transition collar ignored |
| Line3-LineL1 | Keep existing weak labels | 10.474 ns | 10.474 ns | active weak |
| Line3-Line7 | Keep | 0.170 ns | 0.170 ns | active |
| Line6-Line9 | Keep Line9; ignore ambiguous Line6 neighborhood | 33.651 ns | 33.651 ns | Line6 excluded from losses |
| Line6-LineL1 | Keep; surface-reference evidence supports consistency | 15.662 ns | 15.662 ns | active |
| Line6-Line7 | Keep | 2.012 ns | 2.012 ns | active weak |
| Line9-LineX1 | Keep Line9; ignore ambiguous X1 neighborhood | 16.057 ns | 16.057 ns | X1 excluded from losses |
| LineL1-LineX1 | Relabel X1 weakly to the accepted ~327.5 ns ridge | 21.471 ns | 0.689 ns | active weak; X1 remains excluded globally |

“Resolved” means a crossing is either accepted/relabelled or explicitly excluded from supervised losses. It does not claim that ambiguous geological truth was recovered in excluded regions.

## Changed and ignored traces

| Line | Changed traces | Ignored traces | Active strong | Active weak | Split |
|---|---:|---:|---:|---:|---|
| Line3 | 126 | 42 | 949 | 822 | unassigned |
| Line6 | 0 | 129 | 949 | 583 | unassigned |
| Line7 | 0 | 0 | 515 | 1139 | unassigned |
| Line9 | 0 | 0 | 937 | 1441 | test |
| LineL1 | 0 | 0 | 1406 | 544 | unassigned |
| LineX1 | 134 | 139 | 300 | 499 | exclude |

## Safety guarantees

- No whole-line time shift was applied.
- V14 masks, statuses, and weights are preserved in each V15 line file for rollback.
- Line9 geometry, status, weight, and split remain unchanged.
- Accepted relocations remain weak labels and are not promoted to strong.
- Ignored traces have `soft_mask_train=0`, `label_weight=0`, `status_code=2`, and an explicit `ignore_mask`.
- All 78 windows are regenerated from the V15 full-line arrays and match them element-for-element.
- Original waveform, GNSS, ground elevation, flight height, profile chainage, and orientation metadata are unchanged.

## Validation

- V15 final validator: passed, no failures.
- Generic dataset validator: passed structurally; formal-ready remains false by policy.
- Regression tests excluding the historically slow `tests/test_gprmambasep.py`: **85 passed**.
- Modified Python files: `py_compile` passed.
- Normal project contract validation: passed with the expected formal-training warning.
- `--require-formal-ready`: failed as expected because formal training remains blocked.

## Remaining formal-training blockers

1. No confirmed real true-negative windows.
2. No approved simulation independent of Line9 conditioning.
3. Formal line-level train/validation split is not locked.
4. Batch 3 simulation cases still require case-wise review.

The V15 label task is complete. These remaining items are separate dataset-release tasks, not unresolved V15 label-writing defects.
