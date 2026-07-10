# V15 Integration Acceptance

Date: 2026-07-10

## Imported Deliverables

- `MyNet_Codex_精简源码_7a250a7.zip`
- `营山V15最终标签数据包_7a250a7.zip`
- `MyNet_Codex_精简证据资料_7a250a7.zip`

The active measured-data release is `data_yingshan_v15_final_20260710`.
V14 arrays remain only as explicit rollback and audit inputs. Line9 remains
test-only; LineX1 remains excluded/review-only.

## Acceptance Results

- All six V15 full-line NPZ files and all 78 V15 windows passed exact slice
  consistency checks.
- V15 final release validation passed. The accepted Line3-Line9 and
  LineL1-LineX1 air-corrected differences are 6.70 ns and 0.69 ns.
- Original CSV reconstruction completed for all six lines. Window-cache
  waveform correlations are greater than 0.999999987, with maximum absolute
  differences no greater than 1.30e-4 from floating-point normalisation.
- 44 targeted governance, dataset, orientation, loss, and evaluation tests
  passed. Python compilation passed.
- The ordinary project-contract check passed. The formal-ready check remains
  blocked by design: no confirmed measured true negatives, no approved
  non-Line9-conditioned simulations, and no released formal line split.

## Integration Fixes

- Added the missing full-line dataset contract used by formal training.
- Preserved V15 index fields during raw-CSV metadata enrichment.
- Made synthetic-case promotion require a matching human-audit source hash.
- Treated `LINE9_TERRAIN_*` as Line9 terrain family data.
- Made the project validator audit frozen configurations without requiring
  runnable formal split files from disabled historical configurations.

## Cleanup

Deleted obsolete, non-weight simulation outputs:

- `outputs/v4_quick_test`
- `workspace/pilot_validation_v5`

No `.pt`, `.pth`, `.ckpt`, or `.onnx` weights existed in the repository at
cleanup time. Historical Line9-conditioned simulation data remains quarantined
as audit evidence and is not approved for formal training.
