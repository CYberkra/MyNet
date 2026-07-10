# YingShan V15 Final Label Release

- Version: `YINGSHAN_V15_FINAL_20260710`
- Release status: final label release; not a formal training release.
- Line9: preserved as the highest-quality, test-only crossing anchor.
- X1: remains review-only/excluded.

## Final crossing decisions

- Line3-Line9: weak relabel Line3 to the locally supported ~453 ns ridge; transition collars ignored.
- Line6-Line9: preserve Line9; ignore the ambiguous Line6 neighborhood.
- Line9-LineX1: preserve Line9; ignore the ambiguous X1 neighborhood.
- LineL1-LineX1: weak relabel X1 to the locally supported ~327.5 ns ridge.
- Other four crossings retain existing labels.

## Safety properties

- No whole-line time shift was applied.
- Accepted relocations remain weak; no ambiguous label was promoted to strong.
- Unresolved regions have zero label weight and are excluded from all supervised label losses.
- V14 geometry is preserved in every final line NPZ for rollback/audit.

## Remaining formal-training blockers

- No confirmed true-negative measured windows.
- No approved simulation independent of Line9 conditioning.
- No final line-level train/validation split.
