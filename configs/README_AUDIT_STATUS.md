# Configuration audit status — 2026-07-10

All JSON configurations in this directory are intentionally frozen with
`"enabled": false` until `data/dataset_contract_v2/dataset_manifest.json`
reports `formal_training_allowed: true` and the selected configuration passes
`scripts/validate_project_contracts.py --require-formal-ready`.

The freeze prevents historical experiment files from being mistaken for valid
paper-training contracts. The principal blockers are:

- missing canonical real `window_index.csv` and `lines/*.npz`;
- no confirmed real negative traces (`status_code=0`);
- Line9-conditioned simulation data quarantined from Line9 holdout training;
- V4 visible-phase relabeling not completed;
- Batch 3 case-wise geometry review not completed.

Before re-enabling one configuration, update its paths and split lists, verify
that it has a nonempty validation split, provide an explicit `run_type`, and
remove its `training_block_reason` only after the validator succeeds.
