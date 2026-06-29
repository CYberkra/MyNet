---
name: pgda-transfer-validate
description: Validate a PGDA-CSNet transfer zip or extracted bundle without overwriting existing work; checks integrity, stale paths, data roots, Python syntax, CPU smoke, and frozen checkpoint inference.
---

# PGDA Transfer Validate

Use this skill when the user provides a PGDA-CSNet transfer zip, model bundle, freeze package, or extracted handoff directory and asks to inspect, accept, migrate, or validate it.

## Rules

- Never extract over the current project tree.
- Extract archives only into `workspace/transfer_<timestamp_or_bundle_name>/` unless the user gives another isolated path.
- Do not delete source zips, freeze zips, merged `.out` files, checkpoints, or outputs.
- Treat historical reports as evidence; do not rewrite them during validation.
- Prefer read-only inspection first; only patch files if the user asks to migrate/fix the bundle.

## Validation checklist

1. Inventory the archive or extracted directory:
   - entry count
   - compressed/uncompressed size
   - top-level directories
   - extensions summary
   - largest files
2. Run zip integrity check when applicable:
   - `ZipFile.testzip()` should return `None`.
3. Identify the active PGDA training tree:
   - usually `PGDA_CSNet_v0_9_6_SEARCH_WINDOW_GUARD/`
   - record `requirements.txt`, `scripts/`, `configs/`, `reports/`, `outputs/`, `data*` folders.
4. Check for stale paths:
   - `F:\codex`
   - `F:/codex`
   - `E:\anaconda3`
   - old `data_corrected_v1` defaults
5. Check current data roots:
   - `data_corrected_v1_4_terrain_direction`
   - `data_audited_v16_20260627`
   - `data_audited_v17_line9_consistent`
6. Verify dataset arrays can load:
   - sample one or two `.npz` windows
   - confirm `x_raw`, `y_mask`, `status_code`, `label_weight`
   - note `ignore_mask` for audited datasets
7. Verify frozen checkpoint if present:
   - compute SHA256
   - compare against `reports/CURRENT_MODEL_STATE.md`
8. Run syntax checks:
   - `python -m py_compile scripts/train_raw_only.py scripts/eval_full_line.py scripts/v11_confidence_control.py scripts/v11_train_error_head.py scripts/v12_target_input_adapter.py pgdacsnet/model_raw_unet.py`
9. Run a CPU smoke check:
   - prefer `bash 01_fast_cpu_check_raw_only.sh` after migration
   - otherwise use `configs/fast_cpu_check_corrected_v1_4_terrain_direction.json`
10. Run a frozen checkpoint mini-eval when available:
   - Line9 trace `1664-1700`
   - `--force-cpu --no-plot`
   - output to a smoke eval folder.

## Report format

Return a concise Chinese report:

- `结论`: pass / pass with warnings / fail
- `已验证`: bullet list
- `修复或迁移问题`: bullet list
- `剩余风险`: bullet list
- `推荐下一步`: one or two concrete commands or actions
