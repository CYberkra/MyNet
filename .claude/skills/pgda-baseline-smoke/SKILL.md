---
name: pgda-baseline-smoke
description: Run or explain the minimal PGDA-CSNet baseline smoke checks: raw-only dataset guard, config guard, CPU training smoke, and frozen checkpoint mini-eval.
---

# PGDA Baseline Smoke

Use this skill when the user asks to verify the current PGDA-CSNet baseline, check whether the project still runs after edits, or prepare a minimal reproducibility check.

## Preferred working directory

Use the active PGDA training directory, typically:

```text
PGDA_CSNet_v0_9_6_SEARCH_WINDOW_GUARD
```

In the transferred workspace used on 2026-06-27:

```text
D:\Claude\PGDA-CSNet\workspace\transfer_20260627_142748\PGDA-CSNet_transfer_bundle_20260627_142748\PGDA_CSNet_v0_9_6_SEARCH_WINDOW_GUARD
```

## Smoke sequence

1. Syntax check:

```bash
python -m py_compile scripts/check_dataset.py scripts/check_configs.py scripts/train_raw_only.py scripts/eval_full_line.py pgdacsnet/model_raw_unet.py
```

2. Dataset and config guardrails:

```bash
python scripts/check_dataset.py
python scripts/check_configs.py
```

3. CPU smoke training:

```bash
bash 01_fast_cpu_check_raw_only.sh
```

4. Frozen checkpoint mini-eval:

```bash
python scripts/eval_full_line.py \
  --line Line9 \
  --run-dirs outputs/run_gpu_paper_v1_9d_mambavision_hybrid_final_seed1902_line9holdout \
  --checkpoint final \
  --data-root data_corrected_v1_4_terrain_direction \
  --force-cpu \
  --no-plot \
  --trace-start 1664 \
  --trace-end 1700 \
  --center-fusion-weight 0.5 \
  --presence-thr 0.45 \
  --path-prob-thr 0.50 \
  --dp-breakable \
  --dp-max-jump 6 \
  --dp-smooth-weight 0.16 \
  --dp-min-segment 16 \
  --out-dir outputs/eval_smoke_line9_1664_1700
```

## Pass criteria

- `RAW_ONLY_SCHEMA_OK`
- `CONFIG_GUARDRAILS_OK`
- CPU smoke reaches epoch output
- mini-eval writes `Line9_holdout_tr1664_1700_full_metrics.csv`

Warnings are acceptable when they refer to historical configs or intentionally skipped helper JSON.
