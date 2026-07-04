#!/usr/bin/env python3
"""Generate LOLO-CV configs for GprMambaSep Stage 2 mixed sim-real training.

For each held-out line in [Line3, Line6, Line7, Line9, LineL1]:
  - train_lines = other 4 real lines
  - val_lines = [heldout], test_lines = [heldout]
  - sim_data_root = PGDA_SYNTH_DATASET_V1/05_accepted_dataset (auto-detected sim lines)
  - sim_batch_ratio = 0.3

Outputs:
  configs/gpu_mixed_v2_gprmambasep_loo_<line>_seed<seed>.json  (15 files)
  reports/v2_gprmambasep_loo_manifest.json

Usage:
  "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/make_v2_gprmambasep_loo_configs.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

BASE_CONFIG = Path("configs/gpu_mixed_v2_gprmambasep.json")
OUT_DIR = Path("configs")
MANIFEST_DIR = Path("reports")

LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
SEEDS = [1901, 1902, 1903]


def main() -> int:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []

    for heldout in LINES:
        train_lines = [l for l in LINES if l != heldout]
        val_lines = [heldout]
        test_lines = [heldout]

        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["train_lines"] = train_lines
            cfg["train_trace_ranges"] = {}
            cfg["val_lines"] = val_lines
            cfg["test_lines"] = test_lines
            cfg["test_trace_ranges"] = {}
            cfg["seed"] = seed

            label = f"{heldout}_seed{seed}"
            cfg["version"] = f"v2_gprmambasep_mixed_loo_{label}"
            cfg["run_dir"] = f"outputs/run_gprmambasep_loo_{label}"
            cfg["note"] = (
                f"LOLO-CV fold: heldout={heldout}, "
                f"train_lines={train_lines}, seed={seed}, "
                f"sim_batch_ratio=0.3, "
                f"sim data from PGDA_SYNTH_DATASET_V1/05_accepted_dataset "
                f"(auto-detected sim lines)"
            )

            fname = f"gpu_mixed_v2_gprmambasep_loo_{label}.json"
            path = OUT_DIR / fname
            path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            manifest_entries.append({
                "config": fname,
                "heldout_line": heldout,
                "seed": seed,
                "run_dir": cfg["run_dir"],
            })
            print(f"  {fname}")

    # Write manifest
    manifest = {
        "description": "LOLO-CV manifest for GprMambaSep v2 (mixed sim-real)",
        "base_config": str(BASE_CONFIG),
        "lines": LINES,
        "seeds": SEEDS,
        "n_configs": len(manifest_entries),
        "entries": manifest_entries,
    }
    mp = MANIFEST_DIR / "v2_gprmambasep_loo_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {mp}")
    print(f"Total: {len(manifest_entries)} configs ({len(LINES)} folds x {len(SEEDS)} seeds)")

    # Recommended training order
    print("\nTraining order (recommended):")
    for e in manifest_entries:
        print(f"  {e['config']}  ->  {e['run_dir']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
