#!/usr/bin/env python3
"""Generate LOLO-CV configs for v4 Pilot-Train (100 sim cases).

For each held-out line in [Line3, Line6, Line7, Line9, LineL1]:
  - train_lines = other 4 real lines
  - val_lines = [heldout], test_lines = [heldout]
  - All train_trace_ranges/test_trace_ranges cleared (full-line eval)

Outputs:
  configs/gpu_train_v4_pilot_mixed_loo_<line>_seed<seed>.json  (15 files)
  reports/v4_loo_manifest.json

Usage:
  "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/make_v4_loo_configs.py
"""
import copy
import json
import sys
from pathlib import Path

BASE_CONFIG = Path("configs/gpu_train_v4_pilot_mixed.json")
OUT_DIR = Path("configs")
MANIFEST_DIR = Path("reports")

LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
SEEDS = [1901, 1902, 1903]


def main():
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

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
            cfg["version"] = f"v4_pilot_mixed_loo_{label}"
            cfg["run_dir"] = f"outputs/run_gpu_v4_pilot_mixed_loo_{label}"
            cfg["note"] = (
                f"LOLO-CV fold: heldout={heldout}, "
                f"train_lines={train_lines}, "
                f"seed={seed}, "
                f"100 Pilot-Train sim cases (simulation_pretrain_v3)"
            )

            fname = f"gpu_train_v4_pilot_mixed_loo_{label}.json"
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
        "description": "LOLO-CV manifest for v4 (100 Pilot-Train sim cases)",
        "base_config": str(BASE_CONFIG),
        "lines": LINES,
        "seeds": SEEDS,
        "n_configs": len(manifest_entries),
        "entries": manifest_entries,
    }
    mp = MANIFEST_DIR / "v4_loo_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {mp}")
    print(f"Total: {len(manifest_entries)} configs ({len(LINES)} folds × {len(SEEDS)} seeds)")

    # Show training order
    print("\nTraining order (recommended):")
    for e in manifest_entries:
        print(f"  {e['config']}  →  {e['run_dir']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
