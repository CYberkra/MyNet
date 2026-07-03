from pathlib import Path
import argparse
import csv
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def fail(msg):
    print(msg)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data_corrected_v1_4_terrain_direction")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    bad = 0
    required_dirs = [data_root / "windows", data_root / "lines"]
    for d in required_dirs:
        if not d.exists():
            bad += fail(f"MISSING_DIR {d}")

    index_path = data_root / "window_index.csv"
    if not index_path.exists():
        bad += fail(f"MISSING_INDEX {index_path}")
        print(f"RAW_ONLY_SCHEMA_BAD {bad}")
        return 1

    forbidden = {
        "bg501",
        "bg501_agc9",
        "processed_view",
        "response_teacher",
        "full_background",
        "target_only",
        "qa_view",
        "agc9",
        "bg501_agc",
    }
    required_window_keys = {"x_raw", "y_mask", "status_code", "label_weight"}
    required_line_keys = {"raw_full_normalized", "soft_mask_train", "status_code", "label_weight", "dt_ns", "trace_interval_m"}
    corrected_required_keys = {"correction_rule", "correction_velocity_m_per_ns", "correction_sigma_ns"}

    rows = list(csv.DictReader(open(index_path, encoding="utf-8")))
    sample_ids = set()
    line_names = set()
    for r in rows:
        sample_ids.add(r["sample_id"])
        line_names.add(r["line"])
        wp = data_root / "windows" / f"{r['sample_id']}.npz"
        if not wp.exists():
            bad += fail(f"MISSING_WINDOW {wp}")
        lp = data_root / "lines" / f"{r['line']}.npz"
        if not lp.exists():
            bad += fail(f"MISSING_LINE {lp}")

    for p in sorted((data_root / "windows").glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        keys = set(z.files)
        missing = required_window_keys - keys
        if missing:
            bad += fail(f"MISSING_WINDOW_KEYS {p.name} {sorted(missing)}")
        hit = {k for k in keys for f in forbidden if f.lower() in k.lower()}
        if hit:
            bad += fail(f"FORBIDDEN_INPUT {p.name} {sorted(hit)}")
        if p.stem not in sample_ids:
            bad += fail(f"UNINDEXED_WINDOW {p.name}")

    for p in sorted((data_root / "lines").glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        keys = set(z.files)
        missing = required_line_keys - keys
        if missing:
            bad += fail(f"MISSING_LINE_KEYS {p.name} {sorted(missing)}")
        hit = {k for k in keys for f in forbidden if f.lower() in k.lower()}
        if hit:
            bad += fail(f"FORBIDDEN_LINE_INPUT {p.name} {sorted(hit)}")
        if "corrected" in data_root.name.lower():
            missing_corrected = corrected_required_keys - keys
            if missing_corrected:
                bad += fail(f"MISSING_CORRECTION_METADATA {p.name} {sorted(missing_corrected)}")

    # Dataset-level locks. Config-level training eligibility is checked by check_configs.py.
    if any(r["line"] == "Line9" and r["split"] != "test" for r in rows):
        bad += fail("SPLIT_BAD Line9 rows must remain dataset-level test rows")
    if any(r["line"] == "LineX1" and r["split"] != "exclude" for r in rows):
        bad += fail("SPLIT_BAD LineX1 rows must remain excluded review rows")

    if (data_root / "qa_views").exists():
        print("QA_VIEWS_PRESENT_BUT_ISOLATED")
    if "Line6" in line_names:
        print("LINE6_DATASET_SPLIT_IS_METADATA_ONLY_CONFIG_DECIDES_TRAINING_ELIGIBILITY")

    print("RAW_ONLY_SCHEMA_OK" if bad == 0 else f"RAW_ONLY_SCHEMA_BAD {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
