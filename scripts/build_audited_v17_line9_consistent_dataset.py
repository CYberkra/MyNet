from pathlib import Path
import csv
import json
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_audited_v16_20260627"
LINE9_V17 = ROOT / "reports" / "line9_v17_label_package" / "Line9_audited_masks_v17.npz"
OUT = ROOT / "data_audited_v17_line9_consistent"


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src, dst):
    if src.exists():
        shutil.copytree(src, dst)


def window_rows_for_line(line, z, window=256, stride=128):
    width = int(z["raw_full_normalized"].shape[1])
    starts = list(range(0, max(1, width - window + 1), stride))
    if not starts or starts[-1] != width - window:
        starts.append(max(0, width - window))
    rows = []
    for start in starts:
        end = min(start + window, width)
        if end - start < window:
            start = max(0, end - window)
        sl = slice(start, end)
        status = z["status_code"][sl]
        ignore = z["ignore_mask"][:, sl].max(axis=0) > 0 if "ignore_mask" in z.files else np.zeros(end - start, dtype=bool)
        active = ~ignore
        rows.append(
            {
                "sample_id": f"{line}_tr{start:04d}_{end-1:04d}",
                "line": line,
                "start": start,
                "end": end - 1,
                "split": "train",
                "present": int(((status == 1) & active).sum()),
                "weak": int(((status == 2) & active).sum()),
                "no_pick": int(((status == 0) & active).sum()),
                "ignore": int(ignore.sum()),
            }
        )
    return rows


def write_windows_for_line(line, z):
    for row in window_rows_for_line(line, z):
        start = int(row["start"])
        end = int(row["end"]) + 1
        arrays = {}
        arrays["x_raw"] = z["raw_full_normalized"][:, start:end].astype(np.float32)
        arrays["y_mask"] = z["soft_mask_train"][:, start:end].astype(np.float32)
        if "ignore_mask" in z.files:
            arrays["ignore_mask"] = z["ignore_mask"][:, start:end].astype(np.float32)
        for key in ["status_code", "label_weight"]:
            arrays[key] = z[key][start:end]
        arrays["line"] = np.array(line)
        arrays["start_trace"] = np.array(start, dtype=np.int32)
        arrays["end_trace"] = np.array(end - 1, dtype=np.int32)
        np.savez_compressed(OUT / "windows" / f"{row['sample_id']}.npz", **arrays)


def main():
    reset_dir(OUT)
    (OUT / "lines").mkdir(parents=True, exist_ok=True)
    (OUT / "windows").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    copy_tree(SOURCE / "terrain_features", OUT / "terrain_features")
    copy_tree(SOURCE / "terrain_features_zero_material_v1", OUT / "terrain_features_zero_material_v1")

    line_rows = []
    all_window_rows = []
    v17 = np.load(LINE9_V17, allow_pickle=False)
    for src_path in sorted((SOURCE / "lines").glob("*.npz")):
        line = src_path.stem
        src = np.load(src_path, allow_pickle=False)
        arrays = {k: src[k] for k in src.files}
        note = "copied_from_data_audited_v16_20260627"
        if line == "Line9":
            arrays["label_weight"] = v17["label_weight_v17"].astype(np.float32)
            arrays["status_code"] = v17["status_code_v17"].astype(np.int16)
            arrays["label_time_center_v17_ns"] = v17["label_time_center_v17_ns"].astype(np.float32)
            arrays["label_depth_center_v17_m"] = v17["label_depth_center_v17_m"].astype(np.float32)
            arrays["review_priority_v17"] = v17["review_priority_v17"]
            arrays["decision_v17"] = v17["decision_v17"]
            arrays["confidence_v17"] = v17["confidence_v17"].astype(np.float32)
            arrays["v17_source_package"] = np.array("reports/line9_v17_label_package")
            arrays["v17_training_policy"] = np.array("Line9 geometry kept; high-risk holdout/tail downgraded to weak labels")
            note = "line9_v17_reaudit_applied"
        elif "v17_source_package" not in arrays:
            arrays["v17_source_package"] = np.array("not_line9_v17")
            arrays["v17_training_policy"] = np.array("unchanged from data_audited_v16_20260627")

        out_line = OUT / "lines" / f"{line}.npz"
        np.savez_compressed(out_line, **arrays)
        z = np.load(out_line, allow_pickle=False)
        write_windows_for_line(line, z)
        all_window_rows.extend(window_rows_for_line(line, z))
        ignore_count = int(z["ignore_mask"].max(axis=0).sum()) if "ignore_mask" in z.files else 0
        line_rows.append(
            {
                "line": line,
                "width": int(z["raw_full_normalized"].shape[1]),
                "height": int(z["raw_full_normalized"].shape[0]),
                "source_note": note,
                "ignore_trace_count": ignore_count,
                "label_weight_sum": float(z["label_weight"].sum()),
                "strong_count": int((z["status_code"] == 1).sum()),
                "weak_count": int((z["status_code"] == 2).sum()),
            }
        )

    with open(OUT / "window_index.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "line", "start", "end", "split", "present", "weak", "no_pick", "ignore"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_window_rows)
    with open(OUT / "manifests" / "v17_dataset_line_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(line_rows[0].keys()))
        w.writeheader()
        w.writerows(line_rows)
    policy = {
        "dataset": "data_audited_v17_line9_consistent",
        "base": "data_audited_v16_20260627",
        "line9_v17_package": str(LINE9_V17.relative_to(ROOT)),
        "line9_decision": "keep v1.4 geometry after visual/PDF review; downgrade high-risk holdout/tail uncertainty to weak labels",
        "do_not_use_as": "automatic model-generated labels",
    }
    (OUT / "manifests" / "v17_training_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "DATASET_README.md").write_text(
        "# data_audited_v17_line9_consistent\n\n"
        "Built from `data_audited_v16_20260627` plus the Line9 v17 review package.\n\n"
        "Line9 centerline geometry is retained from the v1.4 label because the reviewed high-risk segments remain inside the PDF-constrained 12-16 m interface band and follow the visible continuous reflector. High-risk holdout/tail traces are downgraded to weak supervision with reduced weights.\n",
        encoding="utf-8",
    )
    print(OUT)
    print(OUT / "manifests" / "v17_dataset_line_summary.csv")


if __name__ == "__main__":
    main()
