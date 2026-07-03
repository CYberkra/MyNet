from pathlib import Path
import csv
import json
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"
LABEL_PACKAGE = ROOT.parent / "label_audit_v16_20260627" / "PGDA_L3_L1_L6_L7_AUDITED_LABELS_V16_20260627"
OUT_ROOT = ROOT / "data_audited_v16_20260627"
V16_LINES = ["Line3", "Line6", "Line7", "LineL1"]


def read_csv_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def bool_col(rows, name):
    return np.array([str(r.get(name, "")).lower() == "true" for r in rows], dtype=bool)


def int_col(rows, name):
    return np.array([int(float(r[name])) for r in rows], dtype=np.int16)


def review_band_mask(time_ns, center_ns, half_width_ns):
    time_ns = time_ns.astype(np.float32)
    center_ns = center_ns.astype(np.float32)
    half_width_ns = half_width_ns.astype(np.float32)
    return (np.abs(time_ns[:, None] - center_ns[None, :]) <= half_width_ns[None, :]).astype(np.uint8)


def build_line(line, source_line_path, out_line_path):
    src = np.load(source_line_path, allow_pickle=False)
    arrays = {k: src[k] for k in src.files}
    raw = arrays["raw_full_normalized"].astype(np.float32)
    h, w = raw.shape

    ignore = np.zeros((h, w), dtype=np.float32)
    source_note = "copied_from_v1_4"

    if line in V16_LINES:
        z = np.load(LABEL_PACKAGE / "labels" / "by_line_npz" / f"{line}_audited_masks_v16.npz", allow_pickle=False)
        rows = read_csv_rows(LABEL_PACKAGE / "labels" / "by_line_csv" / f"{line}_audited_labels_v16.csv")
        if len(rows) != w or z["soft_mask"].shape != raw.shape:
            raise ValueError(f"{line}: source raw and v16 labels do not match")

        context = bool_col(rows, "line6_front_context_ignore_flag_v16")
        ultraweak = bool_col(rows, "line6_front_weak_evidence_flag_v16") & ~context
        bg_suppression = bool_col(rows, "line6_possible_bg_suppression_flag_v16")
        status = int_col(rows, "status_code")
        status[context] = 2
        status[ultraweak] = 2

        soft_review = z["soft_mask"].astype(np.float32)
        hard_review = z["hard_mask"].astype(np.uint8)
        soft_train = soft_review.copy()
        hard_train = hard_review.copy()
        soft_train[:, context] = 0.0
        hard_train[:, context] = 0
        ignore[:, context] = 1.0

        arrays["soft_mask_train"] = soft_train.astype(np.float32)
        arrays["soft_mask_review_v16"] = soft_review.astype(np.float32)
        arrays["hard_mask_train_v16"] = hard_train.astype(np.uint8)
        arrays["hard_mask_review_v16"] = hard_review.astype(np.uint8)
        arrays["review_band_mask_v16"] = review_band_mask(z["time_ns"], z["label_time_center_v16_ns"], z["mask_half_width_ns_v16"])
        arrays["status_code"] = status.astype(np.int16)
        arrays["label_weight"] = z["final_effective_weight_v16"].astype(np.float32)
        arrays["label_time_center_v16_ns"] = z["label_time_center_v16_ns"].astype(np.float32)
        arrays["mask_half_width_ns_v16"] = z["mask_half_width_ns_v16"].astype(np.float32)
        arrays["final_effective_weight_v16"] = z["final_effective_weight_v16"].astype(np.float32)
        arrays["line6_front_context_ignore_flag_v16"] = context.astype(np.uint8)
        arrays["line6_front_ultraweak_train_flag_v16"] = ultraweak.astype(np.uint8)
        arrays["line6_possible_bg_suppression_flag_v16"] = bg_suppression.astype(np.uint8)
        arrays["ignore_mask"] = ignore.astype(np.float32)
        arrays["v16_source_package"] = np.array("PGDA_L3_L1_L6_L7_AUDITED_LABELS_V16_20260627")
        arrays["v16_training_policy"] = np.array("soft_mask_train masks out Line6 context-ignore; review labels are preserved separately")
        source_note = "v16_audited_labels_applied"
    else:
        arrays["ignore_mask"] = ignore.astype(np.float32)
        arrays["soft_mask_review_v16"] = arrays["soft_mask_train"].astype(np.float32)
        arrays["review_band_mask_v16"] = (arrays["soft_mask_train"] > 0.0).astype(np.uint8)
        arrays["v16_source_package"] = np.array("not_in_v16_package")
        arrays["v16_training_policy"] = np.array("copied unchanged from data_corrected_v1_4_terrain_direction")

    np.savez_compressed(out_line_path, **arrays)
    return {
        "line": line,
        "width": int(w),
        "height": int(h),
        "source_note": source_note,
        "ignore_trace_count": int((ignore.mean(axis=0) > 0.5).sum()),
        "label_weight_sum": float(np.asarray(arrays["label_weight"], dtype=np.float32).sum()),
    }


def write_windows_and_index():
    rows = list(csv.DictReader(open(SOURCE_DATA_ROOT / "window_index.csv", encoding="utf-8")))
    out_rows = []
    for r in rows:
        line = r["line"]
        s = int(r["start"])
        e = int(r["end"]) + 1
        line_z = np.load(OUT_ROOT / "lines" / f"{line}.npz", allow_pickle=False)
        status = line_z["status_code"][s:e].astype(np.int16)
        ignore_col = (line_z["ignore_mask"][:, s:e].mean(axis=0) > 0.5)
        active = ~ignore_col
        present = int(((status == 1) & active).sum())
        weak = int(((status == 2) & active).sum())
        no_pick = int(((status == 0) & active).sum())
        out_id = r["sample_id"]
        np.savez_compressed(
            OUT_ROOT / "windows" / f"{out_id}.npz",
            x_raw=line_z["raw_full_normalized"][:, s:e].astype(np.float32),
            y_mask=line_z["soft_mask_train"][:, s:e].astype(np.float32),
            status_code=status,
            label_weight=line_z["label_weight"][s:e].astype(np.float32),
            ignore_mask=line_z["ignore_mask"][:, s:e].astype(np.float32),
        )
        rr = dict(r)
        rr["present"] = str(present)
        rr["weak"] = str(weak)
        rr["no_pick"] = str(no_pick)
        rr["ignore"] = str(int(ignore_col.sum()))
        out_rows.append(rr)

    fields = list(rows[0].keys())
    if "ignore" not in fields:
        fields.append("ignore")
    with open(OUT_ROOT / "window_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    (OUT_ROOT / "lines").mkdir(parents=True)
    (OUT_ROOT / "windows").mkdir()
    (OUT_ROOT / "manifests").mkdir()

    line_rows = []
    for source_line in sorted((SOURCE_DATA_ROOT / "lines").glob("*.npz")):
        line = source_line.stem
        line_rows.append(build_line(line, source_line, OUT_ROOT / "lines" / source_line.name))

    write_windows_and_index()

    for dname in ["terrain_features", "terrain_features_zero_material_v1"]:
        src = SOURCE_DATA_ROOT / dname
        if src.exists():
            shutil.copytree(src, OUT_ROOT / dname)

    with open(OUT_ROOT / "manifests" / "v16_dataset_line_summary.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(line_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(line_rows)

    policy = {
        "dataset": str(OUT_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "source_data_root": str(SOURCE_DATA_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "source_label_package": str(LABEL_PACKAGE.relative_to(ROOT.parent)).replace("\\", "/"),
        "v16_lines": V16_LINES,
        "line6_policy": {
            "context_ignore": "Line6 context-ignore traces are retained in soft_mask_review_v16 but zeroed in soft_mask_train and marked in ignore_mask.",
            "ultraweak": "line6_front_ultraweak_train_flag_v16 is weak_evidence AND NOT context_ignore.",
        },
        "training_notes": [
            "Use data_root=data_audited_v16_20260627.",
            "train_raw_only.py now reads optional ignore_mask and removes it from pixel, hard-negative, and centerline losses.",
            "Do not treat hard_mask_review_v16 as strong ground truth.",
        ],
    }
    (OUT_ROOT / "manifests" / "v16_training_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "DATASET_README.md").write_text(
        "# PGDA-CSNet Audited V16 Dataset\n\n"
        "This data root applies `PGDA_L3_L1_L6_L7_AUDITED_LABELS_V16_20260627` to the existing measured-data root.\n\n"
        "- `soft_mask_train` is the training mask.\n"
        "- `soft_mask_review_v16` keeps the source weighted review mask where available.\n"
        "- `review_band_mask_v16` keeps an unweighted audited centerline band for visual review, including Line6 context-ignore traces.\n"
        "- `ignore_mask` marks pixels excluded from positive/negative/centerline loss.\n"
        "- Line6 ultraweak traces use weak status for presence supervision.\n"
        "- Line6 context-ignore traces are not counted in window `present/weak/no_pick` oversampling statistics.\n\n"
        "Use this dataset as weak/weighted supervision, not clean strong ground truth.\n",
        encoding="utf-8",
    )
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
