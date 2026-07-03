from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_audited_v16_20260627"
OUT = ROOT / "reports" / "line9_v17_label_package"


def centerline(mask):
    h, _ = mask.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    s = mask.sum(axis=0)
    c = (mask * yy).sum(axis=0) / np.maximum(s, 1e-6)
    valid = s > 1e-3
    c[~valid] = np.nan
    return c, valid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA / "lines" / "Line9.npz", allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    soft = arrays["soft_mask_train"].astype(np.float32)
    label_weight = arrays["label_weight"].astype(np.float32).copy()
    status = arrays["status_code"].astype(np.int16).copy()
    center, valid = centerline(soft)
    dt = float(arrays["dt_ns"])
    center_ns = center * dt
    depth_m = center_ns * 0.074 * 0.5

    # Audit decision:
    # Keep Line9 centerline geometry from v1.4 because the reviewed high-risk
    # segments stay in the PDF-constrained 12-16 m interface band and follow
    # the visible continuous reflector. Downgrade uncertain tail/transition
    # traces to weak supervision instead of redrawing them from model output.
    decision = np.array(["keep_confirmed_v17"] * soft.shape[1], dtype=object)
    confidence = np.full(soft.shape[1], 0.82, dtype=np.float32)
    review_priority = np.array(["baseline"] * soft.shape[1], dtype=object)

    for a, b, priority in [
        (1280, 1535, "medium"),
        (1536, 1791, "high"),
        (1792, 2047, "high"),
        (2048, 2303, "high"),
        (2304, 2377, "high"),
    ]:
        review_priority[a : b + 1] = priority

    # Keep the interface, but mark low-confidence tail segments as weak labels.
    for a, b, conf in [
        (1536, 1791, 0.74),
        (1792, 2047, 0.72),
        (2048, 2303, 0.58),
        (2304, 2377, 0.52),
    ]:
        confidence[a : b + 1] = conf
        decision[a : b + 1] = "keep_centerline_downgrade_uncertainty_v17"
        status[a : b + 1] = 2
        label_weight[a : b + 1] = np.minimum(label_weight[a : b + 1], conf).astype(np.float32)

    # Preserve strong Line9 train segment where audit evidence and frozen model agree.
    status[:1536] = arrays["status_code"][:1536].astype(np.int16)
    label_weight[:1536] = arrays["label_weight"][:1536].astype(np.float32)

    out_npz = OUT / "Line9_audited_masks_v17.npz"
    np.savez_compressed(
        out_npz,
        soft_mask_train=soft.astype(np.float32),
        hard_mask_train=(soft > 0.25).astype(np.uint8),
        label_weight_v17=label_weight.astype(np.float32),
        status_code_v17=status.astype(np.int16),
        label_time_center_v17_ns=center_ns.astype(np.float32),
        label_depth_center_v17_m=depth_m.astype(np.float32),
        review_priority_v17=review_priority.astype("U16"),
        decision_v17=decision.astype("U48"),
        confidence_v17=confidence.astype(np.float32),
        source=np.array("Line9 v17 visual audit keeps v1.4 centerline; downgrades uncertain holdout tail weights"),
    )

    rows = []
    for i in range(soft.shape[1]):
        rows.append(
            {
                "line": "Line9",
                "trace_idx": i,
                "valid": int(bool(valid[i])),
                "center_time_ns": "" if not valid[i] else f"{float(center_ns[i]):.4f}",
                "center_depth_m": "" if not valid[i] else f"{float(depth_m[i]):.4f}",
                "status_code_v17": int(status[i]),
                "label_weight_v17": f"{float(label_weight[i]):.4f}",
                "review_priority_v17": str(review_priority[i]),
                "decision_v17": str(decision[i]),
                "confidence_v17": f"{float(confidence[i]):.4f}",
            }
        )
    out_csv = OUT / "Line9_audited_labels_v17.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for name, a, b in [
        ("train_front", 0, 1407),
        ("guard", 1408, 1663),
        ("holdout_a", 1664, 1791),
        ("holdout_b", 1792, 2047),
        ("holdout_c", 2048, 2303),
        ("holdout_tail", 2304, 2377),
    ]:
        sl = slice(a, b + 1)
        summary.append(
            {
                "segment": name,
                "trace_range": f"{a}-{b}",
                "median_depth_m": float(np.nanmedian(depth_m[sl])),
                "p10_depth_m": float(np.nanpercentile(depth_m[sl], 10)),
                "p90_depth_m": float(np.nanpercentile(depth_m[sl], 90)),
                "strong_count": int((status[sl] == 1).sum()),
                "weak_count": int((status[sl] == 2).sum()),
                "mean_weight": float(label_weight[sl].mean()),
                "decision": "keep old centerline; downgrade uncertainty where needed",
            }
        )
    out_summary = OUT / "Line9_v17_segment_decisions.csv"
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    manifest = {
        "line": "Line9",
        "version": "v17",
        "basis": [
            "PDF/profile constraint: basal/interface 12-16 m",
            "Reject 20-23 m deeper anomaly as basal",
            "Visual review of high-risk holdout segments",
            "Model predictions used only as disagreement signals, not ground truth",
        ],
        "decision": "Keep v1.4 Line9 centerline geometry; downgrade uncertain high-risk tail/holdout traces to weak labels.",
        "npz": str(out_npz.relative_to(ROOT)),
        "csv": str(out_csv.relative_to(ROOT)),
        "segment_decisions": str(out_summary.relative_to(ROOT)),
        "workbench": "reports/line9_v17_reaudit_workbench",
    }
    (OUT / "Line9_v17_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_npz)
    print(out_csv)
    print(out_summary)
    print(OUT / "Line9_v17_manifest.json")


if __name__ == "__main__":
    main()
