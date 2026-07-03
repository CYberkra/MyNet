from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data_corrected_v1_4_terrain_direction"
NEW = ROOT / "data_audited_v16_20260627"
OUT = ROOT / "reports"


def centerline(mask):
    h, w = mask.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    s = mask.sum(axis=0)
    c = (mask * yy).sum(axis=0) / np.maximum(s, 1e-6)
    valid = s > 1e-3
    c[~valid] = np.nan
    return c, valid


def summarize_line(line):
    old = np.load(OLD / "lines" / f"{line}.npz", allow_pickle=False)
    new = np.load(NEW / "lines" / f"{line}.npz", allow_pickle=False)
    dt = float(new["dt_ns"])
    old_mask = old["soft_mask_train"].astype(np.float32)
    new_mask = new["soft_mask_train"].astype(np.float32)
    old_c, old_v = centerline(old_mask)
    new_c, new_v = centerline(new_mask)
    common = old_v & new_v
    status = new["status_code"].astype(np.int16)
    weight = new["label_weight"].astype(np.float32)
    ignore = new["ignore_mask"].max(axis=0).astype(bool) if "ignore_mask" in new.files else np.zeros_like(weight, dtype=bool)
    active = ~ignore
    row = {
        "line": line,
        "trace_count": int(new_mask.shape[1]),
        "old_valid": int(old_v.sum()),
        "new_valid": int(new_v.sum()),
        "active_traces": int(active.sum()),
        "ignored_traces": int(ignore.sum()),
        "new_status_0": int((status == 0).sum()),
        "new_status_1": int((status == 1).sum()),
        "new_status_2": int((status == 2).sum()),
        "new_weight_sum_active": float(weight[active].sum()),
        "new_weight_mean_active": float(weight[active].mean()) if active.any() else float("nan"),
        "new_center_ns_median_active": float(np.nanmedian(new_c[active] * dt)) if np.isfinite(new_c[active]).any() else float("nan"),
        "new_center_ns_p10_active": float(np.nanpercentile(new_c[active] * dt, 10)) if np.isfinite(new_c[active]).any() else float("nan"),
        "new_center_ns_p90_active": float(np.nanpercentile(new_c[active] * dt, 90)) if np.isfinite(new_c[active]).any() else float("nan"),
        "old_new_common_traces": int(common.sum()),
        "old_new_center_delta_ns_median": float(np.nanmedian((new_c[common] - old_c[common]) * dt)) if common.any() else float("nan"),
        "old_new_center_abs_delta_ns_median": float(np.nanmedian(np.abs(new_c[common] - old_c[common]) * dt)) if common.any() else float("nan"),
        "old_new_center_abs_delta_ns_p90": float(np.nanpercentile(np.abs(new_c[common] - old_c[common]) * dt, 90)) if common.any() else float("nan"),
        "old_new_soft_mask_l1_mean": float(np.mean(np.abs(new_mask - old_mask))),
    }
    return row


def summarize_windows(data_root):
    rows = list(csv.DictReader(open(data_root / "window_index.csv", encoding="utf-8")))
    by_line = {}
    for r in rows:
        line = r["line"]
        item = by_line.setdefault(line, {"line": line, "windows": 0, "present_sum": 0, "weak_sum": 0, "no_pick_sum": 0, "ignore_sum": 0})
        item["windows"] += 1
        item["present_sum"] += int(float(r.get("present", 0) or 0))
        item["weak_sum"] += int(float(r.get("weak", 0) or 0))
        item["no_pick_sum"] += int(float(r.get("no_pick", 0) or 0))
        item["ignore_sum"] += int(float(r.get("ignore", 0) or 0))
    return by_line


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["Line3", "Line6", "Line7", "Line9", "LineL1", "LineX1"]
    rows = [summarize_line(line) for line in lines]
    out_csv = OUT / "v16_vs_v14_label_shift_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    old_w = summarize_windows(OLD)
    new_w = summarize_windows(NEW)
    win_rows = []
    for line in lines:
        row = {"line": line}
        for prefix, src in [("old", old_w.get(line, {})), ("v16", new_w.get(line, {}))]:
            for key in ["windows", "present_sum", "weak_sum", "no_pick_sum", "ignore_sum"]:
                row[f"{prefix}_{key}"] = int(src.get(key, 0))
        win_rows.append(row)
    out_win = OUT / "v16_vs_v14_window_shift_summary.csv"
    with open(out_win, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
        w.writeheader()
        w.writerows(win_rows)

    out_json = OUT / "v16_vs_v14_label_shift_summary.json"
    out_json.write_text(json.dumps({"line_summary": rows, "window_summary": win_rows}, indent=2), encoding="utf-8")
    print(out_csv)
    print(out_win)
    print(out_json)


if __name__ == "__main__":
    main()
