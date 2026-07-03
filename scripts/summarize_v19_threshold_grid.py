from pathlib import Path
import csv
import json
import math
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
VALID_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
PATH_THRESHOLDS = [0.20, 0.40, 0.50, 0.65]


def read_metrics(path):
    if not path.exists():
        return {}
    return {row["metric"]: row["value"] for row in csv.DictReader(path.open(encoding="utf-8"))}


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def add_row(rows, arch, label, pthr):
    ptag = f"p{int(pthr * 100):03d}"
    eval_dir = ROOT / "outputs" / f"eval_paper_{arch}_final_{label}_w050_{ptag}"
    if not eval_dir.exists():
        return
    line_maes = []
    line_picks = []
    for line in VALID_LINES:
        m = read_metrics(eval_dir / f"{line}_full_metrics.csv")
        mae = fnum(m.get("dp_center_mae_ns"))
        pick = fnum(m.get("final_pick_rate"))
        if math.isfinite(mae):
            line_maes.append(mae)
        if math.isfinite(pick):
            line_picks.append(pick)
    hm = read_metrics(eval_dir / "Line9_holdout_tr1664_2377_full_metrics.csv")
    rows.append(
        {
            "model_arch": arch,
            "eval_label": label,
            "center_fusion_weight": 0.5,
            "path_prob_threshold": pthr,
            "valid_avg_mae_ns": statistics.mean(line_maes) if line_maes else math.nan,
            "valid_avg_pick_rate": statistics.mean(line_picks) if line_picks else math.nan,
            "line9_holdout_mae_ns": fnum(hm.get("dp_center_mae_ns")),
            "line9_holdout_pick_rate": fnum(hm.get("final_pick_rate")),
            "curve_source": hm.get("curve_source", ""),
        }
    )


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "v1_9d_mambavision_hybrid"
    rows = []
    for label in ("seed1902", "ensemble3seed"):
        for pthr in PATH_THRESHOLDS:
            add_row(rows, arch, label, pthr)
    rows.sort(key=lambda r: (r["eval_label"], r["path_prob_threshold"]))
    if not rows:
        raise SystemExit(f"No threshold-grid metrics found for {arch}")
    out_csv = ROOT / "reports" / f"paper_{arch}_threshold_grid_w050_summary.csv"
    out_json = ROOT / "reports" / f"paper_{arch}_threshold_grid_w050_summary.json"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
