from pathlib import Path
import csv
import json
import math
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
VALID_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
CENTER_WEIGHTS = [0.0, 0.5, 1.0]
SEEDS = [1901, 1902, 1903]


def read_metrics(path):
    if not path.exists():
        return {}
    return {row["metric"]: row["value"] for row in csv.DictReader(path.open(encoding="utf-8"))}


def fnum(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x


def add_rows(rows, arch, label):
    for weight in CENTER_WEIGHTS:
        suffix = f"w{int(weight * 100):03d}"
        eval_dir = ROOT / "outputs" / f"eval_paper_{arch}_final_{label}_{suffix}_p065"
        if not eval_dir.exists():
            continue
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
                "center_fusion_weight": weight,
                "curve_source": hm.get("curve_source", ""),
                "valid_avg_mae_ns": statistics.mean(line_maes) if line_maes else math.nan,
                "valid_avg_pick_rate": statistics.mean(line_picks) if line_picks else math.nan,
                "line9_holdout_mae_ns": fnum(hm.get("dp_center_mae_ns")),
                "line9_holdout_pick_rate": fnum(hm.get("final_pick_rate")),
                "status": "finalist_eval_x1_excluded_from_ranking",
            }
        )


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "v1_9d_mambavision_hybrid"
    rows = []
    for seed in SEEDS:
        add_rows(rows, arch, f"seed{seed}")
    add_rows(rows, arch, "ensemble3seed")
    rows.sort(key=lambda r: (not math.isfinite(r["line9_holdout_mae_ns"]), r["line9_holdout_mae_ns"], r["valid_avg_mae_ns"]))
    out_csv = ROOT / "reports" / f"paper_{arch}_finalist_eval_summary.csv"
    out_json = ROOT / "reports" / f"paper_{arch}_finalist_eval_summary.json"
    if not rows:
        raise SystemExit(f"No finalist metrics found for {arch}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
