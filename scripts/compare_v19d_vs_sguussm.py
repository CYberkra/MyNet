"""
v1.11 SG-USSM vs v1.9D comparison evaluator.
Runs frozen checkpoint eval on all measured lines for both models,
then compares metrics side by side.
"""
from pathlib import Path
import subprocess, sys, csv, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DATA_ROOT = "data_corrected_v1_4_terrain_direction"
LINES = ["Line3", "Line6", "Line7", "LineL1"]
LINE9_RANGE = (1664, 2377)


def run_eval(run_dir, line, out_name, extra_args=None):
    out = ROOT / "outputs" / out_name
    if out.exists() and list(out.glob("*_full_metrics.csv")):
        print(f"  SKIP {out_name}")
        return out
    cmd = [
        PY, "scripts/eval_full_line.py",
        "--line", line, "--run-dirs", run_dir,
        "--checkpoint", "final", "--data-root", DATA_ROOT,
        "--no-plot", "--force-cpu",
        "--out-dir", f"outputs/{out_name}",
    ]
    if extra_args:
        cmd += extra_args
    print(f"  RUN {out_name}")
    subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    return out


def read_metrics(csv_path):
    if not csv_path.exists():
        return {}
    result = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                result[row["metric"]] = float(row["value"])
            except (ValueError, KeyError):
                pass
    return result


def main():
    print("=" * 70)
    print("v1.11 SG-USSM vs v1.9D — Frozen Checkpoint Comparison")
    print("=" * 70)

    # Model configs: (label, run_dir)
    models = [
        ("v1.9D (baseline)", "outputs/run_gpu_paper_v1_9d_mambavision_hybrid_final_seed1902_line9holdout"),
    ]

    # Check if SG-USSM GPU training has been done
    sguussm_dir = ROOT / "outputs/run_gpu_v1_11_sguussm_seed1902_line9holdout"
    if sguussm_dir.exists() and any(sguussm_dir.glob("*/checkpoint_final.pt")):
        models.append(("SG-USSM v1.11", "outputs/run_gpu_v1_11_sguussm_seed1902_line9holdout"))
    else:
        print(f"\n[INFO] SG-USSM GPU checkpoint not found at {sguussm_dir}")
        print("       Comparison will use frozen v1.9D only.")
        print("       Run: python scripts/train_raw_only.py configs/gpu_train_v1_11_sguussm_seed1902_line9holdout.json")

    # Postprocess params
    pp = [
        "--center-fusion-weight", "0.5",
        "--presence-thr", "0.45",
        "--path-prob-thr", "0.50",
        "--dp-breakable",
        "--dp-max-jump", "6",
        "--dp-smooth-weight", "0.16",
        "--dp-min-segment", "16",
    ]

    all_results = []

    for model_label, run_dir in models:
        print(f"\n--- {model_label} ---")

        # Line9 full holdout
        tag = model_label.replace(" ", "_").replace("(", "").replace(")", "")
        out9 = run_eval(run_dir, "Line9", f"compare_{tag}_Line9_holdout", pp + [
            "--trace-start", str(LINE9_RANGE[0]),
            "--trace-end", str(LINE9_RANGE[1]),
        ])
        m9 = read_metrics(out9 / "Line9_holdout_tr1664_2377_full_metrics.csv")
        if m9:
            all_results.append({"model": model_label, "line": "Line9_holdout",
                                "mae_ns": m9.get("dp_center_mae_ns", float("nan")),
                                "pick_rate": m9.get("final_pick_rate", float("nan"))})

        # Other lines (full)
        for line in LINES:
            out = run_eval(run_dir, line, f"compare_{tag}_{line}", pp)
            m = read_metrics(out / f"{line}_full_metrics.csv")
            if m:
                all_results.append({"model": model_label, "line": line,
                                    "mae_ns": m.get("dp_center_mae_ns", float("nan")),
                                    "pick_rate": m.get("final_pick_rate", float("nan"))})

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    if not all_results:
        print("No results to compare.")
        return

    header = f"{'Model':25s} {'Line':15s} {'MAE (ns)':>10s} {'Pick Rate':>10s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        mae = r["mae_ns"]
        pr = r["pick_rate"]
        mae_s = f"{mae:.3f}" if np.isfinite(mae) else "N/A"
        pr_s = f"{pr:.3f}" if np.isfinite(pr) else "N/A"
        print(f"{r['model']:25s} {r['line']:15s} {mae_s:>10s} {pr_s:>10s}")

    # Save CSV
    out_csv = ROOT / "outputs" / "v1_11_model_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "line", "mae_ns", "pick_rate"])
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved: {out_csv}")
    print("COMPARISON_DONE")


if __name__ == "__main__":
    main()
