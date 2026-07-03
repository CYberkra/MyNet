"""
v1.11 Phase 2: Pareto sweep across min_coverage values + visualization.
Takes the feature CSV from Phase 1 and explores multiple min_coverage targets.
"""
from pathlib import Path
import csv, math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "v11_frozen_v19d_confidence"
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]


def safe_float(val, default=float("nan")):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def load_features():
    rows = []
    with (OUT / "trace_confidence_features.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in row:
                if k not in ("line",):
                    row[k] = safe_float(row[k]) if k != "line" else row[k]
            rows.append(row)
    return rows


def metrics(rows, accepted):
    gt = np.array([r["gt_valid"] == 1 for r in rows], dtype=bool)
    picked = np.array([r["picked"] == 1 for r in rows], dtype=bool)
    err = np.array([r["error_ns"] for r in rows], dtype=np.float32)
    ok = gt & picked & accepted & np.isfinite(err)
    n_gt = int(gt.sum())
    n_ok = int(ok.sum())
    mae = float(np.nanmean(err[ok])) if n_ok > 0 else float("nan")
    cov = n_ok / max(1, n_gt)
    severe = float((ok & (err > 10)).sum() / max(1, n_ok)) if n_ok > 0 else float("nan")
    return {"mae_ns": mae, "coverage": cov, "severe_gt10": severe, "n_accept": n_ok, "n_gt": n_gt}


def lolo_for_score(rows, col, min_cov):
    """Run leave-one-line-out for one score and min_coverage."""
    thresholds = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    results = []
    for target in LINES:
        source = [r for r in rows if r["line"] != target]
        target_rows = [r for r in rows if r["line"] == target]

        # Find best threshold on source lines
        best_thr, best_m = None, None
        for thr in thresholds:
            scores = np.array([r[col] for r in source], dtype=np.float32)
            accepted = np.isfinite(scores) & (scores >= thr)
            m = metrics(source, accepted)
            if m["coverage"] >= min_cov and np.isfinite(m["mae_ns"]):
                if best_m is None or m["mae_ns"] < best_m["mae_ns"]:
                    best_m = m
                    best_thr = thr

        if best_thr is None:
            # Relax: find best MAE regardless of coverage
            for thr in thresholds:
                scores = np.array([r[col] for r in source], dtype=np.float32)
                accepted = np.isfinite(scores) & (scores >= thr)
                m = metrics(source, accepted)
                if np.isfinite(m["mae_ns"]):
                    if best_m is None or m["mae_ns"] < best_m["mae_ns"]:
                        best_m = m
                        best_thr = thr

        if best_thr is not None:
            tgt_scores = np.array([r[col] for r in target_rows], dtype=np.float32)
            tgt_accepted = np.isfinite(tgt_scores) & (tgt_scores >= best_thr)
            tgt_m = metrics(target_rows, tgt_accepted)
            base_m = metrics(target_rows, np.ones(len(target_rows), dtype=bool))
        else:
            tgt_m = {"mae_ns": float("nan"), "coverage": float("nan"), "severe_gt10": float("nan")}
            base_m = {"mae_ns": float("nan"), "coverage": float("nan"), "severe_gt10": float("nan")}

        results.append({
            "target_line": target,
            "source_threshold": best_thr,
            "source_mae_ns": best_m["mae_ns"] if best_m else float("nan"),
            "source_coverage": best_m["coverage"] if best_m else float("nan"),
            "baseline_mae_ns": base_m["mae_ns"],
            "baseline_coverage": base_m["coverage"],
            "abstained_mae_ns": tgt_m["mae_ns"],
            "abstained_coverage": tgt_m["coverage"],
            "severe_gt10": tgt_m["severe_gt10"],
        })
    return results


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    print("Loading features...")
    rows = load_features()
    print(f"  {len(rows)} traces, lines: {sorted(set(r['line'] for r in rows))}")

    scores = ["score_composite", "score_crossview", "score_path", "score_presence"]
    min_covs = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    all_results = []
    pareto_points = []  # for visualization

    for col in scores:
        print(f"\n=== {col} ===")
        for mc in min_covs:
            lolo = lolo_for_score(rows, col, mc)
            avg_mae = np.mean([r["abstained_mae_ns"] for r in lolo if np.isfinite(r["abstained_mae_ns"])])
            avg_cov = np.mean([r["abstained_coverage"] for r in lolo if np.isfinite(r["abstained_coverage"])])
            avg_base_mae = np.mean([r["baseline_mae_ns"] for r in lolo if np.isfinite(r["baseline_mae_ns"])])
            avg_base_cov = np.mean([r["baseline_coverage"] for r in lolo if np.isfinite(r["baseline_coverage"])])
            all_results.append({
                "score": col, "min_source_coverage": mc,
                "avg_baseline_mae": avg_base_mae, "avg_baseline_cov": avg_base_cov,
                "avg_abstained_mae": avg_mae, "avg_abstained_cov": avg_cov,
                "delta_mae": avg_mae - avg_base_mae,
                "delta_cov": avg_cov - avg_base_cov,
            })
            pareto_points.append({
                "score": col, "min_cov": mc, "mae": avg_mae, "cov": avg_cov,
            })
            print(f"  min_cov={mc:.2f}: MAE={avg_mae:.3f} (Δ{avg_mae-avg_base_mae:+.3f}), Cov={avg_cov:.3f}")

    write_csv(OUT / "pareto_sweep_results.csv", all_results)

    # ── Best point per score ──
    print("\n=== BEST POINTS (max MAE reduction at coverage >= 0.50) ===")
    best_by_score = []
    for col in scores:
        candidates = [r for r in all_results if r["score"] == col and r["avg_abstained_cov"] >= 0.50 and np.isfinite(r["avg_abstained_mae"])]
        if not candidates:
            candidates = [r for r in all_results if r["score"] == col and np.isfinite(r["avg_abstained_mae"])]
        best = min(candidates, key=lambda r: r["avg_abstained_mae"])
        best_by_score.append(best)
        print(f"  {col}: MAE={best['avg_abstained_mae']:.3f}, Cov={best['avg_abstained_cov']:.3f}, min_cov={best['min_source_coverage']:.2f}")

    # Also find the Pareto-optimal point overall (minimize both MAE and maximize coverage)
    # Use weighted score: MAE * (1 - 0.5 * coverage)
    print("\n=== PARETO-OPTIMAL (weighted: mae * (1.5 - coverage)) ===")
    for col in scores:
        candidates = [r for r in all_results if r["score"] == col and np.isfinite(r["avg_abstained_mae"])]
        for c in candidates:
            c["_weighted"] = c["avg_abstained_mae"] * (1.5 - c["avg_abstained_cov"])
        best = min(candidates, key=lambda r: r["_weighted"])
        print(f"  {col}: MAE={best['avg_abstained_mae']:.3f}, Cov={best['avg_abstained_cov']:.3f}, min_cov={best['min_source_coverage']:.2f}")

    # ── Visualization ──
    print("\n=== Generating Pareto visualization ===")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=140)
        colors = {"score_composite": "#e74c3c", "score_crossview": "#3498db", "score_path": "#2ecc71", "score_presence": "#f39c12"}
        labels = {"score_composite": "Composite", "score_crossview": "Cross-view", "score_path": "Path Prob", "score_presence": "Presence Prob"}

        for col in scores:
            pts = [p for p in pareto_points if p["score"] == col]
            xs = [p["cov"] for p in pts]
            ys = [p["mae"] for p in pts]
            ax.plot(xs, ys, "o-", color=colors.get(col, "gray"), label=labels.get(col, col), markersize=6, linewidth=2)

        # Baseline reference: mean MAE over all picked & gt traces
        base_errs = [r["error_ns"] for r in rows if np.isfinite(r["error_ns"]) and r["picked"] == 1 and r["gt_valid"] == 1]
        base_mae = float(np.mean(base_errs)) if base_errs else 0.0
        ax.axhline(y=base_mae, color="gray", linestyle="--", alpha=0.5, label=f"Baseline MAE={base_mae:.3f}")

        ax.set_xlabel("Coverage (fraction of GT traces picked)", fontsize=12)
        ax.set_ylabel("MAE (ns)", fontsize=12)
        ax.set_title("v1.11 Frozen v1.9D — Leave-One-Line-Out MAE–Coverage Pareto", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # higher coverage on left

        fig.tight_layout()
        fig.savefig(OUT / "pareto_sweep_lolo.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUT / 'pareto_sweep_lolo.png'}")
    except Exception as e:
        print(f"  Visualization failed: {e}")

    print("\nV11_PARETO_SWEEP_DONE")


if __name__ == "__main__":
    main()
