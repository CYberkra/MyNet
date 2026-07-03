"""
v1.11 Confidence Abstention — Line9 holdout-specific analysis.
Re-uses existing full-line evals and features, but filters target=Line9 to
holdout traces (1664-2377) for metric computation, avoiding train-trace inflation.
"""
import csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "outputs/v11_frozen_v19d_confidence/trace_confidence_features.csv"
OUT = ROOT / "outputs/v11_frozen_v19d_confidence_holdout"
OUT.mkdir(parents=True, exist_ok=True)

LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
LINE9_HOLDOUT = (1664, 2377)
SCORES = ["score_path", "score_presence", "score_p050_consistent", "score_composite", "score_crossview"]


def load_features(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["trace_idx"] = int(row["trace_idx"])
            row["gt_valid"] = int(row["gt_valid"]) == 1
            row["picked"] = int(row["picked"]) == 1
            row["error_ns"] = safe_float(row["error_ns"])
            for c in SCORES:
                row[c] = safe_float(row[c])
            rows.append(row)
    return rows


def safe_float(val, default=float("nan")):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def metrics_for(rows, accepted):
    gt = np.array([r["gt_valid"] for r in rows], dtype=bool)
    picked = np.array([r["picked"] for r in rows], dtype=bool)
    err = np.array([r["error_ns"] for r in rows], dtype=np.float32)
    ok = gt & picked & accepted & np.isfinite(err)
    n_gt = int(gt.sum())
    n_ok = int(ok.sum())
    mae = float(np.nanmean(err[ok])) if n_ok > 0 else float("nan")
    cov = n_ok / max(1, n_gt)
    sev10 = float((ok & (err > 10)).sum() / max(1, n_ok)) if n_ok > 0 else float("nan")
    sev20 = float((ok & (err > 20)).sum() / max(1, n_ok)) if n_ok > 0 else float("nan")
    return {"n_gt": n_gt, "n_accept": n_ok, "coverage": cov, "mae_ns": mae, "severe_gt10": sev10, "severe_gt20": sev20}


def score_arr(rows, col):
    return np.array([r[col] for r in rows], dtype=np.float32)


def pareto_by_threshold(rows, col, thresholds):
    scores = score_arr(rows, col)
    out = []
    for thr in thresholds:
        accepted = np.isfinite(scores) & (scores >= thr)
        m = metrics_for(rows, accepted)
        m["threshold"] = float(thr)
        out.append(m)
    return out


def pareto_by_keep_fraction(rows, col, fractions):
    scores = score_arr(rows, col)
    picked = np.array([r["picked"] for r in rows], dtype=bool)
    candidate = picked & np.isfinite(scores)
    out = []
    for frac in fractions:
        accepted = np.zeros(len(rows), dtype=bool)
        idx = np.flatnonzero(candidate)
        n_keep = int(math.ceil(idx.size * max(0.0, min(1.0, frac))))
        if n_keep > 0:
            order = idx[np.argsort(scores[idx])[::-1]]
            accepted[order[:n_keep]] = True
        m = metrics_for(rows, accepted)
        m["keep_fraction"] = float(frac)
        out.append(m)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    print("=== Loading features ===")
    all_rows = load_features(FEATURES_CSV)
    print(f"  Total traces: {len(all_rows)}")

    # Separate Line9 holdout vs rest
    source_rows = [r for r in all_rows if r["line"] != "Line9"]
    line9_all = [r for r in all_rows if r["line"] == "Line9"]
    line9_holdout = [r for r in line9_all if LINE9_HOLDOUT[0] <= r["trace_idx"] <= LINE9_HOLDOUT[1]]
    line9_train = [r for r in line9_all if r["trace_idx"] < LINE9_HOLDOUT[0]]
    print(f"  Source (other lines): {len(source_rows)} traces")
    print(f"  Line9 total: {len(line9_all)} traces")
    print(f"  Line9 holdout ({LINE9_HOLDOUT[0]}-{LINE9_HOLDOUT[1]}): {len(line9_holdout)} traces")
    print(f"  Line9 train (0-{LINE9_HOLDOUT[0]-1}): {len(line9_train)} traces")

    # ── 1. Baseline metrics on holdout only ──────────────────────
    print("\n=== Baseline (no abstention) on Line9 holdout ===")
    baseline_holdout = metrics_for(line9_holdout, np.ones(len(line9_holdout), dtype=bool))
    for k, v in baseline_holdout.items():
        print(f"  {k}: {v}")

    # ── 2. Holdout LOO: Threshold-based ──────────────────────────
    print("\n=== Threshold-based LOO: source=other lines, target=Line9 holdout ===")
    thresholds = np.linspace(0.0, 1.0, 101)
    min_cov = 0.35
    lolo_rows = []
    for col in SCORES:
        # Find best threshold on source lines
        src_pareto = pareto_by_threshold(source_rows, col, thresholds)
        valid_src = [p for p in src_pareto if p["coverage"] >= min_cov and np.isfinite(p["mae_ns"])]
        if not valid_src:
            valid_src = [p for p in src_pareto if np.isfinite(p["mae_ns"])]
        best_src = min(valid_src, key=lambda p: (p["mae_ns"], -p["coverage"])) if valid_src else None
        if best_src:
            thr = best_src["threshold"]
            target_scores = score_arr(line9_holdout, col)
            target_accepted = np.isfinite(target_scores) & (target_scores >= thr)
            target_m = metrics_for(line9_holdout, target_accepted)
            lolo_rows.append({
                "score": col,
                "source_threshold": thr,
                "source_mae_ns": best_src["mae_ns"], "source_coverage": best_src["coverage"],
                "baseline_mae_ns": baseline_holdout["mae_ns"], "baseline_coverage": baseline_holdout["coverage"],
                "abstained_mae_ns": target_m["mae_ns"], "abstained_coverage": target_m["coverage"],
                "severe_gt10": target_m["severe_gt10"], "severe_gt20": target_m["severe_gt20"],
                "delta_mae": target_m["mae_ns"] - baseline_holdout["mae_ns"],
                "delta_cov": target_m["coverage"] - baseline_holdout["coverage"],
            })
            print(f"  {col:<28} thr={thr:.2f} | base MAE={baseline_holdout['mae_ns']:.3f} cov={baseline_holdout['coverage']:.3f} | abstained MAE={target_m['mae_ns']:.3f} cov={target_m['coverage']:.3f} | ΔMAE={target_m['mae_ns']-baseline_holdout['mae_ns']:+.3f} Δcov={target_m['coverage']-baseline_holdout['coverage']:+.3f}")

    # ── 3. Holdout LOO: Keep-fraction ────────────────────────────
    print("\n=== Keep-fraction LOO: source=other lines, target=Line9 holdout ===")
    fractions = np.linspace(0.2, 1.0, 81)
    quantile_rows = []
    for col in SCORES:
        scores_arr_src = score_arr(source_rows, col)
        picked_src = np.array([r["picked"] for r in source_rows], dtype=bool)
        candidate_src = picked_src & np.isfinite(scores_arr_src)

        best_frac_m = None
        best_frac = 0.5
        for frac in fractions:
            acc = np.zeros(len(source_rows), dtype=bool)
            idx = np.flatnonzero(candidate_src)
            nk = int(math.ceil(idx.size * frac))
            if nk > 0:
                order = idx[np.argsort(scores_arr_src[idx])[::-1]]
                acc[order[:nk]] = True
            fm = metrics_for(source_rows, acc)
            if np.isfinite(fm["mae_ns"]) and fm["coverage"] >= min_cov:
                if best_frac_m is None or (fm["mae_ns"] < best_frac_m["mae_ns"]):
                    best_frac_m = fm
                    best_frac = frac

        if best_frac_m:
            tgt_scores = score_arr(line9_holdout, col)
            tgt_picked = np.array([r["picked"] for r in line9_holdout], dtype=bool)
            tgt_cand = tgt_picked & np.isfinite(tgt_scores)
            tgt_acc = np.zeros(len(line9_holdout), dtype=bool)
            tidx = np.flatnonzero(tgt_cand)
            tnk = int(math.ceil(tidx.size * best_frac))
            if tnk > 0:
                torder = tidx[np.argsort(tgt_scores[tidx])[::-1]]
                tgt_acc[torder[:tnk]] = True
            tgt_m = metrics_for(line9_holdout, tgt_acc)
            quantile_rows.append({
                "score": col,
                "keep_fraction": best_frac,
                "source_mae_ns": best_frac_m["mae_ns"], "source_coverage": best_frac_m["coverage"],
                "baseline_mae_ns": baseline_holdout["mae_ns"], "baseline_coverage": baseline_holdout["coverage"],
                "abstained_mae_ns": tgt_m["mae_ns"], "abstained_coverage": tgt_m["coverage"],
                "severe_gt10": tgt_m["severe_gt10"], "severe_gt20": tgt_m["severe_gt20"],
                "delta_mae": tgt_m["mae_ns"] - baseline_holdout["mae_ns"],
                "delta_cov": tgt_m["coverage"] - baseline_holdout["coverage"],
            })
            print(f"  {col:<28} frac={best_frac:.2f} | base MAE={baseline_holdout['mae_ns']:.3f} cov={baseline_holdout['coverage']:.3f} | abstained MAE={tgt_m['mae_ns']:.3f} cov={tgt_m['coverage']:.3f} | ΔMAE={tgt_m['mae_ns']-baseline_holdout['mae_ns']:+.3f} Δcov={tgt_m['coverage']-baseline_holdout['coverage']:+.3f}")

    # ── 4. Also compute: how much does LOO change when we correctly separate holdout? ──
    print("\n=== Comparison: Full-Line9 vs Holdout-Only target metrics ===")
    line9_all_picked_gt = [r for r in line9_all if r["gt_valid"] and r["picked"]]
    line9_holdout_picked_gt = [r for r in line9_holdout if r["gt_valid"] and r["picked"]]
    if line9_all_picked_gt:
        all_mae = np.nanmean([r["error_ns"] for r in line9_all_picked_gt])
        all_n = len(line9_all_picked_gt)
        print(f"  Full Line9 (ALL traces): MAE={all_mae:.3f} on {all_n} picked-gt traces")
    if line9_holdout_picked_gt:
        hold_mae = np.nanmean([r["error_ns"] for r in line9_holdout_picked_gt])
        hold_n = len(line9_holdout_picked_gt)
        print(f"  Line9 holdout ONLY:       MAE={hold_mae:.3f} on {hold_n} picked-gt traces")
    line9_train_picked_gt = [r for r in line9_train if r["gt_valid"] and r["picked"]]
    if line9_train_picked_gt:
        train_mae = np.nanmean([r["error_ns"] for r in line9_train_picked_gt])
        train_n = len(line9_train_picked_gt)
        print(f"  Line9 train ONLY:         MAE={train_mae:.3f} on {train_n} picked-gt traces")

    # ── 5. Write results ─────────────────────────────────────────
    write_csv(OUT / "holdout_threshold_abstention.csv", lolo_rows)
    write_csv(OUT / "holdout_keep_fraction_abstention.csv", quantile_rows)

    # Summary
    print("\n=== Holdout Abstention Summary ===")
    print(f"{'Score':<28} {'Base MAE':>10} {'Base Cov':>10} {'Abst MAE':>10} {'Abst Cov':>10} {'Δ MAE':>10} {'Δ Cov':>10}")
    for r in lolo_rows:
        print(f"{r['score']:<28} {r['baseline_mae_ns']:10.3f} {r['baseline_coverage']:10.3f} {r['abstained_mae_ns']:10.3f} {r['abstained_coverage']:10.3f} {r['delta_mae']:+10.3f} {r['delta_cov']:+10.3f}")

    print(f"\nOutput: {OUT}")
    print("HOLDOUT_ABSTENTION_DONE")


if __name__ == "__main__":
    main()
