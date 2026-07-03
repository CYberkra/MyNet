"""
v1.11 Confidence Abstention Pipeline — frozen v1.9D baseline.
Runs evals on all measured lines, extracts confidence features,
computes MAE-coverage Pareto and leave-one-line-out abstention.
"""
from pathlib import Path
import subprocess, sys, csv, math, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
RUN_DIR = "outputs/run_gpu_paper_v1_9d_mambavision_hybrid_final_seed1902_line9holdout"
DATA_ROOT = "data_corrected_v1_4_terrain_direction"
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
# Line9 holdout ranges
LINE9_TRAIN = (0, 1407)
LINE9_GUARD = (1408, 1663)
LINE9_HOLD = (1664, 2377)
# Default postprocess
DEFAULT_POSTPROC = dict(
    center_fusion_weight=0.5, presence_thr=0.45,
    dp_max_jump=6, dp_smooth_weight=0.16, dp_min_segment=16,
    dp_breakable=True,
)


def run_eval(line, path_prob_thr, robust_norm, out_dir_name):
    """Run eval_full_line.py for one line/threshold/view combination."""
    out = ROOT / "outputs" / out_dir_name
    if out.exists() and (out / f"{line}_pred_centerline.csv").exists():
        print(f"  SKIP {out_dir_name} (exists)")
        return out
    cmd = [
        PY, "scripts/eval_full_line.py",
        "--line", line,
        "--run-dirs", RUN_DIR,
        "--checkpoint", "final",
        "--data-root", DATA_ROOT,
        "--no-plot", "--force-cpu",
        "--path-prob-thr", str(path_prob_thr),
        "--out-dir", f"outputs/{out_dir_name}",
    ]
    for k, v in DEFAULT_POSTPROC.items():
        flag = f"--{k.replace('_', '-')}"
        cmd += [flag, str(v)] if not isinstance(v, bool) else ([flag] if v else [])
    if robust_norm:
        cmd += ["--override-cfg-json", json.dumps({"per_trace_robust_norm": True, "per_trace_robust_clip": 6.0})]
    print(f"  RUN {out_dir_name}")
    subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True, timeout=1200)
    return out


def eval_name(line, threshold_tag, robust=False):
    """Directory name matching v11 convention."""
    rob = "_robust" if robust else ""
    return f"eval_v11_frozen_{line}{rob}_p{threshold_tag}"


def safe_load_centerline(path):
    """Load pred_centerline.csv into a dict keyed by trace_idx."""
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["trace_idx"])] = row
    return rows


def safe_float(val, default=float("nan")):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def local_contrast(arr, y, x, radius=9):
    if not np.isfinite(y):
        return float("nan")
    yi = int(round(float(y)))
    if yi < 0 or yi >= arr.shape[0] or x < 0 or x >= arr.shape[1]:
        return float("nan")
    lo, hi = max(0, yi - radius), min(arr.shape[0], yi + radius + 1)
    band = arr[lo:hi, x]
    med = float(np.nanmedian(band))
    mad = float(np.nanmedian(np.abs(band - med))) + 1e-6
    return (float(arr[yi, x]) - med) / mad


def segment_lengths(valid):
    out = np.zeros(valid.shape[0], dtype=np.int32)
    i = 0
    while i < valid.shape[0]:
        if not valid[i]:
            i += 1; continue
        j = i + 1
        while j < valid.shape[0] and valid[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


def extract_features(line, p020_dir, p050_dir, rob_p020_dir, rob_p050_dir, data_root):
    """Extract per-trace confidence features for one line."""
    # Find files with glob (handles naming variations like Line9_holdout_tr1664_2377_* vs Line3_*)
    def find_file(directory, pattern):
        matches = list(directory.glob(pattern))
        return matches[0] if matches else None

    def load_centerline(d):
        f = find_file(d, f"*pred_centerline.csv")
        return safe_load_centerline(f) if f else {}

    p020 = load_centerline(p020_dir)
    p050 = load_centerline(p050_dir)
    rob020 = load_centerline(rob_p020_dir)
    rob050 = load_centerline(rob_p050_dir)

    pred_path = find_file(p020_dir, f"*pred_softmask.npy")
    path_path = find_file(p020_dir, f"*path_softmask.npy")
    center_path = find_file(p020_dir, f"*center_softmask.npy")

    pred = np.load(pred_path).astype(np.float32) if pred_path else None
    path_arr = np.load(path_path).astype(np.float32) if path_path else None
    center = np.load(center_path).astype(np.float32) if center_path else None

    data = np.load(data_root / "lines" / f"{line}.npz")
    dt_ns = float(data["dt_ns"])
    n_traces = int(data["raw_full_normalized"].shape[1])

    # DP sample positions
    dp_valid = np.zeros(n_traces, dtype=bool)
    dp_sample = np.full(n_traces, np.nan, dtype=np.float32)
    for idx, row in p020.items():
        if 0 <= idx < n_traces:
            dp_valid[idx] = int(row["dp_valid"]) == 1
            dp_sample[idx] = safe_float(row["dp_center_sample"])
    seg_len = segment_lengths(dp_valid)

    # Jump / curvature
    jump_ns = np.full(n_traces, np.nan, dtype=np.float32)
    curvature_ns = np.full(n_traces, np.nan, dtype=np.float32)
    for i in range(n_traces):
        if not dp_valid[i]:
            continue
        vals = []
        if i > 0 and dp_valid[i - 1]:
            vals.append(abs(dp_sample[i] - dp_sample[i - 1]) * dt_ns)
        if i + 1 < n_traces and dp_valid[i + 1]:
            vals.append(abs(dp_sample[i] - dp_sample[i + 1]) * dt_ns)
        if vals:
            jump_ns[i] = max(vals)
        if i > 0 and i + 1 < n_traces and dp_valid[i - 1] and dp_valid[i + 1]:
            curvature_ns[i] = abs(dp_sample[i - 1] - 2 * dp_sample[i] + dp_sample[i + 1]) * dt_ns

    rows = []
    for ti in range(n_traces):
        r020 = p020.get(ti, {})
        r050 = p050.get(ti, {})
        rr020 = rob020.get(ti, {})
        rr050 = rob050.get(ti, {})

        valid020 = int(r020.get("dp_valid", 0)) == 1
        valid050 = int(r050.get("dp_valid", 0)) == 1
        rob_valid020 = int(rr020.get("dp_valid", 0)) == 1
        rob_valid050 = int(rr050.get("dp_valid", 0)) == 1
        gt_valid = int(r020.get("gt_valid", 0)) == 1

        dp_s = safe_float(r020.get("dp_center_sample"))
        gt_s = safe_float(r020.get("gt_center_sample"))
        mean_s = safe_float(r020.get("mean_center_sample"))
        p050_s = safe_float(r050.get("dp_center_sample"))
        rob_s = safe_float(rr020.get("dp_center_sample"))

        err_ns = abs(dp_s - gt_s) * dt_ns if valid020 and gt_valid else float("nan")
        path_prob = safe_float(r020.get("dp_path_prob"))
        presence_prob = safe_float(r020.get("presence_prob"))
        rob_path_prob = safe_float(rr020.get("dp_path_prob", "nan"))
        rob_presence_prob = safe_float(rr020.get("presence_prob", "nan"))

        # Feature: mask/center/path values at DP pick location
        def safe_at(arr, y, x):
            if arr is None or not np.isfinite(y):
                return float("nan")
            yi = int(round(float(y)))
            if yi < 0 or yi >= arr.shape[0] or x < 0 or x >= arr.shape[1]:
                return float("nan")
            return float(arr[yi, x])

        mask_val = safe_at(pred, dp_s, ti)
        center_val = safe_at(center, dp_s, ti)
        path_val = safe_at(path_arr, dp_s, ti)
        agreement = 1.0 - min(1.0, abs(mask_val - center_val)) if np.isfinite(mask_val) and np.isfinite(center_val) else float("nan")

        # Feature: p050 consistency
        p050_agree_ns = abs(p050_s - dp_s) * dt_ns if valid020 and valid050 and np.isfinite(p050_s) else float("nan")
        p050_score = 1.0 if valid050 and np.isfinite(p050_agree_ns) and p050_agree_ns <= 8.0 else 0.0

        # Feature: robust cross-view
        rob_agree_ns = abs(rob_s - dp_s) * dt_ns if valid020 and rob_valid020 and np.isfinite(rob_s) else float("nan")
        rob_agree_score = math.exp(-rob_agree_ns / 10.0) if np.isfinite(rob_agree_ns) else 0.0

        # Feature: local contrast
        contrast = local_contrast(path_arr if path_arr is not None else pred, dp_s, ti)

        # Feature: jump, curvature, segment
        j_ns = float(jump_ns[ti]) if np.isfinite(jump_ns[ti]) else float("nan")
        c_ns = float(curvature_ns[ti]) if np.isfinite(curvature_ns[ti]) else float("nan")
        s_len = int(seg_len[ti])

        # Derived scores
        smooth_score = math.exp(-j_ns / 8.0) if np.isfinite(j_ns) else 0.5
        contrast_score = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, contrast)))) if np.isfinite(contrast) else 0.5

        composite = (
            max(0.0, min(1.0, path_prob))
            * max(0.0, min(1.0, presence_prob))
            * max(0.0, min(1.0, agreement if np.isfinite(agreement) else 0.5))
            * (0.75 + 0.25 * p050_score)
            * (0.65 + 0.35 * smooth_score)
            * (0.65 + 0.35 * contrast_score)
        )
        rob_prob_score = max(0.0, min(1.0, rob_path_prob)) if np.isfinite(rob_path_prob) else 0.0
        crossview = composite * (0.40 + 0.35 * rob_agree_score + 0.15 * (1.0 if rob_valid020 else 0.0) + 0.10 * (1.0 if rob_valid050 else 0.0)) * (0.75 + 0.25 * rob_prob_score)

        rows.append({
            "line": line, "trace_idx": ti,
            "gt_valid": int(gt_valid), "picked": int(valid020),
            "error_ns": err_ns,
            "path_prob": path_prob, "presence_prob": presence_prob,
            "robust_path_prob": rob_path_prob, "robust_presence_prob": rob_presence_prob,
            "mask_val": mask_val, "center_val": center_val, "path_val": path_val,
            "mask_center_agreement": agreement,
            "local_contrast": contrast,
            "segment_len": s_len, "jump_ns": j_ns, "curvature_ns": c_ns,
            "score_path": path_prob,
            "score_presence": presence_prob,
            "score_p050_consistent": p050_score,
            "score_composite": composite,
            "score_crossview": crossview,
        })
    return rows


def metrics_for(rows, accepted):
    gt = np.array([r["gt_valid"] == 1 for r in rows], dtype=bool)
    picked = np.array([r["picked"] == 1 for r in rows], dtype=bool)
    err = np.array([safe_float(r["error_ns"]) for r in rows], dtype=np.float32)
    ok = gt & picked & accepted & np.isfinite(err)
    n_gt = int(gt.sum())
    n_ok = int(ok.sum())
    mae = float(np.nanmean(err[ok])) if n_ok > 0 else float("nan")
    cov = n_ok / max(1, n_gt)
    sev10 = float((ok & (err > 10)).sum() / max(1, n_ok)) if n_ok > 0 else float("nan")
    sev20 = float((ok & (err > 20)).sum() / max(1, n_ok)) if n_ok > 0 else float("nan")
    return {"n_gt": n_gt, "n_accept": n_ok, "coverage": cov, "mae_ns": mae, "severe_gt10": sev10, "severe_gt20": sev20}


def score_arr(rows, col):
    return np.array([safe_float(r[col]) for r in rows], dtype=np.float32)


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
    picked = np.array([r["picked"] == 1 for r in rows], dtype=bool)
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
    data_root = ROOT / DATA_ROOT
    out_base = ROOT / "outputs" / "v11_frozen_v19d_confidence"
    out_base.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Run all evals ──────────────────────────────────────
    print("=== Phase 1: Running evals ===")
    dirs = {}  # (line, threshold, robust) -> out_dir_path
    for line in LINES:
        for pthr in [0.20, 0.50]:
            tag = f"p{int(pthr*100):03d}"
            for robust in [False, True]:
                name = eval_name(line, tag, robust)
                out_dir = run_eval(line, pthr, robust, name)
                dirs[(line, tag, robust)] = out_dir

    # ── Phase 2: Extract features ──────────────────────────────────
    print("\n=== Phase 2: Extracting features ===")
    all_rows = []
    for line in LINES:
        print(f"  {line}...")
        rows = extract_features(
            line,
            p020_dir=dirs[(line, "p020", False)],
            p050_dir=dirs[(line, "p050", False)],
            rob_p020_dir=dirs[(line, "p020", True)],
            rob_p050_dir=dirs[(line, "p050", True)],
            data_root=data_root,
        )
        all_rows.extend(rows)
    write_csv(out_base / "trace_confidence_features.csv", all_rows)
    print(f"  Total traces: {len(all_rows)}")

    # ── Phase 3: Threshold Pareto ──────────────────────────────────
    print("\n=== Phase 3: Threshold Pareto ===")
    scores = ["score_path", "score_presence", "score_p050_consistent", "score_composite", "score_crossview"]
    thresholds = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    pareto_rows = []
    for col in scores:
        for line_tag in ["ALL"] + LINES:
            subset = all_rows if line_tag == "ALL" else [r for r in all_rows if r["line"] == line_tag]
            for m in pareto_by_threshold(subset, col, thresholds):
                m["score"] = col
                m["line"] = line_tag
                pareto_rows.append(m)
    write_csv(out_base / "score_threshold_pareto.csv", pareto_rows)
    print(f"  Pareto entries: {len(pareto_rows)}")

    # ── Phase 4: Leave-One-Line-Out Abstention ─────────────────────
    print("\n=== Phase 4: Leave-One-Line-Out Abstention ===")
    lolo_rows = []
    quantile_rows = []
    min_cov = 0.35
    for col in scores:
        for target in LINES:
            source = [r for r in all_rows if r["line"] != target]
            target_rows = [r for r in all_rows if r["line"] == target]

            # Threshold-based: find best threshold on source lines
            src_pareto = pareto_by_threshold(source, col, thresholds)
            valid_src = [p for p in src_pareto if p["coverage"] >= min_cov and np.isfinite(p["mae_ns"])]
            if not valid_src:
                valid_src = [p for p in src_pareto if np.isfinite(p["mae_ns"])]
            best_src = min(valid_src, key=lambda p: (p["mae_ns"], -p["coverage"])) if valid_src else None
            if best_src:
                thr = best_src["threshold"]
                target_scores = score_arr(target_rows, col)
                target_accepted = np.isfinite(target_scores) & (target_scores >= thr)
                target_m = metrics_for(target_rows, target_accepted)
                baseline_m = metrics_for(target_rows, np.ones(len(target_rows), dtype=bool))
                lolo_rows.append({
                    "score": col, "target_line": target,
                    "source_threshold": thr,
                    "source_mae_ns": best_src["mae_ns"], "source_coverage": best_src["coverage"],
                    "baseline_mae_ns": baseline_m["mae_ns"], "baseline_coverage": baseline_m["coverage"],
                    "abstained_mae_ns": target_m["mae_ns"], "abstained_coverage": target_m["coverage"],
                    "severe_gt10": target_m["severe_gt10"], "severe_gt20": target_m["severe_gt20"],
                })

            # Keep-fraction: find best fraction on source lines
            scores_arr = score_arr(source, col)
            picked = np.array([r["picked"] == 1 for r in source], dtype=bool)
            candidate = picked & np.isfinite(scores_arr)
            fracs = np.linspace(0.2, 1.0, 81, dtype=np.float32)
            best_frac_m = None
            best_frac = 0.5
            for frac in fracs:
                acc = np.zeros(len(source), dtype=bool)
                idx = np.flatnonzero(candidate)
                nk = int(math.ceil(idx.size * frac))
                if nk > 0:
                    order = idx[np.argsort(scores_arr[idx])[::-1]]
                    acc[order[:nk]] = True
                fm = metrics_for(source, acc)
                if np.isfinite(fm["mae_ns"]) and fm["coverage"] >= min_cov:
                    if best_frac_m is None or (fm["mae_ns"] < best_frac_m["mae_ns"]):
                        best_frac_m = fm
                        best_frac = frac
            if best_frac_m:
                tgt_scores = score_arr(target_rows, col)
                tgt_picked = np.array([r["picked"] == 1 for r in target_rows], dtype=bool)
                tgt_cand = tgt_picked & np.isfinite(tgt_scores)
                tgt_acc = np.zeros(len(target_rows), dtype=bool)
                tidx = np.flatnonzero(tgt_cand)
                tnk = int(math.ceil(tidx.size * best_frac))
                if tnk > 0:
                    torder = tidx[np.argsort(tgt_scores[tidx])[::-1]]
                    tgt_acc[torder[:tnk]] = True
                tgt_m = metrics_for(target_rows, tgt_acc)
                baseline_m2 = metrics_for(target_rows, np.ones(len(target_rows), dtype=bool))
                quantile_rows.append({
                    "score": col, "target_line": target,
                    "keep_fraction": best_frac,
                    "source_mae_ns": best_frac_m["mae_ns"], "source_coverage": best_frac_m["coverage"],
                    "baseline_mae_ns": baseline_m2["mae_ns"], "baseline_coverage": baseline_m2["coverage"],
                    "abstained_mae_ns": tgt_m["mae_ns"], "abstained_coverage": tgt_m["coverage"],
                    "severe_gt10": tgt_m["severe_gt10"], "severe_gt20": tgt_m["severe_gt20"],
                })

    write_csv(out_base / "source_tuned_leave_one_line_abstention.csv", lolo_rows)
    write_csv(out_base / "source_tuned_keep_fraction_abstention.csv", quantile_rows)

    # ── Phase 5: Summary ──────────────────────────────────────────
    print("\n=== Phase 5: Summary ===")
    summary_rows = []
    for col in scores:
        sub = [r for r in lolo_rows if r["score"] == col]
        if not sub:
            continue
        base_mae = np.mean([safe_float(r["baseline_mae_ns"]) for r in sub])
        base_cov = np.mean([safe_float(r["baseline_coverage"]) for r in sub])
        abst_mae = np.mean([safe_float(r["abstained_mae_ns"]) for r in sub if np.isfinite(safe_float(r["abstained_mae_ns"]))])
        abst_cov = np.mean([safe_float(r["abstained_coverage"]) for r in sub if np.isfinite(safe_float(r["abstained_coverage"]))])
        summary_rows.append({
            "score": col,
            "baseline_mae_ns": base_mae, "baseline_coverage": base_cov,
            "abstained_mae_ns": abst_mae, "abstained_coverage": abst_cov,
            "delta_mae_ns": abst_mae - base_mae, "delta_coverage": abst_cov - base_cov,
        })
    write_csv(out_base / "lolo_summary.csv", summary_rows)

    # Print results
    print("\n=== RESULTS: Leave-One-Line-Out Summary ===")
    print(f"{'Score':<28} {'Base MAE':>10} {'Base Cov':>10} {'Abst MAE':>10} {'Abst Cov':>10} {'Δ MAE':>10} {'Δ Cov':>10}")
    for r in summary_rows:
        print(f"{r['score']:<28} {r['baseline_mae_ns']:10.3f} {r['baseline_coverage']:10.3f} {r['abstained_mae_ns']:10.3f} {r['abstained_coverage']:10.3f} {r['delta_mae_ns']:+10.3f} {r['delta_coverage']:+10.3f}")

    print(f"\nOutput dir: {out_base}")
    print("V11_CONFIDENCE_ABSTENTION_DONE")


if __name__ == "__main__":
    main()
