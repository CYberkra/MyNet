from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]


@dataclass
class EvalCase:
    line: str
    eval_dir_p020: Path
    eval_dir_p050: Path
    robust_dir_p020: Path
    robust_dir_p050: Path


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_centerline(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["trace_idx"])] = row
    return rows


def to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_at(arr: np.ndarray, y: float, x: int) -> float:
    if not np.isfinite(y):
        return float("nan")
    yi = int(round(float(y)))
    if yi < 0 or yi >= arr.shape[0] or x < 0 or x >= arr.shape[1]:
        return float("nan")
    return float(arr[yi, x])


def local_contrast(arr: np.ndarray, y: float, x: int, radius: int = 9) -> float:
    if not np.isfinite(y):
        return float("nan")
    yi = int(round(float(y)))
    if yi < 0 or yi >= arr.shape[0] or x < 0 or x >= arr.shape[1]:
        return float("nan")
    lo = max(0, yi - radius)
    hi = min(arr.shape[0], yi + radius + 1)
    center = float(arr[yi, x])
    band = arr[lo:hi, x]
    med = float(np.nanmedian(band))
    mad = float(np.nanmedian(np.abs(band - med))) + 1e-6
    return (center - med) / mad


def segment_lengths(valid: np.ndarray) -> np.ndarray:
    out = np.zeros(valid.shape[0], dtype=np.int32)
    i = 0
    while i < valid.shape[0]:
        if not valid[i]:
            i += 1
            continue
        j = i + 1
        while j < valid.shape[0] and valid[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


def add_neighbor_features(samples: np.ndarray, valid: np.ndarray, dt_ns: float) -> tuple[np.ndarray, np.ndarray]:
    n = samples.shape[0]
    jump = np.full(n, np.nan, dtype=np.float32)
    curvature = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if not valid[i]:
            continue
        vals = []
        if i > 0 and valid[i - 1]:
            vals.append(abs(samples[i] - samples[i - 1]) * dt_ns)
        if i + 1 < n and valid[i + 1]:
            vals.append(abs(samples[i] - samples[i + 1]) * dt_ns)
        if vals:
            jump[i] = float(max(vals))
        if i > 0 and i + 1 < n and valid[i - 1] and valid[i + 1]:
            curvature[i] = float(abs(samples[i - 1] - 2 * samples[i] + samples[i + 1]) * dt_ns)
    return jump, curvature


def build_cases() -> list[EvalCase]:
    cases = []
    for line in LINES:
        base = f"outputs/eval_paper_v1_10d_{line}_sourceval_v19d_baseline_last_breakable"
        robust = f"outputs/eval_paper_v1_11_{line}_sourceval_v19d_baseline_last_robust_breakable"
        cases.append(
            EvalCase(
                line=line,
                eval_dir_p020=ROOT / f"{base}_p020",
                eval_dir_p050=ROOT / f"{base}_p050",
                robust_dir_p020=ROOT / f"{robust}_p020",
                robust_dir_p050=ROOT / f"{robust}_p050",
            )
        )
    return cases


def load_trace_rows(case: EvalCase, data_root: Path) -> list[dict[str, float | int | str]]:
    line = case.line
    p020 = read_centerline(case.eval_dir_p020 / f"{line}_pred_centerline.csv")
    p050 = read_centerline(case.eval_dir_p050 / f"{line}_pred_centerline.csv")
    robust020 = read_centerline(case.robust_dir_p020 / f"{line}_pred_centerline.csv") if (case.robust_dir_p020 / f"{line}_pred_centerline.csv").exists() else {}
    robust050 = read_centerline(case.robust_dir_p050 / f"{line}_pred_centerline.csv") if (case.robust_dir_p050 / f"{line}_pred_centerline.csv").exists() else {}
    pred = np.load(case.eval_dir_p020 / f"{line}_pred_softmask.npy").astype(np.float32)
    path = np.load(case.eval_dir_p020 / f"{line}_path_softmask.npy").astype(np.float32)
    center = np.load(case.eval_dir_p020 / f"{line}_center_softmask.npy").astype(np.float32)
    data = np.load(data_root / "lines" / f"{line}.npz")
    dt_ns = float(data["dt_ns"])
    n = pred.shape[1]

    dp_valid = np.zeros(n, dtype=bool)
    dp_sample = np.full(n, np.nan, dtype=np.float32)
    for idx, row in p020.items():
        if 0 <= idx < n:
            dp_valid[idx] = int(row["dp_valid"]) == 1
            dp_sample[idx] = to_float(row["dp_center_sample"])
    seg_len = segment_lengths(dp_valid)
    jump_ns, curvature_ns = add_neighbor_features(dp_sample, dp_valid, dt_ns)

    rows: list[dict[str, float | int | str]] = []
    for trace_idx in range(n):
        row020 = p020[trace_idx]
        row050 = p050.get(trace_idx, {})
        rowr020 = robust020.get(trace_idx, {})
        rowr050 = robust050.get(trace_idx, {})
        valid020 = int(row020["dp_valid"]) == 1
        valid050 = int(row050.get("dp_valid", "0")) == 1
        robust_valid020 = int(rowr020.get("dp_valid", "0")) == 1
        robust_valid050 = int(rowr050.get("dp_valid", "0")) == 1
        gt_valid = int(row020["gt_valid"]) == 1
        dp_s = to_float(row020["dp_center_sample"])
        gt_s = to_float(row020["gt_center_sample"])
        mean_s = to_float(row020["mean_center_sample"])
        p050_s = to_float(row050.get("dp_center_sample", "nan"))
        robust_s = to_float(rowr020.get("dp_center_sample", "nan"))
        err_ns = abs(dp_s - gt_s) * dt_ns if valid020 and gt_valid else float("nan")
        mean_dp_disagree_ns = abs(mean_s - dp_s) * dt_ns if valid020 and np.isfinite(mean_s) else float("nan")
        p050_agree_ns = abs(p050_s - dp_s) * dt_ns if valid020 and valid050 else float("nan")
        robust_agree_ns = abs(robust_s - dp_s) * dt_ns if valid020 and robust_valid020 else float("nan")
        path_prob = to_float(row020["dp_path_prob"])
        presence_prob = to_float(row020["presence_prob"])
        robust_path_prob = to_float(rowr020.get("dp_path_prob", "nan"))
        robust_presence_prob = to_float(rowr020.get("presence_prob", "nan"))
        mask_val = safe_at(pred, dp_s, trace_idx)
        center_val = safe_at(center, dp_s, trace_idx)
        path_val = safe_at(path, dp_s, trace_idx)
        contrast = local_contrast(path, dp_s, trace_idx)
        agreement = 1.0 - min(1.0, abs(mask_val - center_val)) if np.isfinite(mask_val) and np.isfinite(center_val) else float("nan")
        smooth_score = math.exp(-float(jump_ns[trace_idx]) / 8.0) if np.isfinite(jump_ns[trace_idx]) else 0.5
        p050_score = 1.0 if valid050 and p050_agree_ns <= 8.0 else 0.0
        robust_agree_score = math.exp(-robust_agree_ns / 10.0) if np.isfinite(robust_agree_ns) else 0.0
        robust_pick_score = 1.0 if robust_valid020 else 0.0
        robust_p050_score = 1.0 if robust_valid050 else 0.0
        contrast_score = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, contrast)))) if np.isfinite(contrast) else 0.5
        composite = (
            max(0.0, min(1.0, path_prob))
            * max(0.0, min(1.0, presence_prob))
            * max(0.0, min(1.0, agreement if np.isfinite(agreement) else 0.5))
            * (0.75 + 0.25 * p050_score)
            * (0.65 + 0.35 * smooth_score)
            * (0.65 + 0.35 * contrast_score)
        )
        robust_prob_score = max(0.0, min(1.0, robust_path_prob)) if np.isfinite(robust_path_prob) else 0.0
        crossview = composite * (0.40 + 0.35 * robust_agree_score + 0.15 * robust_pick_score + 0.10 * robust_p050_score) * (0.75 + 0.25 * robust_prob_score)
        rows.append(
            {
                "line": line,
                "trace_idx": trace_idx,
                "gt_valid": int(gt_valid),
                "picked_p020": int(valid020),
                "picked_p050": int(valid050),
                "picked_robust_p020": int(robust_valid020),
                "picked_robust_p050": int(robust_valid050),
                "error_ns": err_ns,
                "path_prob": path_prob,
                "presence_prob": presence_prob,
                "robust_path_prob": robust_path_prob,
                "robust_presence_prob": robust_presence_prob,
                "mask_val": mask_val,
                "center_val": center_val,
                "path_val": path_val,
                "mask_center_agreement": agreement,
                "mean_dp_disagree_ns": mean_dp_disagree_ns,
                "p050_agree_ns": p050_agree_ns,
                "robust_agree_ns": robust_agree_ns,
                "local_contrast": contrast,
                "segment_len": int(seg_len[trace_idx]),
                "jump_ns": float(jump_ns[trace_idx]) if np.isfinite(jump_ns[trace_idx]) else float("nan"),
                "curvature_ns": float(curvature_ns[trace_idx]) if np.isfinite(curvature_ns[trace_idx]) else float("nan"),
                "score_path": path_prob,
                "score_presence": presence_prob,
                "score_p050_consistent": p050_score,
                "score_composite": composite,
                "score_crossview": crossview,
            }
        )
    return rows


def finite_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def metrics_for(rows: list[dict[str, float | int | str]], accepted: np.ndarray) -> dict[str, float]:
    gt = np.array([int(r["gt_valid"]) == 1 for r in rows], dtype=bool)
    picked = np.array([int(r["picked_p020"]) == 1 for r in rows], dtype=bool)
    err = np.array([finite_float(r["error_ns"]) for r in rows], dtype=np.float32)
    ok = gt & picked & accepted & np.isfinite(err)
    base_pickable = int(np.count_nonzero(gt))
    mae = float(np.nanmean(err[ok])) if ok.any() else float("nan")
    coverage = float(np.count_nonzero(ok) / max(1, base_pickable))
    severe10 = float(np.count_nonzero(ok & (err > 10.0)) / max(1, np.count_nonzero(ok))) if ok.any() else float("nan")
    severe20 = float(np.count_nonzero(ok & (err > 20.0)) / max(1, np.count_nonzero(ok))) if ok.any() else float("nan")
    severe40 = float(np.count_nonzero(ok & (err > 40.0)) / max(1, np.count_nonzero(ok))) if ok.any() else float("nan")
    return {
        "n_gt": float(base_pickable),
        "n_accept": float(np.count_nonzero(ok)),
        "coverage": coverage,
        "mae_ns": mae,
        "severe_gt10_rate": severe10,
        "severe_gt20_rate": severe20,
        "severe_gt40_rate": severe40,
    }


def accept_top_fraction(rows: list[dict[str, float | int | str]], score_col: str, keep_fraction: float) -> np.ndarray:
    scores = score_array(rows, score_col)
    picked = np.array([int(r["picked_p020"]) == 1 for r in rows], dtype=bool)
    candidate = picked & np.isfinite(scores)
    accepted = np.zeros(len(rows), dtype=bool)
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return accepted
    n_keep = int(math.ceil(idx.size * max(0.0, min(1.0, keep_fraction))))
    if n_keep <= 0:
        return accepted
    order = idx[np.argsort(scores[idx])[::-1]]
    accepted[order[:n_keep]] = True
    return accepted


def score_array(rows: list[dict[str, float | int | str]], score_col: str) -> np.ndarray:
    return np.array([finite_float(r[score_col]) for r in rows], dtype=np.float32)


def evaluate_thresholds(rows: list[dict[str, float | int | str]], score_col: str, thresholds: np.ndarray) -> list[dict[str, float | str]]:
    scores = score_array(rows, score_col)
    out = []
    for thr in thresholds:
        accepted = np.isfinite(scores) & (scores >= float(thr))
        m = metrics_for(rows, accepted)
        m["score"] = score_col
        m["threshold"] = float(thr)
        out.append(m)
    return out


def choose_source_threshold(source_rows: list[dict[str, float | int | str]], score_col: str, thresholds: np.ndarray, min_coverage: float) -> tuple[float, dict[str, float]]:
    candidates = evaluate_thresholds(source_rows, score_col, thresholds)
    valid = [c for c in candidates if c["coverage"] >= min_coverage and np.isfinite(c["mae_ns"])]
    if not valid:
        valid = [c for c in candidates if np.isfinite(c["mae_ns"])]
    best = min(valid, key=lambda c: (c["mae_ns"], -c["coverage"]))
    return float(best["threshold"]), best


def choose_source_keep_fraction(source_rows: list[dict[str, float | int | str]], score_col: str, keep_fracs: np.ndarray, min_coverage: float) -> tuple[float, dict[str, float]]:
    candidates = []
    for frac in keep_fracs:
        m = metrics_for(source_rows, accept_top_fraction(source_rows, score_col, float(frac)))
        m["keep_fraction"] = float(frac)
        candidates.append(m)
    valid = [c for c in candidates if c["coverage"] >= min_coverage and np.isfinite(c["mae_ns"])]
    if not valid:
        valid = [c for c in candidates if np.isfinite(c["mae_ns"])]
    best = min(valid, key=lambda c: (c["mae_ns"], -c["coverage"]))
    return float(best["keep_fraction"]), best


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data_corrected_v1_4_terrain_direction")
    ap.add_argument("--out-dir", default="reports/v1_11_confidence_control")
    ap.add_argument("--min-source-coverage", type=float, default=0.35)
    args = ap.parse_args()

    data_root = resolve(args.data_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | int | str]] = []
    for case in build_cases():
        all_rows.extend(load_trace_rows(case, data_root))

    feature_csv = out_dir / "trace_confidence_features_v19d_last_p020.csv"
    write_csv(feature_csv, all_rows)

    scores = ["score_path", "score_presence", "score_p050_consistent", "score_composite", "score_crossview"]
    thresholds = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    pareto_rows: list[dict] = []
    for score in scores:
        for line in ["ALL"] + LINES:
            rows = all_rows if line == "ALL" else [r for r in all_rows if r["line"] == line]
            for m in evaluate_thresholds(rows, score, thresholds):
                m["line"] = line
                pareto_rows.append(m)
    write_csv(out_dir / "score_threshold_pareto.csv", pareto_rows)

    lolo_rows: list[dict] = []
    quantile_rows: list[dict] = []
    for score in scores:
        for target in LINES:
            source = [r for r in all_rows if r["line"] != target]
            target_rows = [r for r in all_rows if r["line"] == target]
            thr, source_metric = choose_source_threshold(source, score, thresholds, args.min_source_coverage)
            target_scores = score_array(target_rows, score)
            target_metric = metrics_for(target_rows, np.isfinite(target_scores) & (target_scores >= thr))
            baseline_metric = metrics_for(target_rows, np.ones(len(target_rows), dtype=bool))
            lolo_rows.append(
                {
                    "score": score,
                    "target_line": target,
                    "source_threshold": thr,
                    "source_mae_ns": source_metric["mae_ns"],
                    "source_coverage": source_metric["coverage"],
                    "target_baseline_mae_ns": baseline_metric["mae_ns"],
                    "target_baseline_coverage": baseline_metric["coverage"],
                    "target_mae_ns": target_metric["mae_ns"],
                    "target_coverage": target_metric["coverage"],
                    "target_severe_gt10_rate": target_metric["severe_gt10_rate"],
                    "target_severe_gt20_rate": target_metric["severe_gt20_rate"],
                    "target_severe_gt40_rate": target_metric["severe_gt40_rate"],
                }
            )
            keep_fracs = np.linspace(0.2, 1.0, 81, dtype=np.float32)
            keep_frac, source_keep_metric = choose_source_keep_fraction(source, score, keep_fracs, args.min_source_coverage)
            target_keep_metric = metrics_for(target_rows, accept_top_fraction(target_rows, score, keep_frac))
            quantile_rows.append(
                {
                    "score": score,
                    "target_line": target,
                    "source_keep_fraction": keep_frac,
                    "source_mae_ns": source_keep_metric["mae_ns"],
                    "source_coverage": source_keep_metric["coverage"],
                    "target_baseline_mae_ns": baseline_metric["mae_ns"],
                    "target_baseline_coverage": baseline_metric["coverage"],
                    "target_mae_ns": target_keep_metric["mae_ns"],
                    "target_coverage": target_keep_metric["coverage"],
                    "target_severe_gt10_rate": target_keep_metric["severe_gt10_rate"],
                    "target_severe_gt20_rate": target_keep_metric["severe_gt20_rate"],
                    "target_severe_gt40_rate": target_keep_metric["severe_gt40_rate"],
                }
            )
    write_csv(out_dir / "source_tuned_leave_one_line_abstention.csv", lolo_rows)
    write_csv(out_dir / "source_tuned_keep_fraction_abstention.csv", quantile_rows)

    best_rows = []
    for score in scores:
        subset = [r for r in lolo_rows if r["score"] == score]
        mae_vals = [finite_float(r["target_mae_ns"]) for r in subset if np.isfinite(finite_float(r["target_mae_ns"]))]
        cov_vals = [finite_float(r["target_coverage"]) for r in subset if np.isfinite(finite_float(r["target_coverage"]))]
        base_mae = [finite_float(r["target_baseline_mae_ns"]) for r in subset if np.isfinite(finite_float(r["target_baseline_mae_ns"]))]
        base_cov = [finite_float(r["target_baseline_coverage"]) for r in subset if np.isfinite(finite_float(r["target_baseline_coverage"]))]
        best_rows.append(
            {
                "score": score,
                "avg_baseline_mae_ns": float(np.mean(base_mae)),
                "avg_baseline_coverage": float(np.mean(base_cov)),
                "avg_target_mae_ns": float(np.mean(mae_vals)),
                "avg_target_coverage": float(np.mean(cov_vals)),
                "delta_mae_ns": float(np.mean(mae_vals) - np.mean(base_mae)),
                "delta_coverage": float(np.mean(cov_vals) - np.mean(base_cov)),
            }
        )
    write_csv(out_dir / "source_tuned_summary.csv", best_rows)

    keep_summary_rows = []
    for score in scores:
        subset = [r for r in quantile_rows if r["score"] == score]
        mae_vals = [finite_float(r["target_mae_ns"]) for r in subset if np.isfinite(finite_float(r["target_mae_ns"]))]
        cov_vals = [finite_float(r["target_coverage"]) for r in subset if np.isfinite(finite_float(r["target_coverage"]))]
        base_mae = [finite_float(r["target_baseline_mae_ns"]) for r in subset if np.isfinite(finite_float(r["target_baseline_mae_ns"]))]
        base_cov = [finite_float(r["target_baseline_coverage"]) for r in subset if np.isfinite(finite_float(r["target_baseline_coverage"]))]
        keep_summary_rows.append(
            {
                "score": score,
                "avg_baseline_mae_ns": float(np.mean(base_mae)),
                "avg_baseline_coverage": float(np.mean(base_cov)),
                "avg_target_mae_ns": float(np.mean(mae_vals)),
                "avg_target_coverage": float(np.mean(cov_vals)),
                "delta_mae_ns": float(np.mean(mae_vals) - np.mean(base_mae)),
                "delta_coverage": float(np.mean(cov_vals) - np.mean(base_cov)),
            }
        )
    write_csv(out_dir / "source_tuned_keep_fraction_summary.csv", keep_summary_rows)

    report = [
        "# PGDA-CSNet v1.11 Confidence-Control First Pass",
        "",
        "## Inputs",
        "",
        "- Strict source-validation outputs: `eval_paper_v1_10d_*_sourceval_v19d_baseline_last_breakable_p020/p050`",
        "- Robust inference views: `eval_paper_v1_11_*_sourceval_v19d_baseline_last_robust_breakable_p020/p050`",
        "- Candidate picks are taken from the fixed `P=0.20` baseline.",
        "- `P=0.50` outputs are used only as a consistency feature.",
        "",
        "## Outputs",
        "",
        f"- Trace features: `{feature_csv.relative_to(ROOT).as_posix()}`",
        f"- Threshold Pareto: `{(out_dir / 'score_threshold_pareto.csv').relative_to(ROOT).as_posix()}`",
        f"- Source-tuned LOO abstention: `{(out_dir / 'source_tuned_leave_one_line_abstention.csv').relative_to(ROOT).as_posix()}`",
        f"- Summary: `{(out_dir / 'source_tuned_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- Source-tuned keep-fraction abstention: `{(out_dir / 'source_tuned_keep_fraction_abstention.csv').relative_to(ROOT).as_posix()}`",
        f"- Keep-fraction summary: `{(out_dir / 'source_tuned_keep_fraction_summary.csv').relative_to(ROOT).as_posix()}`",
        "",
        "## Absolute-Threshold Summary",
        "",
        "| Score | Baseline MAE ns | Baseline coverage | Abstained MAE ns | Abstained coverage | Delta MAE ns | Delta coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        report.append(
            "| {score} | {avg_baseline_mae_ns:.3f} | {avg_baseline_coverage:.3f} | {avg_target_mae_ns:.3f} | {avg_target_coverage:.3f} | {delta_mae_ns:.3f} | {delta_coverage:.3f} |".format(
                **row
            )
        )
    report.extend(
        [
            "",
            "## Target-Calibrated Keep-Fraction Summary",
            "",
            "| Score | Baseline MAE ns | Baseline coverage | Abstained MAE ns | Abstained coverage | Delta MAE ns | Delta coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in keep_summary_rows:
        report.append(
            "| {score} | {avg_baseline_mae_ns:.3f} | {avg_baseline_coverage:.3f} | {avg_target_mae_ns:.3f} | {avg_target_coverage:.3f} | {delta_mae_ns:.3f} | {delta_coverage:.3f} |".format(
                **row
            )
        )
    report.extend(
        [
            "",
            "## Interpretation Placeholder",
            "",
            "This first pass only tests whether simple confidence scores can improve strict leave-one-line MAE-coverage behavior without target labels. Promotion requires visual review and comparison against the frozen v1.9D baseline.",
            "",
        ]
    )
    (out_dir / "V1_11_CONFIDENCE_CONTROL_FIRST_PASS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(out_dir / "V1_11_CONFIDENCE_CONTROL_FIRST_PASS_REPORT.md")


if __name__ == "__main__":
    main()
