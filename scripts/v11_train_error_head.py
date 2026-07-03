from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]


FEATURES = [
    "path_prob",
    "presence_prob",
    "robust_path_prob",
    "robust_presence_prob",
    "mask_center_agreement",
    "local_contrast",
    "segment_len",
    "mean_dp_disagree_ns",
    "p050_agree_ns",
    "robust_agree_ns",
    "jump_ns",
    "curvature_ns",
    "score_composite",
    "score_crossview",
]


LOG_FEATURES = {
    "segment_len",
    "mean_dp_disagree_ns",
    "p050_agree_ns",
    "robust_agree_ns",
    "jump_ns",
    "curvature_ns",
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def ffloat(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if int(row["gt_valid"]) != 1 or int(row["picked_p020"]) != 1:
            continue
        err = ffloat(row["error_ns"])
        if not np.isfinite(err):
            continue
        row["error_ns_float"] = err
        out.append(row)
    return out


def read_all_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def matrix(rows: list[dict], medians: np.ndarray | None = None, means: np.ndarray | None = None, stds: np.ndarray | None = None):
    raw = np.zeros((len(rows), len(FEATURES)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(FEATURES):
            value = ffloat(row.get(name, "nan"))
            if name == "local_contrast" and np.isfinite(value):
                value = max(-8.0, min(8.0, value))
            if name in LOG_FEATURES and np.isfinite(value):
                value = math.log1p(max(0.0, value))
            raw[i, j] = value
    if medians is None:
        medians = np.nanmedian(raw, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    raw = np.where(np.isfinite(raw), raw, medians[None, :])
    if means is None:
        means = raw.mean(axis=0)
    if stds is None:
        stds = raw.std(axis=0)
        stds = np.where(stds > 1e-6, stds, 1.0)
    x = (raw - means[None, :]) / stds[None, :]
    return x.astype(np.float32), medians.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def train_logistic(x: np.ndarray, y: np.ndarray, steps: int = 1800, lr: float = 0.04, l2: float = 0.02):
    n, d = x.shape
    xb = np.concatenate([np.ones((n, 1), dtype=np.float32), x], axis=1)
    w = np.zeros(d + 1, dtype=np.float32)
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float((1.0 - y).sum()))
    weights = np.where(y > 0.5, 0.5 / pos, 0.5 / neg).astype(np.float32) * n
    for _ in range(steps):
        p = sigmoid(xb @ w)
        grad = (xb.T @ ((p - y) * weights)) / n
        grad[1:] += l2 * w[1:]
        w -= lr * grad.astype(np.float32)
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float32), x], axis=1)
    return sigmoid(xb @ w).astype(np.float32)


def metrics(rows: list[dict], accepted: np.ndarray, total_gt: int | None = None) -> dict[str, float]:
    err = np.array([ffloat(r["error_ns_float"]) for r in rows], dtype=np.float32)
    ok = accepted & np.isfinite(err)
    mae = float(np.nanmean(err[ok])) if ok.any() else float("nan")
    coverage = float(np.count_nonzero(ok) / max(1, len(rows)))
    overall_coverage = float(np.count_nonzero(ok) / max(1, total_gt if total_gt is not None else len(rows)))
    severe10 = float(np.count_nonzero(ok & (err > 10.0)) / max(1, np.count_nonzero(ok))) if ok.any() else float("nan")
    severe20 = float(np.count_nonzero(ok & (err > 20.0)) / max(1, np.count_nonzero(ok))) if ok.any() else float("nan")
    severe40 = float(np.count_nonzero(ok & (err > 40.0)) / max(1, np.count_nonzero(ok))) if ok.any() else float("nan")
    return {
        "mae_ns": mae,
        "coverage_among_picked": coverage,
        "overall_coverage": overall_coverage,
        "severe_gt10_rate": severe10,
        "severe_gt20_rate": severe20,
        "severe_gt40_rate": severe40,
        "n_accept": float(np.count_nonzero(ok)),
        "n_candidate": float(len(rows)),
    }


def best_threshold(rows: list[dict], prob: np.ndarray, min_coverage: float):
    thresholds = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    candidates = []
    for thr in thresholds:
        m = metrics(rows, prob >= float(thr))
        m["threshold"] = float(thr)
        candidates.append(m)
    valid = [m for m in candidates if m["coverage_among_picked"] >= min_coverage and np.isfinite(m["mae_ns"])]
    if not valid:
        valid = [m for m in candidates if np.isfinite(m["mae_ns"])]
    return min(valid, key=lambda m: (m["mae_ns"], -m["coverage_among_picked"]))


def accept_top_fraction(prob: np.ndarray, keep_fraction: float) -> np.ndarray:
    accepted = np.zeros(prob.shape[0], dtype=bool)
    if prob.size == 0:
        return accepted
    n = int(math.ceil(prob.size * max(0.0, min(1.0, keep_fraction))))
    if n <= 0:
        return accepted
    order = np.argsort(prob)[::-1]
    accepted[order[:n]] = True
    return accepted


def best_keep_fraction(rows: list[dict], prob: np.ndarray, min_coverage: float):
    candidates = []
    for frac in np.linspace(0.2, 1.0, 81, dtype=np.float32):
        m = metrics(rows, accept_top_fraction(prob, float(frac)))
        m["keep_fraction"] = float(frac)
        candidates.append(m)
    valid = [m for m in candidates if m["coverage_among_picked"] >= min_coverage and np.isfinite(m["mae_ns"])]
    if not valid:
        valid = [m for m in candidates if np.isfinite(m["mae_ns"])]
    return min(valid, key=lambda m: (m["mae_ns"], -m["coverage_among_picked"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict], prefix: str) -> dict:
    maes = [ffloat(r[f"{prefix}_mae_ns"]) for r in rows if np.isfinite(ffloat(r[f"{prefix}_mae_ns"]))]
    covs = [ffloat(r[f"{prefix}_coverage_among_picked"]) for r in rows if np.isfinite(ffloat(r[f"{prefix}_coverage_among_picked"]))]
    ocovs = [ffloat(r[f"{prefix}_overall_coverage"]) for r in rows if np.isfinite(ffloat(r[f"{prefix}_overall_coverage"]))]
    sev20 = [ffloat(r[f"{prefix}_severe_gt20_rate"]) for r in rows if np.isfinite(ffloat(r[f"{prefix}_severe_gt20_rate"]))]
    sev40 = [ffloat(r[f"{prefix}_severe_gt40_rate"]) for r in rows if np.isfinite(ffloat(r[f"{prefix}_severe_gt40_rate"]))]
    return {
        f"{prefix}_avg_mae_ns": float(np.mean(maes)),
        f"{prefix}_avg_coverage_among_picked": float(np.mean(covs)),
        f"{prefix}_avg_overall_coverage": float(np.mean(ocovs)),
        f"{prefix}_avg_severe_gt20_rate": float(np.mean(sev20)),
        f"{prefix}_avg_severe_gt40_rate": float(np.mean(sev40)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", default="reports/v1_11_confidence_control_multiview/trace_confidence_features_v19d_last_p020.csv")
    ap.add_argument("--out-dir", default="reports/v1_11_error_head")
    ap.add_argument("--good-error-ns", type=float, default=20.0)
    ap.add_argument("--min-source-coverage", type=float, default=0.55)
    args = ap.parse_args()

    all_rows = read_all_rows(resolve(args.features_csv))
    total_gt_by_line = {
        line: sum(1 for r in all_rows if r["line"] == line and int(r["gt_valid"]) == 1)
        for line in LINES
    }
    rows = read_rows(resolve(args.features_csv))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    coef_rows = []
    decision_rows = []
    for target in LINES:
        train_rows = [r for r in rows if r["line"] != target]
        test_rows = [r for r in rows if r["line"] == target]
        x_train, med, mean, std = matrix(train_rows)
        y_train = np.array([1.0 if ffloat(r["error_ns_float"]) <= args.good_error_ns else 0.0 for r in train_rows], dtype=np.float32)
        w = train_logistic(x_train, y_train)
        x_test, _, _, _ = matrix(test_rows, med, mean, std)
        p_train = predict(w, x_train)
        p_test = predict(w, x_test)

        baseline = metrics(test_rows, np.ones(len(test_rows), dtype=bool), total_gt_by_line[target])
        source_thr = best_threshold(train_rows, p_train, args.min_source_coverage)
        target_thr = metrics(test_rows, p_test >= source_thr["threshold"], total_gt_by_line[target])
        source_frac = best_keep_fraction(train_rows, p_train, args.min_source_coverage)
        keep_accept = accept_top_fraction(p_test, source_frac["keep_fraction"])
        thr_accept = p_test >= source_thr["threshold"]
        target_frac = metrics(test_rows, keep_accept, total_gt_by_line[target])

        row = {
            "target_line": target,
            "baseline_mae_ns": baseline["mae_ns"],
            "baseline_coverage_among_picked": baseline["coverage_among_picked"],
            "baseline_overall_coverage": baseline["overall_coverage"],
            "baseline_severe_gt20_rate": baseline["severe_gt20_rate"],
            "baseline_severe_gt40_rate": baseline["severe_gt40_rate"],
            "threshold": source_thr["threshold"],
            "threshold_source_mae_ns": source_thr["mae_ns"],
            "threshold_source_coverage_among_picked": source_thr["coverage_among_picked"],
            "threshold_mae_ns": target_thr["mae_ns"],
            "threshold_coverage_among_picked": target_thr["coverage_among_picked"],
            "threshold_overall_coverage": target_thr["overall_coverage"],
            "threshold_severe_gt20_rate": target_thr["severe_gt20_rate"],
            "threshold_severe_gt40_rate": target_thr["severe_gt40_rate"],
            "keep_fraction": source_frac["keep_fraction"],
            "keep_source_mae_ns": source_frac["mae_ns"],
            "keep_source_coverage_among_picked": source_frac["coverage_among_picked"],
            "keep_mae_ns": target_frac["mae_ns"],
            "keep_coverage_among_picked": target_frac["coverage_among_picked"],
            "keep_overall_coverage": target_frac["overall_coverage"],
            "keep_severe_gt20_rate": target_frac["severe_gt20_rate"],
            "keep_severe_gt40_rate": target_frac["severe_gt40_rate"],
        }
        result_rows.append(row)
        for name, coef in zip(["bias"] + FEATURES, w.tolist()):
            coef_rows.append({"target_line": target, "feature": name, "coef": coef})
        for row_src, prob, accept_thr, accept_keep in zip(test_rows, p_test.tolist(), thr_accept.tolist(), keep_accept.tolist()):
            decision_rows.append(
                {
                    "line": target,
                    "trace_idx": int(row_src["trace_idx"]),
                    "error_prob_good_le20": float(prob),
                    "accepted_threshold": int(bool(accept_thr)),
                    "accepted_keep_fraction": int(bool(accept_keep)),
                    "error_ns": ffloat(row_src["error_ns_float"]),
                    "picked_p020": int(row_src["picked_p020"]),
                    "gt_valid": int(row_src["gt_valid"]),
                    "source_threshold": source_thr["threshold"],
                    "source_keep_fraction": source_frac["keep_fraction"],
                }
            )

    write_csv(out_dir / "source_trained_error_head_lolo.csv", result_rows)
    write_csv(out_dir / "source_trained_error_head_coefficients.csv", coef_rows)
    write_csv(out_dir / "source_trained_error_head_trace_decisions.csv", decision_rows)

    base = summarize(result_rows, "baseline")
    thr = summarize(result_rows, "threshold")
    keep = summarize(result_rows, "keep")
    summary = [{**base, **thr, **keep}]
    write_csv(out_dir / "source_trained_error_head_summary.csv", summary)

    report = [
        "# PGDA-CSNet v1.11 Source-Trained Error Head",
        "",
        "## Purpose",
        "",
        "Train a lightweight NumPy logistic error head on source-line predictions only, then apply it to the held-out target line without target labels.",
        "",
        "Positive target: pick error <= `{:.1f} ns`.".format(args.good_error_ns),
        "",
        "## Summary",
        "",
        "| Method | Avg MAE ns | Overall coverage | Candidate coverage | Severe >20 ns | Severe >40 ns |",
        "|---|---:|---:|---:|---:|---:|",
        "| baseline | {baseline_avg_mae_ns:.3f} | {baseline_avg_overall_coverage:.3f} | {baseline_avg_coverage_among_picked:.3f} | {baseline_avg_severe_gt20_rate:.3f} | {baseline_avg_severe_gt40_rate:.3f} |".format(**base),
        "| source-threshold error head | {threshold_avg_mae_ns:.3f} | {threshold_avg_overall_coverage:.3f} | {threshold_avg_coverage_among_picked:.3f} | {threshold_avg_severe_gt20_rate:.3f} | {threshold_avg_severe_gt40_rate:.3f} |".format(**thr),
        "| source-keep-fraction error head | {keep_avg_mae_ns:.3f} | {keep_avg_overall_coverage:.3f} | {keep_avg_coverage_among_picked:.3f} | {keep_avg_severe_gt20_rate:.3f} | {keep_avg_severe_gt40_rate:.3f} |".format(**keep),
        "",
        "## Artifacts",
        "",
        f"- Results: `{(out_dir / 'source_trained_error_head_lolo.csv').relative_to(ROOT).as_posix()}`",
        f"- Summary: `{(out_dir / 'source_trained_error_head_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- Coefficients: `{(out_dir / 'source_trained_error_head_coefficients.csv').relative_to(ROOT).as_posix()}`",
        f"- Trace decisions: `{(out_dir / 'source_trained_error_head_trace_decisions.csv').relative_to(ROOT).as_posix()}`",
        "",
    ]
    (out_dir / "V1_11_SOURCE_TRAINED_ERROR_HEAD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir / "V1_11_SOURCE_TRAINED_ERROR_HEAD_REPORT.md")


if __name__ == "__main__":
    main()
