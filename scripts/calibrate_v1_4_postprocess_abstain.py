from pathlib import Path
import argparse
import csv
import itertools
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_full_line import centerline, dp_ridge_centerline


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


PATH_CACHE = {}


def load_case(data, eval_dir, name, line, trace_start=0, trace_end=None, role="train"):
    z = np.load(data / "lines" / f"{line}.npz", allow_pickle=False)
    full_w = z["soft_mask_train"].shape[1]
    if trace_end is None:
        trace_end = full_w - 1
    sl = slice(trace_start, trace_end + 1)
    pred = np.load(eval_dir / f"{name}_pred_softmask.npy").astype(np.float32)
    pres = np.load(eval_dir / f"{name}_presence_prob.npy").astype(np.float32)
    gt = z["soft_mask_train"].astype(np.float32)[:, sl]
    status = z["status_code"].astype(np.int16)[sl]
    dt_ns = float(z["dt_ns"])
    cgt, vgt = centerline(gt, 1e-3)
    return {
        "name": name,
        "line": line,
        "role": role,
        "pred": pred,
        "pres": pres,
        "gt": gt,
        "status": status,
        "dt_ns": dt_ns,
        "cgt": cgt,
        "vgt": vgt,
    }


def fragmentation(valid):
    if valid.size < 2:
        return 0.0
    return float(np.count_nonzero(valid[1:] != valid[:-1])) / float(valid.size)


def eval_case(case, params):
    pred = case["pred"]
    pres = case["pres"]
    dt_ns = case["dt_ns"]
    lo = int(round(params["search_min_ns"] / dt_ns))
    hi = int(round(params["search_max_ns"] / dt_ns))
    cache_key = (case["name"], lo, hi, int(params["dp_max_jump"]), float(params["dp_smooth_weight"]))
    if cache_key not in PATH_CACHE:
        path, _ = dp_ridge_centerline(
            pred,
            max_jump=int(params["dp_max_jump"]),
            smooth_weight=float(params["dp_smooth_weight"]),
            search_min_sample=lo,
            search_max_sample=hi,
        )
        PATH_CACHE[cache_key] = path
    path = PATH_CACHE[cache_key]
    h, w = pred.shape
    yi = np.clip(np.round(path).astype(np.int32), 0, h - 1)
    path_prob = pred[yi, np.arange(w)]
    valid = (pres >= params["presence_thr"]) & (path_prob >= params["path_prob_thr"])

    status = case["status"]
    strong = status == 1
    weak = status == 2
    no_pick = status == 0
    vgt = case["vgt"]
    cgt = case["cgt"]
    both = valid & vgt
    strong_both = valid & vgt & strong
    mae_all = float(np.nanmean(np.abs(path[both] - cgt[both])) * dt_ns) if both.any() else 999.0
    mae_strong = float(np.nanmean(np.abs(path[strong_both] - cgt[strong_both])) * dt_ns) if strong_both.any() else 999.0
    strong_recall = float((valid & strong).sum() / max(int(strong.sum()), 1))
    weak_pick_rate = float((valid & weak).sum() / max(int(weak.sum()), 1)) if weak.any() else 0.0
    no_pick_rate = float((valid & no_pick).sum() / max(int(no_pick.sum()), 1)) if no_pick.any() else 0.0
    pick_rate = float(valid.mean())
    frag = fragmentation(valid)
    return {
        "name": case["name"],
        "role": case["role"],
        "mae_ns": mae_all,
        "mae_strong_ns": mae_strong,
        "strong_recall": strong_recall,
        "weak_pick_rate": weak_pick_rate,
        "no_pick_false_rate": no_pick_rate,
        "pick_rate": pick_rate,
        "gap_penalty": frag,
    }


def train_score(metrics):
    return (
        metrics["mae_strong_ns"]
        + 9.0 * (1.0 - metrics["strong_recall"])
        + 3.0 * metrics["no_pick_false_rate"]
        + 2.0 * metrics["gap_penalty"]
        + 0.35 * abs(metrics["pick_rate"] - 0.65)
    )


def aggregate(params, cases):
    per = [eval_case(c, params) for c in cases]
    train = [r for r in per if r["role"] == "train"]
    review = [r for r in per if r["role"] == "review"]
    holdout = [r for r in per if r["role"] == "holdout"]
    train_scores = [train_score(r) for r in train]
    review_pick = float(np.mean([r["pick_rate"] for r in review])) if review else 0.0
    holdout_recall_penalty = float(np.mean([max(0.0, 0.62 - r["strong_recall"]) for r in holdout])) if holdout else 0.0
    holdout_mae_penalty = float(np.mean([max(0.0, r["mae_strong_ns"] - 5.0) for r in holdout])) if holdout else 0.0
    balanced_score = float(np.mean(train_scores))
    abstain_score = balanced_score + 4.0 * review_pick + 6.0 * holdout_recall_penalty + 0.4 * holdout_mae_penalty
    rec = {
        **params,
        "balanced_score": balanced_score,
        "abstain_score": abstain_score,
        "train_mae_strong_ns": float(np.mean([r["mae_strong_ns"] for r in train])),
        "train_strong_recall": float(np.mean([r["strong_recall"] for r in train])),
        "train_no_pick_false_rate": float(np.mean([r["no_pick_false_rate"] for r in train])),
        "train_pick_rate": float(np.mean([r["pick_rate"] for r in train])),
        "review_pick_rate": review_pick,
        "holdout_mae_strong_ns": float(np.mean([r["mae_strong_ns"] for r in holdout])) if holdout else 999.0,
        "holdout_strong_recall": float(np.mean([r["strong_recall"] for r in holdout])) if holdout else 0.0,
        "holdout_pick_rate": float(np.mean([r["pick_rate"] for r in holdout])) if holdout else 0.0,
    }
    return rec, per


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data_corrected_v1_4_terrain_direction")
    ap.add_argument("--eval-dir", default="outputs/eval_corrected_v1_4_calibration")
    ap.add_argument("--report-dir", default="reports/v1_4_postprocess_calibration")
    args = ap.parse_args()
    data = resolve(args.data_root)
    eval_dir = resolve(args.eval_dir)
    report = resolve(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    cases = [
        load_case(data, eval_dir, "Line3", "Line3", role="train"),
        load_case(data, eval_dir, "Line6", "Line6", role="train"),
        load_case(data, eval_dir, "Line7", "Line7", role="train"),
        load_case(data, eval_dir, "LineL1", "LineL1", role="train"),
        load_case(data, eval_dir, "LineX1", "LineX1", role="review"),
        load_case(data, eval_dir, "Line9_holdout_tr1664_2377", "Line9", 1664, 2377, role="holdout"),
    ]
    grid = {
        "presence_thr": [0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90],
        "path_prob_thr": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        "search_min_ns": [240.0, 260.0, 280.0],
        "search_max_ns": [500.0, 520.0, 540.0],
        "dp_max_jump": [6, 8],
        "dp_smooth_weight": [0.08, 0.16, 0.24],
    }
    rows = []
    best_balanced = None
    best_abstain = None
    best_balanced_per = None
    best_abstain_per = None
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        rec, per = aggregate(params, cases)
        rows.append(rec)
        if best_balanced is None or rec["balanced_score"] < best_balanced["balanced_score"]:
            best_balanced = rec
            best_balanced_per = per
        if best_abstain is None or rec["abstain_score"] < best_abstain["abstain_score"]:
            best_abstain = rec
            best_abstain_per = per

    write_csv(report / "v1_4_postprocess_grid.csv", sorted(rows, key=lambda r: r["abstain_score"]))
    write_csv(report / "v1_4_balanced_best_by_line.csv", best_balanced_per)
    write_csv(report / "v1_4_abstain_best_by_line.csv", best_abstain_per)

    for name, best, per, score_key in [
        ("balanced", best_balanced, best_balanced_per, "balanced_score"),
        ("abstain", best_abstain, best_abstain_per, "abstain_score"),
    ]:
        result = {
            "presence_thr": best["presence_thr"],
            "path_prob_thr": best["path_prob_thr"],
            "search_min_ns": best["search_min_ns"],
            "search_max_ns": best["search_max_ns"],
            "dp_max_jump": int(best["dp_max_jump"]),
            "dp_smooth_weight": best["dp_smooth_weight"],
            "source": "v1_4_terrain_direction_line9_holdout_and_lineX1_review",
            "selection_note": "Balanced optimizes corrected measured training lines. Abstain penalizes LineX1 review picks while protecting Line9 holdout recall/MAE.",
            "score_key": score_key,
            "best": best,
            "per_line": per,
            "eval_dir": str(eval_dir.relative_to(ROOT)),
        }
        (report / f"v1_4_{name}_postprocess_thresholds.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# PGDA-CSNet v1.4 Postprocess Calibration",
        "",
        "Calibration for the terrain/direction corrected label set and v1.4 checkpoint.",
        "",
        f"- Balanced: presence `{best_balanced['presence_thr']}`, path `{best_balanced['path_prob_thr']}`, search `{best_balanced['search_min_ns']}-{best_balanced['search_max_ns']} ns`, jump `{int(best_balanced['dp_max_jump'])}`, smooth `{best_balanced['dp_smooth_weight']}`.",
        f"- Abstain: presence `{best_abstain['presence_thr']}`, path `{best_abstain['path_prob_thr']}`, search `{best_abstain['search_min_ns']}-{best_abstain['search_max_ns']} ns`, jump `{int(best_abstain['dp_max_jump'])}`, smooth `{best_abstain['dp_smooth_weight']}`.",
        "",
        "Use balanced for visual preview and recall-oriented analysis. Use abstain when low-confidence review-line picks should be suppressed.",
    ]
    (report / "V1_4_POSTPROCESS_CALIBRATION_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(report / "v1_4_balanced_postprocess_thresholds.json")
    print(report / "v1_4_abstain_postprocess_thresholds.json")


if __name__ == "__main__":
    main()
