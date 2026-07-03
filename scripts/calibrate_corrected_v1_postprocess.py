from pathlib import Path
import csv
import itertools
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_corrected_v1"
EVAL = ROOT / "outputs" / "eval_corrected_v1_cv_validation"
REPORT = ROOT / "reports"


def centerline(arr, min_sum=1e-4):
    h, _ = arr.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    s = arr.sum(axis=0)
    c = (arr * yy).sum(axis=0) / np.maximum(s, 1e-6)
    v = s > min_sum
    c[~v] = np.nan
    return c, v


def dp_ridge_centerline(prob, max_jump=8, smooth_weight=0.08, search_min_sample=None, search_max_sample=None):
    h, w = prob.shape
    p = np.clip(prob.astype(np.float32), 1e-6, 1.0)
    unary = -np.log(p)
    lo = 0 if search_min_sample is None else max(0, int(search_min_sample))
    hi = h - 1 if search_max_sample is None else min(h - 1, int(search_max_sample))
    mask = np.ones(h, dtype=bool)
    mask[lo : hi + 1] = False
    unary[mask, :] += 20.0
    dp = np.empty((h, w), np.float32)
    back = np.zeros((h, w), np.int16)
    dp[:, 0] = unary[:, 0]
    offsets = np.arange(-max_jump, max_jump + 1, dtype=np.int16)
    big = np.float32(1e6)
    for x in range(1, w):
        prev = dp[:, x - 1]
        cand = np.full((len(offsets), h), big, dtype=np.float32)
        for oi, off in enumerate(offsets):
            penalty = np.float32(smooth_weight * (int(off) ** 2))
            if off < 0:
                cand[oi, -off:] = prev[:off] + penalty
            elif off > 0:
                cand[oi, :-off] = prev[off:] + penalty
            else:
                cand[oi, :] = prev
        arg = np.argmin(cand, axis=0).astype(np.int16)
        dp[:, x] = unary[:, x] + cand[arg, np.arange(h)]
        back[:, x] = np.clip(np.arange(h, dtype=np.int32) + offsets[arg].astype(np.int32), 0, h - 1).astype(np.int16)
    path = np.zeros(w, np.float32)
    y = int(np.argmin(dp[:, -1]))
    path[-1] = y
    for x in range(w - 1, 0, -1):
        y = int(back[y, x])
        path[x - 1] = y
    return path


def load_line(line):
    z = np.load(DATA / "lines" / f"{line}.npz")
    pred = np.load(EVAL / f"{line}_pred_softmask.npy").astype(np.float32)
    pres = np.load(EVAL / f"{line}_presence_prob.npy").astype(np.float32)
    gt = z["soft_mask_train"].astype(np.float32)
    status = z["status_code"].astype(np.int16)
    dt_ns = float(z["dt_ns"])
    cgt, vgt = centerline(gt)
    return {"line": line, "pred": pred, "pres": pres, "gt": gt, "status": status, "dt_ns": dt_ns, "cgt": cgt, "vgt": vgt}


def contiguous_pick_penalty(valid):
    if valid.size == 0:
        return 0.0
    changes = np.count_nonzero(valid[1:] != valid[:-1])
    return float(changes) / float(valid.size)


def eval_params(case, params):
    pred = case["pred"]
    pres = case["pres"]
    dt_ns = case["dt_ns"]
    lo = int(round(params["search_min_ns"] / dt_ns))
    hi = int(round(params["search_max_ns"] / dt_ns))
    path = dp_ridge_centerline(pred, params["dp_max_jump"], params["dp_smooth_weight"], lo, hi)
    h, w = pred.shape
    idx = np.clip(np.round(path).astype(int), 0, h - 1)
    path_prob = pred[idx, np.arange(w)]
    valid = (pres >= params["presence_thr"]) & (path_prob >= params["path_prob_thr"])
    cgt = case["cgt"]
    vgt = case["vgt"]
    status = case["status"]
    strong = status == 1
    weak = status == 2
    both = valid & vgt
    strong_both = valid & vgt & strong
    mae_all = float(np.nanmean(np.abs(path[both] - cgt[both])) * dt_ns) if both.any() else 999.0
    mae_strong = float(np.nanmean(np.abs(path[strong_both] - cgt[strong_both])) * dt_ns) if strong_both.any() else mae_all
    strong_recall = float((valid & strong).sum() / max(strong.sum(), 1))
    weak_pick_rate = float((valid & weak).sum() / max(weak.sum(), 1)) if weak.any() else 0.0
    pick_rate = float(valid.mean())
    gap_penalty = contiguous_pick_penalty(valid)
    # Weak traces are allowed but not forced. Penalize missing strong labels and excessive fragmentation.
    score = mae_strong + 10.0 * (1.0 - strong_recall) + 2.0 * gap_penalty + 0.5 * abs(pick_rate - 0.65)
    return {
        "line": case["line"],
        "mae_ns": mae_all,
        "mae_strong_ns": mae_strong,
        "strong_recall": strong_recall,
        "weak_pick_rate": weak_pick_rate,
        "pick_rate": pick_rate,
        "gap_penalty": gap_penalty,
        "score": float(score),
    }


def main():
    REPORT.mkdir(exist_ok=True)
    cases = [load_line(line) for line in ["Line3", "Line6", "Line7", "LineL1"]]
    grid = {
        "presence_thr": [0.30, 0.45, 0.60],
        "path_prob_thr": [0.10, 0.20, 0.30, 0.40],
        "search_min_ns": [260.0, 280.0, 300.0],
        "search_max_ns": [500.0, 540.0],
        "dp_max_jump": [6, 8],
        "dp_smooth_weight": [0.08, 0.16],
    }
    rows = []
    best = None
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        per_line = [eval_params(case, params) for case in cases]
        avg = {
            "avg_mae_ns": float(np.mean([r["mae_ns"] for r in per_line])),
            "avg_mae_strong_ns": float(np.mean([r["mae_strong_ns"] for r in per_line])),
            "avg_strong_recall": float(np.mean([r["strong_recall"] for r in per_line])),
            "avg_weak_pick_rate": float(np.mean([r["weak_pick_rate"] for r in per_line])),
            "avg_pick_rate": float(np.mean([r["pick_rate"] for r in per_line])),
            "avg_gap_penalty": float(np.mean([r["gap_penalty"] for r in per_line])),
            "avg_score": float(np.mean([r["score"] for r in per_line])),
        }
        rec = {**params, **avg}
        rows.append(rec)
        if best is None or rec["avg_score"] < best["avg_score"]:
            best = rec
            best_per_line = per_line

    out_csv = REPORT / "corrected_v1_postprocess_grid.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["avg_score"]))

    per_line_csv = REPORT / "corrected_v1_postprocess_best_by_line.csv"
    with open(per_line_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_per_line[0].keys()))
        writer.writeheader()
        writer.writerows(best_per_line)

    result = {
        "presence_thr": best["presence_thr"],
        "path_prob_thr": best["path_prob_thr"],
        "search_min_ns": best["search_min_ns"],
        "search_max_ns": best["search_max_ns"],
        "dp_max_jump": int(best["dp_max_jump"]),
        "dp_smooth_weight": best["dp_smooth_weight"],
        "source": "corrected_v1_leave_one_line_out_cv_grid",
        "selection_note": "Score minimizes strong-label centerline MAE, penalizes missed present traces and fragmented picks; weak traces are allowed but not forced.",
        "best": best,
        "per_line": best_per_line,
        "cv_eval_dir": str(EVAL.relative_to(ROOT)),
    }
    out_json = REPORT / "corrected_v1_postprocess_thresholds.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_csv)
    print(per_line_csv)
    print(out_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
