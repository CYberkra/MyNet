from pathlib import Path
import argparse
import csv

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def centerline(arr, min_sum=1e-3):
    h, _ = arr.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    s = arr.sum(axis=0)
    c = (arr * yy).sum(axis=0) / np.maximum(s, 1e-6)
    valid = s > min_sum
    c[~valid] = np.nan
    return c, valid


def dp_ridge_centerline(prob, max_jump=6, smooth_weight=0.16, search_min_sample=None, search_max_sample=None):
    h, w = prob.shape
    p = np.clip(prob.astype(np.float32), 1e-6, 1.0)
    unary = -np.log(p)
    if search_min_sample is not None or search_max_sample is not None:
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
                cand[oi, :] = prev + penalty
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


def breakable_dp(prob, pres, presence_thr, path_prob_thr, min_segment=16, max_jump=6, smooth_weight=0.16, search_min_sample=None, search_max_sample=None):
    h, w = prob.shape
    lo = 0 if search_min_sample is None else max(0, int(search_min_sample))
    hi = h - 1 if search_max_sample is None else min(h - 1, int(search_max_sample))
    peak = np.nanmax(prob[lo : hi + 1, :], axis=0)
    gate = (pres >= presence_thr) & (peak >= path_prob_thr)
    path = np.full(w, np.nan, np.float32)
    valid = np.zeros(w, dtype=bool)
    start = None
    for i, ok in enumerate(np.r_[gate, False]):
        if ok and start is None:
            start = i
        if (not ok) and start is not None:
            end = i
            if end - start >= int(min_segment):
                sub = dp_ridge_centerline(
                    prob[:, start:end],
                    max_jump=max_jump,
                    smooth_weight=smooth_weight,
                    search_min_sample=search_min_sample,
                    search_max_sample=search_max_sample,
                )
                path[start:end] = sub
                valid[start:end] = True
            start = None
    path_prob = np.full(w, np.nan, np.float32)
    final_valid = np.zeros(w, dtype=bool)
    for i in range(w):
        if valid[i] and np.isfinite(path[i]):
            yi = int(np.clip(round(float(path[i])), 0, h - 1))
            path_prob[i] = prob[yi, i]
            final_valid[i] = path_prob[i] >= path_prob_thr and pres[i] >= presence_thr
    path[~final_valid] = np.nan
    return path, final_valid, path_prob


def load_metrics(eval_dir, eval_name, data_root, line, trace_start, trace_end):
    pred_path = eval_dir / f"{eval_name}_path_softmask.npy"
    pred = np.load(pred_path if pred_path.exists() else eval_dir / f"{eval_name}_pred_softmask.npy").astype(np.float32)
    pres = np.load(eval_dir / f"{eval_name}_presence_prob.npy").astype(np.float32)
    z = np.load(data_root / "lines" / f"{line}.npz", allow_pickle=False)
    gt = z["soft_mask_train"].astype(np.float32)[:, trace_start : trace_end + 1]
    dt = float(z["dt_ns"])
    return pred, pres, gt, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--eval-name", default="Line9_holdout_tr1664_2377")
    ap.add_argument("--data-root", default="data_corrected_v1_4_terrain_direction")
    ap.add_argument("--line", default="Line9")
    ap.add_argument("--trace-start", type=int, default=1664)
    ap.add_argument("--trace-end", type=int, default=2377)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    pred, pres, gt, dt = load_metrics(eval_dir, args.eval_name, data_root, args.line, args.trace_start, args.trace_end)
    cgt, vgt = centerline(gt)
    search_min = int(round(240.0 / dt))
    search_max = int(round(500.0 / dt))
    rows = []
    for presence_thr in [0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]:
        for path_thr in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
            path, valid, path_prob = breakable_dp(
                pred,
                pres,
                presence_thr=presence_thr,
                path_prob_thr=path_thr,
                max_jump=6,
                smooth_weight=0.16,
                search_min_sample=search_min,
                search_max_sample=search_max,
            )
            ok = valid & vgt & np.isfinite(path)
            mae = float(np.nanmean(np.abs(path[ok] - cgt[ok]) * dt)) if ok.any() else float("nan")
            signed = float(np.nanmean((path[ok] - cgt[ok]) * dt)) if ok.any() else float("nan")
            rows.append(
                {
                    "presence_thr": presence_thr,
                    "path_prob_thr": path_thr,
                    "mae_ns": mae,
                    "signed_error_ns": signed,
                    "pick_rate": float(valid.mean()),
                    "valid_trace_count": int(valid.sum()),
                    "mean_path_prob": float(np.nanmean(path_prob[valid])) if valid.any() else float("nan"),
                }
            )
    finite = [r for r in rows if np.isfinite(float(r["mae_ns"]))]
    for r in rows:
        dominated = False
        if np.isfinite(float(r["mae_ns"])):
            for q in finite:
                if q is r:
                    continue
                if float(q["mae_ns"]) <= float(r["mae_ns"]) and float(q["pick_rate"]) >= float(r["pick_rate"]) and (
                    float(q["mae_ns"]) < float(r["mae_ns"]) or float(q["pick_rate"]) > float(r["pick_rate"])
                ):
                    dominated = True
                    break
        r["pareto"] = int(not dominated and np.isfinite(float(r["mae_ns"])))
    out = Path(args.out_csv)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(out)
    for r in sorted([r for r in rows if int(r["pareto"])], key=lambda x: (float(x["mae_ns"]), -float(x["pick_rate"])))[:12]:
        print(r)


if __name__ == "__main__":
    main()
