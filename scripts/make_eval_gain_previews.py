from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def moving_average_axis(a, win, axis):
    if win <= 1:
        return a
    pad = win // 2
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad, pad)
    ap = np.pad(a, pad_width, mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis, ap)


def gained_bscan(raw):
    x = raw.astype(np.float32)
    x = x - np.nanmedian(x, axis=1, keepdims=True)
    base = float(np.nanpercentile(np.abs(x), 50))
    env = np.sqrt(moving_average_axis(x * x, 41, axis=0))
    agc = x / (env + max(base * 0.08, 1e-6))
    time_gain = np.linspace(0.75, 1.55, x.shape[0], dtype=np.float32)[:, None]
    y = moving_average_axis(agc * time_gain, 3, axis=1)
    vmax = float(np.nanpercentile(np.abs(y), 99.2))
    if not np.isfinite(vmax) or vmax <= 1e-6:
        vmax = 1.0
    return np.clip(y / vmax, -1.0, 1.0)


def centerline(mask, min_sum=1e-4):
    h, _ = mask.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    s = mask.sum(axis=0)
    c = (mask * yy).sum(axis=0) / np.maximum(s, 1e-6)
    v = s > min_sum
    c[~v] = np.nan
    return c, v


def plot_one(data_root, eval_dir, eval_name, line_name, trace_start, trace_end):
    z = np.load(data_root / "lines" / f"{line_name}.npz")
    raw_full = z["raw_full_normalized"].astype(np.float32)
    gt_full = z["soft_mask_train"].astype(np.float32)
    dt_ns = float(z["dt_ns"])
    trace_end = raw_full.shape[1] - 1 if trace_end < 0 else min(trace_end, raw_full.shape[1] - 1)
    sl = slice(trace_start, trace_end + 1)
    raw = raw_full[:, sl]
    gt = gt_full[:, sl]
    pred = np.load(eval_dir / f"{eval_name}_pred_softmask.npy").astype(np.float32)
    pres = np.load(eval_dir / f"{eval_name}_presence_prob.npy").astype(np.float32)
    gain = gained_bscan(raw)
    pred_c, pred_v = centerline(pred * (pred > 0.15))
    gt_c, gt_v = centerline(gt)
    t = np.arange(raw.shape[0], dtype=np.float32) * dt_ns
    x = np.arange(trace_start, trace_end + 1)
    extent = (trace_start, trace_end, t[-1], t[0])

    fig, ax = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
    ax = ax.ravel()
    ax[0].imshow(gain, aspect="auto", cmap="gray", vmin=-1, vmax=1, extent=extent)
    ax[0].set_title(f"{eval_name}: gained raw B-scan")

    ax[1].imshow(gain, aspect="auto", cmap="gray", vmin=-1, vmax=1, extent=extent)
    ax[1].imshow(gt, aspect="auto", cmap="viridis", alpha=np.clip(gt * 0.85, 0, 0.65), extent=extent)
    ax[1].plot(x[gt_v], gt_c[gt_v] * dt_ns, color="#facc15", lw=1.0)
    ax[1].set_title("gained raw + corrected label")

    ax[2].imshow(gain, aspect="auto", cmap="gray", vmin=-1, vmax=1, extent=extent)
    ax[2].imshow(pred, aspect="auto", cmap="magma", alpha=np.clip(pred * 0.85, 0, 0.65), extent=extent)
    ax[2].plot(x[pred_v], pred_c[pred_v] * dt_ns, color="#38bdf8", lw=1.0)
    ax[2].set_title("gained raw + prediction")

    ax[3].imshow(gt, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.6, float(gt.max())), extent=extent)
    ax[3].set_title("corrected label")

    ax[4].imshow(pred, aspect="auto", cmap="magma", vmin=0, vmax=max(0.6, float(pred.max())), extent=extent)
    ax[4].set_title("prediction probability")

    ax[5].plot(x, pres, lw=1.0)
    ax[5].set_ylim(-0.05, 1.05)
    ax[5].set_title("presence probability")
    ax[5].set_xlabel("trace")
    ax[5].set_ylabel("probability")

    for a in ax[:5]:
        a.set_xlabel("trace")
        a.set_ylabel("time ns")
    out = eval_dir / f"{eval_name}_gain_prediction_preview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--data-root", default="data_corrected_v1")
    ap.add_argument("--items-json", default="")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if args.items_json:
        items = json.loads(Path(args.items_json).read_text(encoding="utf-8"))
    else:
        items = [
            {"eval_name": "Line3", "line": "Line3", "trace_start": 0, "trace_end": -1},
            {"eval_name": "Line6", "line": "Line6", "trace_start": 0, "trace_end": -1},
            {"eval_name": "Line7", "line": "Line7", "trace_start": 0, "trace_end": -1},
            {"eval_name": "LineL1", "line": "LineL1", "trace_start": 0, "trace_end": -1},
            {"eval_name": "LineX1", "line": "LineX1", "trace_start": 0, "trace_end": -1},
            {"eval_name": "Line9_holdout_tr1664_2377", "line": "Line9", "trace_start": 1664, "trace_end": 2377},
        ]
    for item in items:
        out = plot_one(data_root, eval_dir, item["eval_name"], item["line"], int(item["trace_start"]), int(item["trace_end"]))
        print(out)


if __name__ == "__main__":
    main()
