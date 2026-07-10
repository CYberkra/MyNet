from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_optional(path: Path):
    return np.load(path) if path.exists() else None


def overlay_line(ax, y, **kw):
    if y is None:
        return
    x = np.arange(len(y))
    ax.plot(x, y, **kw)


def centerline(mask: np.ndarray, min_sum: float = 1e-4):
    h, w = mask.shape
    ys = np.arange(h, dtype=np.float32)[:, None]
    s = np.nansum(mask, axis=0)
    c = np.nansum(mask * ys, axis=0) / np.maximum(s, 1e-6)
    c[s <= min_sum] = np.nan
    return c


def load_centerline_csv(path: Path):
    """Load actual DP/mean/GT centerlines written by eval_full_line.py.

    Returns arrays in local plot coordinates when possible.  This avoids the
    misleading fallback of taking a center-of-mass over the probability image,
    which is not the same as the DP-decoded path.
    """
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None
    trace_idx = []
    for i, r in enumerate(rows):
        try:
            trace_idx.append(int(float(r.get("trace_idx", i))))
        except Exception:
            trace_idx.append(i)
    offset = min(trace_idx) if trace_idx else 0
    n = len(rows)
    mean = np.full(n, np.nan, dtype=np.float32)
    dp = np.full(n, np.nan, dtype=np.float32)
    gt = np.full(n, np.nan, dtype=np.float32)
    pres = np.full(n, np.nan, dtype=np.float32)
    status = []

    def put(arr, j, value):
        if value is None or value == "":
            return
        try:
            arr[j] = float(value)
        except Exception:
            return

    for i, r in enumerate(rows):
        j = int(trace_idx[i] - offset)
        if j < 0 or j >= n:
            j = i
        put(mean, j, r.get("mean_center_sample"))
        put(dp, j, r.get("dp_center_sample"))
        put(gt, j, r.get("gt_center_sample"))
        put(pres, j, r.get("presence_prob"))
        status.append(r.get("pick_status", ""))
    return {"mean": mean, "dp": dp, "gt": gt, "presence": pres, "status": status, "offset": offset}


def show_bscan(ax, arr, title):
    vmax = np.nanpercentile(np.abs(arr), 99.0) if np.isfinite(arr).any() else 1.0
    vmax = max(float(vmax), 1e-6)
    ax.imshow(arr, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("trace")
    ax.set_ylabel("sample")


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Create six diagnostic figures for GprMambaSep/CurveMamba outputs.")
    ap.add_argument("--line", default="Line9")
    ap.add_argument("--data-root", default="data_corrected_v1_4_terrain_direction")
    ap.add_argument("--eval-dir", required=True, help="Directory from eval_full_line.py")
    ap.add_argument("--sep-dir", default="", help="Directory from eval_gprmambasep_separation.py; optional")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data_root = resolve(args.data_root)
    eval_dir = resolve(args.eval_dir)
    sep_dir = resolve(args.sep_dir) if args.sep_dir else None
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    line = args.line

    line_npz = np.load(data_root / "lines" / f"{line}.npz")
    raw = line_npz["raw_full_normalized"] if "raw_full_normalized" in line_npz.files else line_npz[line_npz.files[0]]
    gt = None
    for key in ("soft_mask_train", "y_mask", "label_mask", "mask", "soft_mask"):
        if key in line_npz.files:
            gt = line_npz[key]
            break

    # Prefer explicit post-audit artifact names; retain old aliases only for
    # historical runs generated before the metric-contract split.
    pred = load_optional(eval_dir / f"{line}_mask_prob.npy")
    if pred is None:
        pred = load_optional(eval_dir / f"{line}_pred_softmask.npy")
    center = load_optional(eval_dir / f"{line}_center_response_prob.npy")
    if center is None:
        center = load_optional(eval_dir / f"{line}_center_softmask.npy")
    path_mask = load_optional(eval_dir / f"{line}_path_prob_image.npy")
    if path_mask is None:
        path_mask = load_optional(eval_dir / f"{line}_path_softmask.npy")
    pres = load_optional(eval_dir / f"{line}_presence_prob.npy")
    curve = load_optional(eval_dir / f"{line}_curve_prob.npy")
    if curve is None:
        curve = load_optional(eval_dir / f"{line}_curve_logits.npy")
    csv_lines = load_centerline_csv(eval_dir / f"{line}_pred_centerline.csv")
    gt_line = centerline(gt) if gt is not None and gt.ndim == 2 else None
    pred_line = centerline(center) if center is not None and center.ndim == 2 else None
    dp_line = centerline(path_mask) if path_mask is not None and path_mask.ndim == 2 else None
    if csv_lines is not None:
        # Prefer exact decoded/evaluated paths from CSV over probability-map
        # center-of-mass approximations.
        gt_line = csv_lines["gt"]
        pred_line = csv_lines["mean"]
        dp_line = csv_lines["dp"]
        if pres is None and np.isfinite(csv_lines["presence"]).any():
            pres = csv_lines["presence"]

    # V1 raw + GT + DP
    fig, ax = plt.subplots(figsize=(12, 5))
    show_bscan(ax, raw, f"{line}: raw + GT/center/DP (CSV decoded path when available)")
    overlay_line(ax, gt_line, color="lime", lw=1.2, label="GT")
    overlay_line(ax, pred_line, color="orange", lw=1.0, label="center")
    overlay_line(ax, dp_line, color="red", lw=1.2, label="DP")
    ax.legend(loc="lower right")
    save(fig, out_dir / "01_raw_gt_dp_overlay.png")

    # V2 G_hat if present
    if sep_dir:
        g_candidates = list(sep_dir.glob("*G_hat*.npy")) + list(sep_dir.glob("*g_hat*.npy"))
        if g_candidates:
            ghat = np.load(g_candidates[0])
            if ghat.ndim == 3:
                ghat = ghat[0]
            fig, ax = plt.subplots(figsize=(12, 5))
            show_bscan(ax, ghat, f"{line}: G_hat + GT")
            overlay_line(ax, gt_line, color="lime", lw=1.2, label="GT")
            ax.legend(loc="lower right")
            save(fig, out_dir / "02_ghat_gt_overlay.png")

    # V3 heatmaps
    panels = [("mask", pred), ("center", center), ("path", path_mask), ("curve", curve)]
    panels = [(n, a) for n, a in panels if a is not None]
    if panels:
        fig, axs = plt.subplots(len(panels), 1, figsize=(12, 3.2 * len(panels)))
        if len(panels) == 1:
            axs = [axs]
        for ax, (name, arr) in zip(axs, panels):
            if arr.ndim == 3:
                arr = arr[0]
            ax.imshow(arr, aspect="auto", cmap="magma")
            ax.set_title(f"{line}: {name} heatmap")
            overlay_line(ax, gt_line, color="cyan", lw=1.0, label="GT")
            ax.legend(loc="lower right")
        save(fig, out_dir / "03_mask_center_curve_heatmap.png")

    # V4 presence
    if pres is not None:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(np.arange(len(pres)), pres, lw=1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{line}: trace presence probability")
        ax.set_xlabel("trace")
        ax.set_ylabel("presence")
        save(fig, out_dir / "04_presence_trace.png")

    # V5 A/S/G panels if present
    if sep_dir:
        comp = []
        for name in ("A_hat", "S_hat", "G_hat"):
            files = list(sep_dir.glob(f"*{name}*.npy")) + list(sep_dir.glob(f"*{name.lower()}*.npy"))
            if files:
                arr = np.load(files[0])
                comp.append((name, arr[0] if arr.ndim == 3 else arr))
        if comp:
            fig, axs = plt.subplots(len(comp), 1, figsize=(12, 3.2 * len(comp)))
            if len(comp) == 1:
                axs = [axs]
            for ax, (name, arr) in zip(axs, comp):
                show_bscan(ax, arr, f"{line}: {name}")
                overlay_line(ax, gt_line, color="lime", lw=1.0)
            save(fig, out_dir / "05_asg_residual_panels.png")

    # V6 compact casebook PDF
    with PdfPages(out_dir / "06_false_path_casebook.pdf") as pdf:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        show_bscan(axs[0], raw, f"{line}: raw")
        overlay_line(axs[0], gt_line, color="lime", lw=1.0, label="GT")
        overlay_line(axs[0], dp_line, color="red", lw=1.0, label="DP")
        axs[0].legend(loc="lower right")
        if pred is not None:
            axs[1].imshow(pred, aspect="auto", cmap="magma")
            axs[1].set_title("pred softmask")
        if center is not None:
            axs[2].imshow(center, aspect="auto", cmap="magma")
            axs[2].set_title("center heatmap")
        pdf.savefig(fig)
        plt.close(fig)

    print(out_dir)


if __name__ == "__main__":
    main()
