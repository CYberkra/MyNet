from pathlib import Path
import csv
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def font(size=20, bold=False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def gray_image(x):
    y = ((np.clip(x, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(y, "L").convert("RGB")


def heat_image(x, color=(230, 63, 42)):
    x = np.clip(x.astype(np.float32), 0.0, max(0.6, float(np.nanmax(x))))
    x = x / max(1e-6, float(np.nanmax(x)))
    r = (x * color[0]).astype(np.uint8)
    g = (x * color[1]).astype(np.uint8)
    b = (x * color[2]).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def overlay(base, mask, color, alpha_scale=155):
    base = base.convert("RGBA")
    hm = heat_image(mask, color=color).convert("RGBA")
    alpha = (np.clip(mask, 0, 1) * alpha_scale).astype(np.uint8)
    hm.putalpha(Image.fromarray(alpha, "L"))
    return Image.alpha_composite(base, hm).convert("RGB")


def read_metrics(path):
    return {r["metric"]: r["value"] for r in csv.DictReader(open(path, encoding="utf-8"))}


def read_centerline(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        def val(key):
            try:
                return float(r[key]) if r[key] != "" else np.nan
            except ValueError:
                return np.nan
        rows.append(
            {
                "trace": int(r["trace_idx"]),
                "dp_time": val("dp_time_ns"),
                "gt_time": val("gt_time_ns"),
                "dp_sample": val("dp_center_sample"),
                "gt_sample": val("gt_center_sample"),
                "pick": r["pick_status"] == "pick",
                "gt_valid": r["gt_valid"] == "1",
            }
        )
    return rows


def draw_line_segments(img, points, fill, width=3, max_trace_gap=2):
    d = ImageDraw.Draw(img)
    segment = []
    prev_trace = None
    for trace, x, y in points:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if prev_trace is not None and trace - prev_trace > max_trace_gap:
            if len(segment) >= 2:
                d.line([(int(px), int(py)) for px, py in segment], fill=fill, width=width, joint="curve")
            segment = []
        segment.append((x, y))
        prev_trace = trace
    if len(segment) >= 2:
        d.line([(int(px), int(py)) for px, py in segment], fill=fill, width=width, joint="curve")


def scale_panel(img, width, height):
    return img.resize((width, height), Image.Resampling.BILINEAR)


def title_panel(img, title, subtitle=""):
    pad = 62
    out = Image.new("RGB", (img.width, img.height + pad), "white")
    out.paste(img, (0, pad))
    d = ImageDraw.Draw(out)
    d.text((10, 8), title, fill=(20, 20, 20), font=font(21, True))
    if subtitle:
        d.text((10, 35), subtitle, fill=(75, 75, 75), font=font(15))
    return out


def curve_panel(center_rows, width, height):
    out = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(out)
    left, top, right, bottom = 54, 24, width - 16, height - 38
    d.rectangle((left, top, right, bottom), outline=(170, 170, 170))
    traces = np.array([r["trace"] for r in center_rows], dtype=np.float32)
    errors = np.array(
        [
            abs(r["dp_time"] - r["gt_time"])
            if r["pick"] and r["gt_valid"] and np.isfinite(r["dp_time"]) and np.isfinite(r["gt_time"])
            else np.nan
            for r in center_rows
        ],
        dtype=np.float32,
    )
    finite = np.isfinite(errors)
    ymax = float(np.nanpercentile(errors[finite], 95)) if finite.any() else 1.0
    ymax = max(2.0, min(40.0, ymax * 1.25))
    d.text((8, top - 3), "ns", fill=(70, 70, 70), font=font(12))
    for frac in [0.0, 0.5, 1.0]:
        y = bottom - frac * (bottom - top)
        d.line((left, y, right, y), fill=(225, 225, 225))
        d.text((8, int(y) - 7), f"{ymax * frac:.0f}", fill=(90, 90, 90), font=font(12))
    if len(traces) > 1 and finite.any():
        x0, x1 = float(traces.min()), float(traces.max())
        pts = []
        for tr, err in zip(traces, errors):
            if not np.isfinite(err):
                continue
            x = left + (tr - x0) / max(1.0, x1 - x0) * (right - left)
            y = bottom - np.clip(err / ymax, 0, 1) * (bottom - top)
            pts.append((x, y))
        if len(pts) >= 2:
            d.line(pts, fill=(31, 94, 180), width=2)
    d.text((left, height - 28), "trace", fill=(70, 70, 70), font=font(12))
    d.text((left + 70, 6), "absolute centerline error", fill=(35, 35, 35), font=font(16, True))
    return out


def make_figure(eval_dir, data_root, line_name, eval_name, output_name):
    z = np.load(data_root / "lines" / f"{line_name}.npz")
    raw = z["raw_full_normalized"].astype(np.float32)
    gt = z["soft_mask_train"].astype(np.float32)
    pred = np.load(eval_dir / f"{eval_name}_pred_softmask.npy").astype(np.float32)
    metrics = read_metrics(eval_dir / f"{eval_name}_full_metrics.csv")
    center = read_centerline(eval_dir / f"{eval_name}_pred_centerline.csv")
    gain = gained_bscan(raw)
    base = gray_image(gain)
    panel_w, panel_h = 540, 310
    raw_img = scale_panel(base, panel_w, panel_h)
    label_img = scale_panel(overlay(base, gt, (244, 184, 30)), panel_w, panel_h)
    pred_img = scale_panel(overlay(base, pred, (225, 58, 42)), panel_w, panel_h)
    prob_img = scale_panel(heat_image(pred, (225, 58, 42)), panel_w, panel_h)

    H, W = pred.shape
    sx, sy = panel_w / float(W), panel_h / float(H)
    pred_with_lines = pred_img.copy()
    gt_points = [(r["trace"], r["trace"] * sx, r["gt_sample"] * sy) for r in center if r["gt_valid"]]
    dp_points = [(r["trace"], r["trace"] * sx, r["dp_sample"] * sy) for r in center if r["pick"]]
    draw_line_segments(pred_with_lines, gt_points, (250, 218, 75), width=3, max_trace_gap=2)
    draw_line_segments(pred_with_lines, dp_points, (47, 111, 237), width=3, max_trace_gap=2)

    curve = curve_panel(center, panel_w, panel_h)
    subtitle = (
        f"MAE={float(metrics.get('dp_center_mae_ns', 'nan')):.3f} ns, "
        f"pick={float(metrics.get('final_pick_rate', 'nan')):.3f}, "
        f"dice={float(metrics.get('soft_dice_weighted', 'nan')):.3f}"
    )
    panels = [
        title_panel(raw_img, f"{line_name} gained raw B-scan", "display gain only; model input remains raw"),
        title_panel(label_img, "corrected label overlay", "yellow = corrected target interface"),
        title_panel(pred_img, "prediction overlay", subtitle),
        title_panel(prob_img, "prediction probability", "soft mask probability"),
        title_panel(pred_with_lines, "centerline comparison", "yellow = label, blue = prediction"),
        title_panel(curve, "error curve", "absolute timing error on picked traces"),
    ]
    gap = 16
    header_h = 74
    cols, rows = 3, 2
    canvas = Image.new(
        "RGB",
        (cols * panel_w + (cols + 1) * gap, header_h + rows * panels[0].height + (rows + 1) * gap),
        "white",
    )
    d = ImageDraw.Draw(canvas)
    d.text((gap, 14), f"PGDA-CSNet v1.7-A few-shot calibration: {line_name}", fill=(15, 15, 15), font=font(30, True))
    d.text((gap, 48), subtitle, fill=(70, 70, 70), font=font(17))
    for i, p in enumerate(panels):
        x = gap + (i % cols) * (panel_w + gap)
        y = header_h + gap + (i // cols) * (p.height + gap)
        canvas.paste(p, (x, y))
    out = eval_dir / output_name
    canvas.save(out)
    return out


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: make_publication_fewshot_figures.py <eval_dir> <data_root> <line_name> <eval_name>"
        )
    eval_dir = Path(sys.argv[1])
    data_root = Path(sys.argv[2])
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    line_name = sys.argv[3]
    eval_name = sys.argv[4]
    out = make_figure(eval_dir, data_root, line_name, eval_name, f"{eval_name}_publication_figure.png")
    print(out)


if __name__ == "__main__":
    main()
