from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]


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


def gray_image(x):
    y = ((np.clip(x, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(y, "L").convert("RGB")


def read_centerline(path):
    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["trace_idx"])] = row
    return out


def read_decisions(path):
    out = {line: {} for line in LINES}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["line"]][int(row["trace_idx"])] = row
    return out


def draw_curve(img, samples, valid, color, width=2):
    d = ImageDraw.Draw(img)
    h = img.height
    w = img.width
    n = len(samples)
    prev = None
    for i, y in enumerate(samples):
        if not valid[i] or not np.isfinite(y):
            prev = None
            continue
        x = int(round(i * (w - 1) / max(1, n - 1)))
        yy = int(round(y * (h - 1) / 500.0))
        yy = max(0, min(h - 1, yy))
        if prev is not None and abs(prev[1] - yy) < 60:
            d.line([prev, (x, yy)], fill=color, width=width)
        else:
            d.ellipse((x - width, yy - width, x + width, yy + width), fill=color)
        prev = (x, yy)


def title_panel(img, title, subtitle=""):
    canvas = Image.new("RGB", (img.width, img.height + 42), "white")
    canvas.paste(img, (0, 42))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
        small = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    d.text((8, 5), title, fill=(20, 20, 20), font=font)
    if subtitle:
        d.text((8, 25), subtitle, fill=(80, 80, 80), font=small)
    return canvas


def make_line_preview(line, data_root, decisions, out_dir):
    data = np.load(data_root / "lines" / f"{line}.npz")
    raw = data["raw_full_normalized"].astype(np.float32)
    gt = data["soft_mask_train"].astype(np.float32)
    gain = gained_bscan(raw)
    base = gray_image(gain).resize((1200, 320), Image.Resampling.BILINEAR)
    label_panel = base.copy()
    accepted_panel = base.copy()
    rejected_panel = base.copy()
    center = read_centerline(ROOT / f"outputs/eval_paper_v1_10d_{line}_sourceval_v19d_baseline_last_breakable_p020/{line}_pred_centerline.csv")
    n = raw.shape[1]
    gt_y = np.full(n, np.nan, dtype=np.float32)
    gt_valid = np.zeros(n, dtype=bool)
    pred_y = np.full(n, np.nan, dtype=np.float32)
    pred_valid = np.zeros(n, dtype=bool)
    acc = np.zeros(n, dtype=bool)
    rej = np.zeros(n, dtype=bool)
    for i in range(n):
        row = center.get(i)
        if row:
            gt_valid[i] = int(row["gt_valid"]) == 1
            pred_valid[i] = int(row["dp_valid"]) == 1
            gt_y[i] = float(row["gt_center_sample"]) if gt_valid[i] else np.nan
            pred_y[i] = float(row["dp_center_sample"]) if pred_valid[i] else np.nan
        dec = decisions.get(line, {}).get(i)
        if dec:
            acc[i] = int(dec["accepted_keep_fraction"]) == 1
            rej[i] = pred_valid[i] and not acc[i]
    draw_curve(label_panel, gt_y, gt_valid, (250, 204, 21), width=2)
    draw_curve(accepted_panel, pred_y, pred_valid & acc, (34, 197, 94), width=2)
    draw_curve(rejected_panel, pred_y, pred_valid & rej, (239, 68, 68), width=2)
    accepted = int(np.count_nonzero(pred_valid & acc))
    rejected = int(np.count_nonzero(pred_valid & rej))
    panels = [
        title_panel(label_panel, f"{line}: audited label", "yellow = label"),
        title_panel(accepted_panel, "accepted picks", f"green accepted = {accepted}"),
        title_panel(rejected_panel, "rejected picks", f"red rejected = {rejected}"),
    ]
    sheet = Image.new("RGB", (panels[0].width, panels[0].height * len(panels)), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, i * panel.height))
    out = out_dir / f"{line}_v11_error_head_keep_preview.png"
    sheet.save(out)
    return out


def main():
    data_root = ROOT / "data_corrected_v1_4_terrain_direction"
    decision_csv = ROOT / "reports/v1_11_error_head_cov075/source_trained_error_head_trace_decisions.csv"
    out_dir = ROOT / "reports/v1_11_error_head_cov075/previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = read_decisions(decision_csv)
    outs = []
    for line in LINES:
        out = make_line_preview(line, data_root, decisions, out_dir)
        outs.append(out)
        print(out)
    ims = [Image.open(p).convert("RGB") for p in outs]
    sheet = Image.new("RGB", (ims[0].width, ims[0].height * len(ims)), "white")
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    contact = out_dir / "v11_error_head_keep_contact_sheet.png"
    sheet.save(contact)
    print(contact)


if __name__ == "__main__":
    main()
