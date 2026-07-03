from pathlib import Path
import csv
import json
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


def gray_image(x):
    y = ((np.clip(x, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(y, "L").convert("RGB")


def heat_image(x, color=(255, 72, 32)):
    x = np.clip(x.astype(np.float32), 0.0, max(0.6, float(np.nanmax(x))))
    x = x / max(1e-6, float(np.nanmax(x)))
    r = (x * color[0]).astype(np.uint8)
    g = (x * color[1]).astype(np.uint8)
    b = (x * color[2]).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def overlay(base, mask, color):
    base = base.convert("RGBA")
    hm = heat_image(mask, color=color).convert("RGBA")
    alpha = (np.clip(mask, 0, 1) * 155).astype(np.uint8)
    hm.putalpha(Image.fromarray(alpha, "L"))
    return Image.alpha_composite(base, hm).convert("RGB")


def resize_panel(img, size=(420, 260)):
    return img.resize(size, Image.Resampling.BILINEAR)


def read_metrics(path):
    if not path.exists():
        return {}
    return {r["metric"]: r["value"] for r in csv.DictReader(open(path, encoding="utf-8"))}


def draw_title(img, title, subtitle=""):
    canvas = Image.new("RGB", (img.width, img.height + 46), "white")
    canvas.paste(img, (0, 46))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    d.text((8, 6), title, fill=(20, 20, 20), font=font)
    if subtitle:
        d.text((8, 27), subtitle, fill=(80, 80, 80), font=small)
    return canvas


def make_preview(data_root, eval_dir, item):
    eval_name = item["eval_name"]
    line_name = item["line"]
    z = np.load(data_root / "lines" / f"{line_name}.npz")
    raw_full = z["raw_full_normalized"].astype(np.float32)
    gt_full = z["soft_mask_train"].astype(np.float32)
    start = int(item.get("trace_start", 0))
    end = int(item.get("trace_end", -1))
    end = raw_full.shape[1] - 1 if end < 0 else min(end, raw_full.shape[1] - 1)
    sl = slice(start, end + 1)
    raw = raw_full[:, sl]
    gt = gt_full[:, sl]
    path_pred = eval_dir / f"{eval_name}_path_softmask.npy"
    pred_source = "path probability" if path_pred.exists() else "prediction probability"
    pred = np.load(path_pred if path_pred.exists() else eval_dir / f"{eval_name}_pred_softmask.npy").astype(np.float32)
    pres = np.load(eval_dir / f"{eval_name}_presence_prob.npy").astype(np.float32)
    gain = gained_bscan(raw)
    metrics = read_metrics(eval_dir / f"{eval_name}_full_metrics.csv")
    sub = f"MAE={float(metrics.get('dp_center_mae_ns', 'nan')):.3f} ns, pick={float(metrics.get('final_pick_rate', 'nan')):.3f}"

    raw_img = resize_panel(gray_image(gain))
    gt_img = resize_panel(overlay(gray_image(gain), gt, (250, 204, 21)))
    pred_img = resize_panel(overlay(gray_image(gain), pred, (255, 72, 32)))
    prob_img = resize_panel(heat_image(pred, (255, 72, 32)))
    pres_img = Image.new("RGB", (420, 260), "white")
    d = ImageDraw.Draw(pres_img)
    pts = []
    if len(pres) > 1:
        for i, v in enumerate(pres):
            x = int(i * 419 / (len(pres) - 1))
            y = int(245 - np.clip(v, 0, 1) * 220)
            pts.append((x, y))
        d.line(pts, fill=(37, 99, 235), width=2)
    d.rectangle((0, 25, 419, 245), outline=(180, 180, 180))
    d.text((10, 8), "presence probability", fill=(40, 40, 40))

    panels = [
        draw_title(raw_img, "gained raw B-scan", f"{line_name} trace {start}-{end}"),
        draw_title(gt_img, "raw + corrected label"),
        draw_title(pred_img, "raw + prediction"),
        draw_title(prob_img, pred_source),
        draw_title(pres_img, "presence curve"),
    ]
    w = panels[0].width
    h = panels[0].height
    canvas = Image.new("RGB", (w * 5, h), "white")
    for i, p in enumerate(panels):
        canvas.paste(p, (i * w, 0))
    out = eval_dir / f"{eval_name}_pillow_preview.png"
    canvas.save(out)
    return out, sub


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: make_paper_v17b_pillow_previews.py <eval_dir> <data_root> <items_json>")
    eval_dir = Path(sys.argv[1])
    data_root = Path(sys.argv[2])
    items_json = Path(sys.argv[3])
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if not items_json.is_absolute():
        items_json = ROOT / items_json
    items = json.loads(items_json.read_text(encoding="utf-8"))
    previews = []
    for item in items:
        out, sub = make_preview(data_root, eval_dir, item)
        previews.append((out, item["eval_name"], sub))
        print(out)
    if previews:
        ims = [Image.open(p[0]).convert("RGB") for p in previews]
        sheet = Image.new("RGB", (ims[0].width, ims[0].height * len(ims)), "white")
        for i, im in enumerate(ims):
            sheet.paste(im, (0, i * im.height))
        sheet_out = eval_dir / "paper_v1_7b_contact_sheet.png"
        sheet.save(sheet_out)
        print(sheet_out)


if __name__ == "__main__":
    main()
