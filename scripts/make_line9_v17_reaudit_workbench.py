from pathlib import Path
import csv
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_audited_v16_20260627"
PRIMARY_EVAL = ROOT / "outputs" / "eval_paper_v1_9d_mambavision_hybrid_final_seed1902_w050_p050_breakable_p050"
LEGACY_EVAL = ROOT / "outputs" / "eval_paper_v1_9d_mambavision_hybrid_final_seed1902_w050_p050"
V16_EVAL = ROOT / "outputs" / "eval_gpu_paper_v1_9d_v16_audited_full_line9_holdout_w050_p050"
OUT = ROOT / "reports" / "line9_v17_reaudit_workbench"
VELOCITY_M_PER_NS = 0.074


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


def centerline(mask, min_sum=1e-3):
    h, _ = mask.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    s = mask.sum(axis=0)
    c = (mask * yy).sum(axis=0) / np.maximum(s, 1e-6)
    v = s > min_sum
    c[~v] = np.nan
    return c, v


def gray_image(x):
    y = ((np.clip(x, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(y, "L").convert("RGB")


def heat_rgba(mask, color, alpha_scale=150):
    x = mask.astype(np.float32)
    mx = max(0.6, float(np.nanmax(x))) if x.size else 0.6
    x = np.clip(x / max(mx, 1e-6), 0.0, 1.0)
    rgba = np.zeros((*x.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (x * color[0]).astype(np.uint8)
    rgba[..., 1] = (x * color[1]).astype(np.uint8)
    rgba[..., 2] = (x * color[2]).astype(np.uint8)
    rgba[..., 3] = (x * alpha_scale).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def overlay(base, mask, color):
    b = base.convert("RGBA")
    return Image.alpha_composite(b, heat_rgba(mask, color)).convert("RGB")


def resize(img, size=(420, 280)):
    return img.resize(size, Image.Resampling.BILINEAR)


def font(size=16):
    for name in ["arial.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def panel(img, title, subtitle=""):
    canvas = Image.new("RGB", (img.width, img.height + 48), "white")
    canvas.paste(img, (0, 48))
    d = ImageDraw.Draw(canvas)
    d.text((8, 5), title, fill=(20, 20, 20), font=font(17))
    if subtitle:
        d.text((8, 27), subtitle, fill=(80, 80, 80), font=font(12))
    return canvas


def read_pred_center(path):
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        tr = int(r["trace_idx"])
        val = r.get("dp_time_ns", "")
        if val:
            out[tr] = float(val)
    return out


def draw_curve(img, traces, times_ns, dt_ns, color, start, end):
    d = ImageDraw.Draw(img)
    pts = []
    for tr in range(start, end + 1):
        t = times_ns.get(tr)
        if t is None or not np.isfinite(t):
            if len(pts) > 1:
                d.line(pts, fill=color, width=2)
            pts = []
            continue
        x = int((tr - start) * (img.width - 1) / max(1, end - start))
        y_sample = t / dt_ns
        y = int(y_sample * (img.height - 1) / 500.0)
        pts.append((x, y))
    if len(pts) > 1:
        d.line(pts, fill=color, width=2)


def make_segment(start, end, line, old_center_ns, primary_center_ns, v16_center_ns, old_mask, primary_pred, v16_pred):
    raw = line["raw_full_normalized"].astype(np.float32)[:, start : end + 1]
    gain = gained_bscan(raw)
    base = gray_image(gain)
    old = old_mask[:, start : end + 1]
    primary = primary_pred[:, start : end + 1] if primary_pred.shape[1] == line["raw_full_normalized"].shape[1] else None
    v16 = v16_pred[:, start - 1664 : end - 1664 + 1] if start >= 1664 and v16_pred is not None else None
    dt = float(line["dt_ns"])

    raw_panel = resize(base)
    old_panel = resize(overlay(base, old, (250, 204, 21)))
    primary_panel = resize(overlay(base, primary, (37, 99, 235))) if primary is not None else resize(base)
    v16_panel = resize(overlay(base, v16, (239, 68, 68))) if v16 is not None else resize(base)
    compare = resize(base)
    scale_x = compare.width / max(1, end - start + 1)
    scale_y = compare.height / 501.0
    d = ImageDraw.Draw(compare)
    for tr in range(start, end + 1):
        x = int((tr - start) * scale_x)
        if tr in old_center_ns:
            y = int((old_center_ns[tr] / dt) * scale_y)
            d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(250, 204, 21))
        if tr in primary_center_ns:
            y = int((primary_center_ns[tr] / dt) * scale_y)
            d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(37, 99, 235))
        if tr in v16_center_ns:
            y = int((v16_center_ns[tr] / dt) * scale_y)
            d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(239, 68, 68))

    panels = [
        panel(raw_panel, "gained raw B-scan", f"Line9 traces {start}-{end}"),
        panel(old_panel, "old v1.4/V16-copied label", "yellow = current training/eval label"),
        panel(primary_panel, "frozen v1.9D prediction", "blue = current primary prior"),
        panel(v16_panel, "V16-trained prediction", "red = V16 label-policy response"),
        panel(compare, "centerline comparison", "yellow old, blue frozen v1.9D, red V16"),
    ]
    canvas = Image.new("RGB", (panels[0].width * len(panels), panels[0].height), "white")
    for i, p in enumerate(panels):
        canvas.paste(p, (i * panels[0].width, 0))
    out = OUT / "previews" / f"Line9_tr{start:04d}_{end:04d}_reaudit.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def segment_summary(start, end, old_center_ns, primary_center_ns, v16_center_ns):
    rows = []
    for tr in range(start, end + 1):
        old = old_center_ns.get(tr)
        pri = primary_center_ns.get(tr)
        v16 = v16_center_ns.get(tr)
        if old is None:
            continue
        pri_abs = abs(pri - old) if pri is not None else np.nan
        v16_abs = abs(v16 - old) if v16 is not None else np.nan
        rows.append((pri_abs, v16_abs))
    arr = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 2), dtype=np.float32)
    return {
        "segment": f"{start}-{end}",
        "start": start,
        "end": end,
        "traces": end - start + 1,
        "primary_abs_err_median_ns": float(np.nanmedian(arr[:, 0])) if arr.size else float("nan"),
        "primary_abs_err_p90_ns": float(np.nanpercentile(arr[:, 0], 90)) if arr.size else float("nan"),
        "v16_abs_err_median_ns": float(np.nanmedian(arr[:, 1])) if arr.size else float("nan"),
        "v16_abs_err_p90_ns": float(np.nanpercentile(arr[:, 1], 90)) if arr.size else float("nan"),
        "review_priority": "",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    line = np.load(DATA / "lines" / "Line9.npz", allow_pickle=False)
    old_mask = line["soft_mask_train"].astype(np.float32)
    old_c, old_v = centerline(old_mask)
    dt = float(line["dt_ns"])
    old_center_ns = {i: float(old_c[i] * dt) for i in range(old_c.shape[0]) if old_v[i]}

    primary_center = read_pred_center(LEGACY_EVAL / "Line9_pred_centerline.csv")
    primary_pred = np.load(LEGACY_EVAL / "Line9_path_softmask.npy").astype(np.float32)
    v16_center = read_pred_center(V16_EVAL / "Line9_holdout_tr1664_2377_pred_centerline.csv")
    v16_pred = np.load(V16_EVAL / "Line9_holdout_tr1664_2377_path_softmask.npy").astype(np.float32)

    segments = []
    starts = list(range(0, 2378, 256))
    for start in starts:
        end = min(start + 255, 2377)
        preview = make_segment(start, end, line, old_center_ns, primary_center, v16_center, old_mask, primary_pred, v16_pred)
        row = segment_summary(start, end, old_center_ns, primary_center, v16_center)
        row["preview"] = str(preview.relative_to(ROOT))
        segments.append(row)

    for row in segments:
        v16_p90 = row["v16_abs_err_p90_ns"]
        pri_p90 = row["primary_abs_err_p90_ns"]
        if row["start"] >= 1664 or (np.isfinite(v16_p90) and v16_p90 >= 12.0) or (np.isfinite(pri_p90) and pri_p90 >= 8.0):
            row["review_priority"] = "high"
        elif row["start"] >= 1280:
            row["review_priority"] = "medium"
        else:
            row["review_priority"] = "baseline"

    out_csv = OUT / "line9_v17_segment_review_index.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(segments[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(segments)

    manifest = {
        "line": "Line9",
        "dataset": "data_audited_v16_20260627",
        "label_source": "Line9 copied unchanged from data_corrected_v1_4_terrain_direction",
        "review_rule": "Use PDF/profile constraints: basal/interface 12-16 m; reject 20-23 m deeper anomaly.",
        "frozen_primary_eval": str(LEGACY_EVAL.relative_to(ROOT)),
        "v16_eval": str(V16_EVAL.relative_to(ROOT)),
        "segment_index": str(out_csv.relative_to(ROOT)),
        "previews_dir": str((OUT / "previews").relative_to(ROOT)),
    }
    (OUT / "line9_v17_reaudit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Line9 v17 Re-audit Workbench\n\n"
        "This folder prepares Line9 for a consistent v17 label audit.\n\n"
        "Review priority should focus on the locked holdout region `1664-2377`, where V16-trained models fail despite fitting the training lines.\n\n"
        "Color convention in preview images:\n\n"
        "- Yellow: current Line9 label copied from v1.4.\n"
        "- Blue: frozen v1.9D primary prediction.\n"
        "- Red: V16-trained prediction on Line9 holdout.\n\n"
        "Audit rule: keep basal/interface in the PDF-constrained `12-16 m` band and do not label the deeper `20-23 m` anomaly as basal.\n",
        encoding="utf-8",
    )
    print(out_csv)
    print(OUT / "line9_v17_reaudit_manifest.json")
    print(OUT / "previews")


if __name__ == "__main__":
    main()
