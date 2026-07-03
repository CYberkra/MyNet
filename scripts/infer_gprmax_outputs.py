from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import zipfile

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.model_raw_unet import build_model, compress_raw
from scripts.eval_full_line import dp_ridge_centerline


def sha256_12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_checkpoint(run_dir, checkpoint, device):
    names = {
        "final": "checkpoint_final.pt",
        "best": "checkpoint_best.pt",
        "last": "checkpoint_last.pt",
    }
    ckpt_path = Path(run_dir) / names[checkpoint]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt_path


def unpack_model_output(out):
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    raise ValueError("model output must include mask and presence logits")


def read_gprmax_ez(path):
    with h5py.File(path, "r") as f:
        ez = f["rxs/rx1/Ez"][:].astype(np.float32)
        dt_ns = float(f.attrs["dt"]) * 1e9
        title = str(f.attrs.get("Title", Path(path).stem))
        iterations = int(f.attrs.get("Iterations", ez.shape[0]))
    return ez, dt_ns, title, iterations


def preprocess_ez(ez, mode):
    x = ez.astype(np.float32, copy=True)
    if mode in ("bgr_robust", "bgr_only"):
        x -= np.nanmedian(x, axis=1, keepdims=True).astype(np.float32)
    scale = float(np.nanpercentile(np.abs(x), 99.5))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    if mode in ("bgr_robust", "robust"):
        x = np.clip(x / scale, -2.0, 2.0)
    return x.astype(np.float32), scale


def infer_one(model, cfg, raw_norm, device):
    h0, w0 = raw_norm.shape
    h = int(cfg["height_resize"])
    w = int(cfg["width_resize"])
    log_scale = float(cfg.get("input_log_scale", 1e-3))
    x = torch.from_numpy(raw_norm[None, None]).float().to(device)
    x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
    x = compress_raw(x, log_scale)
    with torch.no_grad():
        logits, pres_logits = unpack_model_output(model(x))
        prob = torch.sigmoid(logits)
        prob = F.interpolate(prob, (h0, w0), mode="bilinear", align_corners=False)
        pres = torch.sigmoid(pres_logits)
        pres = F.interpolate(pres, size=w0, mode="linear", align_corners=False)
    return prob[0, 0].detach().cpu().numpy(), pres[0, 0].detach().cpu().numpy()


def infer_ensemble(model_specs, raw_norm, device):
    probs = []
    presses = []
    for model, cfg, _ in model_specs:
        prob, pres = infer_one(model, cfg, raw_norm, device)
        probs.append(prob)
        presses.append(pres)
    return np.mean(probs, axis=0).astype(np.float32), np.mean(presses, axis=0).astype(np.float32)


def pick_centerline(prob, pres, dt_ns, presence_thr, path_prob_thr, search_min_ns, search_max_ns):
    real_dt_ns = 1.4
    sample_scale = max(1.0, real_dt_ns / max(dt_ns, 1e-9))
    max_jump = max(8, int(round(8 * sample_scale)))
    smooth_weight = 0.08 / (sample_scale * sample_scale)
    search_min = int(round(float(search_min_ns) / dt_ns))
    search_max = int(round(float(search_max_ns) / dt_ns))
    cdp, vdp = dp_ridge_centerline(
        prob,
        max_jump=max_jump,
        smooth_weight=smooth_weight,
        min_presence=(pres >= presence_thr),
        search_min_sample=search_min,
        search_max_sample=search_max,
    )
    path_prob = np.full(prob.shape[1], np.nan, np.float32)
    final_valid = np.zeros(prob.shape[1], dtype=bool)
    for i in range(prob.shape[1]):
        if vdp[i] and np.isfinite(cdp[i]):
            y = int(np.clip(round(float(cdp[i])), 0, prob.shape[0] - 1))
            path_prob[i] = float(prob[y, i])
            final_valid[i] = bool(path_prob[i] >= path_prob_thr)
    cdp = cdp.copy()
    cdp[~final_valid] = np.nan
    return cdp, final_valid, path_prob, max_jump, smooth_weight


def write_trace_csv(path, dt_ns, pres, strict, visual):
    strict_c, strict_v, strict_p = strict
    visual_c, visual_v, visual_p = visual
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trace_idx",
                "presence_prob",
                "strict_valid",
                "strict_sample",
                "strict_time_ns",
                "strict_path_prob",
                "visual_valid",
                "visual_sample",
                "visual_time_ns",
                "visual_path_prob",
            ]
        )
        for i in range(len(pres)):
            sc = "" if not np.isfinite(strict_c[i]) else f"{strict_c[i]:.3f}"
            st = "" if not np.isfinite(strict_c[i]) else f"{strict_c[i] * dt_ns:.3f}"
            sp = "" if not np.isfinite(strict_p[i]) else f"{strict_p[i]:.6f}"
            vc = "" if not np.isfinite(visual_c[i]) else f"{visual_c[i]:.3f}"
            vt = "" if not np.isfinite(visual_c[i]) else f"{visual_c[i] * dt_ns:.3f}"
            vp = "" if not np.isfinite(visual_p[i]) else f"{visual_p[i]:.6f}"
            writer.writerow([i, f"{pres[i]:.6f}", int(strict_v[i]), sc, st, sp, int(visual_v[i]), vc, vt, vp])


def _font(size=14):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _gray_image(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lim = max(0.05, float(np.nanpercentile(np.abs(arr), 98.0)))
    x = np.clip((arr + lim) / (2 * lim), 0, 1)
    return Image.fromarray((x * 255).astype(np.uint8), "L").convert("RGB")


def _prob_image(arr):
    arr = np.asarray(arr, dtype=np.float32)
    scale = max(0.35, float(np.nanpercentile(arr, 99.5)))
    x = np.clip(arr / scale, 0, 1)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * x).astype(np.uint8)
    rgb[..., 1] = (190 * np.clip(1 - np.abs(x - 0.55) * 1.8, 0, 1)).astype(np.uint8)
    rgb[..., 2] = (55 * (1 - x)).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _draw_centerline(draw, centers, sx, sy, ox, oy, color, width):
    seg = []
    for x, y in enumerate(centers):
        if np.isfinite(y):
            seg.append((ox + x * sx, oy + float(y) * sy))
        else:
            if len(seg) >= 2:
                draw.line(seg, fill=color, width=width)
            seg = []
    if len(seg) >= 2:
        draw.line(seg, fill=color, width=width)


def _draw_hline_time(draw, time_ns, dt_ns, sx, sy, width_px, oy, color):
    y = oy + (float(time_ns) / max(float(dt_ns), 1e-9)) * sy
    for x0 in range(0, width_px, 12):
        draw.line((x0, y, min(width_px - 1, x0 + 6), y), fill=color, width=1)


def plot_case(out_png, case_id, raw_norm, prob, pres, dt_ns, strict_c, visual_c, thresholds):
    h, w = raw_norm.shape
    view_w = 1280
    panel_h = 210
    label_h = 22
    gap = 14
    plot_h = 105
    total_h = 38 + 3 * (label_h + panel_h) + 2 * gap + label_h + plot_h + 18
    canvas = Image.new("RGB", (view_w, total_h), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    small = _font(13)
    sx = view_w / max(w, 1)
    sy = panel_h / max(h, 1)

    draw.text((12, 10), f"{case_id} | strict=yellow, visual=cyan", fill=(20, 20, 20), font=font)
    raw_img = _gray_image(raw_norm)
    prob_img = _prob_image(prob)
    overlay = Image.blend(raw_img, prob_img, 0.42)
    panels = [("normalized Ez", raw_img), ("network probability", prob_img), ("overlay + centerlines", overlay)]
    y = 38
    for label, img in panels:
        draw.text((8, y + 3), label, fill=(25, 25, 25), font=small)
        iy = y + label_h
        canvas.paste(img.resize((view_w, panel_h), Image.Resampling.NEAREST), (0, iy))
        draw.rectangle((0, iy, view_w - 1, iy + panel_h - 1), outline=(210, 210, 205), width=1)
        _draw_hline_time(draw, thresholds["search_min_ns"], dt_ns, sx, sy, view_w, iy, (148, 163, 184))
        _draw_hline_time(draw, thresholds["search_max_ns"], dt_ns, sx, sy, view_w, iy, (148, 163, 184))
        if label.startswith("overlay"):
            _draw_centerline(draw, visual_c, sx, sy, 0, iy, (72, 215, 255), 2)
            _draw_centerline(draw, strict_c, sx, sy, 0, iy, (255, 191, 63), 3)
        y = iy + panel_h + gap

    draw.text((8, y + 3), "presence probability", fill=(25, 25, 25), font=small)
    axis_top = y + label_h
    draw.rectangle((0, axis_top, view_w - 1, axis_top + plot_h), outline=(210, 210, 205), width=1)
    for thr, color in [(thresholds["presence_thr"], (255, 191, 63)), (thresholds["visual_presence_thr"], (72, 215, 255))]:
        yy = axis_top + plot_h - int(float(thr) * plot_h)
        for x0 in range(0, view_w, 12):
            draw.line((x0, yy, min(view_w - 1, x0 + 6), yy), fill=color, width=1)
    pts = [(i * sx, axis_top + plot_h - float(np.clip(v, 0, 1)) * plot_h) for i, v in enumerate(pres)]
    if len(pts) >= 2:
        draw.line(pts, fill=(51, 65, 85), width=2)
    strict_rate = float(np.isfinite(strict_c).mean()) if len(strict_c) else 0.0
    visual_rate = float(np.isfinite(visual_c).mean()) if len(visual_c) else 0.0
    draw.text((view_w - 270, axis_top + 8), f"strict {strict_rate:.1%} | visual {visual_rate:.1%}", fill=(20, 20, 20), font=small)
    canvas.save(out_png)


def make_overview(preview_paths, out_png):
    images = []
    for p in preview_paths:
        img = Image.open(p).convert("RGB")
        images.append((Path(p).stem.replace("_network_preview", ""), img))
    if not images:
        return
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    thumb_w = 900
    label_h = 24
    thumbs = []
    for name, img in images:
        thumb_h = max(1, int(img.height * thumb_w / img.width))
        thumbs.append((name, img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)))
    cell_h = max(img.height for _, img in thumbs) + label_h
    overview = Image.new("RGB", (cols * thumb_w, rows * cell_h), (250, 250, 248))
    draw = ImageDraw.Draw(overview)
    small = _font(14)
    for i, (name, img) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = c * thumb_w
        y = r * cell_h
        draw.text((x + 8, y + 4), name, fill=(20, 20, 20), font=small)
        overview.paste(img, (x, y + label_h))
    overview.save(out_png)


def zip_dir(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(Path(src_dir).rglob("*")):
            if path.is_file() and path.resolve() != Path(zip_path).resolve():
                zf.write(path, path.relative_to(src_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gprmax-dir", required=True)
    ap.add_argument("--run-dir", default=str(ROOT / "outputs" / "run_gpu_final_line9dev"))
    ap.add_argument("--run-dirs", nargs="+", default=None, help="Optional ensemble run directories. Overrides --run-dir when provided.")
    ap.add_argument("--checkpoint", choices=["best", "last", "final"], default="final")
    ap.add_argument(
        "--threshold-json",
        default=str(ROOT / "reports" / "v1_4_ensemble_postprocess_calibration" / "v1_4_abstain_postprocess_thresholds.json"),
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preprocess", choices=["bgr_robust", "robust", "bgr_only", "none"], default="bgr_robust")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = out_dir / "arrays"
    previews_dir = out_dir / "previews"
    traces_dir = out_dir / "trace_picks"
    arrays_dir.mkdir(exist_ok=True)
    previews_dir.mkdir(exist_ok=True)
    traces_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    run_dirs = args.run_dirs if args.run_dirs else [args.run_dir]
    model_specs = [load_checkpoint(run_dir, args.checkpoint, device) for run_dir in run_dirs]
    thresholds = json.loads(Path(args.threshold_json).read_text(encoding="utf-8"))
    thresholds["visual_presence_thr"] = 0.30
    thresholds["visual_path_prob_thr"] = 0.20

    out_files = sorted(Path(args.gprmax_dir).glob("*_merged.out"))
    if not out_files:
        raise FileNotFoundError(f"No *_merged.out files found in {args.gprmax_dir}")

    summaries = []
    preview_paths = []
    for path in out_files:
        case_id = path.stem.replace("_merged", "")
        print(f"Running {case_id}...", flush=True)
        ez, dt_ns, title, iterations = read_gprmax_ez(path)
        raw_norm, raw_scale = preprocess_ez(ez, args.preprocess)
        prob, pres = infer_ensemble(model_specs, raw_norm, device)
        strict_c, strict_v, strict_p, max_jump, smooth_weight = pick_centerline(
            prob,
            pres,
            dt_ns,
            thresholds["presence_thr"],
            thresholds["path_prob_thr"],
            thresholds["search_min_ns"],
            thresholds["search_max_ns"],
        )
        visual_c, visual_v, visual_p, _, _ = pick_centerline(
            prob,
            pres,
            dt_ns,
            thresholds["visual_presence_thr"],
            thresholds["visual_path_prob_thr"],
            thresholds["search_min_ns"],
            thresholds["search_max_ns"],
        )
        np.savez_compressed(
            arrays_dir / f"{case_id}_network_outputs.npz",
            prob=prob.astype(np.float32),
            presence=pres.astype(np.float32),
            raw_norm=raw_norm.astype(np.float32),
            dt_ns=np.float32(dt_ns),
            strict_center_sample=strict_c.astype(np.float32),
            visual_center_sample=visual_c.astype(np.float32),
        )
        write_trace_csv(
            traces_dir / f"{case_id}_trace_picks.csv",
            dt_ns,
            pres,
            (strict_c, strict_v, strict_p),
            (visual_c, visual_v, visual_p),
        )
        preview_png = previews_dir / f"{case_id}_network_preview.png"
        plot_case(preview_png, case_id, raw_norm, prob, pres, dt_ns, strict_c, visual_c, thresholds)
        preview_paths.append(preview_png)
        summaries.append(
            {
                "case_id": case_id,
                "title": title,
                "source_file": str(path),
                "iterations": iterations,
                "height": int(ez.shape[0]),
                "width": int(ez.shape[1]),
                "dt_ns": dt_ns,
                "preprocess": args.preprocess,
                "ensemble_size": len(model_specs),
                "raw_scale_p995": raw_scale,
                "prob_mean": float(np.nanmean(prob)),
                "prob_p95": float(np.nanpercentile(prob, 95)),
                "prob_max": float(np.nanmax(prob)),
                "presence_mean": float(np.nanmean(pres)),
                "presence_max": float(np.nanmax(pres)),
                "strict_pick_rate": float(np.nanmean(strict_v)),
                "visual_pick_rate": float(np.nanmean(visual_v)),
                "strict_median_time_ns": float(np.nanmedian(strict_c * dt_ns)) if np.isfinite(strict_c).any() else float("nan"),
                "visual_median_time_ns": float(np.nanmedian(visual_c * dt_ns)) if np.isfinite(visual_c).any() else float("nan"),
                "dp_max_jump_samples": int(max_jump),
                "dp_smooth_weight": float(smooth_weight),
            }
        )

    summary_csv = out_dir / "gprmax_network_inference_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    overview_png = out_dir / "gprmax_network_inference_overview.png"
    make_overview(preview_paths, overview_png)

    manifest = {
        "checkpoints": [str(ckpt_path) for _, _, ckpt_path in model_specs],
        "checkpoint_sha256_12": [sha256_12(ckpt_path) for _, _, ckpt_path in model_specs],
        "run_dirs": [str(Path(run_dir)) for run_dir in run_dirs],
        "threshold_json": str(Path(args.threshold_json)),
        "thresholds": thresholds,
        "gprmax_dir": str(Path(args.gprmax_dir)),
        "out_dir": str(out_dir),
        "device": str(device),
        "configs": [cfg for _, cfg, _ in model_specs],
        "ensemble_size": len(model_specs),
        "cases": [s["case_id"] for s in summaries],
        "note": "Inference-only domain transfer from measured-data training to gprMax simulations; no simulated labels were used.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = out_dir.with_suffix(".zip")
    zip_dir(out_dir, zip_path)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {overview_png}")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
