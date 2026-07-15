#!/usr/bin/env python3
"""Render raw and background-suppressed AGC views of a solved gprMax B-scan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import resample_time_axis  # noqa: E402
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402


def agc_along_time(values: np.ndarray, window: int = 13) -> np.ndarray:
    """Apply trace-local RMS AGC after horizontal common-mode suppression."""
    if window < 3 or window % 2 == 0:
        raise ValueError("AGC window must be an odd integer of at least three samples")
    x = np.asarray(values, dtype=np.float64)
    pad = window // 2
    padded = np.pad(np.square(x), ((pad, pad), (0, 0)), mode="edge")
    cumulative = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(padded, axis=0)))
    rms = np.sqrt((cumulative[window:] - cumulative[:-window]) / window)
    floor = max(float(np.median(rms)) * 0.06, np.finfo(np.float64).tiny)
    return (x / (rms + floor)).astype(np.float32)


def time_power_gain(values: np.ndarray, time_ns: np.ndarray, power: float) -> np.ndarray:
    """Apply a monotone depth-time display gain without trace-local AGC."""
    if power < 0:
        raise ValueError("time-power gain exponent must be non-negative")
    time = np.asarray(time_ns, dtype=np.float64)
    if time.ndim != 1 or time.size != values.shape[0] or time[-1] <= 0:
        raise ValueError("time axis must match values and end after zero")
    gain = np.power(np.clip(time / time[-1], 0.0, 1.0), power)
    return (np.asarray(values, dtype=np.float64) * gain[:, None]).astype(np.float32)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _panel(values: np.ndarray, scale: float, width: int, height: int) -> Image.Image:
    clipped = np.clip(values / max(scale, 1e-30), -1.0, 1.0)
    gray = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )


def _overlay_curve(
    draw: ImageDraw.ImageDraw,
    curve_ns: np.ndarray | None,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    max_time_ns: float,
) -> None:
    if curve_ns is None:
        return
    x = left + np.linspace(0, width - 1, curve_ns.size)
    y = top + np.clip(curve_ns / max_time_ns, 0, 1) * (height - 1)
    draw.line([(float(px), float(py)) for px, py in zip(x, y)], fill=(255, 0, 255), width=3)


def render_preview(
    run_dir: Path,
    output_path: Path,
    *,
    visible_phase_path: Path | None = None,
    component: str = "Ez",
    agc_window: int = 13,
    processing: str = "agc",
    time_power: float = 1.5,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    output_path = output_path.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    dt_s, raw, _ = read_merged_bscan(run_dir / "full_scene_merged.out", component=component)
    source_end_ns = (raw.shape[0] - 1) * dt_s * 1e9
    protected_end_ns = min(
        float(scene.get("grid", {}).get("protected_window_end_ns", 500.0)),
        source_end_ns,
    )
    output_samples = max(2, int(round(protected_end_ns / 1.4)) + 1)
    protected_time, raw = resample_time_axis(
        raw,
        dt_s,
        time_window_ns=protected_end_ns,
        output_samples=output_samples,
    )
    background_removed = raw - np.median(raw, axis=1, keepdims=True)
    if processing == "agc":
        processed = agc_along_time(background_removed, window=agc_window)
        processed_title = f"Horizontal background suppression + AGC({agc_window})"
        processing_note = "AGC changes amplitude balance and must not be used as a physical-amplitude comparison."
    elif processing == "time-power":
        processed = time_power_gain(background_removed, protected_time, time_power)
        processed_title = f"Horizontal background suppression + normalized time^{time_power:g} gain"
        processing_note = "Time-power gain is monotone with time and is for structural display only, not physical-amplitude comparison."
    else:
        raise ValueError("processing must be 'agc' or 'time-power'")
    declared_count = int(scene["grid"]["trace_count"])
    curve = None
    if visible_phase_path is not None:
        candidate = np.load(visible_phase_path)
        if candidate.shape == (raw.shape[1],):
            curve = candidate.astype(np.float64)
        elif candidate.shape == (declared_count,):
            indices = np.asarray(manifest.get("selected_trace_indices_zero_based", []), dtype=np.int64)
            if indices.shape == (raw.shape[1],):
                curve = candidate[indices].astype(np.float64)
    raw_scale = float(np.quantile(np.abs(raw), 0.995))
    processed_scale = float(np.quantile(np.abs(processed), 0.995))
    width, height = 1760, 850
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = _font(31, True)
    body = _font(20)
    panel_w, panel_h = 790, 590
    boxes = ((70, 170), (900, 170))
    titles = (
        "Raw full-scene Ez (native amplitude gain)",
        processed_title,
    )
    values = (raw, processed)
    scales = (raw_scale, processed_scale)
    selected = manifest.get("selected_trace_indices_zero_based", [])
    spacing = float(scene["grid"]["trace_spacing_m"])
    stride = int(manifest.get("trace_stride", 1))
    run_kind = "full run" if raw.shape[1] == declared_count and stride == 1 else "subset"
    draw.text(
        (70, 32),
        f"{scene['case_id']} - raw versus background-suppressed {processing}",
        fill="black",
        font=title,
    )
    draw.text(
        (70, 78),
        f"{raw.shape[1]}-trace {run_kind}; canonical indices {selected[0] if selected else '?'} to {selected[-1] if selected else '?'}; "
        f"effective trace spacing {spacing * stride:.2f} m. Magenta: solved visible-phase reference.",
        fill="black",
        font=body,
    )
    draw.text((70, 112), f"Display only: {processing_note}", fill="black", font=body)
    for (left, top), panel_title, value, scale in zip(boxes, titles, values, scales):
        draw.text((left, top - 32), panel_title, fill="black", font=body)
        canvas.paste(_panel(value, scale, panel_w, panel_h), (left, top))
        draw.rectangle((left, top, left + panel_w, top + panel_h), outline="black", width=2)
        _overlay_curve(
            draw,
            curve,
            left=left,
            top=top,
            width=panel_w,
            height=panel_h,
            max_time_ns=protected_end_ns,
        )
        draw.text((left + 8, top + 8), "0 ns", fill="white", font=body)
        draw.text((left + 8, top + panel_h - 30), "500 ns", fill="white", font=body)
        draw.text((left, top + panel_h + 12), f"P99.5 display gain = {scale:.3e}", fill="black", font=body)
    draw.text(
        (70, 800),
        f"Raw panel preserves direct-wave dominance. Processed panel is for structure reading only; both use the same 0-{protected_end_ns:g} ns protected window.",
        fill="black",
        font=body,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "output": str(output_path),
        "raw_shape": list(raw.shape),
        "raw_p995_gain": raw_scale,
        "processed_p995_gain": processed_scale,
        "agc_window": agc_window,
        "processing": processing,
        "time_power": time_power if processing == "time-power" else None,
        "visible_phase_overlay": curve is not None,
        "source_end_ns": source_end_ns,
        "protected_end_ns": protected_end_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visible-phase", type=Path)
    parser.add_argument("--component", default="Ez")
    parser.add_argument("--agc-window", type=int, default=13)
    parser.add_argument("--processing", choices=("agc", "time-power"), default="agc")
    parser.add_argument("--time-power", type=float, default=1.5)
    args = parser.parse_args()
    print(json.dumps(render_preview(args.run_dir, args.output, visible_phase_path=args.visible_phase, component=args.component, agc_window=args.agc_window, processing=args.processing, time_power=args.time_power), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
