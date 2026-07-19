#!/usr/bin/env python3
"""Render a sparse full/no-basal run without a GUI or interpolation claims."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "basal_shape_family_pilot_20260720"


def read(path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["rxs"]["rx1"]["Ez"], dtype=np.float32), float(handle.attrs["dt"])


def colour(values: np.ndarray, scale: float) -> np.ndarray:
    x = np.clip(values / max(scale, 1e-12), -1.0, 1.0)
    r = np.where(x >= 0, 255, 255 * (1 + x)).astype(np.uint8)
    b = np.where(x <= 0, 255, 255 * (1 - x)).astype(np.uint8)
    g = (255 * (1 - np.abs(x))).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def panel(values: np.ndarray, width: int, height: int) -> Image.Image:
    # Solver arrays are time x trace. Keep traces on the horizontal axis and
    # time on the vertical axis; transposing here creates a false vertical
    # striping impression in sparse pilots.
    values = np.flipud(values)
    scale = float(np.percentile(np.abs(values), 99.5))
    rgb = colour(values, scale)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((width, height), Image.Resampling.NEAREST)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--time-min-ns", type=float, default=0.0)
    parser.add_argument("--time-max-ns", type=float, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output or DEFAULT_OUT / (run_dir.name + "_bscan_preview.png")).resolve()
    full, dt = read(run_dir / "full_scene_merged.out")
    control, dt_control = read(run_dir / "no_basal_contrast_control_merged.out")
    if full.shape != control.shape or abs(dt - dt_control) > 1e-15:
        raise RuntimeError("full/control merged outputs are not shape-compatible")
    time_ns = np.arange(full.shape[0], dtype=np.float32) * dt * 1e9
    time_max_ns = float(time_ns[-1] if args.time_max_ns is None else args.time_max_ns)
    if not 0.0 <= args.time_min_ns < time_max_ns <= float(time_ns[-1]):
        raise ValueError("requested time crop is outside the merged output")
    crop = (time_ns >= args.time_min_ns) & (time_ns <= time_max_ns)
    full = full[crop]
    control = control[crop]
    time_ns = time_ns[crop]
    # Display-only processing: dewow-like trace background removal and bounded
    # power gain. The causal signed difference remains the audit quantity.
    background = gaussian_filter1d(full, sigma=14.0, axis=1, mode="nearest")
    processed = full - background
    gain = np.clip(np.maximum(time_ns - 50.0, 0.0) / 250.0, 0.0, 2.0) ** 1.35
    processed = processed * gain[:, None]
    signed = full - control
    width, height = 1280, 410
    canvas = Image.new("RGB", (width, height * 3 + 100), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (name, values) in enumerate((("raw full", full), ("display full: background removal + t^1.35 gain", processed), ("signed full - no-basal", signed))):
        top = 35 + i * height
        canvas.paste(panel(values, width, height), (0, top))
        draw.text((12, top + 8), name, fill="black", font=font)
        ticks = [args.time_min_ns, 400.0, time_max_ns]
        for ns in ticks:
            if not args.time_min_ns <= ns <= time_max_ns:
                continue
            y = top + height - int((ns - args.time_min_ns) / max(time_max_ns - args.time_min_ns, 1.0) * height)
            draw.line((0, y, width, y), fill=(255, 120, 0), width=1)
            draw.text((width - 60, y - 10), f"{ns} ns", fill=(255, 120, 0), font=font)
    draw.text((12, height * 3 + 70), f"sparse diagnostic only | traces={full.shape[1]} | dt={dt * 1e9:.4f} ns | no interpolation for metrics", fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
