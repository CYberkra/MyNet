"""Render raw and display-processed MACRO06/Line9 panels without Matplotlib."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.postprocess_physical_sim_v2 import read_merged_bscan


def robust_limit(values: np.ndarray, quantile: float = 0.995) -> float:
    return float(max(np.quantile(np.abs(values), quantile), np.finfo(np.float64).eps))


def seismic_rgb(values: np.ndarray, limit: float) -> np.ndarray:
    unit = np.clip(values / limit, -1.0, 1.0)
    red = np.where(unit >= 0.0, 255, np.rint(255 * (1.0 + unit)))
    blue = np.where(unit <= 0.0, 255, np.rint(255 * (1.0 - unit)))
    green = np.rint(255 * (1.0 - np.abs(unit)))
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def resample_time(values: np.ndarray, target_samples: int) -> np.ndarray:
    old = np.linspace(0.0, 1.0, values.shape[0])
    new = np.linspace(0.0, 1.0, target_samples)
    return np.column_stack([np.interp(new, old, values[:, column]) for column in range(values.shape[1])])


def suppress_and_gain(values: np.ndarray, time_ns: np.ndarray, exponent: float) -> np.ndarray:
    suppressed = values - np.median(values, axis=1, keepdims=True)
    gain = np.power(np.clip(time_ns / max(time_ns[-1], 1e-9), 0.02, 1.0), exponent)
    return suppressed * gain[:, None]


def read_simulation(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    merged = case_dir / "full_scene_merged.out"
    if merged.is_file():
        dt_s, values, _ = read_merged_bscan(merged, component="Ez")
        time_ns = np.arange(values.shape[0], dtype=np.float64) * dt_s * 1e9
        return np.asarray(values, dtype=np.float64), time_ns
    pattern = re.compile(r"^full_scene(\d+)\.out$")
    paths = []
    for path in case_dir.glob("full_scene*.out"):
        match = pattern.match(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths.sort()
    if len(paths) < 8:
        raise RuntimeError("MACRO06 requires at least eight completed full-scene traces")
    columns = []
    dt = None
    for _, path in paths:
        with h5py.File(path, "r") as h5:
            columns.append(np.asarray(h5["rxs"]["rx1"]["Ez"], dtype=np.float64))
            dt = float(h5.attrs["dt"])
    values = np.column_stack(columns)
    time_ns = np.arange(values.shape[0]) * dt * 1e9
    return values, time_ns


def panel(canvas: Image.Image, matrix: np.ndarray, title: str, x: int, y: int, width: int, height: int) -> None:
    image = Image.fromarray(seismic_rgb(matrix, robust_limit(matrix))).resize(
        (width, height), Image.Resampling.NEAREST
    )
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + width - 1, y + height - 1), outline="black", width=1)
    draw.text((x + 8, y + 8), title, fill="black")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--line9", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-power", type=float, default=1.6)
    parser.add_argument("--max-time-ns", type=float, default=500.0)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    sim_raw, sim_time = read_simulation(case_dir)
    scene = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    run = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    with np.load(args.line9.resolve(), allow_pickle=False) as line9:
        real_raw = np.asarray(line9["raw_amplitude"], dtype=np.float64)
        real_time = np.asarray(line9["time_ns"], dtype=np.float64)
        distance = np.asarray(line9["gnss_cumulative_distance_m"], dtype=np.float64)

    # Compare an equal physical-width segment, not an arbitrary equal trace count.
    selected = np.asarray(run.get("selected_trace_indices_zero_based", []), dtype=np.int64)
    spacing_m = float(scene["grid"]["trace_spacing_m"])
    if selected.shape == (sim_raw.shape[1],) and selected.size > 1:
        sim_span_m = float(selected[-1] - selected[0]) * spacing_m
    else:
        sim_span_m = (sim_raw.shape[1] - 1) * spacing_m * int(run.get("trace_stride", 1))
    real_indices = np.flatnonzero(distance <= distance[0] + sim_span_m)
    real_raw = real_raw[:, real_indices]
    common_end_ns = min(float(args.max_time_ns), float(sim_time[-1]), float(real_time[-1]))
    real_rows = real_time <= common_end_ns
    real_time = real_time[real_rows]
    real_raw = real_raw[real_rows]
    sim_raw = np.column_stack(
        [np.interp(real_time, sim_time, sim_raw[:, column]) for column in range(sim_raw.shape[1])]
    )
    sim_time = real_time.copy()
    sim_processed = suppress_and_gain(sim_raw, sim_time, args.time_power)
    real_processed = suppress_and_gain(real_raw, real_time, args.time_power)

    width, height, margin, header = 620, 610, 28, 58
    canvas = Image.new("RGB", (margin * 3 + width * 2, header + margin * 3 + height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 16),
        f"{scene['case_id']} vs Line9: raw and background-suppressed time-power gain (t^{args.time_power:g})",
        fill="black",
    )
    panel(canvas, sim_raw, f"Simulation raw: {sim_raw.shape[1]} traces / {sim_span_m:.1f} m", margin, header, width, height)
    panel(canvas, real_raw, f"Line9 raw: {real_raw.shape[1]} traces / {sim_span_m:.1f} m", margin * 2 + width, header, width, height)
    panel(canvas, sim_processed, "Simulation: background suppression + time-power gain", margin, header + margin + height, width, height)
    panel(canvas, real_processed, "Line9: background suppression + time-power gain", margin * 2 + width, header + margin + height, width, height)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)


if __name__ == "__main__":
    main()
