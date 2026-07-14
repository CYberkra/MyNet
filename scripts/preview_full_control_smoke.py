#!/usr/bin/env python3
"""Render an auditable raw/processed/full-minus-control gprMax smoke preview."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def collect(case_dir: Path, stem: str, expected_trace_count: int) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(stem)}(\d+)\.out$")
    indexed = []
    for path in case_dir.glob(f"{stem}*.out"):
        match = pattern.match(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    expected = list(range(1, expected_trace_count + 1))
    if [index for index, _ in indexed] != expected:
        raise RuntimeError(f"Expected {stem} outputs {expected} in {case_dir}")
    return [path for _, path in indexed]


def stack(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    columns: list[np.ndarray] = []
    time_ns: np.ndarray | None = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle["rxs"]["rx1"]["Ez"], dtype=np.float64)
            local_time = np.arange(values.size, dtype=np.float64) * float(handle.attrs["dt"]) * 1e9
        if time_ns is None:
            time_ns = local_time
        elif values.shape != columns[0].shape or not np.allclose(time_ns, local_time):
            raise RuntimeError(f"Inconsistent time contract in {path}")
        columns.append(values)
    if time_ns is None:
        raise RuntimeError("No receiver outputs")
    return np.column_stack(columns), time_ns


def process(matrix: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
    """Use display-only background suppression and a bounded t^1.5 gain."""
    background_removed = matrix - np.median(matrix, axis=1, keepdims=True)
    gain = np.power(np.clip(time_ns / max(float(time_ns[-1]), 1e-9), 0.0, 1.0), 1.5)
    return background_removed * gain[:, None]


def robust_scale(matrix: np.ndarray) -> float:
    return max(float(np.quantile(np.abs(matrix), 0.995)), np.finfo(np.float64).eps)


def seismic_rgb(matrix: np.ndarray, scale: float) -> np.ndarray:
    unit = np.clip(matrix / scale, -1.0, 1.0)
    red = np.where(unit >= 0.0, 255, np.rint(255 * (1.0 + unit)))
    blue = np.where(unit <= 0.0, 255, np.rint(255 * (1.0 - unit)))
    green = np.rint(255 * (1.0 - np.abs(unit)))
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def visible_phase_peaks(difference: np.ndarray, time_ns: np.ndarray, reference_ns: np.ndarray) -> np.ndarray:
    peaks = []
    for column, centre in enumerate(reference_ns):
        target = np.flatnonzero(np.abs(time_ns - centre) <= 55.0)
        if not target.size:
            raise RuntimeError(f"Reference {centre} ns lies outside the simulated time window")
        local = target[np.argmax(np.abs(difference[target, column]))]
        peaks.append(float(time_ns[local]))
    return np.asarray(peaks, dtype=np.float64)


def render_panel(matrix: np.ndarray, scale: float, width: int, height: int) -> Image.Image:
    return Image.fromarray(seismic_rgb(matrix, scale), mode="RGB").resize((width, height), Image.Resampling.BILINEAR)


def overlay(draw: ImageDraw.ImageDraw, x0: int, y0: int, width: int, height: int, time_ns: np.ndarray, arrivals: np.ndarray, colour: str, label: str) -> None:
    points = []
    for trace_index, value in enumerate(arrivals):
        x = x0 + trace_index / max(len(arrivals) - 1, 1) * (width - 1)
        y = y0 + float(value / time_ns[-1]) * (height - 1)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=colour, width=2)
    draw.text((x0 + 8, y0 + 26), label, fill=colour, font=ImageFont.load_default())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--full-stem", default="full_scene")
    parser.add_argument("--control-stem", default="no_basal_contrast_control")
    parser.add_argument("--arrival-label", type=Path, default=Path("labels/geometric_reference_arrival_time_ns.npy"))
    parser.add_argument("--expected-trace-count", type=int, required=True)
    parser.add_argument("--output-stem", default="full_control_smoke")
    parser.add_argument("--continuity-audit", type=Path)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    full, time_ns = stack(collect(case_dir, args.full_stem, args.expected_trace_count))
    control, control_time_ns = stack(collect(case_dir, args.control_stem, args.expected_trace_count))
    if full.shape != control.shape or not np.allclose(time_ns, control_time_ns):
        raise RuntimeError("Full/control output time contract differs")
    reference = np.asarray(np.load(case_dir / args.arrival_label), dtype=np.float64)[: args.expected_trace_count]
    if reference.shape != (args.expected_trace_count,):
        raise RuntimeError("Reference-arrival label count does not match smoke trace count")
    processed = process(full, time_ns)
    difference = full - control
    peaks = visible_phase_peaks(difference, time_ns, reference)
    peak_source = "per-trace signed-pair peak"
    if args.continuity_audit:
        audited = json.loads(args.continuity_audit.resolve().read_text(encoding="utf-8"))
        path = audited["metrics"]["path_constrained_signed_difference_peak_ns"]
        peaks = np.asarray(path, dtype=np.float64)
        if peaks.shape != reference.shape:
            raise RuntimeError("Continuity-audit path count does not match preview trace count")
        peak_source = "path-constrained signed-pair candidate"
    late_window = time_ns >= 120.0
    panels = (
        ("raw receiver field (late-window display scale)", full, robust_scale(full[late_window])),
        ("background removed + t^1.5", processed, robust_scale(processed)),
        ("signed full - no-basal", difference, robust_scale(difference)),
    )
    width, height, left, top = 640, 660, 68, 86
    canvas = Image.new("RGB", (left + 3 * width, top + height + 98), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((left, 22), f"{case_dir.name}: {args.expected_trace_count}-trace runtime pair (display processing only)", fill="black", font=font)
    for panel_index, (name, matrix, scale) in enumerate(panels):
        x0 = left + panel_index * width
        canvas.paste(render_panel(matrix, scale, width, height), (x0, top))
        draw.rectangle((x0, top, x0 + width - 1, top + height - 1), outline="black", width=1)
        draw.text((x0 + 8, top + 8), f"{name}; p99.5={scale:.3g}", fill="black", font=font)
        overlay(draw, x0, top, width, height, time_ns, reference, "gold", "gold: material-interface reference")
        overlay(draw, x0, top, width, height, time_ns, peaks, "cyan", f"cyan: {peak_source}")
    draw.text((left, top + height + 18), "Gold is a geometric/material reference, not a visible-phase training label. Cyan is extracted only from this strict full-control smoke pair.", fill="black", font=font)
    draw.text((left, top + height + 40), "The family remains development-only until 256-trace runs, convergence checks, visible-phase extraction, and human review pass.", fill="black", font=font)
    canvas.save(report_dir / f"{args.output_stem}.png")
    metrics = {
        "case_id": case_dir.name,
        "trace_count": args.expected_trace_count,
        "time_samples": int(full.shape[0]),
        "dt_ns": float(time_ns[1] - time_ns[0]),
        "reference_arrival_ns": reference.tolist(),
        "full_minus_control_peak_ns": peaks.tolist(),
        "reference_to_peak_offset_ns": (peaks - reference).tolist(),
        "peak_source": peak_source,
        "scales": {name: scale for name, _, scale in panels},
    }
    (report_dir / f"{args.output_stem}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(report_dir / f"{args.output_stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
