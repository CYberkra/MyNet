#!/usr/bin/env python3
"""Compare two solved gprMax full scenes at identical canonical trace positions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import resample_time_axis  # noqa: E402
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402
from scripts.preview_gprmax_raw_and_agc import agc_along_time, time_power_gain  # noqa: E402


@dataclass(frozen=True)
class SolvedRun:
    case_id: str
    raw: np.ndarray
    time_ns: np.ndarray
    selected_indices: np.ndarray
    trace_spacing_m: float


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def common_trace_columns(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted common canonical indices and their columns in each run."""

    left_values = np.asarray(left, dtype=np.int64)
    right_values = np.asarray(right, dtype=np.int64)
    if left_values.ndim != 1 or right_values.ndim != 1:
        raise ValueError("selected trace indices must be one-dimensional")
    if np.unique(left_values).size != left_values.size or np.unique(right_values).size != right_values.size:
        raise ValueError("selected trace indices must be unique")
    common = np.intersect1d(left_values, right_values)
    if common.size < 2:
        raise ValueError("comparison requires at least two common canonical trace positions")
    left_lookup = {int(value): index for index, value in enumerate(left_values)}
    right_lookup = {int(value): index for index, value in enumerate(right_values)}
    return (
        common,
        np.asarray([left_lookup[int(value)] for value in common], dtype=np.int64),
        np.asarray([right_lookup[int(value)] for value in common], dtype=np.int64),
    )


def shared_quantile_scale(left: np.ndarray, right: np.ndarray, quantile: float = 0.995) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    combined = np.concatenate((np.abs(left).ravel(), np.abs(right).ravel()))
    return max(float(np.quantile(combined, quantile)), np.finfo(np.float64).tiny)


def _load_run(
    output_path: Path,
    run_manifest_path: Path,
    scene_manifest_path: Path,
    *,
    component: str,
    time_window_ns: float,
    output_samples: int,
) -> SolvedRun:
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    selected = np.asarray(run["selected_trace_indices_zero_based"], dtype=np.int64)
    dt_s, raw, _ = read_merged_bscan(output_path, component=component)
    if raw.shape[1] != selected.size:
        raise ValueError(f"{output_path} trace count does not match its run manifest")
    time_ns, canonical = resample_time_axis(
        raw,
        dt_s,
        time_window_ns=time_window_ns,
        output_samples=output_samples,
    )
    return SolvedRun(
        case_id=str(scene["case_id"]),
        raw=np.asarray(canonical, dtype=np.float64),
        time_ns=np.asarray(time_ns, dtype=np.float64),
        selected_indices=selected,
        trace_spacing_m=float(scene["grid"]["trace_spacing_m"]),
    )


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(font), size=size)
    except OSError:
        return ImageFont.load_default()


def _gray_panel(values: np.ndarray, scale: float, width: int, height: int) -> Image.Image:
    unit = np.clip(values / scale, -1.0, 1.0)
    gray = np.rint((unit + 1.0) * 127.5).astype(np.uint8)
    image = Image.fromarray(gray, mode="L").convert("RGB")
    image = image.resize((width, image.height), Image.Resampling.NEAREST)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _load_reference(path: Path | None, common: np.ndarray) -> np.ndarray | None:
    if path is None:
        return None
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.ndim != 1 or int(np.max(common)) >= values.size:
        raise ValueError(f"reference {path} does not cover all common trace positions")
    return values[common]


def _overlay_reference(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray | None,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    time_window_ns: float,
) -> None:
    if values is None:
        return
    x = left + (np.arange(values.size, dtype=np.float64) + 0.5) / values.size * width
    y = top + np.clip(values / time_window_ns, 0.0, 1.0) * (height - 1)
    draw.line([(float(px), float(py)) for px, py in zip(x, y)], fill=(255, 0, 255), width=3)


def compare(
    *,
    left_output: Path,
    left_run_manifest: Path,
    left_scene_manifest: Path,
    right_output: Path,
    right_run_manifest: Path,
    right_scene_manifest: Path,
    output_path: Path,
    metrics_path: Path,
    left_reference: Path | None,
    right_reference: Path | None,
    component: str,
    time_window_ns: float,
    output_samples: int,
    agc_window: int,
    time_power: float,
) -> dict[str, object]:
    left_run = _load_run(
        left_output,
        left_run_manifest,
        left_scene_manifest,
        component=component,
        time_window_ns=time_window_ns,
        output_samples=output_samples,
    )
    right_run = _load_run(
        right_output,
        right_run_manifest,
        right_scene_manifest,
        component=component,
        time_window_ns=time_window_ns,
        output_samples=output_samples,
    )
    if not np.allclose(left_run.time_ns, right_run.time_ns):
        raise ValueError("resampled time axes differ")
    common, left_columns, right_columns = common_trace_columns(
        left_run.selected_indices,
        right_run.selected_indices,
    )
    left_raw = left_run.raw[:, left_columns]
    right_raw = right_run.raw[:, right_columns]
    left_background = left_raw - np.median(left_raw, axis=1, keepdims=True)
    right_background = right_raw - np.median(right_raw, axis=1, keepdims=True)
    left_time_power = time_power_gain(left_background, left_run.time_ns, time_power)
    right_time_power = time_power_gain(right_background, right_run.time_ns, time_power)
    left_agc = agc_along_time(left_background, window=agc_window)
    right_agc = agc_along_time(right_background, window=agc_window)
    rows = (
        ("Raw full-scene Ez", left_raw, right_raw),
        (f"Background suppression + time^{time_power:g}", left_time_power, right_time_power),
        (f"Background suppression + AGC({agc_window})", left_agc, right_agc),
    )
    scales = {name: shared_quantile_scale(left, right) for name, left, right in rows}
    left_curve = _load_reference(left_reference, common)
    right_curve = _load_reference(right_reference, common)
    overlay_enabled = left_curve is not None or right_curve is not None

    panel_w, panel_h = 820, 340
    left_x, right_x, top = 120, 1010, 180
    row_gap = 430
    canvas = Image.new("RGB", (1920, 1500), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(31, True)
    heading_font = _font(24, True)
    body_font = _font(19)
    mode = "EXPLANATION WITH MATERIAL REFERENCES" if overlay_enabled else "BLIND - NO LABEL OR REFERENCE OVERLAY"
    draw.text((120, 28), "FORMAL06C versus FORMAL07A common-trace comparison", fill="black", font=title_font)
    draw.text((120, 75), mode, fill=(150, 0, 0) if overlay_enabled else "black", font=heading_font)
    draw.text(
        (120, 115),
        f"Canonical traces {common.tolist()}; shared 0-{time_window_ns:g} ns axis; shared P99.5 scale in every row; no horizontal interpolation.",
        fill="black",
        font=body_font,
    )
    draw.text((left_x, 148), left_run.case_id, fill="black", font=heading_font)
    draw.text((right_x, 148), right_run.case_id, fill="black", font=heading_font)
    for row_index, (name, left_values, right_values) in enumerate(rows):
        y = top + row_index * row_gap
        scale = scales[name]
        draw.text((25, y + 8), name, fill="black", font=body_font)
        for x, values, curve in (
            (left_x, left_values, left_curve),
            (right_x, right_values, right_curve),
        ):
            canvas.paste(_gray_panel(values, scale, panel_w, panel_h), (x, y))
            draw.rectangle((x, y, x + panel_w, y + panel_h), outline="black", width=2)
            _overlay_reference(
                draw,
                curve,
                left=x,
                top=y,
                width=panel_w,
                height=panel_h,
                time_window_ns=time_window_ns,
            )
            draw.text((x + 7, y + 7), "0 ns", fill="white", font=body_font)
            draw.text((x + 7, y + panel_h - 28), f"{time_window_ns:g} ns", fill="white", font=body_font)
        draw.text((left_x, y + panel_h + 8), f"Shared P99.5 scale: {scale:.4e}", fill="black", font=body_font)
    note = (
        "Magenta is a material/source reference and not a visible-phase training label."
        if overlay_enabled
        else "Judge continuity, wavelet character, target visibility, and background realism before opening the explanation figure."
    )
    draw.text((120, 1460), note, fill="black", font=body_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    result: dict[str, object] = {
        "schema": "gprmax_common_trace_comparison_v1",
        "left_case_id": left_run.case_id,
        "right_case_id": right_run.case_id,
        "common_canonical_trace_indices": common.tolist(),
        "common_trace_count": int(common.size),
        "time_window_ns": time_window_ns,
        "output_samples": output_samples,
        "component": component,
        "agc_window": agc_window,
        "time_power": time_power,
        "shared_p995_scales": scales,
        "reference_overlay": overlay_enabled,
        "horizontal_interpolation": "none; nearest-neighbour expansion of sparse trace columns",
        "left_output": _portable(left_output),
        "right_output": _portable(right_output),
        "output": _portable(output_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-output", type=Path, required=True)
        parser.add_argument(f"--{side}-run-manifest", type=Path, required=True)
        parser.add_argument(f"--{side}-scene-manifest", type=Path, required=True)
        parser.add_argument(f"--{side}-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--component", default="Ez")
    parser.add_argument("--time-window-ns", type=float, default=500.0)
    parser.add_argument("--output-samples", type=int, default=358)
    parser.add_argument("--agc-window", type=int, default=13)
    parser.add_argument("--time-power", type=float, default=1.5)
    args = parser.parse_args()
    result = compare(
        left_output=args.left_output,
        left_run_manifest=args.left_run_manifest,
        left_scene_manifest=args.left_scene_manifest,
        right_output=args.right_output,
        right_run_manifest=args.right_run_manifest,
        right_scene_manifest=args.right_scene_manifest,
        output_path=args.output,
        metrics_path=args.metrics,
        left_reference=args.left_reference,
        right_reference=args.right_reference,
        component=args.component,
        time_window_ns=args.time_window_ns,
        output_samples=args.output_samples,
        agc_window=args.agc_window,
        time_power=args.time_power,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
