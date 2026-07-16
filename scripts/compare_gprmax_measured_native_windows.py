#!/usr/bin/env python3
"""Compare a solved native-spacing gprMax subset with equal-width measured windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESEARCH_ROLES = {
    "Line3": "fit",
    "Line7": "fit",
    "LineL1": "fit",
    "Line6": "validation",
    "Line9": "heldout",
    "LineX1": "review-only",
}

from pgdacsnet.simulation_v2 import resample_time_axis  # noqa: E402
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402
from scripts.preview_gprmax_raw_and_agc import time_power_gain  # noqa: E402


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        if start is not None and (not value or index == mask.size - 1):
            end = index if not value else index + 1
            if end - start > best[1] - best[0]:
                best = (start, end)
            start = None
    return best


def _centered_window(start: int, end: int, width: int, total: int) -> tuple[int, int]:
    if end - start < width:
        raise ValueError(f"longest eligible run has {end - start} traces, fewer than {width}")
    center = (start + end) // 2
    low = max(0, min(total - width, center - width // 2))
    return low, low + width


def _processed(values: np.ndarray, time_ns: np.ndarray, power: float) -> np.ndarray:
    common_removed = values - np.median(values, axis=1, keepdims=True)
    return time_power_gain(common_removed, time_ns, power).astype(np.float64)


def _short_case_label(case_id: str) -> str:
    parts = case_id.split("_")
    return "_".join(parts[:3]) if len(case_id) > 36 else case_id


def _panel(values: np.ndarray, scale: float, width: int, height: int) -> Image.Image:
    clipped = np.clip(values / max(scale, np.finfo(np.float64).tiny), -1.0, 1.0)
    gray = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    image = Image.fromarray(gray, mode="L")
    image = image.resize((image.width, height), Image.Resampling.BILINEAR)
    return image.resize((width, height), Image.Resampling.NEAREST).convert("RGB")


def compare(
    run_dir: Path,
    measured_root: Path,
    output_path: Path,
    metrics_path: Path,
    *,
    lines: list[str],
    time_low_ns: float,
    time_high_ns: float,
    time_power: float,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    measured_root = measured_root.resolve()
    output_path = output_path.resolve()
    metrics_path = metrics_path.resolve()
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    dt_s, simulated, _ = read_merged_bscan(run_dir / "full_scene_merged.out", component="Ez")
    trace_count = simulated.shape[1]
    output_samples = int(round(time_high_ns / 1.4)) + 1
    sim_time, simulated = resample_time_axis(
        simulated,
        dt_s,
        time_window_ns=time_high_ns,
        output_samples=output_samples,
    )
    sim_rows = (sim_time >= time_low_ns) & (sim_time <= time_high_ns)
    panels: list[tuple[str, np.ndarray, float]] = []
    sim_view = _processed(simulated, sim_time, time_power)[sim_rows]
    panels.append(
        (
            f"{_short_case_label(str(scene['case_id']))} (development)",
            sim_view,
            float(np.quantile(np.abs(sim_view), 0.995)),
        )
    )

    selections: list[dict[str, object]] = []
    for line in lines:
        path = measured_root / f"{line}.npz"
        with np.load(path, allow_pickle=False) as data:
            status = data["status_code"]
            weight = data["label_weight"]
            ignore = data["v15_final_ignore_trace"].astype(bool)
            eligible = (status == 1) & (weight > 0) & ~ignore
            run_start, run_end = _longest_true_run(eligible)
            low, high = _centered_window(run_start, run_end, trace_count, status.size)
            time_ns = data["time_ns"].astype(np.float64)
            raw = data["raw_full_normalized"][:, low:high].astype(np.float64)
            rows = (time_ns >= time_low_ns) & (time_ns <= time_high_ns)
            view = _processed(raw, time_ns, time_power)[rows]
            scale = float(np.quantile(np.abs(view), 0.995))
            split = str(data["split"].item())
            trace_distance = float(data["trace_interval_m"].item())
        research_role = RESEARCH_ROLES.get(line, "unassigned")
        label = f"{line} ({research_role}; traces {low}-{high - 1})"
        panels.append((label, view, scale))
        selections.append(
            {
                "line": line,
                "split": split,
                "research_role": research_role,
                "source": str(path.relative_to(ROOT).as_posix()),
                "selection_rule": "centered in longest contiguous strong-positive non-ignore run",
                "eligible_run": [run_start, run_end],
                "selected_trace_range_half_open": [low, high],
                "trace_count": trace_count,
                "nominal_width_m": (trace_count - 1) * trace_distance,
            }
        )

    columns = 3
    rows_count = (len(panels) + columns - 1) // columns
    panel_w, panel_h = 520, 420
    margin_x, margin_y = 40, 150
    gap_x, gap_y = 30, 85
    canvas_w = margin_x * 2 + columns * panel_w + (columns - 1) * gap_x
    canvas_h = margin_y + rows_count * panel_h + (rows_count - 1) * gap_y + 75
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_x, 24), "Native-width morphology comparison", fill="black", font=_font(30, True))
    draw.text(
        (margin_x, 70),
        f"{trace_count} consecutive traces; {time_low_ns:g}-{time_high_ns:g} ns; common-mode suppression + time^{time_power:g}; independent P99.5 scales",
        fill="black",
        font=_font(18),
    )
    draw.text(
        (margin_x, 102),
        "Line9 is held-out diagnostic evidence only. No measured pixel, coordinate, or waveform patch is used by the generator.",
        fill="black",
        font=_font(17),
    )
    for index, (label, values, scale) in enumerate(panels):
        row, column = divmod(index, columns)
        left = margin_x + column * (panel_w + gap_x)
        top = margin_y + row * (panel_h + gap_y)
        canvas.paste(_panel(values, scale, panel_w, panel_h), (left, top))
        draw.rectangle((left, top, left + panel_w, top + panel_h), outline="black", width=2)
        draw.text((left, top - 28), label, fill="black", font=_font(18, True))
        draw.text((left + 8, top + 8), f"{time_low_ns:g} ns", fill="white", font=_font(16))
        draw.text((left + 8, top + panel_h - 24), f"{time_high_ns:g} ns", fill="white", font=_font(16))
        draw.text((left, top + panel_h + 10), f"independent P99.5={scale:.3e}", fill="black", font=_font(16))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    result = {
        "audit_type": "NATIVE_WIDTH_MEASURED_MORPHOLOGY_COMPARISON_V1",
        "run": str(run_dir.relative_to(ROOT).as_posix()),
        "case_id": scene["case_id"],
        "selected_trace_indices_zero_based": run.get("selected_trace_indices_zero_based", []),
        "trace_count": trace_count,
        "time_crop_ns": [time_low_ns, time_high_ns],
        "processing": {"common_mode_suppression": True, "time_power": time_power},
        "display_scale": "independent absolute-value P99.5 per panel; morphology only",
        "heldout_restriction": "Line9 is diagnostic only and must not condition the generator",
        "measured_selections": selections,
        "output": str(output_path.relative_to(ROOT).as_posix()),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--measured-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--lines", nargs="+", default=["Line3", "Line6", "Line7", "Line9", "LineL1"])
    parser.add_argument("--time-low-ns", type=float, default=120.0)
    parser.add_argument("--time-high-ns", type=float, default=500.0)
    parser.add_argument("--time-power", type=float, default=1.5)
    args = parser.parse_args()
    result = compare(
        args.run_dir,
        args.measured_root,
        args.output,
        args.metrics,
        lines=args.lines,
        time_low_ns=args.time_low_ns,
        time_high_ns=args.time_high_ns,
        time_power=args.time_power,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
