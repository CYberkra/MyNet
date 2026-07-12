#!/usr/bin/env python3
"""Create a like-for-like audit of a macro case, Pilot 001, and Line9."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = "MACRO01_GENTLE_LONG_LINE_DIAGNOSTIC"


def resample_time(a: np.ndarray, n: int = 501) -> np.ndarray:
    if a.shape[0] == n:
        return np.asarray(a, np.float32)
    x0 = np.linspace(0.0, 1.0, a.shape[0])
    x1 = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(x1, x0, a[:, j]) for j in range(a.shape[1])], axis=1).astype(np.float32)


def display(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    b = a - np.median(a, axis=1, keepdims=True)
    gain = np.maximum(t / 700.0, 0.02) ** 3
    b = b * gain[:, None]
    scale = float(np.percentile(np.abs(b), 99.5)) or 1.0
    return np.clip(b / scale, -1.0, 1.0)


def metrics(a: np.ndarray, t: np.ndarray, curve: np.ndarray, x: np.ndarray) -> dict[str, float]:
    b = a - np.median(a, axis=1, keepdims=True)
    display_gain = np.maximum(t / 700.0, 0.02) ** 3
    display_scale = float(np.percentile(np.abs(b * display_gain[:, None]), 99.5)) or 1.0
    env = np.abs(hilbert(b, axis=0))
    idx = np.clip(np.searchsorted(t, curve), 0, len(t) - 1)
    amp = env[idx, np.arange(a.shape[1])]
    d1 = np.gradient(curve, x)
    d2 = np.gradient(d1, x)
    sign = np.sign(np.diff(curve))
    extrema = int(np.count_nonzero(sign[1:] * sign[:-1] < 0))
    target_band = np.abs(t[:, None] - curve[None, :]) <= 18.0
    bg_band = (t[:, None] >= 300.0) & (t[:, None] <= 500.0) & ~target_band
    return {
        "trace_count": int(a.shape[1]),
        "scan_length_m": float(x[-1] - x[0]),
        "curve_min_ns": float(np.min(curve)),
        "curve_max_ns": float(np.max(curve)),
        "curve_range_ns": float(np.ptp(curve)),
        "curve_extrema_count": extrema,
        "curve_abs_slope_p95_ns_per_m": float(np.percentile(np.abs(d1), 95)),
        "curve_abs_curvature_p95_ns_per_m2": float(np.percentile(np.abs(d2), 95)),
        "target_envelope_cv": float(np.std(amp) / max(np.mean(amp), 1e-12)),
        "target_to_background_rms": float(np.sqrt(np.mean(b[target_band] ** 2)) / max(np.sqrt(np.mean(b[bg_band] ** 2)), 1e-12)),
        "display_scale_p99_5_after_t3_gain": display_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-case", default=DEFAULT_CASE)
    parser.add_argument("--macro-path", type=Path)
    args = parser.parse_args()
    case_id = args.macro_case
    out = ROOT / "reports" / "simulation_v2_control_stage_20260711" / "macro_pilot_line9_audit" / case_id
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 700.0, 501, dtype=np.float32)

    if args.macro_path:
        macro_case = args.macro_path.resolve()
        if macro_case.name != case_id:
            raise ValueError(f"--macro-case {case_id!r} does not match --macro-path leaf {macro_case.name!r}")
    else:
        candidates = (
            ROOT / "outputs" / "simulation_v2_controls" / "official_audited_20260711" / case_id,
            ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "00_controls" / case_id,
        )
        macro_case = next((path for path in candidates if path.is_dir()), candidates[0])
    pair = macro_case / "pair_audit"
    labels = macro_case / "labels"
    if (pair / "full_501x128.npy").is_file():
        macro_a = np.load(pair / "full_501x128.npy")
        macro_curve = np.load(pair / "visible_phase_time_ns.npy")
    else:
        candidates = sorted(labels.glob("full_scene_501x*.npy"))
        if not candidates:
            raise FileNotFoundError(f"no canonical full B-scan under {labels}")
        macro_a = np.load(candidates[0])
        macro_curve = np.load(labels / "visible_phase_time_ns.npy")
    macro_x = np.load(labels / "trace_midpoint_x_m.npy")
    macro_x = macro_x - macro_x[0]

    pilot = ROOT / "data/PGDA_SYNTH_DATASET_V1/05_accepted_dataset/line9_style/mixed/LINE9_STYLE_001"
    pilot_a = resample_time(np.load(pilot / "input/raw_bscan.npy"))
    pilot_curve = np.load(pilot / "label/target_visible_phase_time_ns.npy")
    pilot_x = np.linspace(0.0, 216.141, pilot_a.shape[1])

    line = np.load(ROOT / "data_yingshan_v15_final_20260710/lines/Line9.npz", allow_pickle=True)
    source_x = line["profile_chainage_m"]
    target_x = np.linspace(source_x[0], source_x[-1], 128)
    line_a = np.stack([np.interp(target_x, source_x, row) for row in line["raw_full_normalized"]], axis=0)
    line_curve = np.interp(target_x, source_x, line["v15_final_center_time_ns"])
    line_x = target_x - target_x[0]

    datasets = [
        (case_id, macro_a, macro_curve, macro_x),
        ("Pilot 001", pilot_a, pilot_curve, pilot_x),
        ("Line9 V15", line_a, line_curve, line_x),
    ]
    summary = {name: metrics(a, t, curve, x) for name, a, curve, x in datasets}
    summary["_comparison_contract"] = {
        "morphology_display": "same transform with independent per-dataset P99.5 scale",
        "brightness_comparable_across_rows": False,
        "reason": "simulation, Pilot 001, and measured Line9 do not share calibrated amplitude units",
        "full_control_causal_display": "separate pair audit uses one shared physical scale",
    }
    (out / "comparison_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    canvas = Image.new("RGB", (1800, 1320), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (45, 8),
        "Morphology comparison: identical transform, independent P99.5 scale per row; brightness is not comparable across rows",
        fill="black",
    )
    panel_w, panel_h = 850, 360
    for row, (name, a, curve, x) in enumerate(datasets):
        shown = display(a, t)
        band = (t >= 300) & (t <= 500)
        env = np.abs(hilbert(a - np.median(a, axis=1, keepdims=True), axis=0))
        left = ((shown[band] + 1.0) * 127.5).astype(np.uint8)
        right_f = np.log1p(env[band])
        right = (255.0 * right_f / max(float(np.percentile(right_f, 99.5)), 1e-12)).clip(0, 255).astype(np.uint8)
        for col, arr in enumerate((left, right)):
            rgb = np.repeat(arr[:, :, None], 3, axis=2)
            if col == 1:
                rgb = np.stack((rgb[:, :, 0], (rgb[:, :, 0] * 0.55).astype(np.uint8), np.zeros_like(arr)), axis=2)
            image = Image.fromarray(rgb).resize((panel_w, panel_h), Image.Resampling.BILINEAR)
            px, py = 45 + col * 900, 65 + row * 425
            canvas.paste(image, (px, py))
            points = []
            for xx, yy in zip(x, curve):
                xp = px + int((float(xx) - float(x[0])) / max(float(x[-1] - x[0]), 1e-9) * (panel_w - 1))
                yp = py + int((float(yy) - 300.0) / 200.0 * (panel_h - 1))
                points.append((xp, yp))
            draw.line(points, fill=(0, 230, 235), width=3)
            draw.rectangle((px, py, px + panel_w, py + panel_h), outline=(70, 70, 70), width=1)
            draw.text(
                (px, py - 25),
                f"{name}: {'signed t^3 gain' if col == 0 else 'log envelope'} (per-dataset robust scale)",
                fill="black",
            )
            draw.text((px, py + panel_h + 5), f"0 - {x[-1] - x[0]:.1f} m", fill="black")
        draw.text((5, 65 + row * 425), "300 ns", fill="black")
        draw.text((5, 65 + row * 425 + panel_h - 12), "500 ns", fill="black")
    canvas.save(out / "macro_pilot_line9_same_processing.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
