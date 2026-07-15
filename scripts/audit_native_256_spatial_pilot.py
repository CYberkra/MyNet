#!/usr/bin/env python3
"""Audit a completed native-256 short causal-pair or triplet B-scan pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import extract_visible_phase, resample_time_axis  # noqa: E402
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    right_centered = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    return float(np.sum(left_centered * right_centered) / denominator) if denominator else 0.0


def _panel(values: np.ndarray, scale: float, size: tuple[int, int]) -> Image.Image:
    clipped = np.clip(values / max(scale, 1e-30), -1.0, 1.0)
    gray = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _draw_path(
    draw: ImageDraw.ImageDraw,
    path_ns: np.ndarray,
    time_range: tuple[float, float],
    box: tuple[int, int, int, int],
    colour: str,
    width: int,
) -> None:
    left, top, right, bottom = box
    x = np.linspace(left, right, path_ns.size)
    y = top + (path_ns - time_range[0]) / (time_range[1] - time_range[0]) * (bottom - top)
    draw.line(list(zip(x.tolist(), y.tolist())), fill=colour, width=width)


def _position_tolerance_m(grid_spacing_m: float) -> float:
    """Allow serialization noise while remaining far below one FDTD cell."""
    return max(1e-8, float(grid_spacing_m) * 1e-3)


def _contract_summary(
    path: Path,
    expected: int,
    source_x: np.ndarray,
    receiver_x: np.ndarray,
    position_tolerance_m: float,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("traces", [])
    actual_source = np.asarray([row["source_positions_m"][0][0] for row in rows], dtype=np.float64)
    actual_receiver = np.asarray([row["receiver_positions_m"][0][0] for row in rows], dtype=np.float64)
    indices = [int(row["trace_index"]) for row in rows]
    failures = payload.get("failures_tail", [])
    summary = {
        "complete": bool(payload.get("complete")),
        "captured": int(payload.get("captured_trace_count", 0)),
        "expected": expected,
        "trace_indices_contiguous": indices == list(range(1, expected + 1)),
        "max_source_x_error_m": float(np.max(np.abs(actual_source - source_x[: len(rows)]))) if rows else None,
        "max_receiver_x_error_m": float(np.max(np.abs(actual_receiver - receiver_x[: len(rows)]))) if rows else None,
        "position_tolerance_m": position_tolerance_m,
        "failures": failures,
    }
    summary["ok"] = bool(
        summary["complete"]
        and summary["captured"] == expected
        and summary["trace_indices_contiguous"]
        and summary["max_source_x_error_m"] is not None
        and summary["max_source_x_error_m"] <= position_tolerance_m
        and summary["max_receiver_x_error_m"] is not None
        and summary["max_receiver_x_error_m"] <= position_tolerance_m
        and not failures
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-air-reference",
        action="store_true",
        help="Audit the strict full/control causal pair without requiring an air diagnostic.",
    )
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    run_manifest_path = case_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.is_file() else {}

    paths = {
        "full": case_dir / "full_scene_merged.out",
        "control": case_dir / "no_basal_contrast_control_merged.out",
    }
    if not args.skip_air_reference:
        paths["air"] = case_dir / "air_reference_merged.out"
    loaded = {name: read_merged_bscan(path) for name, path in paths.items()}
    dt_values = [item[0] for item in loaded.values()]
    raw_values = [item[1] for item in loaded.values()]
    attrs = [item[2] for item in loaded.values()]
    aligned = len({value.shape for value in raw_values}) == 1 and np.allclose(dt_values, dt_values[0])
    finite = all(np.isfinite(value).all() for value in raw_values)
    if not aligned:
        raise RuntimeError("full/control/air merged arrays are not aligned")

    raw_end_ns = min((values.shape[0] - 1) * dt * 1e9 for values, dt in zip(raw_values, dt_values))
    analysis_end_ns = min(700.0, float(raw_end_ns))
    time_ns, full = resample_time_axis(
        raw_values[0], dt_values[0], time_window_ns=analysis_end_ns, output_samples=501
    )
    _, control = resample_time_axis(
        raw_values[1], dt_values[1], time_window_ns=analysis_end_ns, output_samples=501
    )
    air = None
    if "air" in loaded:
        _, air = resample_time_axis(
            loaded["air"][1], loaded["air"][0], time_window_ns=analysis_end_ns, output_samples=501
        )
    trace_count = full.shape[1]
    declared_trace_count = int(manifest["grid"]["trace_count"])
    trace_spacing_m = float(manifest["grid"]["trace_spacing_m"])
    declared_span_m = trace_spacing_m * max(declared_trace_count - 1, 0)
    labels = case_dir / "labels"
    selected_indices = np.asarray(
        run_manifest.get("selected_trace_indices_zero_based", list(range(trace_count))),
        dtype=np.int64,
    )
    if selected_indices.shape != (trace_count,):
        raise RuntimeError("run manifest trace selection does not match merged output")
    covered_span_m = trace_spacing_m * int(selected_indices[-1] - selected_indices[0]) if trace_count > 1 else 0.0
    reference_path = labels / "reference_arrival_time_ns.npy"
    if not reference_path.is_file():
        reference_path = labels / "geometric_reference_arrival_time_ns.npy"
    reference = np.load(reference_path)[selected_indices]
    source_x = np.load(labels / "source_x_m.npy")[selected_indices]
    receiver_x = np.load(labels / "receiver_x_m.npy")[selected_indices]
    position_tolerance_m = _position_tolerance_m(float(manifest["grid"]["dl_m"]))
    label_contract = manifest.get("labels", {})
    search_half = float(label_contract.get("visible_phase_search_half_width_ns", 35.0))
    phase_half = float(label_contract.get("visible_phase_phase_half_width_ns", 8.0))
    visible, support, contrast = extract_visible_phase(
        full,
        control,
        time_ns,
        reference,
        search_half_width_ns=search_half,
        phase_half_width_ns=phase_half,
        enforce_continuity=True,
        max_trace_step_ns=5.6,
    )

    distance = np.abs(time_ns[:, None] - visible[None, :])
    target = distance <= 25.0
    background = (time_ns[:, None] >= 220.0) & (time_ns[:, None] <= 550.0) & (distance >= 70.0)
    late = time_ns >= 600.0
    target_per_trace = np.sqrt(np.nanmean(np.where(target, contrast**2, np.nan), axis=0))
    target_rms = _rms(contrast[target])
    background_rms = _rms(contrast[background])
    target_ratio = target_rms / max(background_rms, 1e-30)
    full_spatial = full - np.median(full, axis=1, keepdims=True)
    control_spatial = control - np.median(control, axis=1, keepdims=True)
    control_target_rms = _rms(control_spatial[target])
    reference_range = float(np.ptp(reference))
    visible_range = float(np.ptp(visible))
    steps = np.abs(np.diff(visible))
    residual = visible - reference
    contracts = {
        name: _contract_summary(
            case_dir / "run_logs" / filename,
            trace_count,
            source_x,
            receiver_x,
            position_tolerance_m,
        )
        for name, filename in {
            "full": "full_scene_trace_contract.json",
            "control": "no_basal_contrast_control_trace_contract.json",
            **({"air": "air_reference_trace_contract.json"} if air is not None else {}),
        }.items()
    }
    trace_stride = int(run_manifest.get("trace_stride", 1))
    distributed_coverage_ok = bool(trace_stride > 1 and covered_span_m / max(declared_span_m, 1e-30) >= 0.95)
    result = {
        "schema": "native_256_spatial_pilot_audit_v2",
        "case_id": manifest["case_id"],
        "formal_training_allowed": False,
        "trace_count": trace_count,
        "trace_stride": trace_stride,
        "selected_trace_indices_zero_based": selected_indices.tolist(),
        "declared_trace_count": declared_trace_count,
        "covered_scan_span_m": covered_span_m,
        "declared_scan_span_m": declared_span_m,
        "covered_scan_fraction": covered_span_m / max(declared_span_m, 1e-30),
        "distributed_span_coverage_ok": distributed_coverage_ok,
        "raw_shape": list(raw_values[0].shape),
        "canonical_shape": list(full.shape),
        "analysis_time_window_ns": analysis_end_ns,
        "alignment_ok": bool(aligned),
        "air_reference_included": air is not None,
        "finite_ok": bool(finite),
        "gprmax_versions": sorted({str(item.get("gprMax")) for item in attrs}),
        "trace_contracts": contracts,
        "visible_phase_semantics": "continuous_signed_lobe_within_continuous_matched-contrast_envelope",
        "reference_range_ns": [float(np.min(reference)), float(np.max(reference))],
        "visible_range_ns": [float(np.min(visible)), float(np.max(visible))],
        "visible_to_geometric_path_correlation": _correlation(reference, visible),
        "visible_dynamic_range_retention": visible_range / max(reference_range, 1e-30),
        "median_visible_minus_reference_ns": float(np.median(residual)),
        "visible_residual_p95_ns": float(np.percentile(np.abs(residual - np.median(residual)), 95)),
        "visible_step_p95_ns": float(np.percentile(steps, 95)),
        "visible_step_max_ns": float(np.max(steps)),
        "support_median": float(np.median(support)),
        "support_min": float(np.min(support)),
        "signed_difference_target_rms": target_rms,
        "signed_difference_background_rms": background_rms,
        "signed_difference_target_to_background_rms": target_ratio,
        "control_spatial_target_rms": control_target_rms,
        "control_spatial_to_difference_target_rms": control_target_rms / max(target_rms, 1e-30),
        "target_amplitude_cv": float(np.std(target_per_trace) / max(np.mean(target_per_trace), 1e-30)),
        "target_dropout_fraction_below_25pct_median": float(np.mean(target_per_trace < 0.25 * np.median(target_per_trace))),
        "contrast_late_to_target_rms": _rms(contrast[late]) / max(target_rms, 1e-30),
        "full_control_direct_rms_relative_error": abs(_rms(full[time_ns <= 150]) - _rms(control[time_ns <= 150])) / max(_rms(full[time_ns <= 150]), 1e-30),
        "air_direct_rms_to_full": (
            _rms(air[time_ns <= 150]) / max(_rms(full[time_ns <= 150]), 1e-30)
            if air is not None
            else None
        ),
    }
    result["spatial_pilot_gate"] = {
        "passed": bool(
            finite
            and aligned
            and all(item["ok"] for item in contracts.values())
            and target_ratio >= 1.5
            and result["visible_step_max_ns"] <= 5.6
            and result["visible_to_geometric_path_correlation"] >= 0.75
            and result["visible_dynamic_range_retention"] >= 0.5
            and result["control_spatial_to_difference_target_rms"] <= 0.1
        ),
        "formal_promotion": False,
        "scope": "distributed_full_span_subset" if trace_stride > 1 else "contiguous_local_start_segment",
        "full_span_morphology_validated": trace_count == declared_trace_count,
        "note": (
            "Distributed full-span sampling gate only; full-resolution 256-trace execution and human morphology review remain required."
            if trace_stride > 1
            else "Local continuity gate only; distributed/full-span execution and human morphology review remain required."
        ),
    }
    (output_dir / "spatial_pilot_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.save(output_dir / "visible_phase_time_ns.npy", visible.astype(np.float32))
    np.save(output_dir / "visible_phase_support_score.npy", support.astype(np.float32))

    plot_range = (max(220.0, float(np.min(visible)) - 70.0), min(550.0, float(np.max(visible)) + 70.0))
    rows = (time_ns >= plot_range[0]) & (time_ns <= plot_range[1])
    common_scale = float(np.percentile(np.abs(np.concatenate((full_spatial[rows], control_spatial[rows]), axis=1)), 99.5)) or 1.0
    contrast_scale = float(np.percentile(np.abs(contrast[rows]), 99.5)) or 1.0
    canvas = Image.new("RGB", (2100, 780), "white")
    draw = ImageDraw.Draw(canvas)
    boxes = ((60, 120, 690, 650), (735, 120, 1365, 650), (1410, 120, 2040, 650))
    for values, box, title, scale, draw_paths in (
        (full_spatial, boxes[0], "Full scene", common_scale, True),
        (control_spatial, boxes[1], "No-basal control (no path overlays)", common_scale, False),
        (contrast, boxes[2], "Signed full - control", contrast_scale, True),
    ):
        canvas.paste(_panel(values[rows], scale, (box[2] - box[0], box[3] - box[1])), (box[0], box[1]))
        draw.rectangle(box, outline="#222222", width=2)
        draw.text((box[0], 80), title, fill="#111111", font=_font(24, True))
        if draw_paths:
            _draw_path(draw, reference, plot_range, box, "#ffd92f", 3)
            _draw_path(draw, visible, plot_range, box, "#d730f0", 3)
    draw.text((60, 25), f"{manifest['case_id']} - {trace_count}-trace strict-pair spatial pilot", fill="#111111", font=_font(34, True))
    draw.text((60, 690), "Yellow: independent geometric reference; magenta: continuous visible signed phase.", fill="#222222", font=_font(20))
    draw.text(
        (60, 725),
        f"Target/background={target_ratio:.2f}; max path step={result['visible_step_max_ns']:.2f} ns; "
        f"path corr={result['visible_to_geometric_path_correlation']:.2f}; "
        f"control/target={result['control_spatial_to_difference_target_rms']:.3g}; "
        f"span={covered_span_m:.2f}/{declared_span_m:.2f} m; gate={'PASS' if result['spatial_pilot_gate']['passed'] else 'HOLD'}",
        fill="#0b5d1e" if result["spatial_pilot_gate"]["passed"] else "#991b1b",
        font=_font(24, True),
    )
    canvas.save(output_dir / "spatial_pilot_fixed_gain.png")
    print(json.dumps(result["spatial_pilot_gate"], indent=2))
    return 0 if result["spatial_pilot_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
