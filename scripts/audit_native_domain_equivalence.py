#!/usr/bin/env python3
"""Audit a causally protected small-domain CV01 run against its exact source run."""

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


PROTECTED_END_NS = 500.0


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            size=size,
        )
    except OSError:
        return ImageFont.load_default()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _corr(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        return float("nan")
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    left_scale = float(np.sqrt(np.sum(left_centered * left_centered)))
    right_scale = float(np.sqrt(np.sum(right_centered * right_centered)))
    if left_scale == 0.0 or right_scale == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / (left_scale * right_scale))


def _comparison(reference: np.ndarray, candidate: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    baseline = reference[rows]
    cropped = candidate[rows]
    delta = cropped - baseline
    return {
        "correlation": _corr(baseline, cropped),
        "relative_rms_error": _rms(delta) / max(_rms(baseline), 1e-30),
        "reference_rms": _rms(baseline),
        "candidate_rms": _rms(cropped),
    }


def _load_pair(case_dir: Path) -> dict[str, object]:
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    groups = set(run_manifest.get("input_groups", []))
    required_outputs = {
        "full_scene_merged.out",
        "no_basal_contrast_control_merged.out",
    }
    missing_outputs = [name for name in required_outputs if not (case_dir / name).is_file()]
    if missing_outputs:
        raise RuntimeError(f"missing causal pair outputs in {case_dir}: {missing_outputs}")
    if groups and not {"full_scene", "no_basal_contrast_control"}.issubset(groups):
        raise RuntimeError(f"run manifest does not declare a causal full/control pair: {sorted(groups)}")
    full_dt, full_raw, full_attrs = read_merged_bscan(case_dir / "full_scene_merged.out")
    control_dt, control_raw, control_attrs = read_merged_bscan(case_dir / "no_basal_contrast_control_merged.out")
    if full_raw.shape != control_raw.shape or not np.isclose(full_dt, control_dt):
        raise RuntimeError(f"unaligned full/control pair in {case_dir}")
    time_ns, full = resample_time_axis(full_raw, full_dt, time_window_ns=700.0, output_samples=501)
    _, control = resample_time_axis(control_raw, control_dt, time_window_ns=700.0, output_samples=501)
    indices = np.asarray(run_manifest["selected_trace_indices_zero_based"], dtype=np.int64)
    if indices.shape != (full.shape[1],):
        raise RuntimeError(f"trace selection does not match merged data in {case_dir}")
    labels = case_dir / "labels"
    reference_path = labels / "reference_arrival_time_ns.npy"
    if not reference_path.is_file():
        reference_path = labels / "geometric_reference_arrival_time_ns.npy"
    reference = np.load(reference_path)[indices]
    label_contract = manifest.get("labels", {})
    visible, support, contrast = extract_visible_phase(
        full,
        control,
        time_ns,
        reference,
        search_half_width_ns=float(label_contract.get("visible_phase_search_half_width_ns", 35.0)),
        phase_half_width_ns=float(label_contract.get("visible_phase_phase_half_width_ns", 8.0)),
        enforce_continuity=True,
        max_trace_step_ns=5.6,
    )
    return {
        "case_dir": str(case_dir),
        "manifest": manifest,
        "run_manifest": run_manifest,
        "time_ns": time_ns,
        "full": full,
        "control": control,
        "contrast": contrast,
        "reference": reference,
        "visible": visible,
        "support": support,
        "raw_shape": list(full_raw.shape),
        "gprmax_versions": sorted({str(full_attrs.get("gprMax")), str(control_attrs.get("gprMax"))}),
    }


def _panel(values: np.ndarray, scale: float, size: tuple[int, int]) -> Image.Image:
    clipped = np.clip(values / max(scale, 1e-30), -1.0, 1.0)
    gray = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _draw_path(draw: ImageDraw.ImageDraw, path_ns: np.ndarray, time_range: tuple[float, float], box: tuple[int, int, int, int], colour: str) -> None:
    left, top, right, bottom = box
    x = np.linspace(left, right, path_ns.size)
    y = top + (path_ns - time_range[0]) / (time_range[1] - time_range[0]) * (bottom - top)
    draw.line(list(zip(x.tolist(), y.tolist())), fill=colour, width=3)


def _render(output_path: Path, baseline: dict[str, object], cropped: dict[str, object], result: dict[str, object]) -> None:
    time_ns = np.asarray(baseline["time_ns"])
    rows = (time_ns >= 300.0) & (time_ns <= PROTECTED_END_NS)
    base = np.asarray(baseline["contrast"])[rows]
    small = np.asarray(cropped["contrast"])[rows]
    delta = small - base
    scale = float(np.percentile(np.abs(np.concatenate((base, small), axis=1)), 99.5)) or 1.0
    delta_scale = float(np.percentile(np.abs(delta), 99.5)) or 1.0
    canvas = Image.new("RGB", (2100, 850), "white")
    draw = ImageDraw.Draw(canvas)
    boxes = ((60, 135, 690, 720), (735, 135, 1365, 720), (1410, 135, 2040, 720))
    paths = ((baseline, boxes[0], "110 m inner-PML guard baseline", scale), (cropped, boxes[1], "80 m inner-PML guard crop", scale))
    for data, box, title, local_scale in paths:
        values = np.asarray(data["contrast"])[rows]
        canvas.paste(_panel(values, local_scale, (box[2] - box[0], box[3] - box[1])), (box[0], box[1]))
        draw.rectangle(box, outline="#222222", width=2)
        draw.text((box[0], 95), title, fill="#111111", font=_font(22, True))
        _draw_path(draw, np.asarray(data["reference"]), (300.0, PROTECTED_END_NS), box, "#ffd92f")
        _draw_path(draw, np.asarray(data["visible"]), (300.0, PROTECTED_END_NS), box, "#d730f0")
    canvas.paste(_panel(delta, delta_scale, (boxes[2][2] - boxes[2][0], boxes[2][3] - boxes[2][1])), (boxes[2][0], boxes[2][1]))
    draw.rectangle(boxes[2], outline="#222222", width=2)
    draw.text((boxes[2][0], 95), "Cropped minus baseline contrast", fill="#111111", font=_font(22, True))
    draw.text((60, 25), "CV01 exact small-domain equivalence: protected 0-500 ns only", fill="#111111", font=_font(32, True))
    draw.text((60, 55), "Yellow: copied geometric reference. Magenta: independently extracted continuous visible signed phase.", fill="#222222", font=_font(19))
    contrast = result["protected_window"]["contrast"]
    gate = result["equivalence_gate"]
    text = (
        f"contrast r={contrast['correlation']:.6f}; rel-RMS={contrast['relative_rms_error']:.4%}; "
        f"visible-path P95={result['visible_path_difference_ns']['p95_absolute_ns']:.2f} ns; "
        f"gate={'PASS' if gate['passed'] else 'HOLD'}"
    )
    draw.text((60, 760), text, fill="#0b5d1e" if gate["passed"] else "#991b1b", font=_font(23, True))
    draw.text((60, 795), "The 500-700 ns interval is recorded as diagnostic only and cannot pass or fail this equivalence gate.", fill="#222222", font=_font(18))
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run_dir", type=Path)
    parser.add_argument("cropped_run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = _load_pair(args.baseline_run_dir.resolve())
    cropped = _load_pair(args.cropped_run_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    time_ns = np.asarray(baseline["time_ns"])
    if not np.array_equal(time_ns, cropped["time_ns"]):
        raise RuntimeError("canonical time axes differ")
    baseline_indices = np.asarray(baseline["run_manifest"]["selected_trace_indices_zero_based"])
    cropped_indices = np.asarray(cropped["run_manifest"]["selected_trace_indices_zero_based"])
    if not np.array_equal(baseline_indices, cropped_indices):
        raise RuntimeError("distributed trace selections differ")
    protected = time_ns <= PROTECTED_END_NS
    diagnostic = time_ns > PROTECTED_END_NS
    target = np.abs(time_ns[:, None] - np.asarray(baseline["visible"])[None, :]) <= 25.0
    phase_difference = np.asarray(cropped["visible"]) - np.asarray(baseline["visible"])
    crop_grid = cropped["manifest"]["grid"]
    earliest_roundtrip = float(crop_grid["earliest_free_space_side_roundtrip_ns"])
    result: dict[str, object] = {
        "schema": "native_256_domain_equivalence_audit_v1",
        "baseline_case_id": baseline["manifest"]["case_id"],
        "cropped_case_id": cropped["manifest"]["case_id"],
        "formal_training_allowed": False,
        "baseline_run_dir": baseline["case_dir"],
        "cropped_run_dir": cropped["case_dir"],
        "same_distributed_trace_indices": True,
        "selected_trace_indices_zero_based": baseline_indices.tolist(),
        "protected_window_ns": [0.0, PROTECTED_END_NS],
        "cropped_earliest_free_space_side_roundtrip_ns": earliest_roundtrip,
        "causal_guard_exceeds_protected_window": earliest_roundtrip > PROTECTED_END_NS,
        "protected_window": {
            name: _comparison(np.asarray(baseline[name]), np.asarray(cropped[name]), protected)
            for name in ("full", "control", "contrast")
        },
        "protected_target_band": _comparison(np.asarray(baseline["contrast"]), np.asarray(cropped["contrast"]), target & protected[:, None]),
        "visible_path_difference_ns": {
            "median_signed_ns": float(np.median(phase_difference)),
            "p95_absolute_ns": float(np.percentile(np.abs(phase_difference), 95)),
            "max_absolute_ns": float(np.max(np.abs(phase_difference))),
        },
        "diagnostic_unprotected_500_700_ns": {
            name: _comparison(np.asarray(baseline[name]), np.asarray(cropped[name]), diagnostic)
            for name in ("full", "control", "contrast")
        },
    }
    contrast_delta = np.asarray(cropped["contrast"]) - np.asarray(baseline["contrast"])
    target_reference_rms = _rms(np.asarray(baseline["contrast"])[target & protected[:, None]])
    result["protected_contrast_delta_to_target_rms"] = _rms(contrast_delta[protected]) / max(target_reference_rms, 1e-30)
    protected_values = result["protected_window"]
    assert isinstance(protected_values, dict)
    passed = bool(
        result["causal_guard_exceeds_protected_window"]
        and all(item["correlation"] >= 0.999 for item in protected_values.values())
        and protected_values["full"]["relative_rms_error"] <= 0.02
        and protected_values["control"]["relative_rms_error"] <= 0.02
        and protected_values["contrast"]["relative_rms_error"] <= 0.04
        and result["protected_contrast_delta_to_target_rms"] <= 0.02
        and result["protected_target_band"]["correlation"] >= 0.999
        and result["protected_target_band"]["relative_rms_error"] <= 0.02
        and result["visible_path_difference_ns"]["p95_absolute_ns"] <= 1.4
    )
    result["equivalence_gate"] = {
        "passed": passed,
        "formal_promotion": False,
        "thresholds": {
            "protected_correlation_min": 0.999,
            "protected_full_control_relative_rms_error_max": 0.02,
            "protected_contrast_relative_rms_error_max": 0.04,
            "protected_contrast_delta_to_target_rms_max": 0.02,
            "visible_path_p95_absolute_ns_max": 1.4,
        },
        "note": "Only 0-500 ns can establish the cropped-domain equivalence claim."
    }
    (output_dir / "domain_equivalence_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.save(output_dir / "visible_phase_difference_ns.npy", phase_difference.astype(np.float32))
    _render(output_dir / "domain_equivalence_protected_window.png", baseline, cropped, result)
    print(json.dumps(result["equivalence_gate"], indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
