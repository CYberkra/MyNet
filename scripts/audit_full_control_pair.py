#!/usr/bin/env python3
"""Audit a completed full/no-basal pair without claiming full three-way postprocess."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import extract_visible_phase, resample_time_axis, sha256_file
from scripts.postprocess_physical_sim_v2 import read_merged_bscan, validate_hdf5_contract


def panel(
    values: np.ndarray,
    width: int = 900,
    height: int = 380,
    signed: bool = True,
    scale: float | None = None,
) -> Image.Image:
    scale = float(scale if scale is not None else np.percentile(np.abs(values), 99.5)) or 1.0
    clipped = np.clip(values / scale, -1.0, 1.0)
    if signed:
        gray = ((clipped + 1.0) * 127.5).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        pos = (255.0 * np.abs(clipped)).astype(np.uint8)
        rgb = np.stack((pos, (pos * 0.35).astype(np.uint8), np.zeros_like(pos)), axis=2)
    return Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)


def evaluate_trace_contract(
    path: Path,
    *,
    manifest: dict[str, object],
    output_attrs: dict[str, object],
    output_dt_s: float,
    expected_trace_count: int,
) -> dict[str, object]:
    if not path.is_file():
        return {"present": False, "complete": False, "ok": False, "reason": "capture report is missing"}
    captured = json.loads(path.read_text(encoding="utf-8"))
    rows = captured.get("traces", [])
    try:
        spec = manifest["spec"]
        grid = manifest["grid"]
        source_expected = float(spec["scan_start_x_m"]) + np.arange(
            int(spec["trace_count"]), dtype=np.float64
        ) * float(spec["trace_spacing_m"])
        receiver_expected = source_expected + float(spec["tx_rx_offset_m"])
        source_actual = np.asarray([row["source_positions_m"][0][0] for row in rows], dtype=np.float64)
        receiver_actual = np.asarray([row["receiver_positions_m"][0][0] for row in rows], dtype=np.float64)
        roots = [row["root_attributes"] for row in rows]
        trace_indices = [int(row["trace_index"]) for row in rows]
        grid_shapes = sorted({tuple(int(value) for value in root["nx_ny_nz"]) for root in roots})
        grid_steps = sorted({tuple(float(value) for value in root["dx_dy_dz"]) for root in roots})
        source_steps = sorted({tuple(int(value) for value in root["srcsteps"]) for root in roots})
        receiver_steps = sorted({tuple(int(value) for value in root["rxsteps"]) for root in roots})
        expected_components = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
        component_shapes_ok = all(
            set(row["receiver_shapes"].get("rx1", {})) == expected_components
            and all(
                shape == [int(row["root_attributes"]["Iterations"])]
                for shape in row["receiver_shapes"]["rx1"].values()
            )
            for row in rows
        )
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return {
            "present": True,
            "complete": bool(captured.get("complete")),
            "ok": False,
            "reason": f"malformed trace contract: {exc}",
        }
    report: dict[str, object] = {
        "present": True,
        "complete": bool(captured.get("complete")),
        "captured_trace_count": int(captured.get("captured_trace_count", 0)),
        "max_source_x_error_m": float(np.max(np.abs(source_actual - source_expected[: len(rows)]))) if rows else None,
        "max_receiver_x_error_m": float(np.max(np.abs(receiver_actual - receiver_expected[: len(rows)]))) if rows else None,
        "iterations_unique": sorted({int(root["Iterations"]) for root in roots}),
        "dt_s_unique": sorted({float(root["dt"]) for root in roots}),
        "gprmax_versions": sorted({str(root["gprMax"]) for root in roots}),
        "trace_indices_contiguous": trace_indices == list(range(1, len(rows) + 1)),
        "grid_shapes_unique": [list(value) for value in grid_shapes],
        "grid_steps_m_unique": [list(value) for value in grid_steps],
        "source_steps_cells_unique": [list(value) for value in source_steps],
        "receiver_steps_cells_unique": [list(value) for value in receiver_steps],
        "six_component_shape_contract_ok": component_shapes_ok,
        "failures_tail": captured.get("failures_tail", []),
    }
    expected_grid = [int(value) for value in grid["nx_ny_nz"]]
    expected_dl = float(grid["dl_m"])
    expected_step_cells = int(round(float(grid["trace_spacing_m"]) / expected_dl))
    position_tolerance_m = max(1e-9, expected_dl * 1e-6)
    report["position_tolerance_m"] = position_tolerance_m
    report["ok"] = bool(
        report["complete"]
        and report["captured_trace_count"] == expected_trace_count
        and report["max_source_x_error_m"] <= position_tolerance_m
        and report["max_receiver_x_error_m"] <= position_tolerance_m
        and report["iterations_unique"] == [int(output_attrs["Iterations"])]
        and len(report["dt_s_unique"]) == 1
        and np.isclose(report["dt_s_unique"][0], output_dt_s)
        and report["gprmax_versions"] == [str(output_attrs["gprMax"])]
        and report["trace_indices_contiguous"]
        and report["grid_shapes_unique"] == [expected_grid]
        and len(report["grid_steps_m_unique"]) == 1
        and np.allclose(report["grid_steps_m_unique"][0], [expected_dl] * 3)
        and report["source_steps_cells_unique"] == [[expected_step_cells, 0, 0]]
        and report["receiver_steps_cells_unique"] == [[expected_step_cells, 0, 0]]
        and report["six_component_shape_contract_ok"]
        and not report["failures_tail"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir")
    args = parser.parse_args()
    case_dir = Path(args.case_dir).resolve()
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    full_path = case_dir / "full_scene_merged.out"
    control_path = case_dir / "no_basal_contrast_control_merged.out"
    full_dt, full_raw, full_attrs = read_merged_bscan(full_path)
    control_dt, control_raw, control_attrs = read_merged_bscan(control_path)
    errors = validate_hdf5_contract(path=full_path, data=full_raw, dt_s=full_dt, attrs=full_attrs, manifest=manifest)
    errors += validate_hdf5_contract(path=control_path, data=control_raw, dt_s=control_dt, attrs=control_attrs, manifest=manifest)
    if errors or full_raw.shape != control_raw.shape or not np.isclose(full_dt, control_dt):
        raise RuntimeError("; ".join(errors or ["full/control mismatch"]))
    time_ns, full = resample_time_axis(full_raw, full_dt, time_window_ns=700.0, output_samples=501)
    _, control = resample_time_axis(control_raw, control_dt, time_window_ns=700.0, output_samples=501)
    reference = np.load(case_dir / "labels" / "reference_arrival_time_ns.npy")
    label_contract = manifest.get("labels", {})
    search_half_width_ns = float(label_contract.get("visible_phase_search_half_width_ns", 35.0))
    phase_half_width_ns = float(label_contract.get("visible_phase_phase_half_width_ns", 8.0))
    visible, support, contrast = extract_visible_phase(
        full,
        control,
        time_ns,
        reference,
        search_half_width_ns=search_half_width_ns,
        phase_half_width_ns=phase_half_width_ns,
        enforce_continuity=True,
        max_trace_step_ns=5.6,
    )
    delta = visible - reference
    residual = delta - np.median(delta)
    steps = np.abs(np.diff(visible))
    full_suppressed = full - np.median(full, axis=1, keepdims=True)
    control_suppressed = control - np.median(control, axis=1, keepdims=True)
    distance = np.abs(time_ns[:, None] - visible[None, :])
    target_band = distance <= 25.0
    background_band = (time_ns[:, None] >= 220.0) & (time_ns[:, None] <= 550.0) & (distance >= 70.0)
    comparison_target_band = distance <= 18.0
    comparison_background_band = (
        (time_ns[:, None] >= 300.0)
        & (time_ns[:, None] <= 500.0)
        & ~comparison_target_band
    )
    late_time_band = time_ns >= 600.0
    endpoint_columns = np.zeros(full.shape[1], dtype=bool)
    endpoint_width = min(8, max(1, full.shape[1] // 8))
    endpoint_columns[:endpoint_width] = True
    endpoint_columns[-endpoint_width:] = True
    endpoint_target_band = target_band & endpoint_columns[None, :]
    interior_target_band = target_band & ~endpoint_columns[None, :]
    full_raw_target_rms = float(np.sqrt(np.mean(full[target_band] ** 2)))
    full_raw_background_rms = float(np.sqrt(np.mean(full[background_band] ** 2)))
    full_target_rms = float(np.sqrt(np.mean(full_suppressed[target_band] ** 2)))
    full_background_rms = float(np.sqrt(np.mean(full_suppressed[background_band] ** 2)))
    control_raw_target_rms = float(np.sqrt(np.mean(control[target_band] ** 2)))
    control_raw_background_rms = float(np.sqrt(np.mean(control[background_band] ** 2)))
    control_target_rms = float(np.sqrt(np.mean(control_suppressed[target_band] ** 2)))
    control_background_rms = float(np.sqrt(np.mean(control_suppressed[background_band] ** 2)))
    contrast_target_rms = float(np.sqrt(np.mean(contrast[target_band] ** 2)))
    contrast_background_rms = float(np.sqrt(np.mean(contrast[background_band] ** 2)))
    full_late_rms = float(np.sqrt(np.mean(full_suppressed[late_time_band, :] ** 2)))
    control_late_rms = float(np.sqrt(np.mean(control_suppressed[late_time_band, :] ** 2)))
    contrast_late_rms = float(np.sqrt(np.mean(contrast[late_time_band, :] ** 2)))
    full_endpoint_target_rms = float(np.sqrt(np.mean(full_suppressed[endpoint_target_band] ** 2)))
    full_interior_target_rms = float(np.sqrt(np.mean(full_suppressed[interior_target_band] ** 2)))
    contrast_endpoint_target_rms = float(np.sqrt(np.mean(contrast[endpoint_target_band] ** 2)))
    contrast_interior_target_rms = float(np.sqrt(np.mean(contrast[interior_target_band] ** 2)))
    strict_pair = manifest.get("strict_pair", {})
    artifact_hash_contract = {
        "geometry_index": {
            "expected": strict_pair.get("shared_geometry_sha256"),
            "actual": sha256_file(case_dir / "geology_indices.h5"),
        },
        "full_materials": {
            "expected": strict_pair.get("full_materials_sha256"),
            "actual": sha256_file(case_dir / "materials_full.txt"),
        },
        "control_materials": {
            "expected": strict_pair.get("control_materials_sha256"),
            "actual": sha256_file(case_dir / "materials_no_basal.txt"),
        },
    }
    for item in artifact_hash_contract.values():
        item["ok"] = bool(item["expected"] and item["actual"] == item["expected"])
    artifact_hash_contract_ok = all(bool(item["ok"]) for item in artifact_hash_contract.values())
    control_trace_contract = evaluate_trace_contract(
        case_dir / "run_logs" / "control_trace_contract.json",
        manifest=manifest,
        output_attrs=control_attrs,
        output_dt_s=control_dt,
        expected_trace_count=control.shape[1],
    )
    full_trace_contract = evaluate_trace_contract(
        case_dir / "run_logs" / "full_trace_contract.json",
        manifest=manifest,
        output_attrs=full_attrs,
        output_dt_s=full_dt,
        expected_trace_count=full.shape[1],
    )
    if not full_trace_contract["present"]:
        full_trace_contract.update(
            {
                "reason": "current run merged with --remove-files before the new capture guard was installed",
                "fallback_evidence": "input static audit, model-1 source/receiver log, 128/128 completion log, merged 128-column output",
            }
        )
    numerical_pair_alignment_ok = bool(
        full_raw.shape == control_raw.shape
        and np.isclose(full_dt, control_dt)
        and full_attrs.get("gprMax") == control_attrs.get("gprMax")
    )
    pair_physics_contract_ok = bool(numerical_pair_alignment_ok and artifact_hash_contract_ok)
    formal_provenance_complete = bool(control_trace_contract.get("ok") and full_trace_contract.get("ok"))
    available_trace_contracts_ok = bool(
        control_trace_contract.get("ok")
        and (not full_trace_contract.get("present") or full_trace_contract.get("ok"))
    )
    result = {
        "case_id": manifest["case_id"],
        "ok": bool(pair_physics_contract_ok and available_trace_contracts_ok),
        "audit_type": "full_no_basal_pair_only",
        "air_reference_completed": False, "formal_training_allowed": False,
        "output_shape_cfl": list(full_raw.shape), "output_shape_canonical": list(full.shape),
        "gprmax_version_consistent": full_attrs.get("gprMax") == control_attrs.get("gprMax"),
        "full_sha256": sha256_file(full_path), "control_sha256": sha256_file(control_path),
        "visible_phase_search_half_width_ns": search_half_width_ns,
        "visible_phase_phase_half_width_ns": phase_half_width_ns,
        "visible_phase_semantics": "continuous_signed_lobe_within_continuous_matched-contrast_envelope",
        "median_visible_minus_reference_ns": float(np.median(delta)),
        "max_abs_trace_residual_ns": float(np.max(np.abs(residual))),
        "visible_step_max_ns": float(np.max(steps)), "visible_step_p95_ns": float(np.percentile(steps, 95)),
        "support_median": float(np.median(support)), "support_min": float(np.min(support)),
        "full_target_rms": full_target_rms,
        "full_background_rms": full_background_rms,
        "full_target_to_background_rms": full_target_rms / max(full_background_rms, 1e-30),
        "full_raw_target_rms": full_raw_target_rms,
        "full_raw_background_rms": full_raw_background_rms,
        "full_raw_target_to_background_rms": full_raw_target_rms / max(full_raw_background_rms, 1e-30),
        "full_median_suppressed_target_to_broad_background_rms": full_target_rms / max(full_background_rms, 1e-30),
        "control_target_rms": control_target_rms,
        "control_background_rms": control_background_rms,
        "control_target_to_background_rms": control_target_rms / max(control_background_rms, 1e-30),
        "control_raw_target_rms": control_raw_target_rms,
        "control_raw_background_rms": control_raw_background_rms,
        "control_raw_target_to_background_rms": control_raw_target_rms / max(control_raw_background_rms, 1e-30),
        "contrast_target_rms": contrast_target_rms,
        "contrast_background_rms": contrast_background_rms,
        "contrast_target_to_background_rms": contrast_target_rms / max(contrast_background_rms, 1e-30),
        "contrast_to_full_target_rms": contrast_target_rms / max(full_target_rms, 1e-30),
        "contrast_to_full_raw_target_rms": contrast_target_rms / max(full_raw_target_rms, 1e-30),
        "late_time_window_ns": [600.0, 700.0],
        "full_late_to_target_rms": full_late_rms / max(full_target_rms, 1e-30),
        "control_late_to_target_rms": control_late_rms / max(control_target_rms, 1e-30),
        "contrast_late_to_target_rms": contrast_late_rms / max(contrast_target_rms, 1e-30),
        "endpoint_trace_count_each_side": endpoint_width,
        "full_endpoint_to_interior_target_rms": full_endpoint_target_rms / max(full_interior_target_rms, 1e-30),
        "contrast_endpoint_to_interior_target_rms": contrast_endpoint_target_rms / max(contrast_interior_target_rms, 1e-30),
        "comparison_window_definition": {
            "target_half_width_ns": 18.0,
            "background_time_ns": [300.0, 500.0],
            "purpose": "same numeric window used by audit_macro_pilot_line9.py",
        },
        "comparison_full_target_to_background_rms": float(
            np.sqrt(np.mean(full_suppressed[comparison_target_band] ** 2))
            / max(np.sqrt(np.mean(full_suppressed[comparison_background_band] ** 2)), 1e-30)
        ),
        "comparison_control_target_to_background_rms": float(
            np.sqrt(np.mean(control_suppressed[comparison_target_band] ** 2))
            / max(np.sqrt(np.mean(control_suppressed[comparison_background_band] ** 2)), 1e-30)
        ),
        "comparison_contrast_target_to_background_rms": float(
            np.sqrt(np.mean(contrast[comparison_target_band] ** 2))
            / max(np.sqrt(np.mean(contrast[comparison_background_band] ** 2)), 1e-30)
        ),
        "artifact_hash_contract": artifact_hash_contract,
        "artifact_hash_contract_ok": artifact_hash_contract_ok,
        "numerical_pair_alignment_ok": numerical_pair_alignment_ok,
        "pair_physics_contract_ok": pair_physics_contract_ok,
        "formal_provenance_complete": formal_provenance_complete,
        "control_per_trace_contract": control_trace_contract,
        "full_per_trace_contract": full_trace_contract,
        "pair_contract_ok": bool(pair_physics_contract_ok and formal_provenance_complete),
        "quality_gate_decision": "manual_review_required_no_preregistered_promotion_threshold",
    }
    out = case_dir / "pair_audit"
    out.mkdir(exist_ok=True)
    np.save(out / "full_501x128.npy", full.astype(np.float32))
    np.save(out / "no_basal_501x128.npy", control.astype(np.float32))
    np.save(out / "contrast_501x128.npy", contrast.astype(np.float32))
    np.save(out / "visible_phase_time_ns.npy", visible.astype(np.float32))
    (out / "pair_audit_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    band = (time_ns >= max(0.0, float(np.min(visible)) - 70.0)) & (time_ns <= min(700.0, float(np.max(visible)) + 70.0))
    raw = full[band] - np.median(full[band], axis=1, keepdims=True)
    control_raw = control[band] - np.median(control[band], axis=1, keepdims=True)
    diff = contrast[band]
    canvas = Image.new("RGB", (1840, 500), "white")
    canvas.paste(panel(raw), (10, 60)); canvas.paste(panel(diff), (930, 60))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 20), "Full scene: trace-median suppressed", fill="black")
    draw.text((930, 20), "Matched contrast: full - no-basal", fill="black")
    lo, hi = float(time_ns[band][0]), float(time_ns[band][-1])
    for x0 in (10, 930):
        points = [(x0 + int(i / max(len(visible)-1, 1) * 899), 60 + int((float(v)-lo) / max(hi-lo, 1e-9) * 379)) for i, v in enumerate(visible)]
        draw.line(points, fill=(0, 230, 235), width=3)
    canvas.save(out / "pair_target_review.png")

    common_scale = float(np.percentile(np.abs(np.concatenate((raw, control_raw), axis=1)), 99.5)) or 1.0
    matched = Image.new("RGB", (2760, 520), "white")
    matched.paste(panel(raw, scale=common_scale), (10, 70))
    matched.paste(panel(control_raw, scale=common_scale), (930, 70))
    matched.paste(panel(diff, scale=common_scale), (1850, 70))
    draw = ImageDraw.Draw(matched)
    draw.text((10, 18), "Full scene (shared full/control scale)", fill="black")
    draw.text((930, 18), "No-basal control (same scale)", fill="black")
    draw.text((1850, 18), "Full - control (same scale, no independent gain)", fill="black")
    draw.text(
        (10, 42),
        f"contrast/full target RMS={result['contrast_to_full_target_rms']:.3f}; "
        f"contrast target/background={result['contrast_target_to_background_rms']:.3f}",
        fill="black",
    )
    for x0 in (10, 930, 1850):
        visible_points = [
            (x0 + int(i / max(len(visible) - 1, 1) * 899), 70 + int((float(v) - lo) / max(hi - lo, 1e-9) * 379))
            for i, v in enumerate(visible)
        ]
        reference_points = [
            (x0 + int(i / max(len(reference) - 1, 1) * 899), 70 + int((float(v) - lo) / max(hi - lo, 1e-9) * 379))
            for i, v in enumerate(reference)
        ]
        draw.line(reference_points, fill=(255, 45, 30), width=2)
        draw.line(visible_points, fill=(0, 230, 235), width=3)
    draw.text(
        (10, 470),
        "Red: independent geometric reference (not a label); cyan: continuous signed phase from full - control",
        fill="black",
    )
    matched.save(out / "pair_target_review_matched_scale.png")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
