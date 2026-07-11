#!/usr/bin/env python3
"""Postprocess gprMax V2 controls into canonical auditable arrays.

Positive controls use a matched no-basal-contrast simulation to identify the
visible target phase. Negative controls produce a zero target mask only after
the full-scene HDF5 output has passed shape, time-axis, and metadata checks.
No case is automatically approved for formal training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import (  # noqa: E402
    extract_visible_phase,
    gaussian_curve_mask,
    resample_time_axis,
    sha256_file,
    write_json,
)


def _decode_attr(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_merged_bscan(path: Path, component: str = "Ez") -> tuple[float, np.ndarray, dict[str, Any]]:
    """Read an official gprMax single or merged HDF5 receiver dataset."""

    with h5py.File(path, "r") as f:
        dt_s = float(f.attrs["dt"])
        dataset_path = f"/rxs/rx1/{component}"
        if dataset_path not in f:
            raise KeyError(f"Missing receiver component {dataset_path} in {path}")
        data = np.asarray(f[dataset_path])
        attrs: dict[str, Any] = {
            "gprMax": _decode_attr(f.attrs.get("gprMax", "")),
            "Iterations": int(f.attrs.get("Iterations", data.shape[0])),
            "dx_dy_dz": np.asarray(f.attrs.get("dx_dy_dz", [])).astype(float).tolist(),
            "srcsteps": np.asarray(f.attrs.get("srcsteps", [])).astype(float).tolist(),
            "rxsteps": np.asarray(f.attrs.get("rxsteps", [])).astype(float).tolist(),
            "nrx": int(f.attrs.get("nrx", 0)),
        }
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim == 2 and data.shape[0] < data.shape[1]:
        data = data.T
    if data.ndim != 2:
        raise ValueError(f"Expected [time, trace] receiver data, got {data.shape}")
    return dt_s, data.astype(np.float32), attrs


def validate_hdf5_contract(
    *,
    path: Path,
    data: np.ndarray,
    dt_s: float,
    attrs: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    grid = manifest["grid"]
    if data.shape[1] != int(grid["trace_count"]):
        errors.append(f"{path.name}: expected {grid['trace_count']} traces, got {data.shape[1]}")
    if dt_s <= 0:
        errors.append(f"{path.name}: invalid dt={dt_s}")
    iterations = int(attrs.get("Iterations", -1))
    if iterations != data.shape[0]:
        errors.append(
            f"{path.name}: Iterations={iterations} does not match receiver samples={data.shape[0]}"
        )
    canonical_end_ns = float(grid.get("canonical_time_window_ns", 700.0))
    solver_window_ns = float(grid.get("solver_time_window_ns", grid.get("gprmax_time_window_ns", 701.0)))
    stored_last_ns = (data.shape[0] - 1) * dt_s * 1e9
    solver_coverage_ns = data.shape[0] * dt_s * 1e9
    if stored_last_ns < canonical_end_ns - 1e-6:
        errors.append(
            f"{path.name}: last stored sample {stored_last_ns:.6f} ns is before canonical endpoint {canonical_end_ns:.6f} ns"
        )
    if solver_coverage_ns < solver_window_ns - 1e-6:
        errors.append(
            f"{path.name}: Iterations*dt {solver_coverage_ns:.6f} ns is shorter than requested solver window {solver_window_ns:.6f} ns"
        )
    expected_dl = float(grid["dl_m"])
    dx = np.asarray(attrs.get("dx_dy_dz", []), dtype=float)
    if dx.size and (dx.size != 3 or not np.allclose(dx, expected_dl, atol=1e-12, rtol=0)):
        errors.append(f"{path.name}: dx_dy_dz={dx.tolist()} does not match dl={expected_dl}")
    expected_step = np.array([float(grid["trace_spacing_m"]), 0.0, 0.0])
    for key in ("srcsteps", "rxsteps"):
        value = np.asarray(attrs.get(key, []), dtype=float)
        if value.size and (value.size != 3 or not np.allclose(value, expected_step, atol=1e-12, rtol=0)):
            errors.append(f"{path.name}: {key}={value.tolist()} does not match {expected_step.tolist()}")
    if not str(attrs.get("gprMax", "")).strip():
        errors.append(f"{path.name}: missing gprMax version attribute")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir")
    parser.add_argument("--full", default="full_scene_merged.out")
    parser.add_argument("--control", default="no_basal_contrast_control_merged.out")
    parser.add_argument("--air", default="air_reference_merged.out")
    parser.add_argument("--component", default="Ez")
    parser.add_argument("--arrival-tolerance-ns", type=float, default=5.6)
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    target_presence = bool(manifest["target_presence"])

    full_path = case_dir / args.full
    air_path = case_dir / args.air
    required = [full_path, air_path]
    control_path = case_dir / args.control
    if target_presence:
        required.append(control_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    full_dt, full_raw, full_attrs = read_merged_bscan(full_path, args.component)
    air_dt, air_raw, air_attrs = read_merged_bscan(air_path, args.component)
    runtime_errors = validate_hdf5_contract(
        path=full_path, data=full_raw, dt_s=full_dt, attrs=full_attrs, manifest=manifest
    )
    runtime_errors.extend(
        validate_hdf5_contract(path=air_path, data=air_raw, dt_s=air_dt, attrs=air_attrs, manifest=manifest)
    )
    if not np.isclose(full_dt, air_dt):
        runtime_errors.append("full and air CFL time steps differ")
    if full_raw.shape != air_raw.shape:
        runtime_errors.append("full and air output shapes differ")

    control_raw = None
    control_attrs: dict[str, Any] | None = None
    if target_presence:
        control_dt, control_raw, control_attrs = read_merged_bscan(control_path, args.component)
        runtime_errors.extend(
            validate_hdf5_contract(
                path=control_path,
                data=control_raw,
                dt_s=control_dt,
                attrs=control_attrs,
                manifest=manifest,
            )
        )
        if not np.isclose(full_dt, control_dt):
            runtime_errors.append("full and matched-control CFL time steps differ")
        if full_raw.shape != control_raw.shape:
            runtime_errors.append("full and matched-control output shapes differ")

    if runtime_errors:
        raise RuntimeError("; ".join(runtime_errors))

    canonical_end_ns = float(manifest["grid"].get("canonical_time_window_ns", 700.0))
    canonical_samples = int(manifest["grid"].get("canonical_output_samples", 501))
    time_ns, full = resample_time_axis(
        full_raw, full_dt, time_window_ns=canonical_end_ns, output_samples=canonical_samples
    )
    _, air = resample_time_axis(
        air_raw, air_dt, time_window_ns=canonical_end_ns, output_samples=canonical_samples
    )
    labels = case_dir / "labels"
    labels.mkdir(exist_ok=True)
    np.save(labels / "full_scene_501x256.npy", full.astype(np.float32))
    np.save(labels / "air_reference_501x256.npy", air.astype(np.float32))

    source_hashes = {"full": sha256_file(full_path), "air": sha256_file(air_path)}
    hdf5_attrs: dict[str, Any] = {"full": full_attrs, "air": air_attrs}

    if target_presence:
        assert control_raw is not None and control_attrs is not None
        _, control = resample_time_axis(
            control_raw, full_dt, time_window_ns=canonical_end_ns, output_samples=canonical_samples
        )
        reference_path = labels / "reference_arrival_time_ns.npy"
        if not reference_path.is_file():
            reference_path = labels / "geometric_arrival_time_ns.npy"
        reference = np.load(reference_path, allow_pickle=False)
        visible, support, contrast = extract_visible_phase(full, control, time_ns, reference)
        delta = visible - reference
        if not np.isfinite(delta).all():
            raise RuntimeError("visible phase extraction failed on one or more traces")
        phase_offset = float(np.median(delta))
        residual = delta - phase_offset
        max_abs_residual = float(np.max(np.abs(residual)))
        arrival_model = str(manifest.get("geometry", {}).get("arrival_model", ""))
        exact_reference = arrival_model == "horizontal_layered_bistatic_exact"
        continuity_p95_ns = float(np.percentile(np.abs(np.diff(visible)), 95)) if visible.size > 1 else 0.0
        if exact_reference:
            passed = max_abs_residual <= args.arrival_tolerance_ns
        else:
            # A columnar layered time is a search reference for curved scenes,
            # not an exact specular-ray truth. Gate on extraction support and
            # continuity; do not report a geometric-arrival accuracy claim.
            passed = bool(np.median(support) > 1.0 and continuity_p95_ns <= 14.0)
        target_mask = gaussian_curve_mask(time_ns, visible, sigma_ns=8.4)

        np.save(labels / "visible_phase_time_ns.npy", visible.astype(np.float32))
        np.save(labels / "target_mask_visible_phase_501x256.npy", target_mask)
        np.save(labels / "contrast_response_501x256.npy", contrast.astype(np.float32))
        np.save(labels / "no_basal_contrast_501x256.npy", control.astype(np.float32))
        np.save(labels / "visible_phase_support_ratio.npy", support.astype(np.float32))
        source_hashes["control"] = sha256_file(control_path)
        hdf5_attrs["control"] = control_attrs
        details = {
            "arrival_model": arrival_model,
            "reference_is_exact_flat_layered_model": exact_reference,
            "median_visible_minus_reference_ns": phase_offset,
            "max_abs_trace_residual_after_phase_offset_ns": max_abs_residual,
            "arrival_tolerance_ns": args.arrival_tolerance_ns if exact_reference else None,
            "visible_curve_abs_step_p95_ns": continuity_p95_ns,
            "support_median": float(np.median(support)),
            "support_min": float(np.min(support)),
            "target_mask_path": "labels/target_mask_visible_phase_501x256.npy",
        }
    else:
        passed = True
        target_mask = np.zeros_like(full, dtype=np.float32)
        np.save(labels / "target_mask_confirmed_negative_501x256.npy", target_mask)
        np.save(labels / "visible_phase_time_ns.npy", np.full(full.shape[1], np.nan, dtype=np.float32))
        details = {
            "confirmed_negative_semantics": True,
            "target_mask_nonzero_count": 0,
            "target_mask_path": "labels/target_mask_confirmed_negative_501x256.npy",
        }

    versions = {str(v.get("gprMax", "")) for v in hdf5_attrs.values()}
    version_consistent = len(versions) == 1 and "" not in versions
    passed = passed and version_consistent
    result = {
        "case_id": manifest["case_id"],
        "ok": passed,
        "postprocess_validated": passed,
        "metadata_trusted": False,
        "human_approved": False,
        "formal_training_allowed": False,
        "training_block_reason": "control solver output still requires visual/physical review and explicit contract promotion",
        "target_presence": target_presence,
        "component": args.component,
        "gprmax_version": full_attrs.get("gprMax"),
        "gprmax_version_consistent": version_consistent,
        "output_shape_cfl": list(full_raw.shape),
        "output_shape_canonical": list(full.shape),
        "cfl_dt_s": full_dt,
        "canonical_dt_ns": float(time_ns[1] - time_ns[0]),
        "source_hashes": source_hashes,
        "hdf5_attrs": hdf5_attrs,
        **details,
    }
    write_json(case_dir / "postprocess_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
