#!/usr/bin/env python3
"""Generate a small, independent basal-interface shape-family pilot.

The pilot changes only the basal geometry. Source, grid, cover field,
transition thickness, material tables, and acquisition remain matched across
each positive/full-control pair. The output is deliberately pre-solver and
formal-training blocked until runtime and signed-pair audits pass.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from generate_independent_v2_family01 import (
    C0,
    ROOT,
    Spec,
    _quadratic_metrics,
    _write_case,
    acquisition_arrays,
    build_indices,
    correlated_cover_bins,
    material_rows,
    render_preview,
    scan_arrays,
    sha256,
)


DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls" / "SHAPE01_BASAL_GEOMETRY_PILOT"
DEFAULT_CONTRACT = ROOT / "data" / "contracts" / "simulation_v2" / "basal_shape_family_pilot_v1.json"


def _transition_profile(spec: Spec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter1d

    noise = gaussian_filter1d(rng.standard_normal(spec.nx), sigma=8.0 / spec.dl_m, mode="reflect")
    noise = (noise - noise.mean()) / max(float(noise.std()), 1e-8)
    return np.clip(1.35 + 0.22 * noise, 1.0, 1.8).astype(np.float32)


def _shape_profiles(spec: Spec, shape_name: str, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    midpoint = spec.scan_start_x_m + spec.tx_rx_offset_m / 2 + np.arange(spec.trace_count) * spec.trace_spacing_m
    base = np.full(spec.nx, 15.2, dtype=np.float64)
    scan_center = spec.scan_start_x_m + spec.scan_span_m / 2 + spec.tx_rx_offset_m / 2

    if shape_name == "flat_reference":
        basal = base
    elif shape_name == "broad_rise":
        # The rise must be visible inside the 23 m acquisition span, while
        # remaining smooth enough to avoid a synthetic corner reflector.
        basal = base + 0.82 * np.exp(-((x - scan_center) / 10.5) ** 2)
    elif shape_name == "double_relief":
        basal = (
            base
            + 0.58 * np.exp(-((x - (scan_center - 5.5)) / 4.8) ** 2)
            - 0.42 * np.exp(-((x - (scan_center + 5.5)) / 5.2) ** 2)
        )
    elif shape_name == "gentle_multiscale":
        from generate_independent_v2_family01 import build_profiles

        inherited, inherited_metrics = build_profiles(spec, seed)
        return inherited, {**inherited_metrics, "shape_name": shape_name, "shape_family_role": "natural_multiscale"}
    else:
        raise ValueError(f"unknown shape: {shape_name}")

    transition = _transition_profile(spec, seed + 101)
    scan_basal = np.interp(midpoint, x, basal)
    r2, extrema, slope_p95 = _quadratic_metrics(midpoint, scan_basal)
    shape_metrics = {
        "shape_name": shape_name,
        "scan_depth_min_m": float(np.min(scan_basal)),
        "scan_depth_median_m": float(np.median(scan_basal)),
        "scan_depth_max_m": float(np.max(scan_basal)),
        "scan_depth_range_m": float(np.ptp(scan_basal)),
        "smoothed_extrema_count": extrema,
        "quadratic_fit_r2": r2,
        "abs_slope_p95": slope_p95,
        "broad_shape_gate_ok": bool(
            0.0 <= float(np.ptp(scan_basal)) <= 2.2 and extrema <= 8 and slope_p95 <= 0.20
        ),
        "flat_ground": True,
        "generated_from_measured_arrays": False,
    }
    if not shape_metrics["broad_shape_gate_ok"]:
        raise RuntimeError(f"shape failed broad geometry gate: {shape_metrics}")
    return {
        "x_m": x.astype(np.float32),
        "ground_y_m": np.full(spec.nx, spec.ground_y_m, dtype=np.float32),
        "basal_depth_m": basal.astype(np.float32),
        "transition_thickness_m": transition,
        "basal_y_m": (spec.ground_y_m - basal).astype(np.float32),
        "transition_top_y_m": (spec.ground_y_m - basal + transition).astype(np.float32),
    }, shape_metrics


def _contract(shape_name: str, scene_family_id: str, seed: int) -> dict[str, Any]:
    return {
        "contract_id": "PGDA_BASAL_SHAPE_FAMILY_PILOT_V1",
        "scene_family_id": scene_family_id,
        "formal_training_allowed": False,
        "promotion_allowed": False,
        "line9_conditioned": False,
        "provenance": {
            "line9_conditioned": False,
            "measured_files_read_by_generator": [],
            "shape_prior": "generic analytic and seeded multiscale geometry only",
            "shape_name": shape_name,
            "profile_seed": seed,
        },
        "source": {
            "model": "ideal_hertzian_line_source",
            "waveform": "55 MHz Ricker pulse proxy",
            "proxy_only": True,
            "not_sfcw": True,
        },
        "cases": [
            {
                "case_id": f"{scene_family_id}_POS",
                "target_presence": True,
                "negative_semantics": "not_a_negative_sample",
                "profile_seed_base": seed,
                "field_seed": 2026072001,
            },
            {
                "case_id": f"{scene_family_id}_MATCHED_NEG",
                "target_presence": False,
                "negative_semantics": "designed_true_negative_exactly_equal_to_positive_no_basal_physical_state",
                "profile_seed_base": seed,
                "field_seed": 2026072001,
            },
        ],
    }


def generate(output_root: Path, contract_path: Path, *, overwrite: bool) -> dict[str, Any]:
    spec = Spec()
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    shape_specs = (
        ("flat_reference", "BS01_FLAT_REFERENCE", 2026072001),
        ("broad_rise", "BS02_BROAD_RISE", 2026072011),
        ("double_relief", "BS03_DOUBLE_RELIEF", 2026072021),
        ("gentle_multiscale", "BS04_GENTLE_MULTISCALE", 2026072031),
    )
    all_results: list[dict[str, Any]] = []
    for shape_name, scene_family_id, seed in shape_specs:
        family_dir = output_root / scene_family_id
        if family_dir.exists():
            if not overwrite:
                raise FileExistsError(f"family exists; pass --overwrite: {family_dir}")
            shutil.rmtree(family_dir)
        family_dir.mkdir(parents=True)
        contract = _contract(shape_name, scene_family_id, seed)
        family_contract_path = family_dir / "shape_contract.json"
        profiles, shape_metrics = _shape_profiles(spec, shape_name, seed)
        bins, field_stats = correlated_cover_bins(spec, 2026072001)
        data = build_indices(spec, bins, profiles)
        full_rows = material_rows(control=False)
        control_rows = material_rows(control=True)
        geometry_source = family_dir / "_shared_geology_indices.h5"
        with h5py.File(geometry_source, "w") as handle:
            handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
            handle.attrs["generator"] = "scripts/generate_basal_shape_family_pilot.py"
            handle.attrs["scene_family_id"] = scene_family_id
            handle.create_dataset("data", data=data, dtype=np.int16, compression="gzip", compression_opts=4, shuffle=True)
        family_contract_path.write_text(json.dumps(contract, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        manifests = []
        for case in contract["cases"]:
            manifests.append(
                _write_case(
                    family_dir / case["case_id"], case, contract, spec, data, profiles,
                    shape_metrics, field_stats, full_rows, control_rows, geometry_source, family_contract_path,
                )
            )
        positive_dir = family_dir / contract["cases"][0]["case_id"]
        negative_dir = family_dir / contract["cases"][1]["case_id"]
        if sha256(positive_dir / "geology_indices.h5") != sha256(negative_dir / "geology_indices.h5"):
            raise RuntimeError("shape pair geometry is not byte-identical")
        preview = render_preview(
            family_dir, spec, data, profiles, full_rows, control_rows, shape_metrics,
            output_name=f"{scene_family_id}_GEOMETRY_PREVIEW.png",
            heading=f"Basal shape pilot: {shape_name}",
            positive_title="Positive full: matched cover + transition + basal contrast",
        )
        family_manifest = {
            "contract_id": contract["contract_id"],
            "scene_family_id": scene_family_id,
            "shape_name": shape_name,
            "formal_training_allowed": False,
            "promotion_allowed": False,
            "line9_conditioned": False,
            "case_ids": [case["case_id"] for case in contract["cases"]],
            "shared_geometry_sha256": sha256(positive_dir / "geology_indices.h5"),
            "shared_geometry_array_sha256": __import__("generate_independent_v2_family01").array_sha256(data),
            "exact_negative_equivalence": True,
            "profile_shape_gate": shape_metrics,
            "cover_field_statistics": field_stats,
            "preview": preview.name,
            "next_gate": "static and geometry-only audit, then one-trace full/no-basal pair",
        }
        (family_dir / "family_manifest.json").write_text(json.dumps(family_manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        from generate_independent_v2_family01 import write_checksums

        write_checksums(family_dir)
        all_results.append(family_manifest)
        geometry_source.unlink()
    report = {
        "pilot_id": "SHAPE01_BASAL_GEOMETRY_PILOT",
        "formal_training_allowed": False,
        "promotion_allowed": False,
        "shape_count": len(all_results),
        "families": all_results,
        "controls": "full_scene/no_basal_contrast_control/air_reference",
        "same_cover_field_across_shapes": True,
    }
    (output_root / "shape_family_pilot_manifest.json").write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_root.resolve(), args.contract.resolve(), overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
