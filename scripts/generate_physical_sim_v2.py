#!/usr/bin/env python3
"""Generate physics-audited gprMax control scenes for simulation contract V2.

The generated controls are *not* training-approved. They establish the correct
geometry -> propagation-time -> post-solver visible-phase chain before a pilot
batch is allowed.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import (  # noqa: E402
    GridSpec,
    Material,
    SourceSpec,
    canonical_json_sha256,
    compress_column_boxes,
    control_point_interface_depth,
    gaussian_curve_mask,
    make_scene_arrays,
    sha256_file,
    smooth_interface_depth,
    snap_to_grid,
    write_json,
)

CONTRACT_DIR = ROOT / "data" / "contracts" / "simulation_v2"
DEFAULT_CASES = CONTRACT_DIR / "control_cases_v1.json"
DEFAULT_MATERIALS = CONTRACT_DIR / "materials_v1.json"
DEFAULT_OUT = ROOT / "data" / "simulations" / "v2" / "00_controls"


def _material(payload: dict[str, Any]) -> Material:
    return Material(
        name=str(payload["name"]),
        epsilon_r=float(payload["epsilon_r"]),
        conductivity_s_per_m=float(payload["conductivity_s_per_m"]),
    )


def _load_inputs(cases_path: Path, materials_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    materials = json.loads(materials_path.read_text(encoding="utf-8"))
    return cases, materials


def _input_header(
    *,
    title: str,
    domain_x_m: float,
    domain_y_m: float,
    grid: GridSpec,
    materials: list[Material],
    source: SourceSpec,
    src_x_m: float,
    rx_x_m: float,
    antenna_y_m: float,
    geometry_include: str | None,
    geometry_view_name: str | None = None,
) -> str:
    lines = [
        f"#title: {title}",
        f"#domain: {domain_x_m:.9g} {domain_y_m:.9g} {grid.dl_m:.9g}",
        f"#dx_dy_dz: {grid.dl_m:.9g} {grid.dl_m:.9g} {grid.dl_m:.9g}",
        f"#time_window: {grid.solver_time_window_ns * 1e-9:.9g}",
        "#pml_cells: " + " ".join(str(v) for v in grid.pml_cells),
        "#messages: y",
    ]
    lines.extend(m.gprmax_command() for m in materials)
    lines.extend(
        [
            f"#waveform: {source.waveform} {source.amplitude:.9g} "
            f"{source.center_frequency_hz:.9g} uavgpr_v2_wavelet",
            f"#hertzian_dipole: {source.polarization} {src_x_m:.9g} "
            f"{antenna_y_m:.9g} 0 uavgpr_v2_wavelet",
            f"#rx: {rx_x_m:.9g} {antenna_y_m:.9g} 0 rx1 Ez",
            f"#src_steps: {grid.trace_spacing_m:.9g} 0 0",
            f"#rx_steps: {grid.trace_spacing_m:.9g} 0 0",
        ]
    )
    if geometry_include:
        lines.append(f"#include_file: {geometry_include}")
    if geometry_view_name:
        lines.append(
            f"#geometry_view: 0 0 0 {domain_x_m:.9g} {domain_y_m:.9g} {grid.dl_m:.9g} "
            f"{grid.dl_m:.9g} {grid.dl_m:.9g} {grid.dl_m:.9g} {geometry_view_name} n"
        )
    return "\n".join(lines) + "\n"


def _build_geometry_commands(
    *,
    grid: GridSpec,
    domain_x_m: float,
    ground_y_at_cells: np.ndarray,
    cover_bottom_y_at_cells: np.ndarray,
    basal_y_at_cells: np.ndarray,
    cover_name: str,
    weathered_name: str | np.ndarray,
    bedrock_name: str | np.ndarray,
) -> list[str]:
    nx = int(round(domain_x_m / grid.dl_m))
    if any(arr.shape != (nx,) for arr in (ground_y_at_cells, cover_bottom_y_at_cells, basal_y_at_cells)):
        raise ValueError("geometry arrays do not match domain x cells")
    bottom = np.zeros(nx, dtype=np.float64)
    commands: list[str] = []
    # Object commands are processed in order. Fill all bedrock first, then
    # overwrite its upper part with weathered material and cover.
    def append_layer(lower: np.ndarray, upper: np.ndarray, names: str | np.ndarray) -> None:
        name_array = np.full(nx, names, dtype=object) if isinstance(names, str) else np.asarray(names, dtype=object)
        if name_array.shape != (nx,):
            raise ValueError("layer material names do not match domain x cells")
        start = 0
        for end in range(1, nx + 1):
            if end == nx or name_array[end] != name_array[start]:
                commands.extend(compress_column_boxes(
                    x0_m=start * grid.dl_m, dl_m=grid.dl_m,
                    lower_y_m=lower[start:end], upper_y_m=upper[start:end],
                    material_name=str(name_array[start]), z_size_m=grid.dl_m,
                ))
                start = end

    append_layer(bottom, basal_y_at_cells, bedrock_name)
    append_layer(basal_y_at_cells, cover_bottom_y_at_cells, weathered_name)
    commands.extend(
        compress_column_boxes(
            x0_m=0.0,
            dl_m=grid.dl_m,
            lower_y_m=cover_bottom_y_at_cells,
            upper_y_m=ground_y_at_cells,
            material_name=cover_name,
            z_size_m=grid.dl_m,
        )
    )
    return commands


def _render_preview(
    case_dir: Path,
    case: dict[str, Any],
    arrays,
    domain_x_m: float,
    domain_y_m: float,
    target_presence: bool,
) -> None:
    """Write a dependency-light geometry preview for control-case review.

    This generator runs in the simulation environment, where importing the
    project Matplotlib build may abort the process before any case is written.
    Pillow is sufficient for an audit preview and keeps case generation usable
    on both the solver machine and CI.
    """
    x = arrays.trace_midpoint_x_m
    ground = arrays.ground_y_m
    cover_bottom = arrays.cover_bottom_y_m
    basal = arrays.basal_interface_y_m
    antenna = arrays.antenna_y_m
    geom_t = arrays.geometric_arrival_time_ns

    canvas = Image.new("RGB", (1300, 800), "white")
    draw = ImageDraw.Draw(canvas)
    left, right = 85, 1240
    top, split, bottom = 55, 440, 745

    def points(values: np.ndarray, y0: int, y1: int, lo: float, hi: float) -> list[tuple[int, int]]:
        denom = max(hi - lo, 1e-9)
        return [
            (int(left + (float(xx) / max(domain_x_m, 1e-9)) * (right - left)),
             int(y1 - ((float(yy) - lo) / denom) * (y1 - y0)))
            for xx, yy in zip(x, values)
        ]

    draw.text((left, 18), f"{case['case_id']} geometry (target_presence={target_presence})", fill="black")
    for label, values, colour in (
        ("ground", ground, "black"), ("cover bottom", cover_bottom, "#c77c00"),
        ("nominal basal", basal, "#d62728"), ("antenna", antenna, "#1f77b4"),
    ):
        draw.line(points(values, top, split - 25, 0.0, domain_y_m), fill=colour, width=2)
        draw.text((left + 8, top + 18 * ("ground cover bottom nominal basal antenna".split().index(label.split()[0])),), label, fill=colour)
    draw.rectangle((left, top, right, split - 25), outline="gray")
    draw.text((left, split - 10), "gprMax y coordinate (m, upward)", fill="black")

    draw.text((left, split + 12), f"Pre-solver arrival ({arrays.arrival_model}); visible phase pending", fill="black")
    draw.rectangle((left, split + 42, right, bottom), outline="gray")
    if target_presence:
        # Time increases downward in the lower panel.
        arrival = [(int(left + (float(xx) / max(domain_x_m, 1e-9)) * (right - left)),
                    int(split + 42 + (float(tt) / 700.0) * (bottom - (split + 42))))
                   for xx, tt in zip(x, geom_t)]
        draw.line(arrival, fill="#d62728", width=2)
    else:
        draw.text((left + 12, split + 60), "No target arrival", fill="gray")
    canvas.save(case_dir / "preview_geometry_and_arrival.png")


def generate_case(
    case: dict[str, Any],
    material_sets: dict[str, Any],
    out_root: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = out_root / case_id
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Case exists: {case_dir}")
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    grid = GridSpec(**case.get("grid", {}))
    source = SourceSpec(**case.get("source", {}))
    material_payload = material_sets["sets"][case["material_set"]]
    cover = _material(material_payload["cover"])
    weathered = _material(material_payload["weathered"])
    bedrock = _material(material_payload["bedrock"])
    weathered_variants = [_material(item) for item in material_payload.get("weathered_variants", [])]
    bedrock_variants = [_material(item) for item in material_payload.get("bedrock_variants", [])]
    materials = [cover, weathered, bedrock, *weathered_variants, *bedrock_variants]
    max_er = max(m.epsilon_r for m in materials)
    grid.validate(max_er, source.center_frequency_hz)
    source.validate(grid)

    x_local = np.arange(grid.trace_count, dtype=np.float64) * grid.trace_spacing_m
    interface = case["interface"]
    if interface.get("kind") == "control_points":
        basal_depth = control_point_interface_depth(
            x_local,
            base_depth_m=float(interface["base_depth_m"]),
            control_point_fractions=interface["control_point_fractions"],
            depth_offsets_m=interface["depth_offsets_m"],
            smoothing_length_m=float(interface.get("smoothing_length_m", 1.0)),
        )
    else:
        basal_depth = smooth_interface_depth(
            x_local,
            base_depth_m=float(interface["base_depth_m"]),
            slope=float(interface.get("slope", 0.0)),
            sinusoid_amplitude_m=float(interface.get("sinusoid_amplitude_m", 0.0)),
            sinusoid_wavelength_m=interface.get("sinusoid_wavelength_m"),
        )
    agl = np.full(grid.trace_count, float(case["flight_height_agl_m"]), dtype=np.float64)

    bedrock_below_m = 3.0
    ground_baseline = snap_to_grid(
        grid.pml_guard_m + bedrock_below_m + float(np.max(basal_depth)), grid.dl_m
    )
    terrain = case["terrain"]
    if terrain.get("kind", "flat") == "control_points":
        terrain_offsets = control_point_interface_depth(
            x_local,
            base_depth_m=1.0,
            control_point_fractions=terrain["control_point_fractions"],
            depth_offsets_m=terrain["elevation_offsets_m"],
            smoothing_length_m=float(terrain.get("smoothing_length_m", 1.0)),
        ) - 1.0
        ground_y = ground_baseline + terrain_offsets
    elif terrain.get("kind", "flat") == "flat":
        ground_y = ground_baseline
    else:
        raise ValueError(f"Unsupported terrain kind: {terrain.get('kind')!r}")
    arrays = make_scene_arrays(
        grid=grid,
        source=source,
        basal_depth_m=basal_depth,
        flight_height_agl_m=agl,
        cover_fraction=float(case["cover_fraction"]) if "cover_fraction" in case else None,
        cover_thickness_m=(
            float(case["cover_boundary"]["base_depth_m"])
            if "cover_boundary" in case else None
        ),
        ground_y_m=ground_y,
        cover_material=cover,
        weathered_material=weathered,
        arrival_model=(
            "horizontal_layered_bistatic_exact"
            if str(interface.get("kind", "flat")) == "flat"
            else "columnar_layered_reference_not_specular_exact"
        ),
        scan_left_margin_m=case.get("scan_left_margin_m"),
    )

    # Mirror the left one-cell snapping slack in the right domain extent.
    right_margin = float(case.get("scan_right_margin_m", grid.pml_guard_m + grid.dl_m))
    if right_margin < grid.pml_guard_m + grid.dl_m:
        raise ValueError("scan_right_margin_m must preserve the PML/guard clearance")
    domain_x = snap_to_grid(float(arrays.receiver_x_m[-1] + right_margin), grid.dl_m)
    domain_y_required = float(np.max(arrays.antenna_y_m) + grid.pml_guard_m)
    domain_y = snap_to_grid(max(domain_y_required, float(case.get("domain_min_y_m", 0.0))), grid.dl_m)
    nx = int(round(domain_x / grid.dl_m))
    x_centers = (np.arange(nx, dtype=np.float64) + 0.5) * grid.dl_m
    ground_cells = np.interp(
        x_centers, arrays.trace_midpoint_x_m, arrays.ground_y_m,
        left=arrays.ground_y_m[0], right=arrays.ground_y_m[-1]
    )
    cover_bottom_cells = np.interp(
        x_centers, arrays.trace_midpoint_x_m, arrays.cover_bottom_y_m,
        left=arrays.cover_bottom_y_m[0], right=arrays.cover_bottom_y_m[-1]
    )
    basal_cells = np.interp(
        x_centers, arrays.trace_midpoint_x_m, arrays.basal_interface_y_m,
        left=arrays.basal_interface_y_m[0], right=arrays.basal_interface_y_m[-1]
    )
    ground_cells = np.asarray(snap_to_grid(ground_cells, grid.dl_m))
    cover_bottom_cells = np.asarray(snap_to_grid(cover_bottom_cells, grid.dl_m))
    basal_cells = np.asarray(snap_to_grid(basal_cells, grid.dl_m))

    def lateral_names(base: Material, variants: list[Material], key: str) -> str | np.ndarray:
        zones = case.get("lateral_material_zones", [])
        if not zones:
            return base.name
        available = {base.name: base, **{m.name: m for m in variants}}
        selected = np.full(nx, base.name, dtype=object)
        for zone in zones:
            material_name = str(zone[key])
            if material_name not in available:
                raise ValueError(f"unknown {key} material: {material_name}")
            lo, hi = float(zone["start_fraction"]), float(zone["end_fraction"])
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError("lateral material zones must be ordered fractions within [0, 1]")
            mask = (x_centers / domain_x >= lo) & (x_centers / domain_x < hi)
            selected[mask] = material_name
        return selected

    weathered_names = lateral_names(weathered, weathered_variants, "weathered_material")
    bedrock_names = lateral_names(bedrock, bedrock_variants, "bedrock_material")

    target_presence = bool(case["target_presence"])
    full_bedrock_name = bedrock.name if target_presence else weathered.name
    full_commands = _build_geometry_commands(
        grid=grid,
        domain_x_m=domain_x,
        ground_y_at_cells=ground_cells,
        cover_bottom_y_at_cells=cover_bottom_cells,
        basal_y_at_cells=basal_cells,
        cover_name=cover.name,
        weathered_name=weathered_names,
        bedrock_name=bedrock_names if target_presence else weathered_names,
    )
    control_commands = _build_geometry_commands(
        grid=grid,
        domain_x_m=domain_x,
        ground_y_at_cells=ground_cells,
        cover_bottom_y_at_cells=cover_bottom_cells,
        basal_y_at_cells=basal_cells,
        cover_name=cover.name,
        weathered_name=weathered_names,
        bedrock_name=weathered_names,
    )

    (case_dir / "full_scene_geometry.inc").write_text("\n".join(full_commands) + "\n", encoding="utf-8")
    (case_dir / "no_basal_contrast_geometry.inc").write_text(
        "\n".join(control_commands) + "\n", encoding="utf-8"
    )

    common = dict(
        domain_x_m=domain_x,
        domain_y_m=domain_y,
        grid=grid,
        materials=materials,
        source=source,
        src_x_m=float(arrays.source_x_m[0]),
        rx_x_m=float(arrays.receiver_x_m[0]),
        antenna_y_m=float(arrays.antenna_y_m[0]),
    )
    (case_dir / "full_scene.in").write_text(
        _input_header(
            title=f"PGDA SIM V2 {case_id} full scene",
            geometry_include="full_scene_geometry.inc",
            **common,
        ),
        encoding="utf-8",
    )
    if target_presence:
        (case_dir / "no_basal_contrast_control.in").write_text(
            _input_header(
                title=f"PGDA SIM V2 {case_id} no basal contrast control",
                geometry_include="no_basal_contrast_geometry.inc",
                **common,
            ),
            encoding="utf-8",
        )
    (case_dir / "air_reference.in").write_text(
        _input_header(
            title=f"PGDA SIM V2 {case_id} air reference",
            geometry_include=None,
            **common,
        ),
        encoding="utf-8",
    )
    (case_dir / "geometry_check_full.in").write_text(
        _input_header(
            title=f"PGDA SIM V2 {case_id} full geometry check",
            geometry_include="full_scene_geometry.inc",
            geometry_view_name="geometry_full",
            **common,
        ),
        encoding="utf-8",
    )
    if target_presence:
        (case_dir / "geometry_check_control.in").write_text(
            _input_header(
                title=f"PGDA SIM V2 {case_id} control geometry check",
                geometry_include="no_basal_contrast_geometry.inc",
                geometry_view_name="geometry_control",
                **common,
            ),
            encoding="utf-8",
        )

    labels = case_dir / "labels"
    labels.mkdir()
    canonical_time = np.linspace(0.0, grid.canonical_time_window_ns, grid.output_samples, dtype=np.float32)
    geometric = arrays.geometric_arrival_time_ns.astype(np.float32)
    if not target_presence:
        geometric_training = np.full_like(geometric, np.nan)
        geometric_prior = np.zeros((grid.output_samples, grid.trace_count), dtype=np.float32)
    else:
        geometric_training = geometric
        geometric_prior = gaussian_curve_mask(canonical_time, geometric, sigma_ns=8.4)
    np.save(labels / "time_501_ns.npy", canonical_time)
    np.save(labels / "trace_midpoint_x_m.npy", arrays.trace_midpoint_x_m.astype(np.float32))
    np.save(labels / "source_x_m.npy", arrays.source_x_m.astype(np.float32))
    np.save(labels / "receiver_x_m.npy", arrays.receiver_x_m.astype(np.float32))
    np.save(labels / "ground_y_m.npy", arrays.ground_y_m.astype(np.float32))
    np.save(labels / "flight_height_agl_m.npy", arrays.flight_height_agl_m.astype(np.float32))
    np.save(labels / "antenna_y_m.npy", arrays.antenna_y_m.astype(np.float32))
    np.save(labels / "basal_interface_depth_m.npy", arrays.basal_depth_m.astype(np.float32))
    np.save(labels / "cover_thickness_m.npy", arrays.cover_thickness_m.astype(np.float32))
    np.save(labels / "weathered_thickness_m.npy", arrays.weathered_thickness_m.astype(np.float32))
    np.save(labels / "reference_arrival_time_ns.npy", geometric_training)
    # Compatibility alias retained for downstream code. For non-flat scenes
    # this is a columnar layered reference, not an exact specular ray time.
    np.save(labels / "geometric_arrival_time_ns.npy", geometric_training)
    np.save(labels / "visible_phase_time_ns.npy", np.full(grid.trace_count, np.nan, dtype=np.float32))
    np.save(labels / "geometric_prior_not_for_training_501x256.npy", geometric_prior)
    np.save(labels / "target_mask_pending_postprocess_501x256.npy", np.zeros_like(geometric_prior))

    source_hash = sha256_file(Path(__file__))
    try:
        generator_git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        generator_worktree_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", str(Path(__file__).relative_to(ROOT)),
                 "pgdacsnet/simulation_v2.py", "data/contracts/simulation_v2"],
                cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        generator_git_commit = "unavailable"
        generator_worktree_dirty = True
    scene_payload: dict[str, Any] = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "case_id": case_id,
        "family": case["family"],
        "purpose": case["purpose"],
        "target_presence": target_presence,
        "formal_training_allowed": False,
        "training_block_reason": "control scene requires actual gprMax run and visible-phase postprocessing",
        "line9_conditioned": False,
        "reference_line": None,
        "generator_path": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": source_hash,
        "generator_git_commit": generator_git_commit,
        "generator_worktree_dirty": generator_worktree_dirty,
        "deterministic_seed": int(case.get("seed", 0)),
        "case_spec_sha256": canonical_json_sha256(case),
        "grid": {
            "dimension": "2D_x_y_with_one_z_cell",
            "dl_m": grid.dl_m,
            "domain_x_m": domain_x,
            "domain_y_m": domain_y,
            "pml_cells": list(grid.pml_cells),
            "guard_cells": grid.guard_cells,
            "trace_count": grid.trace_count,
            "trace_spacing_m": grid.trace_spacing_m,
            "trace_midpoint_span_m": grid.scan_span_m,
            "trace_span_m": grid.scan_span_m,
            "tx_endpoint_span_m": float(arrays.source_x_m[-1] - arrays.source_x_m[0]),
            "rx_endpoint_span_m": float(arrays.receiver_x_m[-1] - arrays.receiver_x_m[0]),
            "solver_time_window_ns": grid.solver_time_window_ns,
            "gprmax_time_window_ns": grid.solver_time_window_ns,
            "canonical_time_window_ns": grid.canonical_time_window_ns,
            "canonical_output_samples": grid.output_samples,
            "canonical_output_dt_ns": grid.output_dt_ns,
        },
        "source": {
            "model": source.model,
            "polarization": source.polarization,
            "waveform": source.waveform,
            "center_frequency_hz": source.center_frequency_hz,
            "tx_rx_offset_m": source.tx_rx_offset_m,
            "assumption_status": source.assumption_status,
        },
        "materials": {
            "set": case["material_set"],
            "cover": cover.__dict__,
            "weathered": weathered.__dict__,
            "bedrock": bedrock.__dict__,
            "full_scene_bedrock_material": full_bedrock_name,
            "control_bedrock_material": weathered.name,
        },
        "geometry": {
            "terrain_kind": case["terrain"]["kind"],
            "nominal_interface": case["interface"],
            "cover_fraction": float(case["cover_fraction"]) if "cover_fraction" in case else None,
            "cover_boundary": case.get("cover_boundary"),
            "cover_model": (
                {"kind": "fraction", "fraction": float(case["cover_fraction"])}
                if "cover_fraction" in case else case["cover_boundary"]
            ),
            "bedrock_below_min_m": bedrock_below_m,
            "arrival_model": arrays.arrival_model,
            "matched_positive_case_id": case.get("matched_positive_case_id"),
            "matched_negative_case_id": case.get("matched_negative_case_id"),
            "coordinate_convention": "x survey direction; y upward from lower-left origin; z invariant one cell",
        },
        "labels": {
            "reference_arrival": arrays.arrival_model if target_presence else "not_applicable",
            "geometric_arrival": "compatibility alias; inspect geometry.arrival_model before interpretation",
            "visible_phase": "pending full minus no-basal-control postprocess",
            "geometric_prior_training_allowed": False,
            "target_mask_training_allowed": False,
        },
    }
    write_json(case_dir / "scene_manifest.json", scene_payload)

    run_lines = [
        f"# {case_id}",
        "",
        "## Geometry-only checks",
        "```bash",
        "python -m gprMax geometry_check_full.in --geometry-only",
    ]
    if target_presence:
        run_lines.append("python -m gprMax geometry_check_control.in --geometry-only")
    run_lines.extend(
        [
            "```",
            "",
            "## B-scan runs (256 traces; source/receiver step = 0.09 m)",
            "```bash",
            "python -m gprMax full_scene.in -n 256 --geometry-fixed",
            "python -m tools.outputfiles_merge full_scene --remove-files",
        ]
    )
    if target_presence:
        run_lines.extend(
            [
                "python -m gprMax no_basal_contrast_control.in -n 256 --geometry-fixed",
                "python -m tools.outputfiles_merge no_basal_contrast_control --remove-files",
            ]
        )
    run_lines.extend(
        [
            "python -m gprMax air_reference.in -n 256 --geometry-fixed",
            "python -m tools.outputfiles_merge air_reference --remove-files",
            "```",
            "",
            "The merged HDF5 outputs must then be passed to scripts/postprocess_physical_sim_v2.py.",
        ]
    )
    (case_dir / "RUN_COMMANDS.md").write_text("\n".join(run_lines) + "\n", encoding="utf-8")

    _render_preview(case_dir, case, arrays, domain_x, domain_y, target_presence)

    hash_rows = []
    for path in sorted(p for p in case_dir.rglob("*") if p.is_file() and p.name != "FILE_SHA256.csv"):
        hash_rows.append((str(path.relative_to(case_dir)), sha256_file(path), path.stat().st_size))
    with (case_dir / "FILE_SHA256.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["relative_path", "sha256", "size_bytes"])
        writer.writerows(hash_rows)
    return scene_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases, materials = _load_inputs(Path(args.cases), Path(args.materials))
    selected = set(args.case_id)
    if selected:
        cases = [case for case in cases if case["case_id"] in selected]
        missing = selected - {case["case_id"] for case in cases}
        if missing:
            raise SystemExit(f"Unknown case IDs: {sorted(missing)}")
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifests = [generate_case(case, materials, out_root, overwrite=args.overwrite) for case in cases]
    generated_entries = [
        {
            "case_id": m["case_id"],
            "family": m["family"],
            "target_presence": m["target_presence"],
            "case_path": (
                (out_root / m["case_id"]).relative_to(ROOT).as_posix()
                if (out_root / m["case_id"]).is_relative_to(ROOT)
                else str((out_root / m["case_id"]).resolve())
            ),
            "manifest_sha256": sha256_file(out_root / m["case_id"] / "scene_manifest.json"),
            "train_allowed": False,
        }
        for m in manifests
    ]
    if selected and (out_root / "control_index.json").is_file():
        previous = json.loads((out_root / "control_index.json").read_text(encoding="utf-8"))
        entries_by_id = {entry["case_id"]: entry for entry in previous.get("cases", [])}
        entries_by_id.update({entry["case_id"]: entry for entry in generated_entries})
        ordered_ids = [case["case_id"] for case in _load_inputs(Path(args.cases), Path(args.materials))[0]]
        generated_entries = [entries_by_id[case_id] for case_id in ordered_ids if case_id in entries_by_id]
    index = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "formal_training_allowed": False,
        "line9_conditioned": False,
        "case_count": len(generated_entries),
        "cases": generated_entries,
    }
    write_json(out_root / "control_index.json", index)
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
