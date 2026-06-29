from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
except Exception:  # pragma: no cover
    plt = None

from uavgpr_simlab.core.config import AppConfig
from uavgpr_simlab.core.scene_world import SceneObject, SceneWorld
from uavgpr_simlab.sim.materials import material_lines


def _fmt(v: float) -> str:
    return f"{float(v):.5g}"


def _safe_box_line(
    *,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    material: str,
    min_cell_m: float,
    domain_x_m: float,
    domain_y_m: float,
    domain_z_m: float,
) -> str | None:
    """Return a gprMax-safe #box line or None if it cannot fit.

    gprMax validates geometry after mapping coordinates to the FDTD grid.
    A positive but sub-cell-thick interval, for example a 0.05 m surface
    proxy in a 0.25 m grid, can be rejected as having equal lower/upper
    coordinates.  This helper clips boxes to the domain and guarantees at
    least one grid-cell of thickness in x/y/z when the domain has enough
    room.  It is intentionally conservative for SceneWorld generated boxes.
    """

    cell = max(float(min_cell_m), 1e-6)
    lo_x = max(0.0, min(float(x0), float(x1)))
    hi_x = min(float(domain_x_m), max(float(x0), float(x1)))
    lo_y = max(0.0, min(float(y0), float(y1)))
    hi_y = min(float(domain_y_m), max(float(y0), float(y1)))
    lo_z = max(0.0, min(float(z0), float(z1)))
    hi_z = min(float(domain_z_m), max(float(z0), float(z1)))

    def widen(lo: float, hi: float, domain: float) -> tuple[float, float] | None:
        domain = float(domain)
        if domain <= 0:
            return None
        if hi - lo >= cell - 1e-9:
            return lo, hi
        mid = (lo + hi) / 2.0
        lo2 = mid - cell / 2.0
        hi2 = mid + cell / 2.0
        if lo2 < 0.0:
            hi2 -= lo2
            lo2 = 0.0
        if hi2 > domain:
            lo2 -= hi2 - domain
            hi2 = domain
        lo2 = max(0.0, lo2)
        hi2 = min(domain, hi2)
        if hi2 - lo2 < max(cell * 0.5, 1e-6):
            return None
        return lo2, hi2

    wx = widen(lo_x, hi_x, float(domain_x_m))
    wy = widen(lo_y, hi_y, float(domain_y_m))
    wz = widen(lo_z, hi_z, float(domain_z_m))
    if wx is None or wy is None or wz is None:
        return None
    lo_x, hi_x = wx
    lo_y, hi_y = wy
    lo_z, hi_z = wz
    if not (lo_x < hi_x and lo_y < hi_y and lo_z < hi_z):
        return None
    return f"#box: {_fmt(lo_x)} {_fmt(lo_y)} {_fmt(lo_z)} {_fmt(hi_x)} {_fmt(hi_y)} {_fmt(hi_z)} {material}"


def validate_gprmax_box_lines(text: str, *, min_cell_m: float = 0.0) -> list[str]:
    """Return diagnostics for invalid or sub-cell #box commands."""

    problems: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("#box:"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            problems.append(f"line {lineno}: malformed #box command: {stripped}")
            continue
        try:
            x0, y0, z0, x1, y1, z1 = [float(v) for v in parts[1:7]]
        except ValueError:
            problems.append(f"line {lineno}: non-numeric #box coordinates: {stripped}")
            continue
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        if not (dx > 0 and dy > 0 and dz > 0):
            problems.append(f"line {lineno}: lower coordinates must be less than upper coordinates: {stripped}")
        if min_cell_m and (dx < min_cell_m - 1e-9 or dy < min_cell_m - 1e-9 or dz < min_cell_m - 1e-9):
            problems.append(f"line {lineno}: #box thinner than grid cell {min_cell_m:g} m: {stripped}")
    return problems


def _variants(vs: list[str] | tuple[str, ...]) -> list[str]:
    alias = {"target": "target_only", "clutter": "clutter_only", "background": "background_only", "air": "air_only"}
    out: list[str] = []
    for v in vs:
        vv = alias.get(str(v), str(v))
        if vv not in out:
            out.append(vv)
    return out


def _ground_at(world: SceneWorld, x: float) -> float:
    return float(np.interp(float(x), np.asarray(world.ground_x, dtype=float), np.asarray(world.ground_y, dtype=float)))


def _antenna_lines(cfg: AppConfig, world: SceneWorld) -> list[str]:
    z = world.domain_z_m / 2.0
    return [
        f"#waveform: ricker 1 {_fmt(cfg.radar.center_frequency_mhz * 1e6)} uavgpr_wavelet",
        f"#hertzian_dipole: z {_fmt(world.trajectory.scan_start_x_m)} {_fmt(world.trajectory.source_y_m)} {_fmt(z)} uavgpr_wavelet",
        f"#rx: {_fmt(world.trajectory.scan_start_x_m + world.trajectory.tx_rx_offset_m)} {_fmt(world.trajectory.receiver_y_m)} {_fmt(z)}",
        f"#src_steps: {_fmt(world.trace_spacing_m)} 0 0",
        f"#rx_steps: {_fmt(world.trace_spacing_m)} 0 0",
    ]


def _object_lines(cfg: AppConfig, world: SceneWorld, obj: SceneObject) -> list[str]:
    zmid = (float(obj.z0_m) + float(obj.z1_m)) / 2.0
    if obj.kind in {"wire", "tree"} and obj.radius_m:
        return [f"#cylinder: {_fmt(obj.x0_m)} {_fmt(obj.y0_m)} {_fmt(zmid)} {_fmt(obj.x1_m)} {_fmt(obj.y1_m)} {_fmt(zmid)} {_fmt(obj.radius_m)} {obj.material}"]
    line = _safe_box_line(
        x0=obj.x0_m, y0=obj.y0_m, z0=obj.z0_m, x1=obj.x1_m, y1=obj.y1_m, z1=obj.z1_m,
        material=obj.material,
        min_cell_m=max(float(cfg.geometry.dx_m), 1e-6),
        domain_x_m=world.domain_x_m,
        domain_y_m=world.domain_y_m,
        domain_z_m=world.domain_z_m,
    )
    return [line] if line else []


def _column_boxes(cfg: AppConfig, world: SceneWorld, variant: str) -> list[str]:
    if variant == "air_only":
        return []
    x = np.asarray(world.ground_x, dtype=float)
    ground = np.asarray(world.ground_y, dtype=float)
    iface = np.asarray(world.bedrock_interface_y, dtype=float)
    width = max(float(cfg.geometry.geometry_column_width_m), float(cfg.geometry.dx_m))
    lines: list[str] = []
    for i, xi in enumerate(x):
        x0 = float(xi)
        x1 = min(float(xi + width), world.domain_x_m)
        gy = float(ground[i])
        iy = float(iface[i])
        cover = world.cover_material_by_column[min(i, len(world.cover_material_by_column) - 1)]
        bed = world.bedrock_material_by_column[min(i, len(world.bedrock_material_by_column) - 1)]

        cell = max(float(cfg.geometry.dx_m), 1e-6)

        def add_box(y0: float, y1: float, material: str) -> None:
            line = _safe_box_line(
                x0=x0, y0=y0, z0=0.0, x1=x1, y1=y1, z1=world.domain_z_m,
                material=material,
                min_cell_m=cell,
                domain_x_m=world.domain_x_m,
                domain_y_m=world.domain_y_m,
                domain_z_m=world.domain_z_m,
            )
            if line:
                lines.append(line)

        if variant in {"raw", "target_only"}:
            add_box(0.0, iy, bed)
            # Overlay continuous interbed surrogates as deterministic column boxes.
            prev_y = iy
            for layer in world.interbed_layers:
                ly = float(np.interp((x0 + x1) / 2.0, np.asarray(layer.x_m), np.asarray(layer.y_m)))
                layer_thickness = max(cell, min(float(layer.thickness_m), max(0.4, cell)))
                if cell <= ly < prev_y - cell:
                    add_box(max(0.0, ly - layer_thickness), ly, layer.material)
                    prev_y = ly
            add_box(iy, gy, cover)
        elif variant == "background_only":
            # No target interface: fill below ground with cover/background material.
            add_box(0.0, gy, cover)
        elif variant == "clutter_only":
            # Keep a one-grid-cell surface proxy only; target interface is excluded.
            # Sub-cell proxies such as 0.05 m in a 0.25 m grid are invalid in gprMax.
            surface_thickness = max(cell, 0.05)
            add_box(max(0.0, gy - surface_thickness), gy, cover)
    return lines


def make_gprmax_input_from_scene(cfg: AppConfig, world: SceneWorld, variant: str) -> str:
    variant = _variants([variant])[0]
    lines = [
        f"#title: UavGPR-SimLab {world.case_id} {variant} SceneWorld {world.family}",
        f"#domain: {_fmt(world.domain_x_m)} {_fmt(world.domain_y_m)} {_fmt(world.domain_z_m)}",
        f"#dx_dy_dz: {_fmt(cfg.geometry.dx_m)} {_fmt(cfg.geometry.dx_m)} {_fmt(world.domain_z_m)}",
        f"#time_window: {_fmt(cfg.radar.time_window_ns * 1e-9)}",
        f"#pml_cells: 10 10 0 10 10 0",
        f"## trajectory_mode={world.trajectory.mode}; not true terrain-following unless metadata says otherwise",
    ]
    lines += material_lines(world.materials)
    lines += _column_boxes(cfg, world, variant)

    for obj in world.water_zones:
        if variant in obj.include_in:
            lines += _object_lines(cfg, world, obj)
    for obj in world.anomaly_objects:
        if variant in obj.include_in:
            lines += _object_lines(cfg, world, obj)
    for obj in world.external_clutter_objects:
        if variant in obj.include_in:
            lines += _object_lines(cfg, world, obj)

    lines += _antenna_lines(cfg, world)
    lines.append(
        f"#geometry_view: 0 0 0 {_fmt(world.domain_x_m)} {_fmt(world.domain_y_m)} {_fmt(world.domain_z_m)} "
        f"{_fmt(max(cfg.geometry.dx_m,0.05))} {_fmt(max(cfg.geometry.dx_m,0.05))} {_fmt(max(world.domain_z_m,cfg.geometry.dz_m))} geometry n"
    )
    text = "\n".join(lines) + "\n"
    problems = validate_gprmax_box_lines(text, min_cell_m=max(float(cfg.geometry.dx_m), 1e-6))
    if problems:
        detail = "\n".join(problems[:20])
        raise ValueError(f"Invalid gprMax #box geometry for {world.case_id}/{variant}:\n{detail}")
    return text




def compute_cover_velocity(world: "SceneWorld") -> float:
    """Compute average cover layer velocity from actual material properties."""
    eps_values = []
    for mat_name in world.cover_material_by_column:
        if mat_name in world.materials:
            eps_values.append(float(world.materials[mat_name].get("eps_r", 10.0)))
    if not eps_values:
        return 0.075  # fallback
    avg_eps = sum(eps_values) / len(eps_values)
    return 0.3 / max((avg_eps ** 0.5), 1.0)


def _write_bscan_aligned_placeholders(cfg: AppConfig, world: SceneWorld, case_dir: Path, variants: list[str]) -> dict[str, Path]:
    """Write B-scan-shaped placeholders and masks before expensive gprMax runs.

    These arrays make the case folder structurally complete for PGDA-CSNet data
    packaging.  They are NaN placeholders until real gprMax outputs are merged,
    but their shape, time axis and distance axis are the same contract that the
    later raw/target/clutter B-scan products must follow.
    """

    out_dir = case_dir / "outputs"
    labels_dir = case_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    samples = int(world.samples)
    traces = int(world.trace_count)
    time_axis_ns = np.linspace(0.0, float(world.time_window_ns), samples, dtype=np.float32)
    tx_x_axis_m = (
        float(world.trajectory.scan_start_x_m)
        + np.arange(traces, dtype=np.float32) * float(world.trace_spacing_m)
    )
    rx_x_axis_m = tx_x_axis_m + float(world.trajectory.tx_rx_offset_m)
    midpoint_x_axis_m = (tx_x_axis_m + rx_x_axis_m) / 2.0
    # Canonical B-scan horizontal coordinate.  Keep the historical file name
    # distance_axis_m.npy for compatibility, but define it explicitly as the
    # TX/RX midpoint axis instead of an implicit TX axis.
    distance_axis_m = midpoint_x_axis_m.astype(np.float32)
    x_world = np.asarray(world.ground_x, dtype=float)
    depth_world = np.asarray(world.bedrock_interface_depth_m, dtype=float)
    interface_depth = np.interp(distance_axis_m, x_world, depth_world)
    # Add air travel time from UAV (constant level) to ground surface.
    ground_surface_y = np.interp(distance_axis_m, np.asarray(world.ground_x, dtype=float),
                                 np.asarray(world.ground_y, dtype=float))
    uav_y = max(ground_surface_y) + float(world.trajectory.nominal_height_m)
    air_twt = 2.0 * (uav_y - ground_surface_y) / 0.3  # speed of light in air
    # Compute cover layer velocity from actual sampled material properties.
    velocity_m_per_ns = compute_cover_velocity(world)
    interface_time_ns = np.clip(air_twt + 2.0 * interface_depth / velocity_m_per_ns, 0.0, float(world.time_window_ns))

    interface_mask = np.zeros((samples, traces), dtype=np.uint8)
    half_width = max(float(world.time_window_ns) / max(samples, 1) * 2.0, 2.0)
    for ix, t0 in enumerate(interface_time_ns):
        interface_mask[:, ix] = (np.abs(time_axis_ns - t0) <= half_width).astype(np.uint8)

    layer_mask = np.zeros((samples, traces), dtype=np.uint8)
    for ix, t0 in enumerate(interface_time_ns):
        layer_mask[time_axis_ns <= t0, ix] = 1  # cover above bedrock interface
        layer_mask[time_axis_ns > t0, ix] = 2   # below interpreted interface

    time_axis_path = out_dir / "time_axis_ns.npy"
    time_axis_501_path = out_dir / "time_axis_501_ns.npy"
    distance_axis_path = out_dir / "distance_axis_m.npy"
    tx_axis_path = out_dir / "tx_x_axis_m.npy"
    rx_axis_path = out_dir / "rx_x_axis_m.npy"
    midpoint_axis_path = out_dir / "midpoint_x_axis_m.npy"
    np.save(time_axis_path, time_axis_ns)
    np.save(time_axis_501_path, np.linspace(0.0, float(world.time_window_ns), 501, dtype=np.float32))
    np.save(distance_axis_path, distance_axis_m)
    np.save(tx_axis_path, tx_x_axis_m.astype(np.float32))
    np.save(rx_axis_path, rx_x_axis_m.astype(np.float32))
    np.save(midpoint_axis_path, midpoint_x_axis_m.astype(np.float32))
    interface_mask_path = labels_dir / "interface_mask_bscan.npy"
    layer_mask_path = labels_dir / "layer_mask_bscan.npy"
    np.save(interface_mask_path, interface_mask)
    np.save(layer_mask_path, layer_mask)

    placeholder = np.full((samples, traces), np.nan, dtype=np.float32)
    alias = {
        "raw": "raw_bscan.npy",
        "target_only": "target_only_bscan.npy",
        "background_only": "background_only_bscan.npy",
        "clutter_only": "clutter_only_bscan.npy",
        "air_only": "air_only_bscan.npy",
    }
    paths: dict[str, Path] = {
        "time_axis_ns_npy": time_axis_path,
        "distance_axis_m_npy": distance_axis_path,
        "tx_x_axis_m_npy": tx_axis_path,
        "rx_x_axis_m_npy": rx_axis_path,
        "midpoint_x_axis_m_npy": midpoint_axis_path,
        "interface_mask_bscan_npy": interface_mask_path,
        "layer_mask_bscan_npy": layer_mask_path,
    }
    for variant in variants:
        name = alias.get(variant, f"{variant}_bscan.npy")
        p = out_dir / name
        np.save(p, placeholder)
        paths[f"{variant}_bscan_npy"] = p
    clutter_gt_path = out_dir / "clutter_gt_bscan.npy"
    np.save(clutter_gt_path, placeholder)
    paths["clutter_gt_bscan_npy"] = clutter_gt_path

    note = {
        "status": "placeholder_before_gprmax_run",
        "message": "B-scan .npy files are NaN placeholders aligned to masks. Replace them with merged gprMax B-scan arrays after running each .in file.",
        "shape_samples_x_traces": [samples, traces],
        "time_axis_ns": "outputs/time_axis_ns.npy",
        "distance_axis_m": "outputs/distance_axis_m.npy",
        "distance_axis_role": "midpoint_x",
        "tx_x_axis_m": "outputs/tx_x_axis_m.npy",
        "rx_x_axis_m": "outputs/rx_x_axis_m.npy",
        "midpoint_x_axis_m": "outputs/midpoint_x_axis_m.npy",
        "interface_mask_bscan": "labels/interface_mask_bscan.npy",
        "layer_mask_bscan": "labels/layer_mask_bscan.npy",
        "flight_height_mode": world.trajectory.mode,
        "trajectory_note": world.trajectory.note,
    }
    status_path = out_dir / "bscan_placeholder_status.json"
    status_path.write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["bscan_placeholder_status_json"] = status_path

    qc_path = case_dir / "bscan_qc_report.json"
    qc = {
        "schema": "uavgpr_simlab.bscan_qc.v1",
        "case_id": world.case_id,
        "status": "not_run",
        "message": "gprMax B-scans have not been merged yet; outputs are placeholders.",
        "expected_shape_samples_x_traces": [samples, traces],
        "variants": {v: {"status": "not_run", "bscan_path": str((out_dir / alias.get(v, f'{v}_bscan.npy')).relative_to(case_dir)).replace("\\", "/")} for v in variants},
        "raw_minus_target_computable": False,
        "clutter_gt_generated": False,
        "clutter_gt_path": "outputs/clutter_gt_bscan.npy",
        "flight_height_mode": world.trajectory.mode,
    }
    qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["bscan_qc_report_json"] = qc_path
    return paths


def write_scene_labels(cfg: AppConfig, world: SceneWorld, dataset_dir: Path, case_dir: Path, variants: list[str]) -> dict[str, Path]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    scene_world_json = case_dir / "scene_world.json"
    scene_world_json.write_text(json.dumps(world.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    label_json = dataset_dir / f"{world.case_id}_labels.json"
    label_json.write_text(
        json.dumps(world.to_label_json(asdict(cfg.radar), asdict(cfg.geometry), asdict(cfg.geology)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (case_dir / f"{world.case_id}_labels.json").write_text(label_json.read_text(encoding="utf-8"), encoding="utf-8")

    iface_csv = dataset_dir / f"{world.case_id}_interface.csv"
    with iface_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["x_m", "ground_y_m", "interface_y_m", "interface_depth_m"])
        for row in zip(world.ground_x, world.ground_y, world.bedrock_interface_y, world.bedrock_interface_depth_m):
            wr.writerow([f"{float(v):.5f}" for v in row])

    nx = max(2, int(math.ceil(world.domain_x_m / cfg.geometry.dx_m)))
    ny = max(2, int(math.ceil(world.domain_y_m / cfg.geometry.dx_m)))
    x_grid = np.linspace(0, world.domain_x_m, nx)
    y_grid = np.linspace(0, world.domain_y_m, ny)
    iy = np.interp(x_grid, np.asarray(world.ground_x, dtype=float), np.asarray(world.bedrock_interface_y, dtype=float))
    interface_mask = np.zeros((ny, nx), dtype=np.uint8)
    for j, y in enumerate(y_grid):
        interface_mask[j, np.abs(y - iy) <= max(cfg.geometry.dx_m * 1.5, 0.1)] = 1
    mask_npy = dataset_dir / f"{world.case_id}_mask.npy"
    np.save(mask_npy, interface_mask)

    interface_gt = labels_dir / "interface_gt.npy"
    np.save(interface_gt, np.asarray([world.ground_x, world.bedrock_interface_depth_m], dtype=float))
    layer_gt = labels_dir / "layer_gt.npy"
    layer = np.zeros((ny, nx), dtype=np.uint8)
    ground_i = np.interp(x_grid, np.asarray(world.ground_x, dtype=float), np.asarray(world.ground_y, dtype=float))
    for ix, (gy, by) in enumerate(zip(ground_i, iy)):
        layer[y_grid <= by, ix] = 2  # bedrock / interbed
        layer[(y_grid > by) & (y_grid <= gy), ix] = 1  # cover
    np.save(layer_gt, layer)
    layer_gt_x_axis = labels_dir / "layer_gt_x_axis_m.npy"
    layer_gt_y_axis = labels_dir / "layer_gt_y_axis_m.npy"
    np.save(layer_gt_x_axis, x_grid.astype(np.float32))
    np.save(layer_gt_y_axis, y_grid.astype(np.float32))

    aligned = _write_bscan_aligned_placeholders(cfg, world, case_dir, variants)

    metadata = world.metadata_summary(variants)
    metadata["outputs"] = {v: f"{v}.in" for v in variants}
    metadata["bscan_contract"] = {
        "status": "placeholder_before_gprmax_run",
        "shape_samples_x_traces": [int(world.samples), int(world.trace_count)],
        "time_window_ns": float(world.time_window_ns),
        "samples": int(world.samples),
        "trace_count": int(world.trace_count),
        "time_axis_ns": "outputs/time_axis_ns.npy",
        "distance_axis_m": "outputs/distance_axis_m.npy",
        "distance_axis_role": "midpoint_x",
        "tx_x_axis_m": "outputs/tx_x_axis_m.npy",
        "rx_x_axis_m": "outputs/rx_x_axis_m.npy",
        "midpoint_x_axis_m": "outputs/midpoint_x_axis_m.npy",
        "mask_alignment": "labels/interface_mask_bscan.npy and labels/layer_mask_bscan.npy share the same samples x traces shape as *_bscan.npy placeholders and later merged gprMax B-scans.",
    }
    metadata["labels"] = {
        "interface_gt": "labels/interface_gt.npy",
        "layer_gt": "labels/layer_gt.npy",
        "layer_gt_coordinate_system": "model_grid",
        "layer_gt_x_axis_m": "labels/layer_gt_x_axis_m.npy",
        "layer_gt_y_axis_m": "labels/layer_gt_y_axis_m.npy",
        "interface_mask_bscan": "labels/interface_mask_bscan.npy",
        "layer_mask_bscan": "labels/layer_mask_bscan.npy",
        "legacy_mask_npy": str(mask_npy.name),
    }
    metadata["case_files"] = {
        "scene_world": "scene_world.json",
        "metadata_summary": "metadata_summary.json",
        "model_preview": "previews/model_preview.png",
        "variant_previews": {v: f"previews/{v}_preview.png" for v in variants},
        "inputs": {v: f"{v}.in" for v in variants},
        "bscan_placeholders": {k: str(v.relative_to(case_dir)).replace("\\", "/") for k, v in aligned.items() if k.endswith("_npy") or k.endswith("_json")},
        "bscan_qc_report": "bscan_qc_report.json",
    }
    metadata_summary = case_dir / "metadata_summary.json"
    metadata_summary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "scene_world_json": scene_world_json,
        "label_json": label_json,
        "interface_csv": iface_csv,
        "mask_npy": mask_npy,
        "interface_gt_npy": interface_gt,
        "layer_gt_npy": layer_gt,
        "layer_gt_x_axis_m_npy": layer_gt_x_axis,
        "layer_gt_y_axis_m_npy": layer_gt_y_axis,
        "metadata_summary_json": metadata_summary,
        **aligned,
    }


def render_scene_preview(world: SceneWorld, out_png: str | Path, *, variant: str = "raw", title: str | None = None, width: int = 900, height: int = 520) -> Path | None:
    if plt is None:
        return None
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(world.ground_x, dtype=float)
    ground = np.asarray(world.ground_y, dtype=float)
    iface = np.asarray(world.bedrock_interface_y, dtype=float)
    ymin = max(0.0, float(np.nanmin(iface)) - 4.0)
    ymax = min(world.domain_y_m, float(np.nanmax(ground)) + world.trajectory.nominal_height_m + 2.0)
    dpi = 120
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    if variant in {"raw", "target_only"}:
        ax.fill_between(x, ymin, iface, alpha=0.28, label="bedrock / interbeds")
        ax.fill_between(x, iface, ground, alpha=0.34, label="cover layer")
        ax.plot(x, iface, linewidth=2.0, label="bedrock interface")
        for layer in world.interbed_layers:
            ax.plot(layer.x_m, layer.y_m, linewidth=1.0, linestyle="--", alpha=0.8, label=layer.layer_id if layer.layer_id.endswith("01") else None)
    elif variant == "background_only":
        ax.fill_between(x, ymin, ground, alpha=0.24, label="background cover without target interface")
    elif variant == "clutter_only":
        ax.plot(x, ground, linewidth=1.6, label="surface proxy")
    ax.plot(x, ground, linewidth=1.8, label="ground")
    ax.plot([world.trajectory.scan_start_x_m, world.trajectory.scan_end_x_m], [world.trajectory.source_y_m, world.trajectory.source_y_m], linestyle="-.", linewidth=1.8, label="constant-level flight path")

    def draw_obj(obj: SceneObject, color_label: str) -> None:
        ax.add_patch(plt.Rectangle((obj.x0_m, obj.y0_m), max(0.05, obj.x1_m - obj.x0_m), max(0.05, obj.y1_m - obj.y0_m), fill=False, linewidth=1.4))
        ax.text(obj.x0_m, obj.y1_m, obj.kind, fontsize=7)

    if variant in {"raw", "background_only", "clutter_only"}:
        for obj in world.external_clutter_objects:
            draw_obj(obj, "clutter")
    if variant in {"raw", "target_only", "background_only"}:
        for obj in world.water_zones:
            draw_obj(obj, "water")
    if variant in {"raw", "target_only"}:
        for obj in world.anomaly_objects:
            draw_obj(obj, "anomaly")

    ax.set_xlim(0, world.domain_x_m)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x / m")
    ax.set_ylabel("model y / m")
    depths = np.asarray(world.bedrock_interface_depth_m, dtype=float)
    relief = float(np.max(ground) - np.min(ground))
    ax.set_title(title or f"{world.case_id} | {world.family} | {variant} | constant-level | relief {relief:.1f}m | interface {depths.min():.1f}-{depths.max():.1f}m")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out.resolve()


def write_scene_world_case(cfg: AppConfig, world: SceneWorld, case_dir: Path, dataset_dir: Path, variants: list[str]) -> dict[str, Any]:
    variants = _variants(variants)
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = write_scene_labels(cfg, world, dataset_dir, case_dir, variants)
    previews = case_dir / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    model_preview = previews / "model_preview.png"
    render_scene_preview(world, model_preview, variant="raw")
    model_preview_alias = case_dir / "model_preview.png"
    if model_preview.exists():
        model_preview_alias.write_bytes(model_preview.read_bytes())
    variant_previews: dict[str, Path] = {}
    for variant in variants:
        input_path = case_dir / f"{variant}.in"
        input_path.write_text(make_gprmax_input_from_scene(cfg, world, variant), encoding="utf-8")
        preview_path = previews / f"{variant}_preview.png"
        render_scene_preview(world, preview_path, variant=variant)
        variant_previews[variant] = preview_path
    raw_preview = variant_previews.get("raw") or model_preview
    variant_preview_alias = case_dir / "variant_preview.png"
    if raw_preview.exists():
        variant_preview_alias.write_bytes(raw_preview.read_bytes())
    return {**paths, "model_preview_png": model_preview_alias, "variant_preview_png_alias": variant_preview_alias, "variant_previews": variant_previews}
