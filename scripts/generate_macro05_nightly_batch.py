#!/usr/bin/env python3
"""Generate ten independent, guarded MACRO05 gprMax night-run families."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = WORKSPACE / "PGDA_gprMax_MACRO05_NIGHTLY_20260712"
C0 = 299_792_458.0
Q_OFFSETS = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
REGIONS = ("topsoil", "deep_cover", "transition_1", "transition_2", "transition_3", "bedrock")


@dataclass(frozen=True)
class GridSpec:
    domain_x_m: float = 382.1
    domain_y_m: float = 45.0
    dl_m: float = 0.05
    pml_cells: int = 60
    trace_count: int = 128
    trace_spacing_m: float = 1.7
    scan_start_x_m: float = 83.0
    source_y_m: float = 29.65
    tx_rx_offset_m: float = 0.2
    center_frequency_hz: float = 55e6
    solver_time_window_s: float = 701e-9

    @property
    def nx(self) -> int:
        return int(round(self.domain_x_m / self.dl_m))

    @property
    def ny(self) -> int:
        return int(round(self.domain_y_m / self.dl_m))

    @property
    def scan_span_m(self) -> float:
        return (self.trace_count - 1) * self.trace_spacing_m

    @property
    def physical_guard_m(self) -> float:
        return self.scan_start_x_m - self.pml_cells * self.dl_m


@dataclass(frozen=True)
class Family:
    case_id: str
    field_seed: int
    profile_seed: int
    basal_mean_m: float
    basal_amplitudes_m: tuple[float, float, float]
    basal_periods_m: tuple[float, float, float]
    transition_base_m: float
    transition_amplitude_m: float
    weakening_centres: tuple[float, ...]
    weakening_strengths_m: tuple[float, ...]
    terrain_amplitude_m: float
    topsoil_mean_m: float
    contrast_scale: float
    conductivity_scale: float
    clutter_count: int
    design_note: str
    domain_equivalence: bool = False


FAMILIES = (
    Family("MACRO05_F01_DOMAIN_EQUIVALENCE", 2026071301, 2026071302, 15.2, (0.34, 0.18, 0.09), (188, 73, 33), 1.55, 0.16, (0.33, 0.67), (1.05, 0.72), 0.15, 3.3, 1.0, 1.0, 3, "Exact cropped/shifted MACRO04 domain-equivalence control", True),
    Family("MACRO05_F02_SHALLOW_DRY_GENTLE", 2026071401, 2026071402, 13.35, (0.28, 0.16, 0.08), (176, 68, 31), 1.35, 0.14, (0.28, 0.76), (0.55, 0.42), 0.12, 2.9, 0.90, 0.78, 2, "Shallower dry cover with restrained relief"),
    Family("MACRO05_F03_DEEP_WEAK_CONTRAST", 2026071501, 2026071502, 16.45, (0.42, 0.20, 0.10), (205, 82, 37), 1.85, 0.20, (0.45,), (0.95,), 0.18, 3.5, 0.62, 0.92, 2, "Deep weak-contrast interface near the detectability limit"),
    Family("MACRO05_F04_THICK_WEATHERING_DROPOUT", 2026071601, 2026071602, 15.25, (0.38, 0.22, 0.12), (194, 74, 29), 2.05, 0.24, (0.22, 0.56, 0.83), (1.00, 1.25, 0.78), 0.16, 3.2, 0.82, 1.02, 3, "Broad weathered zones create smooth causal dropout"),
    Family("MACRO05_F05_MULTISCALE_FOLDED", 2026071701, 2026071702, 14.85, (0.52, 0.31, 0.22), (152, 48, 21), 1.55, 0.20, (0.39, 0.71), (0.62, 0.74), 0.20, 3.1, 0.92, 0.95, 3, "Higher-curvature but continuous multiscale interface"),
    Family("MACRO05_F06_CLUTTER_RICH_LENSES", 2026071801, 2026071802, 14.60, (0.36, 0.24, 0.13), (184, 61, 27), 1.50, 0.18, (0.52,), (0.68,), 0.14, 3.0, 0.88, 1.08, 6, "Finite tapered shallow lenses increase coherent non-target clutter"),
    Family("MACRO05_F07_TERRAIN_COUPLED_HEIGHT", 2026071901, 2026071902, 15.05, (0.33, 0.20, 0.11), (201, 79, 35), 1.65, 0.19, (0.30, 0.64), (0.58, 0.82), 0.42, 3.25, 0.88, 1.00, 3, "Larger terrain and flight-height variation with moderate target relief"),
    Family("MACRO05_F08_LOW_CONTRAST_BROAD_DROPOUT", 2026072001, 2026072002, 15.75, (0.30, 0.17, 0.08), (217, 91, 39), 2.10, 0.16, (0.50,), (1.55,), 0.17, 3.45, 0.45, 1.05, 2, "Low contrast and one broad transition-driven dropout zone"),
    Family("MACRO05_F09_THIN_TRANSITION_SHARP", 2026072101, 2026072102, 14.25, (0.44, 0.24, 0.14), (168, 57, 24), 1.15, 0.12, (0.24, 0.78), (0.30, 0.38), 0.13, 2.8, 1.12, 0.88, 3, "Thin weathering and stronger but nontrivial basal contrast"),
    Family("MACRO05_F10_NEAR_FLAT_LOCAL_NOTCH", 2026072201, 2026072202, 15.10, (0.18, 0.10, 0.05), (238, 96, 43), 1.70, 0.14, (0.37, 0.69), (0.42, 1.10), 0.11, 3.15, 0.76, 0.98, 3, "Near-flat regional interface with a localized weak segment"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot normalise constant/non-finite values")
    return (values - float(np.mean(values))) / scale


def smooth_noise(rng: np.random.Generator, size: int, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * normalise(gaussian_filter1d(rng.standard_normal(size), sigma=sigma, mode="reflect"))


def correlated_quantiles(spec: GridSpec, seed: int) -> tuple[np.ndarray, list[float]]:
    factor = 4
    shape = (math.ceil(spec.nx / factor), math.ceil(spec.ny / factor))
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=np.float32)
    for sigma, weight in (((42.0, 6.0), 0.48), ((15.0, 2.8), 0.35), ((4.5, 1.1), 0.17)):
        component = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=sigma, mode="reflect")
        field += weight * normalise(component)
    field = zoom(normalise(field), (spec.nx / shape[0], spec.ny / shape[1]), order=1, mode="reflect")
    field = normalise(field[: spec.nx, : spec.ny])
    thresholds = np.quantile(field, [0.10, 0.30, 0.70, 0.90]).astype(np.float32)
    return np.digitize(field, thresholds).astype(np.int16), [float(value) for value in thresholds]


def family_properties(family: Family) -> dict[str, tuple[float, float, float, float]]:
    conductivity = family.conductivity_scale
    deep_eps = 11.8
    base = {
        "topsoil": (13.0, 1.20, 0.0043 * conductivity, 0.0009 * conductivity),
        "deep_cover": (deep_eps, 0.95, 0.0022 * conductivity, 0.0005 * conductivity),
        "transition_1": (11.4, 0.80, 0.0020 * conductivity, 0.0004 * conductivity),
        "transition_2": (10.7, 0.70, 0.0018 * conductivity, 0.00035 * conductivity),
        "transition_3": (10.0, 0.60, 0.0015 * conductivity, 0.00025 * conductivity),
        "bedrock": (9.1, 0.60, 0.0010 * conductivity, 0.0002 * conductivity),
    }
    for name in ("transition_1", "transition_2", "transition_3", "bedrock"):
        eps, eps_scale, sigma, sigma_scale = base[name]
        base[name] = (
            deep_eps + family.contrast_scale * (eps - deep_eps),
            eps_scale * family.contrast_scale,
            base["deep_cover"][2] + family.contrast_scale * (sigma - base["deep_cover"][2]),
            sigma_scale * family.contrast_scale,
        )
    return base


def make_profiles(spec: GridSpec, family: Family) -> dict[str, np.ndarray]:
    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    rng = np.random.default_rng(family.profile_seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=8)
    ground = (
        21.65
        + family.terrain_amplitude_m * np.sin(2 * np.pi * x / 198.0 + phase[0])
        + 0.45 * family.terrain_amplitude_m * np.sin(2 * np.pi * x / 79.0 + phase[1])
        + smooth_noise(rng, spec.nx, 220.0, max(0.04, family.terrain_amplitude_m * 0.28))
    )
    basal = np.full(spec.nx, family.basal_mean_m, dtype=np.float64)
    for amplitude, period, offset in zip(family.basal_amplitudes_m, family.basal_periods_m, phase[2:5]):
        basal += amplitude * np.sin(2 * np.pi * x / period + offset)
    basal += smooth_noise(rng, spec.nx, 150.0, 0.10)
    topsoil = (
        family.topsoil_mean_m
        + 0.20 * np.sin(2 * np.pi * x / 127.0 + phase[5])
        + smooth_noise(rng, spec.nx, 190.0, 0.07)
    )
    transition = (
        family.transition_base_m
        + family.transition_amplitude_m * np.sin(2 * np.pi * x / 89.0 + phase[6])
        + smooth_noise(rng, spec.nx, 205.0, 0.07)
    )
    for fraction, strength in zip(family.weakening_centres, family.weakening_strengths_m):
        centre = spec.scan_start_x_m + fraction * spec.scan_span_m
        width = 17.0 + 10.0 * ((family.profile_seed + int(fraction * 100)) % 5) / 4.0
        transition += strength * np.exp(-0.5 * ((x - centre) / width) ** 2)
    transition = np.clip(transition, 0.95, 3.35)
    basal = np.clip(basal, 12.4, 17.2)
    basal = np.maximum(basal, topsoil + transition + 4.2)
    return {
        "x_m": x.astype(np.float32),
        "ground_y_m": ground.astype(np.float32),
        "topsoil_depth_m": topsoil.astype(np.float32),
        "basal_depth_m": basal.astype(np.float32),
        "transition_thickness_m": transition.astype(np.float32),
        "basal_y_m": (ground - basal).astype(np.float32),
        "transition_top_y_m": (ground - basal + transition).astype(np.float32),
    }


def make_lenses(spec: GridSpec, family: Family) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(family.profile_seed + 77)
    lenses = []
    for index in range(family.clutter_count):
        centre = spec.scan_start_x_m + (index + 1) / (family.clutter_count + 1) * spec.scan_span_m
        centre += float(rng.uniform(-9.0, 9.0))
        length = float(rng.uniform(20.0, 48.0))
        lenses.append(
            {
                "name": f"finite_lens_{index + 1}",
                "x0": max(spec.pml_cells * spec.dl_m + 5.0, centre - length / 2),
                "x1": min(spec.domain_x_m - spec.pml_cells * spec.dl_m - 5.0, centre + length / 2),
                "depth": float(rng.uniform(5.2, 9.8)),
                "thickness": float(rng.uniform(0.16, 0.34)),
                "eps": float(rng.uniform(10.0, 14.0)),
                "sigma": float(rng.uniform(0.0015, 0.0042) * family.conductivity_scale),
            }
        )
    return lenses


def material_rows(properties: dict[str, tuple[float, float, float, float]], lenses: list[dict], control: bool) -> list[dict]:
    rows = []
    deep = properties["deep_cover"]
    for region_index, region in enumerate(REGIONS):
        base = deep if control and region.startswith(("transition_", "bedrock")) else properties[region]
        eps_mean, eps_scale, sigma_mean, sigma_scale = base
        for q, offset in enumerate(Q_OFFSETS):
            rows.append(
                {
                    "index": region_index * len(Q_OFFSETS) + q,
                    "id": f"{region}_q{q}",
                    "epsilon_r": round(float(eps_mean + eps_scale * offset), 6),
                    "conductivity_s_per_m": round(float(max(0.0, sigma_mean + sigma_scale * offset)), 8),
                }
            )
    for index, lens in enumerate(lenses):
        rows.append(
            {
                "index": len(REGIONS) * len(Q_OFFSETS) + index,
                "id": lens["name"],
                "epsilon_r": lens["eps"],
                "conductivity_s_per_m": lens["sigma"],
            }
        )
    return rows


def build_indices(spec: GridSpec, quantiles: np.ndarray, profile: dict[str, np.ndarray], lenses: list[dict]) -> np.ndarray:
    y = (np.arange(spec.ny, dtype=np.float32) + 0.5) * spec.dl_m
    depth = profile["ground_y_m"][:, None] - y[None, :]
    data = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    sub = depth >= 0
    basal = profile["basal_depth_m"][:, None]
    transition = profile["transition_thickness_m"][:, None]
    transition_top = basal - transition
    top = depth < profile["topsoil_depth_m"][:, None]
    fraction = np.clip((depth - transition_top) / np.maximum(transition, spec.dl_m), 0.0, 1.0)
    region = np.full_like(data, -1)
    region[sub & top] = 0
    region[sub & ~top & (depth < transition_top)] = 1
    mask = sub & (depth >= transition_top) & (depth < basal)
    region[mask & (fraction < 1 / 3)] = 2
    region[mask & (fraction >= 1 / 3) & (fraction < 2 / 3)] = 3
    region[mask & (fraction >= 2 / 3)] = 4
    region[sub & (depth >= basal)] = 5
    data[sub] = region[sub] * len(Q_OFFSETS) + quantiles[sub]
    x = profile["x_m"]
    for index, lens in enumerate(lenses):
        phase = np.clip((x - lens["x0"]) / (lens["x1"] - lens["x0"]), 0.0, 1.0)
        thickness = lens["thickness"] * np.sin(np.pi * phase) ** 2
        centre = lens["depth"] + 0.13 * np.sin(2 * np.pi * phase + 0.5 * index)
        lens_mask = (
            (x[:, None] >= lens["x0"])
            & (x[:, None] <= lens["x1"])
            & (np.abs(depth - centre[:, None]) <= thickness[:, None] / 2)
            & (depth > profile["topsoil_depth_m"][:, None] + 0.4)
            & (depth < transition_top - 0.5)
        )
        data[lens_mask] = len(REGIONS) * len(Q_OFFSETS) + index
    return data[:, :, None]


def reference_arrival(spec: GridSpec, profile: dict[str, np.ndarray], properties: dict) -> dict[str, np.ndarray]:
    source = spec.scan_start_x_m + np.arange(spec.trace_count) * spec.trace_spacing_m
    receiver = source + spec.tx_rx_offset_m
    midpoint = (source + receiver) / 2
    ground = np.interp(midpoint, profile["x_m"], profile["ground_y_m"])
    top = np.interp(midpoint, profile["x_m"], profile["topsoil_depth_m"])
    basal = np.interp(midpoint, profile["x_m"], profile["basal_depth_m"])
    transition = np.interp(midpoint, profile["x_m"], profile["transition_thickness_m"])
    air = np.maximum(spec.source_y_m - ground, 0.0)
    deep = np.maximum(basal - transition - top, 0.0)
    eps = [properties[name][0] for name in ("topsoil", "deep_cover", "transition_1", "transition_2", "transition_3")]
    one_way = air + top * math.sqrt(eps[0]) + deep * math.sqrt(eps[1])
    one_way += transition / 3 * sum(math.sqrt(value) for value in eps[2:])
    return {
        "source_x_m": source.astype(np.float32),
        "receiver_x_m": receiver.astype(np.float32),
        "trace_midpoint_x_m": midpoint.astype(np.float32),
        "antenna_y_m": np.full(spec.trace_count, spec.source_y_m, dtype=np.float32),
        "ground_y_m": ground.astype(np.float32),
        "flight_height_m": air.astype(np.float32),
        "basal_interface_depth_m": basal.astype(np.float32),
        "transition_thickness_m": transition.astype(np.float32),
        "geometric_reference_arrival_time_ns": (2e9 * one_way / C0).astype(np.float32),
    }


def input_text(spec: GridSpec, title: str, material_file: str | None, geometry_view: str | None = None) -> str:
    lines = [
        f"#title: {title}",
        f"#domain: {spec.domain_x_m:g} {spec.domain_y_m:g} {spec.dl_m:g}",
        f"#dx_dy_dz: {spec.dl_m:g} {spec.dl_m:g} {spec.dl_m:g}",
        f"#time_window: {spec.solver_time_window_s:g}",
        f"#pml_cells: {spec.pml_cells} {spec.pml_cells} 0 {spec.pml_cells} {spec.pml_cells} 0",
        "#messages: y",
        f"#waveform: ricker 1 {spec.center_frequency_hz:g} macro05_wavelet",
        f"#hertzian_dipole: z {spec.scan_start_x_m:g} {spec.source_y_m:g} 0 macro05_wavelet",
        f"#rx: {spec.scan_start_x_m + spec.tx_rx_offset_m:g} {spec.source_y_m:g} 0 rx1 Ez",
        f"#src_steps: {spec.trace_spacing_m:g} 0 0",
        f"#rx_steps: {spec.trace_spacing_m:g} 0 0",
    ]
    if material_file:
        lines.append(f"#geometry_objects_read: 0 0 0 geology_indices.h5 {material_file}")
    if geometry_view:
        lines.append(
            f"#geometry_view: 0 0 0 {spec.domain_x_m:g} {spec.domain_y_m:g} {spec.dl_m:g} "
            f"{spec.dl_m:g} {spec.dl_m:g} {spec.dl_m:g} {geometry_view} n"
        )
    return "\n".join(lines) + "\n"


def write_materials(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(f"#material: {row['epsilon_r']} {row['conductivity_s_per_m']} 1 0 {row['id']}" for row in rows) + "\n",
        encoding="ascii",
    )


def preview(case_dir: Path, spec: GridSpec, profile: dict[str, np.ndarray], labels: dict[str, np.ndarray], family: Family) -> None:
    canvas = Image.new("RGB", (1600, 760), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
        body_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    except OSError:
        title_font = body_font = ImageFont.load_default()
    draw.text((50, 24), family.case_id, fill="#111111", font=title_font)
    boxes = ((60, 100, 760, 610), (840, 100, 1540, 610))
    for box in boxes:
        draw.rectangle(box, outline="#333333", width=2)
    source = labels["source_x_m"] - spec.scan_start_x_m
    depth = labels["basal_interface_depth_m"]
    transition = labels["transition_thickness_m"]
    arrival = labels["geometric_reference_arrival_time_ns"]
    def points(xv: np.ndarray, yv: np.ndarray, box: tuple[int, int, int, int], y_min: float, y_max: float) -> list[tuple[int, int]]:
        x = box[0] + xv / spec.scan_span_m * (box[2] - box[0])
        y = box[1] + (yv - y_min) / max(y_max - y_min, 1e-6) * (box[3] - box[1])
        return list(zip(x.astype(int).tolist(), y.astype(int).tolist()))
    draw.line(points(source, depth, boxes[0], 11.5, 18.0), fill="#b2182b", width=4)
    draw.line(points(source, transition, boxes[0], 0.8, 7.3), fill="#1b7837", width=3)
    draw.line(points(source, arrival, boxes[1], 320.0, 510.0), fill="#2166ac", width=4)
    draw.text((60, 70), "Red: basal depth; green: transition thickness (shared display axis)", fill="#222222", font=body_font)
    draw.text((840, 70), "Independent geometric two-way reference (not a visible-phase label)", fill="#222222", font=body_font)
    boundary_ns = 2e9 * spec.physical_guard_m / C0
    lines = [
        family.design_note,
        f"Domain {spec.domain_x_m:.1f} m; scan {spec.scan_span_m:.1f} m; physical side guard {spec.physical_guard_m:.1f} m",
        f"Earliest free-space side-boundary round trip {boundary_ns:.1f} ns; reference max {float(arrival.max()):.1f} ns",
        f"Depth {float(depth.min()):.2f}-{float(depth.max()):.2f} m; transition {float(transition.min()):.2f}-{float(transition.max()):.2f} m",
        "No Line9 waveform, label, geometry, or timing distribution was read. formal_training_allowed=false.",
    ]
    for index, line in enumerate(lines):
        draw.text((60, 635 + index * 23), line, fill="#222222", font=body_font)
    canvas.save(case_dir / "preview_pre_solver.png")


def write_checksums(case_dir: Path) -> None:
    rows = []
    for path in sorted(case_dir.rglob("*")):
        if path.is_file() and path.name != "FILE_SHA256.csv":
            rows.append((path.relative_to(case_dir).as_posix(), sha256(path), path.stat().st_size))
    with (case_dir / "FILE_SHA256.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        writer.writerows(rows)


def equivalence_geometry(spec: GridSpec) -> tuple[np.ndarray, dict, dict, list[dict], list[dict], list[dict], list[float]]:
    from scripts import generate_macro04_deeper_dropout_voxel as macro04

    old_spec = macro04.Spec()
    quantiles, thresholds = macro04.correlated_field(old_spec)
    old_profile = macro04.profiles(old_spec)
    old_data = macro04.build_indices(old_spec, quantiles, old_profile)
    shift_m = old_spec.scan_start_x_m - spec.scan_start_x_m
    start = int(round(shift_m / spec.dl_m))
    stop = start + spec.nx
    profile = {name: values[start:stop].copy() for name, values in old_profile.items()}
    profile["x_m"] = profile["x_m"] - shift_m
    properties = dict(macro04.FULL_PROPERTIES)
    lenses = [{**lens, "x0": lens["x0"] - shift_m, "x1": lens["x1"] - shift_m} for lens in macro04.LENSES]
    full_rows = macro04.material_rows(control=False)
    control_rows = macro04.material_rows(control=True)
    return old_data[start:stop].copy(), profile, properties, lenses, full_rows, control_rows, thresholds


def generate_case(output_root: Path, spec: GridSpec, family: Family) -> dict:
    case_dir = output_root / family.case_id
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if family.domain_equivalence:
        data, profile, properties, lenses, full_rows, control_rows, thresholds = equivalence_geometry(spec)
    else:
        quantiles, thresholds = correlated_quantiles(spec, family.field_seed)
        profile = make_profiles(spec, family)
        properties = family_properties(family)
        lenses = make_lenses(spec, family)
        data = build_indices(spec, quantiles, profile, lenses)
        full_rows = material_rows(properties, lenses, control=False)
        control_rows = material_rows(properties, lenses, control=True)
    h5_path = case_dir / "geology_indices.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = "scripts/generate_macro05_nightly_batch.py"
        handle.attrs["field_seed"] = family.field_seed
        handle.attrs["profile_seed"] = family.profile_seed
        handle.create_dataset("data", data=data, dtype=np.int16)
    write_materials(case_dir / "materials_full.txt", full_rows)
    write_materials(case_dir / "materials_no_basal.txt", control_rows)
    for name, material, view in (
        ("full_scene.in", "materials_full.txt", None),
        ("no_basal_contrast_control.in", "materials_no_basal.txt", None),
        ("geometry_check_full.in", "materials_full.txt", "geometry_full"),
        ("geometry_check_control.in", "materials_no_basal.txt", "geometry_control"),
    ):
        (case_dir / name).write_text(input_text(spec, f"{family.case_id} {name}", material, view), encoding="ascii")
    (case_dir / "air_reference.in").write_text(input_text(spec, f"{family.case_id} air", None), encoding="ascii")
    labels = reference_arrival(spec, profile, properties)
    for name, values in labels.items():
        np.save(labels_dir / f"{name}.npy", values)
    np.save(labels_dir / "reference_arrival_time_ns.npy", labels["geometric_reference_arrival_time_ns"])
    for name, values in profile.items():
        np.save(labels_dir / f"full_{name}.npy", values)
    changed = [
        index for index, (full, control) in enumerate(zip(full_rows, control_rows))
        if (full["epsilon_r"], full["conductivity_s_per_m"]) != (control["epsilon_r"], control["conductivity_s_per_m"])
    ]
    expected_changed = list(range(2 * len(Q_OFFSETS), len(REGIONS) * len(Q_OFFSETS)))
    if changed != expected_changed:
        raise RuntimeError(f"{family.case_id}: strict-pair material indices changed unexpectedly: {changed}")
    guard = spec.physical_guard_m
    right_guard = spec.domain_x_m - spec.pml_cells * spec.dl_m - (
        spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m
    )
    boundary_ns = 2e9 * min(guard, right_guard) / C0
    reference_max = float(labels["geometric_reference_arrival_time_ns"].max())
    manifest = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "batch_id": "MACRO05_NIGHTLY_10FAMILY_20260712",
        "case_id": family.case_id,
        "scene_family_id": family.case_id,
        "family_design": asdict(family),
        "purpose": family.design_note,
        "line9_conditioned": False,
        "reference_line": None,
        "formal_training_allowed": False,
        "training_block_reason": "nightly pre-promotion family; requires domain equivalence, full/control output audit, visible-phase review, and human promotion",
        "spec": asdict(spec),
        "domain_contract": {
            "physical_left_guard_m": guard,
            "physical_right_guard_m": right_guard,
            "pml_thickness_m": spec.pml_cells * spec.dl_m,
            "earliest_free_space_side_roundtrip_ns": boundary_ns,
            "latest_geometric_reference_ns": reference_max,
            "boundary_clearance_after_reference_ns": boundary_ns - reference_max,
            "protected_target_window_end_ns": 500.0,
            "protected_window_clearance_ns": boundary_ns - 500.0,
            "full_700ns_boundary_isolation": boundary_ns > 700.0,
            "domain_width_ratio_to_macro04": spec.domain_x_m / 480.1,
            "estimated_pair_minutes_from_macro04": 61.0 * spec.domain_x_m / 480.1,
            "domain_equivalence_case": family.domain_equivalence,
        },
        "source": {"waveform": "ricker", "center_frequency_hz": spec.center_frequency_hz, "polarisation": "z"},
        "grid": {
            "dimension": "2D_x_y_with_one_z_cell",
            "shape": list(data.shape),
            "dl_m": spec.dl_m,
            "pml_cells": [spec.pml_cells, spec.pml_cells, 0, spec.pml_cells, spec.pml_cells, 0],
            "canonical_output_samples": 501,
            "canonical_time_window_ns": 700.0,
        },
        "geometry": {
            "field_seed": family.field_seed,
            "profile_seed": family.profile_seed,
            "index_sha256": sha256(h5_path),
            "index_min_max": [int(data.min()), int(data.max())],
            "quantile_thresholds": thresholds,
            "finite_lenses": lenses,
            "basal_depth_min_m": float(labels["basal_interface_depth_m"].min()),
            "basal_depth_max_m": float(labels["basal_interface_depth_m"].max()),
            "transition_min_m": float(labels["transition_thickness_m"].min()),
            "transition_max_m": float(labels["transition_thickness_m"].max()),
            "reference_arrival_min_ns": float(labels["geometric_reference_arrival_time_ns"].min()),
            "reference_arrival_max_ns": reference_max,
        },
        "strict_pair": {
            "shared_geometry_hdf5": True,
            "shared_geometry_sha256": sha256(h5_path),
            "changed_material_indices": changed,
            "only_transition_and_bedrock_changed": True,
            "full_materials_sha256": sha256(case_dir / "materials_full.txt"),
            "control_materials_sha256": sha256(case_dir / "materials_no_basal.txt"),
        },
        "labels": {
            "geometric_reference": "independent columnar reference; not a visible-phase training label",
            "visible_phase": "pending signed full-minus-control extraction",
            "training_allowed": False,
        },
    }
    (case_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (case_dir / "RUN_COMMANDS.md").write_text(
        "# Run contract\n\nUse the package-level `RUN_NIGHTLY_GPU.cmd` or `RUN_ONE_CASE_GPU.cmd`. "
        "Do not train from geometric references. Air reference is deferred until the strict pair passes.\n",
        encoding="utf-8",
    )
    preview(case_dir, spec, profile, labels, family)
    write_checksums(case_dir)
    return manifest


def write_runners(output_root: Path) -> None:
    case_order = "\n".join(family.case_id for family in FAMILIES) + "\n"
    (output_root / "nightly_case_order.txt").write_text(case_order, encoding="ascii")
    runner = r"""@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
if not defined GPRMAX_PYTHON set "GPRMAX_PYTHON=python"
if defined GPRMAX_VCVARS call "%GPRMAX_VCVARS%" >nul
if defined GPRMAX_SOURCE set "PYTHONPATH=%GPRMAX_SOURCE%;%PYTHONPATH%"
if not defined CUDA_DEVICE set "CUDA_DEVICE=0"

if not "%~1"=="" (
  call :run_case "%~1"
  exit /b !errorlevel!
)

for /f "usebackq delims=" %%C in ("%ROOT%nightly_case_order.txt") do (
  call :run_case "%%C"
  if errorlevel 1 exit /b !errorlevel!
)
exit /b 0

:run_case
set "CASE_ID=%~1"
set "CASE_DIR=%ROOT%%CASE_ID%"
set "LOG_DIR=%CASE_DIR%\run_logs"
if exist "%CASE_DIR%\pair_complete.marker" goto :eof
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%CASE_DIR%"

if exist "full_scene_merged.out" goto control
echo [%CASE_ID%] full started %date% %time% > "%LOG_DIR%\pair_status.log"
"%GPRMAX_PYTHON%" -m gprMax full_scene.in -n 128 --geometry-fixed -gpu %CUDA_DEVICE% > "%LOG_DIR%\full_scene.log" 2>&1
if errorlevel 1 exit /b 21
"%GPRMAX_PYTHON%" "%ROOT%tools\capture_gprmax_trace_contract.py" "%CASE_DIR%" --prefix full_scene --expected 128 --output "%LOG_DIR%\full_trace_contract.json" --poll-seconds 0.1 --timeout-seconds 180 >> "%LOG_DIR%\full_scene.log" 2>&1
if errorlevel 1 exit /b 24
"%GPRMAX_PYTHON%" -m tools.outputfiles_merge full_scene --remove-files >> "%LOG_DIR%\full_scene.log" 2>&1
if errorlevel 1 exit /b 21
echo [%CASE_ID%] full complete %date% %time% >> "%LOG_DIR%\pair_status.log"

:control
if exist "no_basal_contrast_control_merged.out" goto complete
echo [%CASE_ID%] control started %date% %time% >> "%LOG_DIR%\pair_status.log"
"%GPRMAX_PYTHON%" -m gprMax no_basal_contrast_control.in -n 128 --geometry-fixed -gpu %CUDA_DEVICE% > "%LOG_DIR%\no_basal.log" 2>&1
if errorlevel 1 exit /b 22
"%GPRMAX_PYTHON%" "%ROOT%tools\capture_gprmax_trace_contract.py" "%CASE_DIR%" --prefix no_basal_contrast_control --expected 128 --output "%LOG_DIR%\control_trace_contract.json" --poll-seconds 0.1 --timeout-seconds 180 >> "%LOG_DIR%\no_basal.log" 2>&1
if errorlevel 1 exit /b 25
"%GPRMAX_PYTHON%" -m tools.outputfiles_merge no_basal_contrast_control --remove-files >> "%LOG_DIR%\no_basal.log" 2>&1
if errorlevel 1 exit /b 22
echo [%CASE_ID%] control complete %date% %time% >> "%LOG_DIR%\pair_status.log"

:complete
echo complete>"%CASE_DIR%\pair_complete.marker"
echo [%CASE_ID%] pair complete %date% %time% >> "%LOG_DIR%\pair_status.log"
goto :eof
"""
    runner = runner.replace("setlocal EnableExtensions", "setlocal EnableExtensions EnableDelayedExpansion")
    (output_root / "RUN_NIGHTLY_GPU.cmd").write_text(runner, encoding="ascii")
    one = r"""@echo off
if "%~1"=="" (
  echo Usage: RUN_ONE_CASE_GPU.cmd CASE_ID
  exit /b 2
)
call "%~dp0RUN_NIGHTLY_GPU.cmd" "%~1"
exit /b %errorlevel%
"""
    (output_root / "RUN_ONE_CASE_GPU.cmd").write_text(one, encoding="ascii")


def write_readme(output_root: Path) -> None:
    (output_root / "README_NIGHTLY.md").write_text(
        """# MACRO05 portable overnight run

This package contains ten independent 2D gprMax full/no-basal pairs. Every scene remains blocked from training until output-pair, visible-phase, and human review gates pass.

## Environment

Set these variables in the same Command Prompt before running:

```bat
set GPRMAX_PYTHON=C:\\path\\to\\python.exe
set GPRMAX_SOURCE=C:\\path\\to\\gprMax-source
set GPRMAX_VCVARS=C:\\path\\to\\vcvars64.bat
set CUDA_DEVICE=0
```

`GPRMAX_SOURCE` and `GPRMAX_VCVARS` are optional when the reviewed environment already provides them.

## Preflight

Run the domain-equivalence family first when moving to a new computer:

```bat
RUN_ONE_CASE_GPU.cmd MACRO05_F01_DOMAIN_EQUIVALENCE
```

Then run or resume all ten families:

```bat
RUN_NIGHTLY_GPU.cmd
```

A case with `pair_complete.marker` is skipped. Existing merged full output resumes at the control. Do not rename case folders or edit `nightly_case_order.txt` while a run is active.

## Physics contract

- Domain: 382.1 x 45 m at 0.05 m cells.
- Scan: 128 traces at 1.7 m spacing, starting at x=83 m.
- Physical side guard: 80 m beyond each PML inner boundary.
- Earliest free-space side-boundary round trip: 533.7 ns.
- Protected target/search window: 0-500 ns.
- Source: 55 MHz z-polarised Ricker pulse.
- Full/control share one HDF5 geometry; only transition/bedrock material indices 10-29 change.
- `air_reference.in` is prepared but deliberately excluded from the overnight runner.

Do not use geometric reference arrays as visible-phase labels.
""",
        encoding="ascii",
    )


def batch_preview(output_root: Path, manifests: list[dict]) -> None:
    canvas = Image.new("RGB", (1800, 1150), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
        body = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    except OSError:
        title = body = ImageFont.load_default()
    draw.text((45, 25), "MACRO05 nightly 10-family pre-solver overview", fill="#111111", font=title)
    for row, manifest in enumerate(manifests):
        y = 90 + row * 100
        geometry = manifest["geometry"]
        domain = manifest["domain_contract"]
        draw.text((45, y), manifest["case_id"], fill="#111111", font=body)
        draw.text((520, y), manifest["purpose"], fill="#333333", font=body)
        draw.text(
            (45, y + 30),
            f"depth {geometry['basal_depth_min_m']:.2f}-{geometry['basal_depth_max_m']:.2f} m | "
            f"transition {geometry['transition_min_m']:.2f}-{geometry['transition_max_m']:.2f} m | "
            f"reference {geometry['reference_arrival_min_ns']:.1f}-{geometry['reference_arrival_max_ns']:.1f} ns | "
            f"boundary clearance after 500 ns {domain['protected_window_clearance_ns']:.1f} ns",
            fill="#1f4e79",
            font=body,
        )
        draw.line((45, y + 75, 1750, y + 75), fill="#dddddd", width=1)
    draw.text((45, 1100), "All cases: 382.1 x 45 m, 55 MHz Ricker, 128 x 1.7 m scan, shared-HDF5 full/no-basal pair, formal training disabled.", fill="#7f1d1d", font=body)
    canvas.save(output_root / "NIGHTLY_10FAMILY_OVERVIEW.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    manifests = [generate_case(output_root, spec, family) for family in FAMILIES]
    tools_dir = output_root / "tools"
    tools_dir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "capture_gprmax_trace_contract.py", tools_dir / "capture_gprmax_trace_contract.py")
    write_runners(output_root)
    write_readme(output_root)
    batch_preview(output_root, manifests)
    batch = {
        "batch_id": "MACRO05_NIGHTLY_10FAMILY_20260712",
        "case_count": len(manifests),
        "case_order": [item["case_id"] for item in manifests],
        "spec": asdict(spec),
        "domain_decision": manifests[0]["domain_contract"],
        "line9_conditioned_case_count": sum(bool(item["line9_conditioned"]) for item in manifests),
        "formal_training_allowed": False,
        "estimated_total_pair_hours": sum(item["domain_contract"]["estimated_pair_minutes_from_macro04"] for item in manifests) / 60.0,
        "run_requirements": {
            "GPRMAX_PYTHON": "Python executable with PyCUDA/h5py and reviewed gprMax environment",
            "GPRMAX_SOURCE": "optional gprMax source root when package is not installed",
            "GPRMAX_VCVARS": "optional vcvars64.bat path on Windows",
            "CUDA_DEVICE": "defaults to 0",
        },
    }
    (output_root / "batch_manifest.json").write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
