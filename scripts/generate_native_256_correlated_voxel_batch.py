#!/usr/bin/env python3
"""Generate native 501x256 correlated-voxel gprMax source decks.

The batch inherits MACRO03's multiscale geology representation without
compressing its long-line shape into the native 22.95 m acquisition window.
Each positive uses one shared HDF5 index volume for the strict full/control
pair; only transition and bedrock constitutive mappings change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "contracts" / "simulation_v2" / "native_256_correlated_voxel_batch_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "data" / "simulations" / "v2" / "01_native_256_correlated_voxel_batch_v1"
)
C0 = 299_792_458.0
Q_OFFSETS = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
REGIONS = ("topsoil", "deep_cover", "transition_1", "transition_2", "transition_3", "bedrock")


@dataclass(frozen=True)
class Spec:
    domain_x_m: float = 243.135
    domain_y_m: float = 36.0
    dl_m: float = 0.0225
    pml_cells: int = 20
    trace_count: int = 256
    trace_spacing_m: float = 0.09
    scan_start_x_m: float = 110.0025
    source_y_m: float = 28.035
    tx_rx_offset_m: float = 0.18
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


MATERIAL_FAMILIES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "balanced": {
        "topsoil": (13.0, 1.0, 0.0045, 0.0010),
        "deep_cover": (11.8, 0.9, 0.0025, 0.0006),
        "transition_1": (11.2, 0.8, 0.0022, 0.0005),
        "transition_2": (10.3, 0.8, 0.0018, 0.0004),
        "transition_3": (9.4, 0.7, 0.0015, 0.0003),
        "bedrock": (8.5, 0.6, 0.0009, 0.0002),
    },
    "low_contrast": {
        "topsoil": (13.2, 0.9, 0.0048, 0.0009),
        "deep_cover": (12.2, 0.8, 0.0030, 0.0006),
        "transition_1": (11.8, 0.7, 0.0028, 0.0005),
        "transition_2": (11.2, 0.7, 0.0025, 0.0004),
        "transition_3": (10.6, 0.6, 0.0021, 0.0003),
        "bedrock": (9.8, 0.5, 0.0015, 0.0002),
    },
    "patchy": {
        "topsoil": (12.8, 1.1, 0.0042, 0.0010),
        "deep_cover": (11.6, 1.0, 0.0027, 0.0007),
        "transition_1": (10.9, 1.0, 0.0024, 0.0006),
        "transition_2": (10.1, 0.9, 0.0020, 0.0005),
        "transition_3": (9.3, 0.8, 0.0016, 0.0004),
        "bedrock": (8.3, 0.7, 0.0010, 0.0003),
    },
    "upper_clutter_negative": {
        "topsoil": (13.4, 1.2, 0.0048, 0.0011),
        "deep_cover": (11.7, 1.1, 0.0027, 0.0007),
        "transition_1": (11.7, 1.1, 0.0027, 0.0007),
        "transition_2": (11.7, 1.1, 0.0027, 0.0007),
        "transition_3": (11.7, 1.1, 0.0027, 0.0007),
        "bedrock": (11.7, 1.1, 0.0027, 0.0007),
    },
}

LENS_MATERIALS = (
    ("moist_lens", 14.4, 0.0048),
    ("dry_lens", 9.7, 0.0016),
    ("mixed_lens", 13.5, 0.0039),
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
        raise ValueError("cannot normalise a constant or non-finite field")
    return (values - float(np.mean(values))) / scale


def smooth_noise_1d(rng: np.random.Generator, size: int, sigma_cells: float) -> np.ndarray:
    return normalise(gaussian_filter1d(rng.standard_normal(size), sigma=sigma_cells, mode="reflect"))


def correlated_quantiles(spec: Spec, seed: int) -> tuple[np.ndarray, list[float]]:
    """Pre-smooth a three-scale property field before material quantisation."""
    factor = 8
    coarse_shape = (math.ceil(spec.nx / factor), math.ceil(spec.ny / factor))
    rng = np.random.default_rng(seed)
    components = (
        ((48.0, 8.0), 0.50),
        ((18.0, 3.5), 0.34),
        ((6.0, 1.3), 0.16),
    )
    field = np.zeros(coarse_shape, dtype=np.float32)
    for sigma, weight in components:
        noise = rng.standard_normal(coarse_shape).astype(np.float32)
        field += weight * normalise(gaussian_filter(noise, sigma=sigma, mode="reflect"))
    field = normalise(field)
    field = zoom(field, (spec.nx / coarse_shape[0], spec.ny / coarse_shape[1]), order=1, mode="reflect")
    field = normalise(field[: spec.nx, : spec.ny])
    thresholds = np.quantile(field, [0.10, 0.30, 0.70, 0.90]).astype(np.float32)
    return np.digitize(field, thresholds).astype(np.int16), [float(value) for value in thresholds]


def build_profiles(spec: Spec, case: dict[str, Any]) -> dict[str, np.ndarray]:
    profile_spec = case["profile"]
    rng = np.random.default_rng(int(case["profile_seed"]))
    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    long = smooth_noise_1d(rng, spec.nx, 42.0 / spec.dl_m)
    meso = smooth_noise_1d(rng, spec.nx, 8.0 / spec.dl_m)
    local = smooth_noise_1d(rng, spec.nx, 1.6 / spec.dl_m)
    ground = 21.45 + 0.16 * long + 0.10 * meso + 0.04 * local

    top_long = smooth_noise_1d(rng, spec.nx, 25.0 / spec.dl_m)
    top_meso = smooth_noise_1d(rng, spec.nx, 5.0 / spec.dl_m)
    topsoil = float(profile_spec["topsoil_depth_m"]) + 0.24 * top_long + 0.13 * top_meso
    topsoil = np.clip(topsoil, 2.4, 5.4)

    result: dict[str, np.ndarray] = {
        "x_m": x.astype(np.float32),
        "ground_y_m": ground.astype(np.float32),
        "topsoil_depth_m": topsoil.astype(np.float32),
    }
    if not bool(case["target_presence"]):
        return result

    amplitudes = np.asarray(profile_spec["basal_amplitudes_m"], dtype=np.float32)
    basal = float(profile_spec["basal_depth_m"]) + amplitudes[0] * long + amplitudes[1] * meso + amplitudes[2] * local
    transition_long = smooth_noise_1d(rng, spec.nx, 19.0 / spec.dl_m)
    transition_meso = smooth_noise_1d(rng, spec.nx, 4.2 / spec.dl_m)
    transition = float(profile_spec["transition_thickness_m"]) + 0.30 * transition_long + 0.18 * transition_meso
    if case["material_family"] == "patchy":
        weakening = np.exp(-0.5 * ((x - 118.0) / 5.5) ** 2) + 0.7 * np.exp(-0.5 * ((x - 130.0) / 3.8) ** 2)
        transition += 0.75 * weakening
    transition = np.clip(transition, 0.9, 3.0)
    basal = np.maximum(basal, topsoil + transition + 5.0)
    result.update(
        {
            "basal_depth_m": basal.astype(np.float32),
            "transition_thickness_m": transition.astype(np.float32),
            "basal_y_m": (ground - basal).astype(np.float32),
            "transition_top_y_m": (ground - basal + transition).astype(np.float32),
        }
    )
    return result


def lens_definitions(spec: Spec, family: str) -> list[dict[str, float | int]]:
    centre = spec.scan_start_x_m + 0.5 * spec.scan_span_m
    base = [
        {"material": 0, "x0": centre - 10.5, "x1": centre - 2.0, "depth": 6.1, "thickness": 0.50},
        {"material": 1, "x0": centre + 1.0, "x1": centre + 9.8, "depth": 7.8, "thickness": 0.42},
    ]
    if family == "sparse":
        return base[:1]
    if family == "lens_rich":
        base.append({"material": 2, "x0": centre - 1.8, "x1": centre + 5.2, "depth": 5.0, "thickness": 0.36})
    return base


def material_rows(family: str, *, control: bool) -> list[dict[str, Any]]:
    properties = MATERIAL_FAMILIES[family]
    deep = properties["deep_cover"]
    rows: list[dict[str, Any]] = []
    for region_index, region in enumerate(REGIONS):
        base = deep if control and region in REGIONS[2:] else properties[region]
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
    for lens_index, (name, eps, conductivity) in enumerate(LENS_MATERIALS):
        rows.append(
            {
                "index": len(REGIONS) * len(Q_OFFSETS) + lens_index,
                "id": name,
                "epsilon_r": eps,
                "conductivity_s_per_m": conductivity,
            }
        )
    return rows


def build_indices(
    spec: Spec,
    case: dict[str, Any],
    quantiles: np.ndarray,
    profiles: dict[str, np.ndarray],
    lenses: list[dict[str, float | int]],
) -> np.ndarray:
    y = (np.arange(spec.ny, dtype=np.float32) + 0.5) * spec.dl_m
    depth = profiles["ground_y_m"][:, None] - y[None, :]
    data = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    subsurface = depth >= 0
    topsoil = subsurface & (depth < profiles["topsoil_depth_m"][:, None])
    region = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    region[topsoil] = 0

    if bool(case["target_presence"]):
        basal = profiles["basal_depth_m"][:, None]
        transition = profiles["transition_thickness_m"][:, None]
        transition_top = basal - transition
        region[subsurface & ~topsoil & (depth < transition_top)] = 1
        transition_mask = subsurface & (depth >= transition_top) & (depth < basal)
        fraction = np.clip((depth - transition_top) / np.maximum(transition, spec.dl_m), 0.0, 1.0)
        region[transition_mask & (fraction < 1 / 3)] = 2
        region[transition_mask & (fraction >= 1 / 3) & (fraction < 2 / 3)] = 3
        region[transition_mask & (fraction >= 2 / 3)] = 4
        region[subsurface & (depth >= basal)] = 5
        lens_ceiling = transition_top[:, 0] - 0.55
    else:
        region[subsurface & ~topsoil] = 1
        lens_ceiling = np.full(spec.nx, 12.5, dtype=np.float32)

    data[subsurface] = region[subsurface] * len(Q_OFFSETS) + quantiles[subsurface]
    x = profiles["x_m"]
    for lens_index, lens in enumerate(lenses):
        x0, x1 = float(lens["x0"]), float(lens["x1"])
        phase = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
        taper = np.sin(np.pi * phase) ** 2
        thickness = float(lens["thickness"]) * taper
        centre = float(lens["depth"]) + 0.14 * np.sin(2 * np.pi * phase + 0.6 * lens_index)
        mask = (
            (x[:, None] >= x0)
            & (x[:, None] <= x1)
            & (np.abs(depth - centre[:, None]) <= thickness[:, None] / 2)
            & (depth > profiles["topsoil_depth_m"][:, None] + 0.45)
            & (depth < lens_ceiling[:, None])
        )
        data[mask] = len(REGIONS) * len(Q_OFFSETS) + int(lens["material"])
    return data[:, :, None]


def write_materials(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(
            f"#material: {row['epsilon_r']} {row['conductivity_s_per_m']} 1 0 {row['id']}"
            for row in rows
        )
        + "\n",
        encoding="ascii",
    )


def input_text(
    spec: Spec,
    title: str,
    material_file: str | None,
    *,
    geometry_view: str | None = None,
) -> str:
    def number(value: float) -> str:
        return f"{value:.12g}"

    lines = [
        f"#title: {title}",
        f"#domain: {number(spec.domain_x_m)} {number(spec.domain_y_m)} {number(spec.dl_m)}",
        f"#dx_dy_dz: {number(spec.dl_m)} {number(spec.dl_m)} {number(spec.dl_m)}",
        f"#time_window: {number(spec.solver_time_window_s)}",
        f"#pml_cells: {spec.pml_cells} {spec.pml_cells} 0 {spec.pml_cells} {spec.pml_cells} 0",
        "#messages: y",
        f"#waveform: ricker 1 {number(spec.center_frequency_hz)} native_cv_wavelet",
        f"#hertzian_dipole: z {number(spec.scan_start_x_m)} {number(spec.source_y_m)} 0 native_cv_wavelet",
        f"#rx: {number(spec.scan_start_x_m + spec.tx_rx_offset_m)} {number(spec.source_y_m)} 0 rx1 Ez",
        f"#src_steps: {number(spec.trace_spacing_m)} 0 0",
        f"#rx_steps: {number(spec.trace_spacing_m)} 0 0",
    ]
    if material_file:
        lines.append(f"#geometry_objects_read: 0 0 0 geology_indices.h5 {material_file}")
    if geometry_view:
        lines.append(
            f"#geometry_view: 0 0 0 {number(spec.domain_x_m)} {number(spec.domain_y_m)} {number(spec.dl_m)} "
            f"{number(spec.dl_m)} {number(spec.dl_m)} {number(spec.dl_m)} {geometry_view} n"
        )
    return "\n".join(lines) + "\n"


def scan_arrays(
    spec: Spec,
    profiles: dict[str, np.ndarray],
    target_presence: bool,
    material_family: str,
) -> dict[str, np.ndarray]:
    source_x = spec.scan_start_x_m + np.arange(spec.trace_count) * spec.trace_spacing_m
    receiver_x = source_x + spec.tx_rx_offset_m
    midpoint = 0.5 * (source_x + receiver_x)
    result = {
        "source_x_m": source_x.astype(np.float32),
        "receiver_x_m": receiver_x.astype(np.float32),
        "trace_midpoint_x_m": midpoint.astype(np.float32),
        "antenna_y_m": np.full(spec.trace_count, spec.source_y_m, dtype=np.float32),
        "ground_y_m": np.interp(midpoint, profiles["x_m"], profiles["ground_y_m"]).astype(np.float32),
        "topsoil_depth_m": np.interp(midpoint, profiles["x_m"], profiles["topsoil_depth_m"]).astype(np.float32),
    }
    result["flight_height_m"] = (spec.source_y_m - result["ground_y_m"]).astype(np.float32)
    if target_presence:
        basal = np.interp(midpoint, profiles["x_m"], profiles["basal_depth_m"])
        transition = np.interp(midpoint, profiles["x_m"], profiles["transition_thickness_m"])
        family_eps = MATERIAL_FAMILIES[material_family]
        one_way = result["flight_height_m"] + result["topsoil_depth_m"] * math.sqrt(family_eps["topsoil"][0])
        deep = np.maximum(basal - transition - result["topsoil_depth_m"], 0.0)
        one_way += deep * math.sqrt(family_eps["deep_cover"][0])
        one_way += transition / 3 * sum(math.sqrt(family_eps[name][0]) for name in REGIONS[2:5])
        result["basal_interface_depth_m"] = basal.astype(np.float32)
        result["transition_thickness_m"] = transition.astype(np.float32)
        result["geometric_reference_arrival_time_ns"] = (2e9 * one_way / C0).astype(np.float32)
    return result


def morphology_metrics(
    spec: Spec,
    profiles: dict[str, np.ndarray],
    target_presence: bool,
    material_family: str,
) -> dict[str, Any]:
    if not target_presence:
        return {"target_presence": False, "single_quadratic_bowl_rejected": True}
    scan = scan_arrays(spec, profiles, True, material_family)
    x = scan["trace_midpoint_x_m"].astype(np.float64)
    y = scan["basal_interface_depth_m"].astype(np.float64)
    smooth = gaussian_filter1d(y, sigma=max(1.0, 0.25 / spec.trace_spacing_m), mode="nearest")
    derivative = np.diff(smooth)
    sign = np.sign(derivative)
    sign[sign == 0] = 1
    extrema = int(np.count_nonzero(sign[1:] != sign[:-1]))
    # The bundled Windows NumPy can load an incompatible LAPACK DLL on
    # ``polyfit``. The trace grid is symmetric, so solve this quadratic least
    # squares system analytically without a BLAS dependency.
    u = x - x.mean()
    u2 = u * u
    sum_u2 = float(np.sum(u2))
    sum_u4 = float(np.sum(u2 * u2))
    determinant = len(u) * sum_u4 - sum_u2 * sum_u2
    a0 = (float(np.sum(y)) * sum_u4 - float(np.sum(y * u2)) * sum_u2) / determinant
    a1 = float(np.sum(y * u)) / sum_u2
    a2 = (len(u) * float(np.sum(y * u2)) - sum_u2 * float(np.sum(y))) / determinant
    fitted = a0 + a1 * u + a2 * u2
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 1.0
    range_m = float(np.ptp(y))
    bowl_only = bool(r2 > 0.985 and extrema <= 1)
    acceptable = bool(range_m >= 0.25 and not bowl_only)
    return {
        "target_presence": True,
        "scan_depth_range_m": range_m,
        "smoothed_extrema_count": extrema,
        "quadratic_fit_r2": float(r2),
        "single_quadratic_bowl_rejected": not bowl_only,
        "broad_morphology_gate_ok": acceptable,
    }


def _colour_map(values: np.ndarray, anchors: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    positions = values * (len(anchors) - 1)
    lower = np.floor(positions).astype(np.int16)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (positions - lower)[..., None]
    colours = np.asarray(anchors, dtype=np.float32)
    return np.rint(colours[lower] * (1 - weight) + colours[upper] * weight).astype(np.uint8)


def preview(
    case_dir: Path,
    spec: Spec,
    case: dict[str, Any],
    data: np.ndarray,
    profiles: dict[str, np.ndarray],
    full_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    full_eps = np.asarray([float(row["epsilon_r"]) for row in full_rows])
    control_eps = np.asarray([float(row["epsilon_r"]) for row in control_rows])
    flat = data[:, :, 0]
    safe = np.clip(flat, 0, len(full_eps) - 1)
    full_map = np.where(flat >= 0, full_eps[safe], 1.0)
    delta_map = np.where(flat >= 0, full_eps[safe] - control_eps[safe], 0.0)
    scan_left = int(round(spec.scan_start_x_m / spec.dl_m))
    scan_right = int(round((spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m) / spec.dl_m)) + 1

    width, height = 1800, 1260
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    boxes = ((80, 70, 1760, 385), (80, 465, 1760, 800), (80, 880, 1760, 1080))
    viridis = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))
    coolwarm = ((49, 54, 149), (116, 173, 209), (247, 247, 247), (244, 109, 67), (165, 0, 38))

    whole = full_map[::12, ::4].T
    crop = full_map[scan_left:scan_right:2, ::3].T
    delta = delta_map[scan_left:scan_right:2, ::3].T
    images = (
        _colour_map((whole - 1.0) / 14.5, viridis),
        _colour_map((crop - 1.0) / 14.5, viridis),
        _colour_map((delta + 4.5) / 9.0, coolwarm),
    )
    for rgb, box in zip(images, boxes):
        image = Image.fromarray(np.flipud(rgb), mode="RGB")
        image = image.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
        canvas.paste(image, (box[0], box[1]))
        draw.rectangle(box, outline="black", width=2)

    whole_scan_left = boxes[0][0] + spec.scan_start_x_m / spec.domain_x_m * (boxes[0][2] - boxes[0][0])
    whole_scan_right = boxes[0][0] + (spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m) / spec.domain_x_m * (boxes[0][2] - boxes[0][0])
    draw.rectangle((whole_scan_left, boxes[0][1], whole_scan_right, boxes[0][3]), outline="white", width=3)
    draw.text((80, 25), f"{case['case_id']} | MACRO03-inherited native correlated-voxel batch", fill="black", font=font)
    draw.text((80, 400), "Full 243.135 m guarded domain; white box is the unscaled native 22.95 m scan", fill="black", font=font)
    draw.text((80, 815), "Native scan crop: full-scene epsilon_r and finite tapered upper lenses", fill="black", font=font)
    draw.text((80, 1095), "Native scan crop: full minus control constitutive contrast (zero for true negative)", fill="black", font=font)
    draw.text(
        (80, 1140),
        f"target={case['target_presence']}  range={metrics.get('scan_depth_range_m', 'n/a')} m  "
        f"extrema={metrics.get('smoothed_extrema_count', 'n/a')}  quadratic_R2={metrics.get('quadratic_fit_r2', 'n/a')}",
        fill="black",
        font=font,
    )
    draw.text((80, 1180), "Geometry only. Geometric arrival is an audit prior; visible-phase labels require solved signed pairs.", fill="black", font=font)
    canvas.save(case_dir / "preview_geometry_and_material_contrast.png")


def write_checksums(case_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file() and item.name != "FILE_SHA256.csv"):
        rows.append((path.relative_to(case_dir).as_posix(), sha256(path), path.stat().st_size))
    with (case_dir / "FILE_SHA256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        writer.writerows(rows)


def generate_case(
    output_root: Path,
    case: dict[str, Any],
    *,
    overwrite: bool,
    spec: Spec | None = None,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    spec = spec or Spec()
    case_dir = output_root / str(case["case_id"])
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"case exists; pass --overwrite: {case_dir}")
        shutil.rmtree(case_dir)
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(parents=True)

    quantiles, thresholds = correlated_quantiles(spec, int(case["field_seed"]))
    profiles = build_profiles(spec, case)
    lenses = lens_definitions(spec, str(case["lens_family"]))
    data = build_indices(spec, case, quantiles, profiles, lenses)
    h5_path = case_dir / "geology_indices.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = "scripts/generate_native_256_correlated_voxel_batch.py"
        handle.attrs["field_seed"] = int(case["field_seed"])
        handle.attrs["profile_seed"] = int(case["profile_seed"])
        handle.create_dataset(
            "data",
            data=data,
            dtype=np.int16,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(256, spec.nx), min(256, spec.ny), 1),
        )

    full_rows = material_rows(str(case["material_family"]), control=False)
    control_rows = material_rows(str(case["material_family"]), control=True)
    write_materials(case_dir / "materials_full.txt", full_rows)
    write_materials(case_dir / "materials_no_basal.txt", control_rows)
    case_id = str(case["case_id"])
    (case_dir / "full_scene.in").write_text(input_text(spec, f"{case_id} full", "materials_full.txt"), encoding="ascii")
    if bool(case["target_presence"]):
        (case_dir / "no_basal_contrast_control.in").write_text(
            input_text(spec, f"{case_id} no basal contrast", "materials_no_basal.txt"), encoding="ascii"
        )
    (case_dir / "air_reference.in").write_text(input_text(spec, f"{case_id} air", None), encoding="ascii")
    (case_dir / "geometry_check_full.in").write_text(
        input_text(spec, f"{case_id} geometry full", "materials_full.txt", geometry_view="geometry_full"),
        encoding="ascii",
    )
    if bool(case["target_presence"]):
        (case_dir / "geometry_check_control.in").write_text(
            input_text(spec, f"{case_id} geometry control", "materials_no_basal.txt", geometry_view="geometry_control"),
            encoding="ascii",
        )

    arrays = scan_arrays(
        spec,
        profiles,
        bool(case["target_presence"]),
        str(case["material_family"]),
    )
    for name, values in arrays.items():
        np.save(labels_dir / f"{name}.npy", values)
    if "geometric_reference_arrival_time_ns" in arrays:
        np.save(labels_dir / "reference_arrival_time_ns.npy", arrays["geometric_reference_arrival_time_ns"])
        np.save(labels_dir / "geometric_arrival_time_ns.npy", arrays["geometric_reference_arrival_time_ns"])
    for name, values in profiles.items():
        np.save(labels_dir / f"full_{name}.npy", values)
    np.save(labels_dir / "target_presence.npy", np.asarray(bool(case["target_presence"]), dtype=np.bool_))

    metrics = morphology_metrics(
        spec,
        profiles,
        bool(case["target_presence"]),
        str(case["material_family"]),
    )
    if bool(case["target_presence"]) and not metrics["broad_morphology_gate_ok"]:
        raise RuntimeError(f"{case_id} failed native morphology gate: {metrics}")
    changed_indices = []
    unchanged_indices = []
    for full, control in zip(full_rows, control_rows):
        changed = any(full[field] != control[field] for field in ("epsilon_r", "conductivity_s_per_m"))
        (changed_indices if changed else unchanged_indices).append(int(full["index"]))
    expected_changed = list(range(2 * len(Q_OFFSETS), 6 * len(Q_OFFSETS)))
    if bool(case["target_presence"]) and changed_indices != expected_changed:
        raise RuntimeError(f"{case_id}: strict pair changed unexpected material indices: {changed_indices}")

    max_eps = max(float(row["epsilon_r"]) for row in full_rows)
    cells_per_lambda = C0 / (2.8 * spec.center_frequency_hz * math.sqrt(max_eps) * spec.dl_m)
    manifest = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "standard_id": "PGDA_NATIVE_256_RELEASE_STANDARD_V1",
        "case_id": case_id,
        "scene_family_id": case["scene_family_id"],
        "purpose": case["purpose"],
        "target_presence": bool(case["target_presence"]),
        "formal_training_allowed": False,
        "training_block_reason": "requires solver outputs, signed-pair label extraction, independent audit, and human promotion",
        "line9_conditioned": False,
        "reference_line": None,
        "generator_path": "scripts/generate_native_256_correlated_voxel_batch.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "catalog_path": str(catalog_path.relative_to(ROOT)).replace("\\", "/")
        if catalog_path.is_relative_to(ROOT)
        else str(catalog_path),
        "catalog_sha256": sha256(catalog_path),
        "spec": asdict(spec),
        "grid": {
            "dimension": "2D_x_y_with_one_z_cell",
            "nx_ny_nz": [spec.nx, spec.ny, 1],
            "dl_m": spec.dl_m,
            "domain_x_m": spec.domain_x_m,
            "domain_y_m": spec.domain_y_m,
            "pml_cells": [spec.pml_cells, spec.pml_cells, 0, spec.pml_cells, spec.pml_cells, 0],
            "trace_count": spec.trace_count,
            "trace_spacing_m": spec.trace_spacing_m,
            "trace_midpoint_span_m": spec.scan_span_m,
            "solver_time_window_ns": spec.solver_time_window_s * 1e9,
            "canonical_time_window_ns": 700.0,
            "canonical_output_samples": 501,
            "canonical_output_dt_ns": 1.4,
            "cells_per_min_wavelength_at_2p8fc": cells_per_lambda,
            "left_scan_margin_m": spec.scan_start_x_m,
            "right_scan_margin_m": spec.domain_x_m - (spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m),
        },
        "source": {
            "model": "ideal_hertzian_line_source",
            "polarization": "z",
            "waveform": "ricker",
            "center_frequency_hz": spec.center_frequency_hz,
            "tx_rx_offset_m": spec.tx_rx_offset_m,
        },
        "geometry": {
            "index_file": "geology_indices.h5",
            "index_file_sha256": sha256(h5_path),
            "index_shape": list(data.shape),
            "index_min_max": [int(data.min()), int(data.max())],
            "field_seed": int(case["field_seed"]),
            "profile_seed": int(case["profile_seed"]),
            "quantile_thresholds": thresholds,
            "correlation_scales_m": {
                "property_field_horizontal": [8.64, 3.24, 1.08],
                "property_field_vertical": [1.44, 0.63, 0.234],
                "interface_horizontal": [42.0, 8.0, 1.6],
            },
            "dielectric_smoothing": "external field pre-smoothed before five-bin quantisation; gprMax import averaging remains off",
            "scan_window_transfer": "native scan crop from full guarded-domain correlated fields; no long-profile rescaling",
            "finite_tapered_lenses": lenses,
            "morphology_metrics": metrics,
        },
        "strict_pair": {
            "required": bool(case["target_presence"]),
            "shared_geometry_hdf5": bool(case["target_presence"]),
            "shared_geometry_sha256": sha256(h5_path),
            "unchanged_material_indices": unchanged_indices,
            "changed_material_indices": changed_indices if bool(case["target_presence"]) else [],
            "only_transition_and_bedrock_changed": bool(case["target_presence"]),
        },
        "labels": {
            "geometric_reference": "columnar layered audit prior only",
            "visible_phase": "pending continuous envelope support plus signed-lobe extraction from solved full-minus-control",
            "target_mask_training_allowed": False,
        },
    }
    (case_dir / "scene_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview(case_dir, spec, case, data, profiles, full_rows, control_rows, metrics)
    write_checksums(case_dir)
    return manifest


def generate_batch(
    catalog_path: Path,
    output_root: Path,
    *,
    selected: set[str],
    overwrite: bool,
) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("standard_id") != "PGDA_NATIVE_256_RELEASE_STANDARD_V1":
        raise ValueError("catalog is not bound to the native 256 standard")
    if catalog.get("formal_training_allowed") is not False:
        raise ValueError("pre-solver catalog must remain blocked from formal training")
    cases = list(catalog.get("cases", []))
    if selected:
        cases = [case for case in cases if str(case["case_id"]) in selected]
        missing = selected - {str(case["case_id"]) for case in cases}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    ids = [str(case["case_id"]) for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("catalog must contain unique cases")
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = [
        generate_case(output_root, case, overwrite=overwrite, catalog_path=catalog_path)
        for case in cases
    ]
    report = {
        "schema": "native_256_correlated_voxel_preflight_v1",
        "formal_training_allowed": False,
        "catalog_sha256": sha256(catalog_path),
        "case_count": len(manifests),
        "positive_case_count": sum(bool(item["target_presence"]) for item in manifests),
        "negative_case_count": sum(not bool(item["target_presence"]) for item in manifests),
        "case_ids": ids,
        "morphology_gates_ok": all(
            not bool(item["target_presence"]) or bool(item["geometry"]["morphology_metrics"]["broad_morphology_gate_ok"])
            for item in manifests
        ),
        "promotion_blockers": [
            "No complete 256-trace solver outputs exist.",
            "No signed visible-phase labels exist.",
            "No independent visual release decision exists.",
        ],
    }
    (output_root / "native_256_correlated_voxel_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = generate_batch(
        args.catalog.resolve(),
        args.output_root.resolve(),
        selected=set(args.case_id),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["morphology_gates_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
