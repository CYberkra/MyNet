#!/usr/bin/env python3
"""Generate the first independent V2 positive and matched true-negative family.

The generator is deliberately self-contained. It reads only the committed
family contract and seeded generic priors. It never reads measured arrays,
Line9 assets, or the Line9-conditioned FORMAL06/07 development geometries.
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
DEFAULT_CONTRACT = ROOT / "data" / "contracts" / "simulation_v2" / "independent_v2_family01_pilot.json"
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
SUPPORTED_CONTRACT_IDS = {
    "PGDA_INDEPENDENT_V2_FAMILY01_PILOT_V1",
    "PGDA_INDEPENDENT_V2_FAMILY01_R2_PILOT_V1",
}
C0 = 299_792_458.0
COVER_BINS = 32
TRANSITION_LEVELS = 8
TRANSITION_START = COVER_BINS
BEDROCK_START = COVER_BINS + TRANSITION_LEVELS * COVER_BINS
MATERIAL_COUNT = BEDROCK_START + COVER_BINS


@dataclass(frozen=True)
class Spec:
    domain_x_m: float = 242.73
    domain_y_m: float = 48.0
    dl_m: float = 0.03
    pml_cells: int = 60
    physical_side_guard_m: float = 108.0
    trace_count: int = 256
    trace_spacing_m: float = 0.09
    scan_start_x_m: float = 109.8
    tx_rx_offset_m: float = 0.18
    ground_y_m: float = 28.5
    source_y_m: float = 36.51
    center_frequency_hz: float = 55e6
    solver_time_window_s: float = 750e-9
    protected_time_window_ns: float = 700.0

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
    def source_reference_delay_ns(self) -> float:
        return math.sqrt(2.0) / self.center_frequency_hz * 1e9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_text(path: Path, text: str, *, encoding: str) -> None:
    """Write hash-protected generated text with platform-independent LF bytes."""
    path.write_text(text, encoding=encoding, newline="\n")


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot normalise a constant or non-finite field")
    return (values - float(np.mean(values))) / scale


def smooth_noise_1d(rng: np.random.Generator, size: int, sigma_cells: float) -> np.ndarray:
    return normalise(gaussian_filter1d(rng.standard_normal(size), sigma=sigma_cells, mode="reflect"))


def _quadratic_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
    smooth = gaussian_filter1d(y.astype(np.float64), sigma=max(1.0, 0.45 / float(np.median(np.diff(x)))), mode="nearest")
    derivative = np.diff(smooth)
    sign = np.sign(derivative)
    sign[sign == 0] = 1
    extrema = int(np.count_nonzero(sign[1:] != sign[:-1]))
    u = x.astype(np.float64) - float(np.mean(x))
    u2 = u * u
    sum_u2 = float(np.sum(u2))
    sum_u4 = float(np.sum(u2 * u2))
    determinant = len(u) * sum_u4 - sum_u2 * sum_u2
    a0 = (float(np.sum(y)) * sum_u4 - float(np.sum(y * u2)) * sum_u2) / determinant
    a1 = float(np.sum(y * u)) / sum_u2
    a2 = (len(u) * float(np.sum(y * u2)) - sum_u2 * float(np.sum(y))) / determinant
    fitted = a0 + a1 * u + a2 * u2
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 1.0
    slope_p95 = float(np.percentile(np.abs(np.diff(smooth) / np.diff(x)), 95))
    return float(r2), extrema, slope_p95


def build_profiles(spec: Spec, seed_base: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select one generic multiscale basal crop using only broad shape gates."""
    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    midpoint = spec.scan_start_x_m + spec.tx_rx_offset_m / 2 + np.arange(spec.trace_count) * spec.trace_spacing_m
    for attempt in range(64):
        accepted_seed = seed_base + attempt
        rng = np.random.default_rng(accepted_seed)
        long = smooth_noise_1d(rng, spec.nx, 24.0 / spec.dl_m)
        meso = smooth_noise_1d(rng, spec.nx, 5.0 / spec.dl_m)
        local = smooth_noise_1d(rng, spec.nx, 1.3 / spec.dl_m)
        basal = np.clip(15.2 + 0.78 * long + 0.48 * meso + 0.16 * local, 13.5, 17.5)
        transition = np.clip(
            1.35
            + 0.24 * smooth_noise_1d(rng, spec.nx, 8.0 / spec.dl_m)
            + 0.11 * smooth_noise_1d(rng, spec.nx, 2.0 / spec.dl_m),
            0.9,
            2.0,
        )
        scan_basal = np.interp(midpoint, x, basal)
        r2, extrema, slope_p95 = _quadratic_metrics(midpoint, scan_basal)
        depth_range = float(np.ptp(scan_basal))
        gate = bool(
            0.45 <= depth_range <= 2.2
            and 2 <= extrema <= 8
            and r2 < 0.985
            and slope_p95 <= 0.20
        )
        if gate:
            profiles = {
                "x_m": x.astype(np.float32),
                "ground_y_m": np.full(spec.nx, spec.ground_y_m, dtype=np.float32),
                "basal_depth_m": basal.astype(np.float32),
                "transition_thickness_m": transition.astype(np.float32),
                "basal_y_m": (spec.ground_y_m - basal).astype(np.float32),
                "transition_top_y_m": (spec.ground_y_m - basal + transition).astype(np.float32),
            }
            metrics = {
                "accepted_seed": accepted_seed,
                "attempt": attempt,
                "scan_depth_min_m": float(np.min(scan_basal)),
                "scan_depth_median_m": float(np.median(scan_basal)),
                "scan_depth_max_m": float(np.max(scan_basal)),
                "scan_depth_range_m": depth_range,
                "smoothed_extrema_count": extrema,
                "quadratic_fit_r2": r2,
                "abs_slope_p95": slope_p95,
                "broad_shape_gate_ok": True,
            }
            return profiles, metrics
    raise RuntimeError("no independent basal profile passed the frozen broad shape gates")


def correlated_cover_bins(spec: Spec, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a smooth, aperiodic 2D field before 32-bin material quantisation."""
    factor = 6
    coarse_shape = (math.ceil(spec.nx / factor), math.ceil(spec.ny / factor))
    coarse_dl = spec.dl_m * factor
    rng = np.random.default_rng(seed)
    components = (
        ((9.0 / coarse_dl, 3.0 / coarse_dl), 0.55),
        ((2.4 / coarse_dl, 1.2 / coarse_dl), 0.30),
        ((0.75 / coarse_dl, 0.45 / coarse_dl), 0.15),
    )
    field = np.zeros(coarse_shape, dtype=np.float32)
    for sigma, weight in components:
        component = gaussian_filter(rng.standard_normal(coarse_shape).astype(np.float32), sigma=sigma, mode="reflect")
        field += weight * normalise(component)
    field = normalise(field)
    field = zoom(field, (spec.nx / coarse_shape[0], spec.ny / coarse_shape[1]), order=1, mode="reflect")
    field = np.clip(normalise(field[: spec.nx, : spec.ny]), -2.75, 2.75)
    thresholds = np.quantile(field, np.arange(1, COVER_BINS) / COVER_BINS).astype(np.float32)
    bins = np.digitize(field, thresholds).astype(np.int16)

    ground_cell = int(round(spec.ground_y_m / spec.dl_m))
    scan_left = int(round(spec.scan_start_x_m / spec.dl_m))
    scan_right = int(round((spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m) / spec.dl_m)) + 1
    cover_crop = field[scan_left:scan_right, max(0, ground_cell - int(12.0 / spec.dl_m)) : ground_cell]
    mean_depth_profile = np.mean(cover_crop, axis=0)
    spectrum = np.abs(np.fft.rfft(mean_depth_profile - float(np.mean(mean_depth_profile)))) ** 2
    vertical_peak_fraction = float(np.max(spectrum[1:]) / np.sum(spectrum[1:])) if spectrum.size > 1 and np.sum(spectrum[1:]) > 0 else 0.0
    stats = {
        "seed": seed,
        "coarse_shape": list(coarse_shape),
        "correlation_scales_m": {
            "long_xy": [9.0, 3.0],
            "meso_xy": [2.4, 1.2],
            "local_xy": [0.75, 0.45],
        },
        "component_weights": [0.55, 0.30, 0.15],
        "used_bins": int(np.unique(bins).size),
        "horizontal_neighbor_bin_change_rate": float(np.mean(bins[1:, :] != bins[:-1, :])),
        "vertical_neighbor_bin_change_rate": float(np.mean(bins[:, 1:] != bins[:, :-1])),
        "vertical_spectral_peak_fraction_in_scan_cover": vertical_peak_fraction,
        "quantile_thresholds": [float(value) for value in thresholds],
        "sinusoidal_stratigraphy": False,
        "isolated_inclusions": 0,
        "point_targets": 0,
        "vertical_partitions": 0,
    }
    return bins, stats


def build_indices(spec: Spec, bins: np.ndarray, profiles: dict[str, np.ndarray]) -> np.ndarray:
    data = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    y = (np.arange(spec.ny, dtype=np.float32) + 0.5) * spec.dl_m
    for start in range(0, spec.nx, 256):
        stop = min(start + 256, spec.nx)
        local_bins = bins[start:stop]
        depth = spec.ground_y_m - y[None, :]
        subsurface = depth >= 0
        block = np.where(subsurface, local_bins, -1).astype(np.int16)
        basal = profiles["basal_depth_m"][start:stop, None]
        transition = profiles["transition_thickness_m"][start:stop, None]
        transition_top = basal - transition
        in_transition = subsurface & (depth >= transition_top) & (depth < basal)
        fraction = np.clip((depth - transition_top) / np.maximum(transition, spec.dl_m), 0.0, 0.999999)
        levels = np.floor(fraction * TRANSITION_LEVELS).astype(np.int16)
        block[in_transition] = (TRANSITION_START + levels * COVER_BINS + local_bins)[in_transition]
        in_bedrock = subsurface & (depth >= basal)
        block[in_bedrock] = (BEDROCK_START + local_bins)[in_bedrock]
        data[start:stop] = block
    return data[:, :, None]


def _cover_properties(local_bin: int) -> tuple[float, float]:
    fraction = local_bin / (COVER_BINS - 1)
    return 11.8 + 0.8 * fraction, 0.0018 + 0.0010 * fraction


def material_rows(*, control: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_bin in range(COVER_BINS):
        epsilon, conductivity = _cover_properties(local_bin)
        rows.append({
            "index": local_bin,
            "region": "cover",
            "local_bin": local_bin,
            "epsilon_r": epsilon,
            "conductivity_s_per_m": conductivity,
            "id": f"cover_{local_bin:02d}",
        })
    for level in range(TRANSITION_LEVELS):
        alpha = (level + 1) / TRANSITION_LEVELS
        for local_bin in range(COVER_BINS):
            cover_epsilon, cover_conductivity = _cover_properties(local_bin)
            local_offset = (local_bin / (COVER_BINS - 1) - 0.5) * 2.0
            cap_epsilon = 11.62 + 0.06 * local_offset
            cap_conductivity = 0.0019 + 0.00008 * local_offset
            if control:
                epsilon, conductivity = cover_epsilon, cover_conductivity
            else:
                epsilon = (1 - alpha) * cover_epsilon + alpha * cap_epsilon
                conductivity = (1 - alpha) * cover_conductivity + alpha * cap_conductivity
            index = TRANSITION_START + level * COVER_BINS + local_bin
            rows.append({
                "index": index,
                "region": f"transition_{level + 1}",
                "local_bin": local_bin,
                "epsilon_r": epsilon,
                "conductivity_s_per_m": conductivity,
                "id": f"transition_{level + 1}_{local_bin:02d}",
            })
    for local_bin in range(COVER_BINS):
        cover_epsilon, cover_conductivity = _cover_properties(local_bin)
        local_offset = (local_bin / (COVER_BINS - 1) - 0.5) * 2.0
        if control:
            epsilon, conductivity = cover_epsilon, cover_conductivity
        else:
            epsilon = 11.32 + 0.05 * local_offset
            conductivity = 0.0012 + 0.00006 * local_offset
        index = BEDROCK_START + local_bin
        rows.append({
            "index": index,
            "region": "bedrock",
            "local_bin": local_bin,
            "epsilon_r": epsilon,
            "conductivity_s_per_m": conductivity,
            "id": f"bedrock_{local_bin:02d}",
        })
    if [row["index"] for row in rows] != list(range(MATERIAL_COUNT)):
        raise RuntimeError("material rows are not contiguous and index-aligned")
    return rows


def write_materials(path: Path, rows: list[dict[str, Any]]) -> None:
    write_canonical_text(
        path,
        "\n".join(
            f"#material: {row['epsilon_r']:.9g} {row['conductivity_s_per_m']:.9g} 1 0 {row['id']}"
            for row in rows
        )
        + "\n",
        encoding="ascii",
    )


def input_text(spec: Spec, title: str, material_file: str | None, *, geometry_view: str | None = None) -> str:
    def number(value: float) -> str:
        return f"{value:.12g}"

    lines = [
        f"#title: {title}",
        f"#domain: {number(spec.domain_x_m)} {number(spec.domain_y_m)} {number(spec.dl_m)}",
        f"#dx_dy_dz: {number(spec.dl_m)} {number(spec.dl_m)} {number(spec.dl_m)}",
        f"#time_window: {number(spec.solver_time_window_s)}",
        f"#pml_cells: {spec.pml_cells} {spec.pml_cells} 0 {spec.pml_cells} {spec.pml_cells} 0",
        "#messages: y",
        f"#waveform: ricker 1 {number(spec.center_frequency_hz)} iv2_f01_ricker55",
        f"#hertzian_dipole: z {number(spec.scan_start_x_m)} {number(spec.source_y_m)} 0 iv2_f01_ricker55",
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


def scan_arrays(spec: Spec, profiles: dict[str, np.ndarray], full_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    source_x = spec.scan_start_x_m + np.arange(spec.trace_count) * spec.trace_spacing_m
    receiver_x = source_x + spec.tx_rx_offset_m
    midpoint = (source_x + receiver_x) / 2
    basal = np.interp(midpoint, profiles["x_m"], profiles["basal_depth_m"])
    transition = np.interp(midpoint, profiles["x_m"], profiles["transition_thickness_m"])
    cover_epsilon = np.mean([float(row["epsilon_r"]) for row in full_rows[:COVER_BINS]])
    cap_rows = full_rows[TRANSITION_START:BEDROCK_START]
    transition_epsilon = np.mean([float(row["epsilon_r"]) for row in cap_rows])
    one_way_optical_m = (
        (spec.source_y_m - spec.ground_y_m)
        + np.maximum(basal - transition, 0.0) * math.sqrt(cover_epsilon)
        + transition * math.sqrt(transition_epsilon)
    )
    geometric = 2e9 * one_way_optical_m / C0
    return {
        "source_x_m": source_x.astype(np.float32),
        "receiver_x_m": receiver_x.astype(np.float32),
        "trace_midpoint_x_m": midpoint.astype(np.float32),
        "flight_height_m": np.full(spec.trace_count, spec.source_y_m - spec.ground_y_m, dtype=np.float32),
        "basal_interface_depth_m": basal.astype(np.float32),
        "transition_thickness_m": transition.astype(np.float32),
        "geometric_reference_arrival_time_ns": geometric.astype(np.float32),
        "source_referenced_arrival_time_ns": (geometric + spec.source_reference_delay_ns).astype(np.float32),
    }


def acquisition_arrays(spec: Spec) -> dict[str, np.ndarray]:
    """Return target-independent acquisition coordinates for every case."""
    source_x = spec.scan_start_x_m + np.arange(spec.trace_count) * spec.trace_spacing_m
    receiver_x = source_x + spec.tx_rx_offset_m
    return {
        "source_x_m": source_x.astype(np.float32),
        "receiver_x_m": receiver_x.astype(np.float32),
        "trace_midpoint_x_m": ((source_x + receiver_x) / 2).astype(np.float32),
        "flight_height_m": np.full(spec.trace_count, spec.source_y_m - spec.ground_y_m, dtype=np.float32),
    }


def _material_maps(data: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    lookup = np.asarray([float(row["epsilon_r"]) for row in rows], dtype=np.float32)
    flat = data[:, :, 0]
    safe = np.clip(flat, 0, len(lookup) - 1)
    return np.where(flat >= 0, lookup[safe], 1.0)


def render_preview(
    output_root: Path,
    spec: Spec,
    data: np.ndarray,
    profiles: dict[str, np.ndarray],
    full_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    shape_metrics: dict[str, Any],
    *,
    output_name: str = "IV2_F01_FAMILY_PRE_SOLVER_PREVIEW.png",
    heading: str = "Independent V2 Family 01 pre-solver geometry audit",
    positive_title: str = (
        "Positive full: smooth aperiodic cover + finite weathered transition + weak bedrock contrast"
    ),
) -> Path:
    full_map = _material_maps(data, full_rows)
    control_map = _material_maps(data, control_rows)
    scan_left = int(round((spec.scan_start_x_m - 3.0) / spec.dl_m))
    scan_right = int(round((spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m + 3.0) / spec.dl_m))
    y_bottom = int(round(6.0 / spec.dl_m))
    y_top = int(round((spec.ground_y_m + 1.0) / spec.dl_m))
    full_crop = full_map[scan_left:scan_right:2, y_bottom:y_top:2].T
    control_crop = control_map[scan_left:scan_right:2, y_bottom:y_top:2].T
    delta_crop = full_crop - control_crop

    width, height = 1700, 1320
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panels = ((80, 110, 1620, 445), (80, 535, 1620, 870), (80, 960, 1620, 1210))
    panel_arrays = (full_crop, control_crop, delta_crop)
    panel_titles = (
        positive_title,
        "Matched background / positive control: exact target-absent physical state",
        "Constitutive difference: only transition and bedrock contrast",
    )
    for array, box, title in zip(panel_arrays, panels, panel_titles):
        if title.startswith("Constitutive"):
            scale = max(float(np.max(np.abs(array))), 1e-6)
            normalised = np.clip(0.5 + 0.5 * array / scale, 0.0, 1.0)
            rgb = np.stack((normalised, 1 - np.abs(normalised - 0.5) * 2, 1 - normalised), axis=-1)
        else:
            normalised = np.clip((array - 10.8) / 2.2, 0.0, 1.0)
            rgb = np.stack((0.15 + 0.75 * normalised, 0.22 + 0.62 * normalised, 0.75 - 0.58 * normalised), axis=-1)
        image = Image.fromarray(np.rint(np.flipud(rgb) * 255).astype(np.uint8), mode="RGB")
        image = image.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
        canvas.paste(image, (box[0], box[1]))
        draw.rectangle(box, outline="black", width=2)
        draw.text((box[0], box[1] - 28), title, fill="black", font=font)

    draw.text((80, 35), heading, fill="black", font=font)
    draw.text(
        (80, 1240),
        f"depth range={shape_metrics['scan_depth_range_m']:.3f} m | extrema={shape_metrics['smoothed_extrema_count']} | "
        f"quadratic R2={shape_metrics['quadratic_fit_r2']:.4f} | no measured or development arrays read",
        fill="black",
        font=font,
    )
    draw.text((80, 1270), "Geometry preview only. No visible-phase label exists before a solved signed pair.", fill="black", font=font)
    preview_path = output_root / output_name
    canvas.save(preview_path)
    return preview_path


def write_checksums(case_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file() and item.name != "FILE_SHA256.csv"):
        rows.append((path.relative_to(case_dir).as_posix(), sha256(path), path.stat().st_size))
    with (case_dir / "FILE_SHA256.csv").open("w", newline="\n", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        writer.writerows(rows)


def _write_case(
    case_dir: Path,
    case: dict[str, Any],
    contract: dict[str, Any],
    spec: Spec,
    data: np.ndarray,
    profiles: dict[str, np.ndarray],
    shape_metrics: dict[str, Any],
    field_stats: dict[str, Any],
    full_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    geometry_source: Path,
    contract_path: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True)
    labels_dir = case_dir / "labels"
    labels_dir.mkdir()
    geometry_path = case_dir / "geology_indices.h5"
    shutil.copy2(geometry_source, geometry_path)
    target_presence = bool(case["target_presence"])
    physical_rows = full_rows if target_presence else control_rows
    write_materials(case_dir / "materials_full.txt", physical_rows)
    write_canonical_text(
        case_dir / "full_scene.in",
        input_text(spec, f"{case['case_id']} full", "materials_full.txt"),
        encoding="ascii",
    )
    write_canonical_text(
        case_dir / "geometry_check_full.in",
        input_text(spec, f"{case['case_id']} geometry", "materials_full.txt", geometry_view="geometry_full"),
        encoding="ascii",
    )
    if target_presence:
        write_materials(case_dir / "materials_no_basal.txt", control_rows)
        write_canonical_text(
            case_dir / "no_basal_contrast_control.in",
            input_text(spec, f"{case['case_id']} no basal", "materials_no_basal.txt"),
            encoding="ascii",
        )
        write_canonical_text(
            case_dir / "geometry_check_control.in",
            input_text(spec, f"{case['case_id']} control geometry", "materials_no_basal.txt", geometry_view="geometry_control"),
            encoding="ascii",
        )
    write_canonical_text(
        case_dir / "air_reference.in", input_text(spec, f"{case['case_id']} air", None), encoding="ascii"
    )

    np.save(labels_dir / "target_presence.npy", np.asarray(target_presence, dtype=np.bool_))
    np.save(labels_dir / "valid_trace_mask.npy", np.ones(spec.trace_count, dtype=np.bool_))
    for name, values in acquisition_arrays(spec).items():
        np.save(labels_dir / f"{name}.npy", values)
    if target_presence:
        arrays = scan_arrays(spec, profiles, full_rows)
        for name, values in arrays.items():
            np.save(labels_dir / f"{name}.npy", values)
        np.save(labels_dir / "reference_arrival_time_ns.npy", arrays["geometric_reference_arrival_time_ns"])
        np.save(labels_dir / "geometric_arrival_time_ns.npy", arrays["geometric_reference_arrival_time_ns"])
        np.save(labels_dir / "target_mask.npy", np.zeros((501, spec.trace_count), dtype=np.float32))
    else:
        np.save(labels_dir / "trace_state.npy", np.zeros(spec.trace_count, dtype=np.int8))
        np.save(labels_dir / "target_mask.npy", np.zeros((501, spec.trace_count), dtype=np.float32))

    pml_m = spec.pml_cells * spec.dl_m
    right_guard = spec.domain_x_m - pml_m - (spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m)
    max_epsilon = max(float(row["epsilon_r"]) for row in full_rows)
    cells_per_min_wavelength = C0 / (2.8 * spec.center_frequency_hz * math.sqrt(max_epsilon)) / spec.dl_m
    manifest = {
        "contract_id": contract["contract_id"],
        "case_id": case["case_id"],
        "scene_family_id": contract["scene_family_id"],
        "lifecycle_state": "pre_solver_blocked_pilot",
        "formal_training_allowed": False,
        "promotion_allowed": False,
        "target_presence": target_presence,
        "negative_semantics": case["negative_semantics"],
        "line9_conditioned": False,
        "strict_line9_holdout_allowed": True,
        "generator_path": "scripts/generate_independent_v2_family01.py",
        "generator_sha256": sha256(Path(__file__)),
        "contract_path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
        "contract_sha256": sha256(contract_path),
        "training_block_reason": "pre-solver pilot; complete runtime, human semantics, and registry promotion gates remain pending",
        "parameter_provenance": contract["provenance"],
        "grid": {
            "trace_count": spec.trace_count,
            "trace_spacing_m": spec.trace_spacing_m,
            "dl_m": spec.dl_m,
            "domain_x_m": spec.domain_x_m,
            "domain_y_m": spec.domain_y_m,
            "nx_ny_nz": [spec.nx, spec.ny, 1],
            "pml_cells": [spec.pml_cells, spec.pml_cells, 0, spec.pml_cells, spec.pml_cells, 0],
            "left_physical_guard_m": spec.scan_start_x_m - pml_m,
            "right_physical_guard_m": right_guard,
            "earliest_lateral_boundary_round_trip_ns": 2e9 * min(spec.scan_start_x_m - pml_m, right_guard) / C0,
            "solver_time_window_ns": spec.solver_time_window_s * 1e9,
            "protected_time_window_ns": spec.protected_time_window_ns,
            "protected_window_end_ns": spec.protected_time_window_ns,
            "canonical_output_samples": 501,
            "canonical_output_dt_ns": 1.4,
            "cells_per_min_wavelength_at_2_8fc": cells_per_min_wavelength,
        },
        "source": {
            **contract["source"],
            "reference_delay_ns": spec.source_reference_delay_ns,
        },
        "geometry": {
            "index_file": "geology_indices.h5",
            "index_file_sha256": sha256(geometry_path),
            "index_array_sha256": array_sha256(data),
            "index_shape": list(data.shape),
            "profile_shape_gate": shape_metrics,
            "cover_field_statistics": field_stats,
            "flat_ground": True,
            "fixed_flight_height_m": spec.source_y_m - spec.ground_y_m,
            "discrete_anomaly_bodies": 0,
            "visible_phase_geometry_written": False,
            "latent_target_partition_constitutively_neutral": not target_presence,
        },
        "materials": {
            "full_materials_sha256": sha256(case_dir / "materials_full.txt"),
            "material_count": MATERIAL_COUNT,
            "cover_bins": COVER_BINS,
            "transition_levels": TRANSITION_LEVELS,
            "only_positive_full_has_basal_contrast": True,
        },
        "strict_pair": {
            "required": target_presence,
            "shared_geometry_hdf5": target_presence,
            "control_restores_transition_and_bedrock_to_each_local_cover_bin": True,
            "positive_control_equals_family_negative_full": True,
        },
        "labels": {
            "geometric_reference": "audit prior only" if target_presence else "absent for true negative",
            "source_referenced_arrival": "audit prior only" if target_presence else "absent for true negative",
            "visible_phase_search_half_width_ns": 55.0,
            "visible_phase_phase_half_width_ns": 10.0,
            "visible_phase": "absent until solved signed pair review",
            "target_mask_training_allowed": False,
        },
        "next_gate": "static and geometry-only audit, then positive one-trace pair plus negative one-trace full smoke",
    }
    if target_presence:
        manifest["materials"]["no_basal_materials_sha256"] = sha256(case_dir / "materials_no_basal.txt")
    write_canonical_text(
        case_dir / "scene_manifest.json", json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    write_canonical_text(
        case_dir / "RUN_COMMANDS.md",
        "# Runtime commands\n\n"
        "Run through `scripts/run_native_256_release_pilot.py`; never execute or overwrite this source deck in place.\n",
        encoding="utf-8",
    )
    write_checksums(case_dir)
    return manifest


def generate_family(contract_path: Path, output_root: Path, *, overwrite: bool, spec: Spec | None = None) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_id") not in SUPPORTED_CONTRACT_IDS:
        raise ValueError("unexpected independent family contract")
    if contract.get("formal_training_allowed") is not False:
        raise ValueError("pre-solver family must remain blocked")
    provenance = contract["provenance"]
    if provenance.get("line9_conditioned") is not False or provenance.get("measured_files_read_by_generator") != []:
        raise ValueError("independent generator contract forbids measured or Line9 inputs")
    spec = spec or Spec()
    family_dir = output_root / contract["scene_family_id"]
    if family_dir.exists():
        if not overwrite:
            raise FileExistsError(f"family exists; pass --overwrite: {family_dir}")
        shutil.rmtree(family_dir)
    family_dir.mkdir(parents=True)

    positive = next(case for case in contract["cases"] if case["target_presence"])
    negative = next(case for case in contract["cases"] if not case["target_presence"])
    if positive["profile_seed_base"] != negative["profile_seed_base"] or positive["field_seed"] != negative["field_seed"]:
        raise ValueError("matched positive/negative must share profile and field seeds")
    profiles, shape_metrics = build_profiles(spec, int(positive["profile_seed_base"]))
    bins, field_stats = correlated_cover_bins(spec, int(positive["field_seed"]))
    data = build_indices(spec, bins, profiles)
    full_rows = material_rows(control=False)
    control_rows = material_rows(control=True)

    common_geometry = family_dir / "_shared_geology_indices.h5"
    with h5py.File(common_geometry, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = "scripts/generate_independent_v2_family01.py"
        handle.attrs["scene_family_id"] = contract["scene_family_id"]
        handle.create_dataset(
            "data",
            data=data,
            dtype=np.int16,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(256, spec.nx), min(256, spec.ny), 1),
        )

    manifests = []
    for case in (positive, negative):
        case_dir = family_dir / case["case_id"]
        manifests.append(
            _write_case(
                case_dir,
                case,
                contract,
                spec,
                data,
                profiles,
                shape_metrics,
                field_stats,
                full_rows,
                control_rows,
                common_geometry,
                contract_path,
            )
        )

    positive_dir = family_dir / positive["case_id"]
    negative_dir = family_dir / negative["case_id"]
    if sha256(positive_dir / "geology_indices.h5") != sha256(negative_dir / "geology_indices.h5"):
        raise RuntimeError("matched family geometry files are not byte-identical")
    if sha256(positive_dir / "materials_no_basal.txt") != sha256(negative_dir / "materials_full.txt"):
        raise RuntimeError("positive control and negative full materials are not byte-identical")
    preview_path = render_preview(family_dir, spec, data, profiles, full_rows, control_rows, shape_metrics)
    common_geometry.unlink()

    family_manifest = {
        "contract_id": contract["contract_id"],
        "scene_family_id": contract["scene_family_id"],
        "lifecycle_state": "pre_solver_blocked_pilot",
        "formal_training_allowed": False,
        "split_group_indivisible": True,
        "line9_conditioned": False,
        "case_ids": [case["case_id"] for case in contract["cases"]],
        "positive_case_id": positive["case_id"],
        "true_negative_case_id": negative["case_id"],
        "shared_geometry_sha256": sha256(positive_dir / "geology_indices.h5"),
        "shared_geometry_array_sha256": array_sha256(data),
        "positive_control_materials_sha256": sha256(positive_dir / "materials_no_basal.txt"),
        "negative_full_materials_sha256": sha256(negative_dir / "materials_full.txt"),
        "exact_negative_equivalence": True,
        "profile_shape_gate": shape_metrics,
        "cover_field_statistics": field_stats,
        "preview": preview_path.name,
        "next_gate": "static, geometry-only, one-trace and sparse runtime audits; no training export",
    }
    write_canonical_text(
        family_dir / "family_manifest.json", json.dumps(family_manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    write_checksums(family_dir)
    return family_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = generate_family(args.contract.resolve(), args.output_root.resolve(), overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
