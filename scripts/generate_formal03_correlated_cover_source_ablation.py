#!/usr/bin/env python3
"""Generate the FORMAL03 shared-geology source-ablation family.

FORMAL03 is a blocked development family.  It uses one generic, seeded,
two-dimensional correlated cover field and one strict full/no-basal indexed
geometry.  Three source cases share that geometry exactly: 65 MHz Ricker,
80 MHz Ricker, and an 80 MHz finite-duration zero-mean Gaussian-modulated
transient.  No measured line, label, waveform, or held-out statistic is read.
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

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "00_controls"
FAMILY_ID = "FORMAL03_CORRELATED_COVER_SOURCE_ABLATION"
C0 = 299_792_458.0


@dataclass(frozen=True)
class Spec:
    """Grid-aligned acquisition with a protected 0-500 ns interval."""

    domain_x_m: float = 179.73
    domain_y_m: float = 48.0
    dl_m: float = 0.03
    pml_cells: int = 60
    physical_side_guard_m: float = 76.5
    trace_count: int = 256
    trace_spacing_m: float = 0.09
    scan_start_x_m: float = 78.3
    tx_rx_offset_m: float = 0.18
    ground_y_m: float = 28.5
    source_y_m: float = 36.51
    solver_time_window_s: float = 650e-9
    protected_window_end_ns: float = 500.0
    cover_bins: int = 24
    transition_levels: int = 20
    profile_seed: int = 2026071517
    field_seed: int = 2026071518

    @property
    def nx(self) -> int:
        return int(round(self.domain_x_m / self.dl_m))

    @property
    def ny(self) -> int:
        return int(round(self.domain_y_m / self.dl_m))

    @property
    def pml_m(self) -> float:
        return self.pml_cells * self.dl_m

    @property
    def scan_span_m(self) -> float:
        return (self.trace_count - 1) * self.trace_spacing_m

    @property
    def flight_height_m(self) -> float:
        return self.source_y_m - self.ground_y_m

    @property
    def right_guard_m(self) -> float:
        endpoint = self.scan_start_x_m + self.scan_span_m + self.tx_rx_offset_m
        return self.domain_x_m - self.pml_m - endpoint

    @property
    def boundary_round_trip_ns(self) -> float:
        return 2e9 * min(self.physical_side_guard_m, self.right_guard_m) / C0


@dataclass(frozen=True)
class Material:
    material_id: str
    epsilon_r: float
    conductivity_s_per_m: float


@dataclass(frozen=True)
class SourceVariant:
    case_id: str
    kind: str
    center_frequency_hz: float
    waveform_id: str
    reference_delay_ns: float
    custom_sigma_ns: float | None = None


BEDROCK = Material("bedrock", 9.0, 0.0018)
SOURCE_VARIANTS = (
    SourceVariant(
        "FORMAL03_CORRELATED_COVER_RICKER65",
        "ricker",
        65e6,
        "formal03_ricker65",
        math.sqrt(2.0) / 65e6 * 1e9,
    ),
    SourceVariant(
        "FORMAL03_CORRELATED_COVER_RICKER80",
        "ricker",
        80e6,
        "formal03_ricker80",
        math.sqrt(2.0) / 80e6 * 1e9,
    ),
    SourceVariant(
        "FORMAL03_CORRELATED_COVER_GABOR80",
        "gaussian_modulated_zero_mean",
        80e6,
        "formal03_gabor80",
        50.0,
        11.5,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(np.std(values))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("cannot normalise a constant or non-finite field")
    return (values - float(np.mean(values))) / standard_deviation


def _correlated_1d(
    rng: np.random.Generator,
    count: int,
    correlation_m: float,
    dl_m: float,
) -> np.ndarray:
    sigma = max(correlation_m / dl_m, 1.0)
    return normalise(gaussian_filter1d(rng.standard_normal(count), sigma=sigma, mode="reflect"))


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float]:
    """Solve the crop-shape fit without depending on a LAPACK runtime."""

    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for pivot in range(3):
        best = max(range(pivot, 3), key=lambda index: abs(augmented[index][pivot]))
        augmented[pivot], augmented[best] = augmented[best], augmented[pivot]
        scale = augmented[pivot][pivot]
        if abs(scale) < 1e-20:
            raise ValueError("singular quadratic-fit system")
        augmented[pivot] = [value / scale for value in augmented[pivot]]
        for row in range(3):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[pivot])
            ]
    return tuple(float(augmented[index][3]) for index in range(3))


def _fit_quadratic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centred = x - float(np.mean(x))
    z2 = centred * centred
    z3 = z2 * centred
    z4 = z2 * z2
    matrix = [
        [float(y.size), float(np.sum(centred)), float(np.sum(z2))],
        [float(np.sum(centred)), float(np.sum(z2)), float(np.sum(z3))],
        [float(np.sum(z2)), float(np.sum(z3)), float(np.sum(z4))],
    ]
    vector = [float(np.sum(y)), float(np.sum(centred * y)), float(np.sum(z2 * y))]
    constant, linear, quadratic = _solve_3x3(matrix, vector)
    return constant + linear * centred + quadratic * z2


def crop_statistics(x: np.ndarray, values: np.ndarray, spec: Spec) -> dict[str, float | int]:
    left = spec.scan_start_x_m + 0.5 * spec.tx_rx_offset_m
    right = left + spec.scan_span_m
    mask = (x >= left) & (x <= right)
    crop_x = x[mask]
    crop = values[mask]
    smooth = gaussian_filter1d(crop, sigma=max(0.45 / spec.dl_m, 1.0), mode="reflect")
    derivative = np.gradient(smooth, crop_x)
    sign = np.sign(derivative)
    sign[sign == 0.0] = 1.0
    extrema = int(np.count_nonzero(np.diff(sign)))
    fit = _fit_quadratic(crop_x, crop)
    residual = float(np.sum((crop - fit) ** 2))
    total = float(np.sum((crop - float(np.mean(crop))) ** 2))
    return {
        "min_m": float(np.min(crop)),
        "median_m": float(np.median(crop)),
        "max_m": float(np.max(crop)),
        "range_m": float(np.ptp(crop)),
        "smoothed_extrema_count": extrema,
        "quadratic_fit_r2": 1.0 - residual / total if total > 0.0 else 1.0,
        "abs_slope_p95": float(np.percentile(np.abs(np.gradient(crop, crop_x)), 95.0)),
    }


def build_profiles(spec: Spec) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Build a deterministic non-periodic basal path and independent transition."""

    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    for attempt in range(128):
        rng = np.random.default_rng(spec.profile_seed + attempt)
        basal = 13.9
        basal += 0.55 * _correlated_1d(rng, spec.nx, 17.0, spec.dl_m)
        basal += 0.25 * _correlated_1d(rng, spec.nx, 5.3, spec.dl_m)
        basal += 0.07 * _correlated_1d(rng, spec.nx, 1.5, spec.dl_m)
        basal = np.clip(basal, 12.0, 15.8)

        transition_rng = np.random.default_rng(spec.profile_seed + 1000 + attempt)
        transition = 1.02
        transition += 0.23 * _correlated_1d(transition_rng, spec.nx, 7.5, spec.dl_m)
        transition += 0.08 * _correlated_1d(transition_rng, spec.nx, 2.1, spec.dl_m)
        transition = np.clip(transition, 0.66, 1.50)

        stats = crop_statistics(x, basal, spec)
        if not (0.85 <= float(stats["range_m"]) <= 2.5):
            continue
        if int(stats["smoothed_extrema_count"]) < 2:
            continue
        if float(stats["quadratic_fit_r2"]) >= 0.98:
            continue
        if float(stats["abs_slope_p95"]) >= 0.22:
            continue
        ground = np.full(spec.nx, spec.ground_y_m, dtype=np.float64)
        return (
            {
                "full_x_m": x.astype(np.float32),
                "full_ground_y_m": ground.astype(np.float32),
                "full_basal_depth_m": basal.astype(np.float32),
                "full_transition_thickness_m": transition.astype(np.float32),
                "full_basal_y_m": (ground - basal).astype(np.float32),
                "full_transition_top_y_m": (ground - basal + transition).astype(np.float32),
            },
            {**stats, "accepted_seed": spec.profile_seed + attempt, "attempt": attempt},
        )
    raise RuntimeError("could not generate a FORMAL03 profile that passes shape gates")


def _upsampled_component(
    rng: np.random.Generator,
    shape: tuple[int, int],
    strides: tuple[int, int],
    smooth_sigma: tuple[float, float],
) -> np.ndarray:
    coarse_shape = (
        int(math.ceil(shape[0] / strides[0])) + 3,
        int(math.ceil(shape[1] / strides[1])) + 3,
    )
    coarse = rng.standard_normal(coarse_shape).astype(np.float32)
    coarse = gaussian_filter(coarse, sigma=smooth_sigma, mode="reflect")
    factors = (shape[0] / coarse.shape[0], shape[1] / coarse.shape[1])
    field = zoom(coarse, factors, order=3, mode="reflect", prefilter=True)
    result = np.empty(shape, dtype=np.float32)
    result.fill(0.0)
    sx = min(shape[0], field.shape[0])
    sy = min(shape[1], field.shape[1])
    result[:sx, :sy] = field[:sx, :sy]
    if sx < shape[0]:
        result[sx:, :sy] = result[sx - 1 : sx, :sy]
    if sy < shape[1]:
        result[:, sy:] = result[:, sy - 1 : sy]
    return result


def build_cover_field(spec: Spec) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Create a smooth 2-D field, then quantise it only for material indexing."""

    rng = np.random.default_rng(spec.field_seed)
    shape = (spec.nx, spec.ny)
    field = _upsampled_component(rng, shape, (24, 9), (2.6, 2.0))
    fine = _upsampled_component(rng, shape, (10, 5), (2.2, 1.8))
    field += 0.36 * fine
    del fine
    field = normalise(field).astype(np.float32)
    field = np.clip(field, -2.6, 2.6)
    unit = (field + 2.6) / 5.2
    bins = np.minimum((unit * spec.cover_bins).astype(np.int16), spec.cover_bins - 1)

    horizontal_change = float(np.mean(bins[1:, :] != bins[:-1, :]))
    vertical_change = float(np.mean(bins[:, 1:] != bins[:, :-1]))
    vertical_std = np.std(field, axis=1)
    stats: dict[str, float | int] = {
        "used_bins": int(np.unique(bins).size),
        "horizontal_neighbor_bin_change_rate": horizontal_change,
        "vertical_neighbor_bin_change_rate": vertical_change,
        "columns_with_near_zero_vertical_variation": int(np.count_nonzero(vertical_std < 1e-3)),
        "latent_min": float(np.min(field)),
        "latent_median": float(np.median(field)),
        "latent_max": float(np.max(field)),
    }
    return field, bins, stats


def base_materials(spec: Spec) -> list[Material]:
    fractions = (np.arange(spec.cover_bins, dtype=np.float64) + 0.5) / spec.cover_bins
    epsilon = 11.2 + 2.6 * fractions
    conductivity = 0.0017 + 0.0016 * fractions
    return [
        Material(f"cover_{index:02d}", float(epsilon[index]), float(conductivity[index]))
        for index in range(spec.cover_bins)
    ]


def material_rows(spec: Spec, control: bool) -> list[Material]:
    """Map every target-state index back to its local cover bin in the control."""

    bases = base_materials(spec)
    rows = list(bases)
    for level in range(spec.transition_levels):
        fraction = (level + 0.5) / spec.transition_levels
        for base_index, base in enumerate(bases):
            if control:
                epsilon = base.epsilon_r
                conductivity = base.conductivity_s_per_m
            else:
                epsilon = base.epsilon_r + fraction * (BEDROCK.epsilon_r - base.epsilon_r)
                conductivity = base.conductivity_s_per_m + fraction * (
                    BEDROCK.conductivity_s_per_m - base.conductivity_s_per_m
                )
            rows.append(
                Material(
                    f"transition_{level:02d}_cover_{base_index:02d}",
                    float(epsilon),
                    float(conductivity),
                )
            )
    for base_index, base in enumerate(bases):
        rows.append(
            Material(
                f"bedrock_cover_{base_index:02d}",
                base.epsilon_r if control else BEDROCK.epsilon_r,
                base.conductivity_s_per_m if control else BEDROCK.conductivity_s_per_m,
            )
        )
    return rows


def build_indices(
    spec: Spec,
    profile: dict[str, np.ndarray],
    cover_bins: np.ndarray,
) -> np.ndarray:
    y = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    depth = profile["full_ground_y_m"][:, None] - y[None, :]
    basal = profile["full_basal_depth_m"][:, None]
    transition = profile["full_transition_thickness_m"][:, None]
    transition_top = basal - transition
    data = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    subsurface = depth >= 0.0
    data[subsurface] = cover_bins[subsurface]
    transition_mask = subsurface & (depth >= transition_top) & (depth < basal)
    fraction = np.clip((depth - transition_top) / np.maximum(transition, spec.dl_m), 0.0, 1.0)
    levels = np.minimum((fraction * spec.transition_levels).astype(np.int16), spec.transition_levels - 1)
    data[transition_mask] = (
        spec.cover_bins
        + levels[transition_mask] * spec.cover_bins
        + cover_bins[transition_mask]
    )
    bedrock_mask = subsurface & (depth >= basal)
    data[bedrock_mask] = (
        spec.cover_bins
        + spec.transition_levels * spec.cover_bins
        + cover_bins[bedrock_mask]
    )
    return data[:, :, None]


def write_materials(path: Path, rows: list[Material]) -> None:
    path.write_text(
        "\n".join(
            f"#material: {row.epsilon_r:.8f} {row.conductivity_s_per_m:.10f} 1 0 {row.material_id}"
            for row in rows
        )
        + "\n",
        encoding="ascii",
    )


def custom_waveform(variant: SourceVariant, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    if variant.custom_sigma_ns is None:
        raise ValueError("custom waveform requested for a non-custom source")
    time_ns = np.arange(0.0, spec.solver_time_window_s * 1e9 + 5.05, 0.1, dtype=np.float64)
    tau_s = (time_ns - variant.reference_delay_ns) * 1e-9
    sigma_s = variant.custom_sigma_ns * 1e-9
    envelope = np.exp(-0.5 * (tau_s / sigma_s) ** 2)
    dc_correction = math.exp(-0.5 * (2.0 * math.pi * variant.center_frequency_hz * sigma_s) ** 2)
    values = envelope * (np.cos(2.0 * math.pi * variant.center_frequency_hz * tau_s) - dc_correction)
    values[np.abs(tau_s) > 4.5 * sigma_s] = 0.0
    residual = float(np.trapz(values, time_ns * 1e-9))
    weight = envelope.copy()
    weight[np.abs(tau_s) > 4.5 * sigma_s] = 0.0
    values -= residual * weight / float(np.trapz(weight, time_ns * 1e-9))
    values /= float(np.max(np.abs(values)))
    return time_ns * 1e-9, values


def write_custom_waveform(path: Path, variant: SourceVariant, spec: Spec) -> dict[str, float]:
    time_s, values = custom_waveform(variant, spec)
    with path.open("w", encoding="ascii", newline="") as handle:
        handle.write(f"time {variant.waveform_id}\n")
        for time_value, amplitude in zip(time_s, values):
            handle.write(f"{time_value:.12e} {amplitude:.12e}\n")
    dt = float(time_s[1] - time_s[0])
    spectrum = np.abs(np.fft.rfft(values))
    frequency = np.fft.rfftfreq(values.size, dt)
    peak = float(frequency[int(np.argmax(spectrum[1:])) + 1])
    return {
        "sample_interval_ns": dt * 1e9,
        "discrete_time_integral": float(np.trapz(values, time_s)),
        "peak_frequency_hz": peak,
        "maximum_absolute_amplitude": float(np.max(np.abs(values))),
    }


def source_lines(variant: SourceVariant) -> list[str]:
    if variant.kind == "ricker":
        return [
            f"#waveform: ricker 1 {variant.center_frequency_hz:g} {variant.waveform_id}",
        ]
    return ["#excitation_file: source_waveform.txt"]


def input_text(
    spec: Spec,
    variant: SourceVariant,
    title: str,
    materials: str | None,
    geometry_view: str | None = None,
) -> str:
    lines = [
        f"#title: {title}",
        f"#domain: {spec.domain_x_m:g} {spec.domain_y_m:g} {spec.dl_m:g}",
        f"#dx_dy_dz: {spec.dl_m:g} {spec.dl_m:g} {spec.dl_m:g}",
        f"#time_window: {spec.solver_time_window_s:g}",
        f"#pml_cells: {spec.pml_cells} {spec.pml_cells} 0 {spec.pml_cells} {spec.pml_cells} 0",
        "#messages: y",
        *source_lines(variant),
        f"#hertzian_dipole: z {spec.scan_start_x_m:g} {spec.source_y_m:g} 0 {variant.waveform_id}",
        f"#rx: {spec.scan_start_x_m + spec.tx_rx_offset_m:g} {spec.source_y_m:g} 0 rx1 Ez",
        f"#src_steps: {spec.trace_spacing_m:g} 0 0",
        f"#rx_steps: {spec.trace_spacing_m:g} 0 0",
    ]
    if materials:
        lines.append(f"#geometry_objects_read: 0 0 0 geology_indices.h5 {materials}")
    if geometry_view:
        lines.append(
            f"#geometry_view: 0 0 0 {spec.domain_x_m:g} {spec.domain_y_m:g} {spec.dl_m:g} "
            f"{spec.dl_m:g} {spec.dl_m:g} {spec.dl_m:g} {geometry_view} n"
        )
    return "\n".join(lines) + "\n"


def reference_arrival(
    spec: Spec,
    profile: dict[str, np.ndarray],
    indices: np.ndarray,
    full_rows: list[Material],
) -> dict[str, np.ndarray]:
    source_x = spec.scan_start_x_m + np.arange(spec.trace_count) * spec.trace_spacing_m
    receiver_x = source_x + spec.tx_rx_offset_m
    midpoint = 0.5 * (source_x + receiver_x)
    basal = np.interp(midpoint, profile["full_x_m"], profile["full_basal_depth_m"])
    transition = np.interp(midpoint, profile["full_x_m"], profile["full_transition_thickness_m"])
    epsilon = np.asarray([row.epsilon_r for row in full_rows], dtype=np.float64)
    y = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    travel = np.empty(spec.trace_count, dtype=np.float64)
    for trace, (x_value, basal_depth) in enumerate(zip(midpoint, basal)):
        ix = int(np.clip(round(x_value / spec.dl_m - 0.5), 0, spec.nx - 1))
        depth = spec.ground_y_m - y
        path = (depth >= 0.0) & (depth < basal_depth)
        material_index = indices[ix, path, 0]
        travel[trace] = spec.flight_height_m + float(np.sum(np.sqrt(epsilon[material_index]))) * spec.dl_m
    return {
        "source_x_m": source_x.astype(np.float32),
        "receiver_x_m": receiver_x.astype(np.float32),
        "trace_midpoint_x_m": midpoint.astype(np.float32),
        "flight_height_m": np.full(spec.trace_count, spec.flight_height_m, dtype=np.float32),
        "basal_interface_depth_m": basal.astype(np.float32),
        "transition_thickness_m": transition.astype(np.float32),
        "geometric_reference_arrival_time_ns": (2e9 * travel / C0).astype(np.float32),
    }


def _physical_transition_step(spec: Spec, full_rows: list[Material]) -> tuple[float, float]:
    max_epsilon = 0.0
    max_conductivity = 0.0
    for base in range(spec.cover_bins):
        sequence = [full_rows[base]]
        sequence.extend(
            full_rows[spec.cover_bins + level * spec.cover_bins + base]
            for level in range(spec.transition_levels)
        )
        sequence.append(full_rows[spec.cover_bins * (spec.transition_levels + 1) + base])
        max_epsilon = max(max_epsilon, max(abs(a.epsilon_r - b.epsilon_r) for a, b in zip(sequence, sequence[1:])))
        max_conductivity = max(
            max_conductivity,
            max(abs(a.conductivity_s_per_m - b.conductivity_s_per_m) for a, b in zip(sequence, sequence[1:])),
        )
    return max_epsilon, max_conductivity


def preview_geometry(
    path: Path,
    spec: Spec,
    indices: np.ndarray,
    profile: dict[str, np.ndarray],
    full_rows: list[Material],
    control_rows: list[Material],
    title: str | None = None,
) -> None:
    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    y = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    x_left = spec.scan_start_x_m - 2.0
    x_right = spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m + 2.0
    y_low = spec.ground_y_m - 18.0
    y_high = spec.ground_y_m + 1.0
    x_mask = (x >= x_left) & (x <= x_right)
    y_mask = (y >= y_low) & (y <= y_high)
    sample = indices[x_mask, :, 0][:, y_mask].T
    full_epsilon = np.asarray([row.epsilon_r for row in full_rows])
    control_epsilon = np.asarray([row.epsilon_r for row in control_rows])
    valid = sample >= 0
    safe = np.clip(sample, 0, full_epsilon.size - 1)
    full = np.where(valid, full_epsilon[safe], 1.0)
    contrast = np.where(valid, full_epsilon[safe] - control_epsilon[safe], 0.0)

    def colour(values: np.ndarray, low: float, high: float, diverging: bool = False) -> np.ndarray:
        unit = np.clip((values - low) / (high - low), 0.0, 1.0)
        if diverging:
            return np.stack((255 * unit, 255 * (1.0 - np.abs(unit - 0.5) * 1.6), 255 * (1.0 - unit)), axis=-1).astype(np.uint8)
        return np.stack((255 * unit, 255 * (1.0 - np.abs(unit - 0.5)), 255 * (1.0 - unit)), axis=-1).astype(np.uint8)

    canvas = Image.new("RGB", (1900, 930), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    boxes = ((70, 75, 1830, 410), (70, 525, 1830, 860))
    for box, image in zip(boxes, (colour(full, 1.0, 14.0), colour(contrast, -5.0, 5.0, True))):
        panel = Image.fromarray(np.flipud(image), mode="RGB").resize(
            (box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR
        )
        canvas.paste(panel, (box[0], box[1]))
        draw.rectangle(box, outline="black", width=2)
    profile_mask = (profile["full_x_m"] >= x_left) & (profile["full_x_m"] <= x_right)
    for key, colour_name in (("full_transition_top_y_m", "cyan"), ("full_basal_y_m", "yellow")):
        points = [
            (
                boxes[0][0] + float((px - x_left) / (x_right - x_left)) * (boxes[0][2] - boxes[0][0]),
                boxes[0][3] - float((py - y_low) / (y_high - y_low)) * (boxes[0][3] - boxes[0][1]),
            )
            for px, py in zip(profile["full_x_m"][profile_mask], profile[key][profile_mask])
        ]
        draw.line(points, fill=colour_name, width=3)
    draw.text(
        (70, 20),
        title or f"{FAMILY_ID}: shared generic geology; pre-solver only",
        fill="black",
        font=font,
    )
    draw.text((70, 440), "Full epsilon: 2-D correlated cover; cyan transition top; yellow basal interface", fill="black", font=font)
    draw.text((70, 885), "Full minus strict local-cover control; no measured conditioning and no visible-phase label", fill="black", font=font)
    canvas.save(path)


def preview_sources(
    path: Path,
    spec: Spec,
    variants: tuple[SourceVariant, ...] = SOURCE_VARIANTS,
) -> None:
    width, height = 1800, 900
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colours = ("#2f5597", "#c65911", "#548235")
    left, right, top, bottom = 95, 1725, 75, 805
    draw.rectangle((left, top, right, bottom), outline="black", width=2)
    time_ns = np.linspace(0.0, 120.0, 2401)
    for variant_index, (variant, colour_name) in enumerate(zip(variants, colours)):
        if variant.kind == "ricker":
            time_s = time_ns * 1e-9
            chi = math.sqrt(2.0) / variant.center_frequency_hz
            zeta = math.pi**2 * variant.center_frequency_hz**2
            delay = time_s - chi
            values = -(2.0 * zeta * (2.0 * zeta * delay**2 - 1.0) * np.exp(-zeta * delay**2)) / (2.0 * zeta)
        else:
            full_time, full_values = custom_waveform(variant, spec)
            values = np.interp(time_ns * 1e-9, full_time, full_values)
        points = [
            (
                left + float(t / 120.0) * (right - left),
                (top + bottom) / 2.0 - float(v) * 0.38 * (bottom - top),
            )
            for t, v in zip(time_ns, values)
        ]
        draw.line(points, fill=colour_name, width=3)
        draw.text((left + 20, top + 20 + 28 * variant_index), variant.case_id, fill=colour_name, font=font)
    draw.line((left, (top + bottom) / 2.0, right, (top + bottom) / 2.0), fill="#777777", width=1)
    draw.text((left, 25), "FORMAL03 source waveforms on shared geology (0-120 ns)", fill="black", font=font)
    draw.text((left, 840), "Reference delays are explicit in each manifest and label array; custom transient is finite-duration and zero-mean.", fill="black", font=font)
    canvas.save(path)


def write_checksums(case_dir: Path) -> None:
    rows: list[tuple[str, str, int]] = []
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file() and item.name != "FILE_SHA256.csv"):
        rows.append((path.relative_to(case_dir).as_posix(), sha256(path), path.stat().st_size))
    with (case_dir / "FILE_SHA256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        writer.writerows(rows)


def validate_spec(
    spec: Spec,
    variants: tuple[SourceVariant, ...] = SOURCE_VARIANTS,
) -> None:
    coordinates = (
        spec.domain_x_m,
        spec.domain_y_m,
        spec.scan_start_x_m,
        spec.source_y_m,
        spec.ground_y_m,
        spec.tx_rx_offset_m,
        spec.trace_spacing_m,
        spec.physical_side_guard_m,
    )
    if any(abs(round(value / spec.dl_m) - value / spec.dl_m) > 1e-8 for value in coordinates):
        raise ValueError("all domain and acquisition coordinates must align to the FDTD grid")
    if abs(spec.right_guard_m - spec.physical_side_guard_m) > 1e-8:
        raise ValueError("left and right physical side guards must match")
    if spec.boundary_round_trip_ns < spec.protected_window_end_ns:
        raise ValueError("side-boundary round trip enters the protected interval")
    max_epsilon = max(material.epsilon_r for material in base_materials(spec))
    for variant in variants:
        cells = C0 / (2.8 * variant.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m)
        if cells < 10.0:
            raise ValueError(f"{variant.case_id} fails the 2.8fc ten-cell wavelength gate")


def generate(
    output_root: Path,
    spec: Spec | None = None,
    variants: tuple[SourceVariant, ...] = SOURCE_VARIANTS,
) -> list[Path]:
    spec = spec or Spec()
    validate_spec(spec, variants)
    output_root.mkdir(parents=True, exist_ok=True)
    profile, crop_stats = build_profiles(spec)
    latent, cover_bins, field_stats = build_cover_field(spec)
    indices = build_indices(spec, profile, cover_bins)
    full_rows = material_rows(spec, control=False)
    control_rows = material_rows(spec, control=True)
    max_epsilon_step, max_conductivity_step = _physical_transition_step(spec, full_rows)

    if not variants:
        raise ValueError("at least one source variant is required")
    first_dir = output_root / variants[0].case_id
    first_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = first_dir / "geology_indices.h5"
    with h5py.File(geometry_path, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = "scripts/generate_formal03_correlated_cover_source_ablation.py"
        handle.attrs["family_id"] = FAMILY_ID
        handle.create_dataset("data", data=indices, dtype=np.int16, compression="gzip", compression_opts=4)
    write_materials(first_dir / "materials_full.txt", full_rows)
    write_materials(first_dir / "materials_no_basal.txt", control_rows)
    shared_geometry_hash = sha256(geometry_path)

    arrival = reference_arrival(spec, profile, indices, full_rows)
    case_dirs: list[Path] = []
    for variant in variants:
        case_dir = output_root / variant.case_id
        labels_dir = case_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        if case_dir != first_dir:
            shutil.copy2(geometry_path, case_dir / "geology_indices.h5")
            shutil.copy2(first_dir / "materials_full.txt", case_dir / "materials_full.txt")
            shutil.copy2(first_dir / "materials_no_basal.txt", case_dir / "materials_no_basal.txt")
        waveform_stats: dict[str, float] | None = None
        if variant.kind != "ricker":
            waveform_stats = write_custom_waveform(case_dir / "source_waveform.txt", variant, spec)

        for filename, title, materials, view in (
            ("full_scene.in", "full scene", "materials_full.txt", None),
            ("no_basal_contrast_control.in", "no-basal contrast control", "materials_no_basal.txt", None),
            ("air_reference.in", "air reference", None, None),
            ("geometry_check_full.in", "geometry full", "materials_full.txt", "geometry_check_full"),
            ("geometry_check_control.in", "geometry control", "materials_no_basal.txt", "geometry_check_control"),
        ):
            (case_dir / filename).write_text(
                input_text(spec, variant, f"{variant.case_id} {title}", materials, view),
                encoding="ascii",
            )

        source_reference = arrival["geometric_reference_arrival_time_ns"] + variant.reference_delay_ns
        for name, values in {**profile, **arrival}.items():
            np.save(labels_dir / f"{name}.npy", values)
        np.save(labels_dir / "source_referenced_arrival_time_ns.npy", source_reference.astype(np.float32))

        max_epsilon = max(row.epsilon_r for row in full_rows)
        cells = C0 / (2.8 * variant.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m)
        manifest = {
            "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
            "family_id": FAMILY_ID,
            "case_id": variant.case_id,
            "lifecycle_state": "pre_solver_review",
            "purpose": "shared-geology source ablation for measured-like multi-cycle morphology",
            "generator_path": "scripts/generate_formal03_correlated_cover_source_ablation.py",
            "generator_sha256": sha256(Path(__file__).resolve()),
            "formal_training_allowed": False,
            "promotion_allowed": False,
            "target_presence": True,
            "strict_line9_holdout_allowed": False,
            "line9_conditioned": False,
            "training_block_reason": "requires static, geometry, one-trace, distributed causal-pair, and human morphology gates",
            "parameter_provenance": "seeded generic 2-D correlated priors and project-wide depth bounds; no measured data are read",
            "terrain_stage": "flat ground and fixed height while source and subsurface response are isolated",
            "spec": asdict(spec),
            "source": {
                **asdict(variant),
                "proxy_only": True,
                "not_sfcw": True,
                "reference_delay_contract": "explicit source-waveform peak delay added only to source_referenced_arrival_time_ns",
                "custom_waveform_statistics": waveform_stats,
            },
            "acquisition": {
                "trace_count": spec.trace_count,
                "spacing_m": spec.trace_spacing_m,
                "span_m": spec.scan_span_m,
                "tx_rx_offset_m": spec.tx_rx_offset_m,
                "flight_height_m": spec.flight_height_m,
            },
            "grid": {
                "trace_count": spec.trace_count,
                "trace_spacing_m": spec.trace_spacing_m,
                "dl_m": spec.dl_m,
                "nx_ny_nz": [spec.nx, spec.ny, 1],
                "pml_thickness_m": spec.pml_m,
                "left_physical_guard_m": spec.physical_side_guard_m,
                "right_physical_guard_m": spec.right_guard_m,
                "earliest_lateral_boundary_round_trip_ns": spec.boundary_round_trip_ns,
                "protected_window_end_ns": spec.protected_window_end_ns,
                "solver_window_end_ns": spec.solver_time_window_s * 1e9,
                "cells_per_min_wavelength_at_2_8fc": cells,
            },
            "geometry": {
                "index_file": "geology_indices.h5",
                "shared_index_file": "geology_indices.h5",
                "shared_index_sha256": shared_geometry_hash,
                "index_shape": list(indices.shape),
                "flat_ground": True,
                "fixed_flight_height_m": spec.flight_height_m,
                "discrete_anomaly_bodies": 0,
                "explicit_periodic_basal_components": False,
                "cover_field": "continuous seeded 2-D latent field quantised only for indexed constitutive mapping",
                "cover_field_statistics": field_stats,
                "crop_shape_gate": crop_stats,
            },
            "materials": {
                "cover_bins": spec.cover_bins,
                "transition_levels": spec.transition_levels,
                "bedrock": asdict(BEDROCK),
                "maximum_physical_transition_epsilon_step": max_epsilon_step,
                "maximum_physical_transition_conductivity_step_s_per_m": max_conductivity_step,
                "full_material_count": len(full_rows),
            },
            "strict_pair": {
                "shared_geometry_hdf5": True,
                "control_restores_each_target_voxel_to_its_local_cover_bin": True,
                "full_materials_sha256": sha256(case_dir / "materials_full.txt"),
                "control_materials_sha256": sha256(case_dir / "materials_no_basal.txt"),
            },
            "reference_statistics": {
                "basal_depth_m": {
                    "min": float(np.min(arrival["basal_interface_depth_m"])),
                    "median": float(np.median(arrival["basal_interface_depth_m"])),
                    "max": float(np.max(arrival["basal_interface_depth_m"])),
                },
                "geometric_arrival_time_ns": {
                    "min": float(np.min(arrival["geometric_reference_arrival_time_ns"])),
                    "median": float(np.median(arrival["geometric_reference_arrival_time_ns"])),
                    "max": float(np.max(arrival["geometric_reference_arrival_time_ns"])),
                },
                "source_referenced_arrival_time_ns": {
                    "min": float(np.min(source_reference)),
                    "median": float(np.median(source_reference)),
                    "max": float(np.max(source_reference)),
                },
            },
            "labels": {
                "geometric_reference": "material-interface estimate only",
                "source_referenced_arrival": "geometric estimate plus explicit source peak delay; not a visible-phase label",
                "visible_phase_search_half_width_ns": 55.0,
                "visible_phase_phase_half_width_ns": 8.0,
                "visible_phase": "absent until a successful signed full-minus-control runtime pair is reviewed",
                "training_allowed": False,
            },
            "next_gate": "static and geometry audits, then one-trace full/control smoke",
        }
        (case_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (case_dir / "RUN_COMMANDS.md").write_text(
            f"""# {variant.case_id}

This source-ablation case is blocked from training. Run from the repository
root and stop at the first failed gate.

```powershell
$case = "data/PGDA_SYNTH_DATASET_V2/00_controls/{variant.case_id}"
$cudaArgs = @()
if ($env:PGDA_CUDA_BIN) {{ $cudaArgs = @("--cuda-bin", $env:PGDA_CUDA_BIN) }}

python scripts/run_native_256_release_pilot.py $case --run-id formal03_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal03_smoke1 --trace-count 1 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal03_distributed8_stride36 --trace-count 8 --trace-stride 36 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal03_full24_stride11_morphology --trace-count 24 --trace-stride 11 --full-scene-only @cudaArgs --execute
```

Run the full-scene-only command only after the eight-trace causal pair selects
this source. Do not run 256 traces until the source and geology pass the
24-trace human morphology gate. The supervision-valid interval ends at
{spec.protected_window_end_ns:g} ns; later samples are diagnostics only.
""",
            encoding="utf-8",
        )
        case_dirs.append(case_dir)

    shared_preview = output_root / "FORMAL03_SHARED_GEOMETRY_PREVIEW.png"
    preview_geometry(shared_preview, spec, indices, profile, full_rows, control_rows)
    source_preview = output_root / "FORMAL03_SOURCE_ABLATION_PREVIEW.png"
    preview_sources(source_preview, spec, variants)
    for case_dir in case_dirs:
        shutil.copy2(shared_preview, case_dir / "preview_geometry_and_strict_pair_contrast.png")
        shutil.copy2(source_preview, case_dir / "preview_source_waveforms.png")
        write_checksums(case_dir)

    policy = {
        "family_id": FAMILY_ID,
        "formal_training_allowed": False,
        "line9_conditioned": False,
        "shared_geology_sha256": shared_geometry_hash,
        "case_ids": [variant.case_id for variant in variants],
        "selection_rule": "compare one-trace causal validity, then 24 distributed full/control traces at source-delay-aligned time; reject over-strong, over-uniform, or branch-dominated morphology",
        "rejection_gates": {
            "target_to_adjacent_background_rms_max": 8.0,
            "path_envelope_cv_min": 0.30,
            "path_dynamic_range_retention_min": 0.60,
            "unexplained_periodic_or_crossing_branches_allowed": False,
        },
        "no_full_256_before_pilot_pass": True,
    }
    (output_root / "FORMAL03_CORRELATED_COVER_SOURCE_ABLATION_POLICY.json").write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )
    return case_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for case_dir in generate(args.output_root.resolve()):
        print(case_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
