#!/usr/bin/env python3
"""Generate the pre-solver FORMAL02 graded cover-bedrock baseline.

FORMAL02 replaces the rejected FORMAL01 morphology while preserving strict
full/no-basal causality. It intentionally starts with flat ground, fixed
height, one homogeneous cover, a fine monotonic weathering gradient, and a
non-periodic multiscale basal surface. No measured line or label is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL02_GRADED_BEDROCK"
CASE_ID = f"{FAMILY_ID}_G0_BASELINE"
C0 = 299_792_458.0


@dataclass(frozen=True)
class Spec:
    """Grid-aligned native acquisition with a protected 0-500 ns window."""

    domain_x_m: float = 188.55
    domain_y_m: float = 49.95
    dl_m: float = 0.045
    pml_cells: int = 60
    physical_side_guard_m: float = 80.01
    trace_count: int = 256
    trace_spacing_m: float = 0.09
    scan_start_x_m: float = 82.71
    tx_rx_offset_m: float = 0.18
    ground_y_m: float = 29.97
    source_y_m: float = 37.98
    center_frequency_hz: float = 55e6
    solver_time_window_s: float = 650e-9
    protected_window_end_ns: float = 500.0
    transition_levels: int = 12
    profile_seed: int = 2026071508

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


COVER = Material("cover", 12.5, 0.0024)
BEDROCK = Material("bedrock", 6.5, 0.0012)


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


def correlated_component(
    rng: np.random.Generator,
    count: int,
    correlation_m: float,
    dl_m: float,
) -> np.ndarray:
    sigma_cells = max(correlation_m / dl_m, 1.0)
    return normalise(gaussian_filter1d(rng.standard_normal(count), sigma=sigma_cells, mode="reflect"))


def solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float]:
    """Solve a small dense system without requiring a BLAS/LAPACK runtime."""

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


def quadratic_fit_values(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    centred = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y = np.asarray(values, dtype=np.float64)
    z2 = centred * centred
    z3 = z2 * centred
    z4 = z2 * z2
    matrix = [
        [float(y.size), float(np.sum(centred)), float(np.sum(z2))],
        [float(np.sum(centred)), float(np.sum(z2)), float(np.sum(z3))],
        [float(np.sum(z2)), float(np.sum(z3)), float(np.sum(z4))],
    ]
    vector = [float(np.sum(y)), float(np.sum(centred * y)), float(np.sum(z2 * y))]
    constant, linear, quadratic = solve_3x3(matrix, vector)
    return constant + linear * centred + quadratic * z2


def crop_statistics(x: np.ndarray, values: np.ndarray, spec: Spec) -> dict[str, float | int]:
    left = spec.scan_start_x_m + 0.5 * spec.tx_rx_offset_m
    right = left + spec.scan_span_m
    mask = (x >= left) & (x <= right)
    crop_x = x[mask]
    crop = values[mask]
    if crop.size < 32:
        raise ValueError("acquisition crop is unexpectedly short")
    smooth = gaussian_filter1d(crop, sigma=max(0.45 / spec.dl_m, 1.0), mode="reflect")
    derivative = np.gradient(smooth, crop_x)
    sign = np.sign(derivative)
    sign[sign == 0.0] = 1.0
    extrema = int(np.count_nonzero(np.diff(sign)))
    fit = quadratic_fit_values(crop_x, crop)
    residual = float(np.sum((crop - fit) ** 2))
    total = float(np.sum((crop - float(np.mean(crop))) ** 2))
    quadratic_r2 = 1.0 - residual / total if total > 0.0 else 1.0
    return {
        "min_m": float(np.min(crop)),
        "median_m": float(np.median(crop)),
        "max_m": float(np.max(crop)),
        "range_m": float(np.ptp(crop)),
        "smoothed_extrema_count": extrema,
        "quadratic_fit_r2": float(quadratic_r2),
        "abs_slope_p95": float(np.percentile(np.abs(np.gradient(crop, crop_x)), 95.0)),
    }


def build_profiles(spec: Spec) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Select a deterministic non-periodic crop that passes shape gates."""

    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    for attempt in range(96):
        rng = np.random.default_rng(spec.profile_seed + attempt)
        basal = 13.8
        basal += 0.46 * correlated_component(rng, spec.nx, 19.0, spec.dl_m)
        basal += 0.24 * correlated_component(rng, spec.nx, 6.2, spec.dl_m)
        basal += 0.08 * correlated_component(rng, spec.nx, 1.8, spec.dl_m)
        basal = np.clip(basal, 12.1, 15.7)

        transition = 0.92
        transition += 0.20 * correlated_component(rng, spec.nx, 8.0, spec.dl_m)
        transition += 0.07 * correlated_component(rng, spec.nx, 2.4, spec.dl_m)
        transition = np.clip(transition, 0.50, 1.55)

        stats = crop_statistics(x, basal, spec)
        if not (0.65 <= float(stats["range_m"]) <= 2.4):
            continue
        if int(stats["smoothed_extrema_count"]) < 2:
            continue
        if float(stats["quadratic_fit_r2"]) >= 0.985:
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
    raise RuntimeError("could not generate a non-periodic acquisition crop that passes FORMAL02 gates")


def material_rows(spec: Spec, control: bool) -> list[Material]:
    rows = [COVER]
    for level in range(spec.transition_levels):
        fraction = (level + 0.5) / spec.transition_levels
        epsilon = COVER.epsilon_r + fraction * (BEDROCK.epsilon_r - COVER.epsilon_r)
        conductivity = COVER.conductivity_s_per_m + fraction * (
            BEDROCK.conductivity_s_per_m - COVER.conductivity_s_per_m
        )
        if control:
            epsilon = COVER.epsilon_r
            conductivity = COVER.conductivity_s_per_m
        rows.append(Material(f"transition_{level:02d}", float(epsilon), float(conductivity)))
    rows.append(
        Material("bedrock", COVER.epsilon_r, COVER.conductivity_s_per_m)
        if control
        else BEDROCK
    )
    return rows


def build_indices(spec: Spec, profile: dict[str, np.ndarray]) -> np.ndarray:
    y = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    depth = profile["full_ground_y_m"][:, None] - y[None, :]
    basal = profile["full_basal_depth_m"][:, None]
    transition = profile["full_transition_thickness_m"][:, None]
    transition_top = basal - transition
    data = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    subsurface = depth >= 0.0
    data[subsurface & (depth < transition_top)] = 0
    transition_mask = subsurface & (depth >= transition_top) & (depth < basal)
    fraction = np.clip((depth - transition_top) / np.maximum(transition, spec.dl_m), 0.0, 1.0)
    levels = np.minimum((fraction * spec.transition_levels).astype(np.int16), spec.transition_levels - 1)
    data[transition_mask] = 1 + levels[transition_mask]
    data[subsurface & (depth >= basal)] = 1 + spec.transition_levels
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


def input_text(spec: Spec, title: str, materials: str | None, geometry_view: str | None = None) -> str:
    lines = [
        f"#title: {title}",
        f"#domain: {spec.domain_x_m:g} {spec.domain_y_m:g} {spec.dl_m:g}",
        f"#dx_dy_dz: {spec.dl_m:g} {spec.dl_m:g} {spec.dl_m:g}",
        f"#time_window: {spec.solver_time_window_s:g}",
        f"#pml_cells: {spec.pml_cells} {spec.pml_cells} 0 {spec.pml_cells} {spec.pml_cells} 0",
        "#messages: y",
        f"#waveform: ricker 1 {spec.center_frequency_hz:g} formal02_ricker55",
        f"#hertzian_dipole: z {spec.scan_start_x_m:g} {spec.source_y_m:g} 0 formal02_ricker55",
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
    full_rows: list[Material],
) -> dict[str, np.ndarray]:
    source_x = spec.scan_start_x_m + np.arange(spec.trace_count) * spec.trace_spacing_m
    receiver_x = source_x + spec.tx_rx_offset_m
    midpoint = 0.5 * (source_x + receiver_x)
    basal = np.interp(midpoint, profile["full_x_m"], profile["full_basal_depth_m"])
    transition = np.interp(midpoint, profile["full_x_m"], profile["full_transition_thickness_m"])
    cover = np.maximum(basal - transition, 0.0)
    transition_sqrt_epsilon = float(np.mean([math.sqrt(row.epsilon_r) for row in full_rows[1:-1]]))
    one_way = spec.flight_height_m
    one_way += cover * math.sqrt(COVER.epsilon_r)
    one_way += transition * transition_sqrt_epsilon
    return {
        "source_x_m": source_x.astype(np.float32),
        "receiver_x_m": receiver_x.astype(np.float32),
        "trace_midpoint_x_m": midpoint.astype(np.float32),
        "flight_height_m": np.full(spec.trace_count, spec.flight_height_m, dtype=np.float32),
        "basal_interface_depth_m": basal.astype(np.float32),
        "transition_thickness_m": transition.astype(np.float32),
        "geometric_reference_arrival_time_ns": (2e9 * one_way / C0).astype(np.float32),
    }


def preview(
    case_dir: Path,
    spec: Spec,
    indices: np.ndarray,
    profile: dict[str, np.ndarray],
    full_rows: list[Material],
    control_rows: list[Material],
) -> None:
    stride_x = max(1, spec.nx // 1800)
    stride_y = max(1, spec.ny // 620)
    sample = indices[::stride_x, ::stride_y, 0].T
    full_epsilon = np.asarray([row.epsilon_r for row in full_rows], dtype=np.float64)
    control_epsilon = np.asarray([row.epsilon_r for row in control_rows], dtype=np.float64)
    valid = sample >= 0
    safe = np.clip(sample, 0, len(full_epsilon) - 1)
    full = np.where(valid, full_epsilon[safe], 1.0)
    contrast = np.where(valid, full_epsilon[safe] - control_epsilon[safe], 0.0)

    def colour(values: np.ndarray, low: float, high: float, diverging: bool = False) -> np.ndarray:
        unit = np.clip((values - low) / (high - low), 0.0, 1.0)
        if diverging:
            red = np.rint(255 * unit)
            blue = np.rint(255 * (1.0 - unit))
            green = np.rint(255 * (1.0 - np.abs(unit - 0.5) * 1.5))
            return np.stack((red, green, blue), axis=-1).astype(np.uint8)
        return np.stack(
            (np.rint(255 * unit), np.rint(255 * (1.0 - np.abs(unit - 0.5))), np.rint(255 * (1.0 - unit))),
            axis=-1,
        ).astype(np.uint8)

    width, height = 1900, 1030
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    boxes = ((70, 90, 1830, 455), (70, 590, 1830, 955))
    images = (colour(full, 1.0, 13.0), colour(contrast, -7.0, 7.0, True))
    for box, image in zip(boxes, images):
        panel = Image.fromarray(np.flipud(image), mode="RGB").resize(
            (box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR
        )
        canvas.paste(panel, (box[0], box[1]))
        draw.rectangle(box, outline="black", width=2)
    for key, colour_name in (("full_transition_top_y_m", "cyan"), ("full_basal_y_m", "gold")):
        points = [
            (
                boxes[0][0] + float(x / spec.domain_x_m) * (boxes[0][2] - boxes[0][0]),
                boxes[0][3] - float(y / spec.domain_y_m) * (boxes[0][3] - boxes[0][1]),
            )
            for x, y in zip(profile["full_x_m"][::stride_x], profile[key][::stride_x])
        ]
        draw.line(points, fill=colour_name, width=3)
    scan_left = boxes[0][0] + spec.scan_start_x_m / spec.domain_x_m * (boxes[0][2] - boxes[0][0])
    scan_right = boxes[0][0] + (
        (spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m) / spec.domain_x_m
    ) * (boxes[0][2] - boxes[0][0])
    for box in boxes:
        draw.rectangle((scan_left, box[1], scan_right, box[3]), outline="white", width=3)
    draw.text((70, 25), f"{CASE_ID}: pre-solver graded baseline, not trainable", fill="black", font=font)
    draw.text(
        (70, 480),
        "Full epsilon; cyan=weathered-transition top, gold=basal interface, white=native 256-trace scan",
        fill="black",
        font=font,
    )
    draw.text(
        (70, 980),
        f"Full minus no-basal constitutive contrast; {spec.transition_levels} small monotonic steps replace FORMAL01's three large jumps",
        fill="black",
        font=font,
    )
    canvas.save(case_dir / "preview_geometry_and_strict_pair_contrast.png")

    x_centres = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    y_centres = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    x_left = spec.scan_start_x_m - 1.8
    x_right = spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m + 1.8
    y_low = spec.ground_y_m - 18.0
    y_high = spec.ground_y_m + 2.0
    x_mask = (x_centres >= x_left) & (x_centres <= x_right)
    y_mask = (y_centres >= y_low) & (y_centres <= y_high)
    zoom_indices = indices[x_mask, :, 0][:, y_mask].T
    zoom_valid = zoom_indices >= 0
    zoom_safe = np.clip(zoom_indices, 0, len(full_epsilon) - 1)
    zoom_full = np.where(zoom_valid, full_epsilon[zoom_safe], 1.0)
    zoom_control = np.where(zoom_valid, control_epsilon[zoom_safe], 1.0)
    zoom_contrast = zoom_full - zoom_control
    zoom_canvas = Image.new("RGB", (1900, 930), "white")
    zoom_draw = ImageDraw.Draw(zoom_canvas)
    zoom_boxes = ((70, 80, 1830, 420), (70, 535, 1830, 875))
    for box, image in zip(
        zoom_boxes,
        (colour(zoom_full, 1.0, 13.0), colour(zoom_contrast, -7.0, 7.0, True)),
    ):
        panel = Image.fromarray(np.flipud(image), mode="RGB").resize(
            (box[2] - box[0], box[3] - box[1]), Image.Resampling.NEAREST
        )
        zoom_canvas.paste(panel, (box[0], box[1]))
        zoom_draw.rectangle(box, outline="black", width=2)
    profile_mask = (profile["full_x_m"] >= x_left) & (profile["full_x_m"] <= x_right)
    for key, colour_name in (("full_transition_top_y_m", "cyan"), ("full_basal_y_m", "gold")):
        points = [
            (
                zoom_boxes[0][0]
                + float((x - x_left) / (x_right - x_left)) * (zoom_boxes[0][2] - zoom_boxes[0][0]),
                zoom_boxes[0][3]
                - float((y - y_low) / (y_high - y_low)) * (zoom_boxes[0][3] - zoom_boxes[0][1]),
            )
            for x, y in zip(profile["full_x_m"][profile_mask], profile[key][profile_mask])
        ]
        zoom_draw.line(points, fill=colour_name, width=3)
    zoom_draw.text((70, 22), f"{CASE_ID}: native-window geometry detail", fill="black", font=font)
    zoom_draw.text((70, 445), "Full epsilon inside native scan and 1.8 m context", fill="black", font=font)
    zoom_draw.text(
        (70, 895),
        "Strict constitutive contrast; no visible-phase label exists before runtime",
        fill="black",
        font=font,
    )
    zoom_canvas.save(case_dir / "preview_native_window_geometry.png")


def write_checksums(case_dir: Path) -> None:
    rows: list[tuple[str, str, int]] = []
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file() and item.name != "FILE_SHA256.csv"):
        rows.append((path.relative_to(case_dir).as_posix(), sha256(path), path.stat().st_size))
    with (case_dir / "FILE_SHA256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        writer.writerows(rows)


def validate_spec(spec: Spec) -> None:
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
        raise ValueError("side-boundary round trip enters the protected supervision window")
    max_epsilon = max(COVER.epsilon_r, BEDROCK.epsilon_r)
    cells_per_wavelength = C0 / (2.8 * spec.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m)
    if cells_per_wavelength < 10.0:
        raise ValueError("grid does not resolve the 2.8fc minimum wavelength with ten cells")


def generate(output_root: Path) -> Path:
    spec = Spec()
    validate_spec(spec)
    case_dir = output_root / CASE_ID
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    profile, crop_stats = build_profiles(spec)
    indices = build_indices(spec, profile)
    geometry_path = case_dir / "geology_indices.h5"
    with h5py.File(geometry_path, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = "scripts/generate_formal02_graded_bedrock.py"
        handle.attrs["case_id"] = CASE_ID
        handle.create_dataset("data", data=indices, dtype=np.int16, compression="gzip", compression_opts=4)

    full_rows = material_rows(spec, control=False)
    control_rows = material_rows(spec, control=True)
    write_materials(case_dir / "materials_full.txt", full_rows)
    write_materials(case_dir / "materials_no_basal.txt", control_rows)
    for filename, title, materials, view in (
        ("full_scene.in", "full scene", "materials_full.txt", None),
        ("no_basal_contrast_control.in", "no-basal contrast control", "materials_no_basal.txt", None),
        ("air_reference.in", "air reference", None, None),
        ("geometry_check_full.in", "geometry full", "materials_full.txt", "geometry_check_full"),
        ("geometry_check_control.in", "geometry control", "materials_no_basal.txt", "geometry_check_control"),
    ):
        (case_dir / filename).write_text(input_text(spec, f"{CASE_ID} {title}", materials, view), encoding="ascii")

    arrival = reference_arrival(spec, profile, full_rows)
    for name, values in {**profile, **arrival}.items():
        np.save(labels_dir / f"{name}.npy", values)

    adjacent_epsilon_steps = np.abs(np.diff([row.epsilon_r for row in full_rows]))
    adjacent_conductivity_steps = np.abs(np.diff([row.conductivity_s_per_m for row in full_rows]))
    max_epsilon = max(row.epsilon_r for row in full_rows)
    cells_per_wavelength = C0 / (2.8 * spec.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m)
    manifest = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "family_id": FAMILY_ID,
        "case_id": CASE_ID,
        "lifecycle_state": "pre_solver_review",
        "purpose": "independent graded cover-weathered-bedrock morphology baseline",
        "generator_path": "scripts/generate_formal02_graded_bedrock.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "formal_training_allowed": False,
        "promotion_allowed": False,
        "target_presence": True,
        "strict_line9_holdout_allowed": False,
        "line9_conditioned": False,
        "training_block_reason": "requires source/grid audit, geometry-only build, boundary equivalence, strict runtime pair, visible-phase validation, and human morphology review",
        "parameter_provenance": "seeded generic multiscale priors and project-wide depth bounds; no measured line, measured label, measured waveform, or held-out statistic is read",
        "source_proxy": "55 MHz Ricker controlled baseline; not an instrument-faithful SFCW synthesis",
        "terrain_stage": "flat ground and fixed height while subsurface and source physics are isolated",
        "spec": asdict(spec),
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
            "cells_per_min_wavelength_at_2_8fc": cells_per_wavelength,
        },
        "geometry": {
            "index_file": "geology_indices.h5",
            "shared_index_file": "geology_indices.h5",
            "shared_index_sha256": sha256(geometry_path),
            "index_shape": list(indices.shape),
            "flat_ground": True,
            "fixed_flight_height_m": spec.flight_height_m,
            "discrete_anomaly_bodies": 0,
            "separate_topsoil_interface": False,
            "explicit_periodic_basal_components": False,
            "transition_levels": spec.transition_levels,
            "crop_shape_gate": crop_stats,
        },
        "materials": {
            "cover": asdict(COVER),
            "bedrock": asdict(BEDROCK),
            "maximum_adjacent_epsilon_step": float(np.max(adjacent_epsilon_steps)),
            "maximum_adjacent_conductivity_step_s_per_m": float(np.max(adjacent_conductivity_steps)),
        },
        "strict_pair": {
            "shared_geometry_hdf5": True,
            "changed_material_indices": list(range(1, spec.transition_levels + 2)),
            "only_transition_and_bedrock_changed": True,
            "full_materials_sha256": sha256(case_dir / "materials_full.txt"),
            "control_materials_sha256": sha256(case_dir / "materials_no_basal.txt"),
        },
        "reference_statistics": {
            "basal_depth_m": {
                "min": float(np.min(arrival["basal_interface_depth_m"])),
                "median": float(np.median(arrival["basal_interface_depth_m"])),
                "max": float(np.max(arrival["basal_interface_depth_m"])),
            },
            "transition_thickness_m": {
                "min": float(np.min(arrival["transition_thickness_m"])),
                "median": float(np.median(arrival["transition_thickness_m"])),
                "max": float(np.max(arrival["transition_thickness_m"])),
            },
            "geometric_arrival_time_ns": {
                "min": float(np.min(arrival["geometric_reference_arrival_time_ns"])),
                "median": float(np.median(arrival["geometric_reference_arrival_time_ns"])),
                "max": float(np.max(arrival["geometric_reference_arrival_time_ns"])),
            },
        },
        "labels": {
            "geometric_reference": "material-interface estimate only; not a visible-phase label",
            "visible_phase": "absent until a successful signed full-minus-control runtime pair is reviewed",
            "training_allowed": False,
        },
        "next_gate": "static audit and geometry-only build, followed by one-trace full/control smoke",
    }
    (case_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (case_dir / "RUN_COMMANDS.md").write_text(
        f"""# {CASE_ID}

This pre-solver baseline is not trainable. Run from the repository root and
stop at the first failed gate. The runner stages a disposable solver copy;
never execute gprMax inside this versioned source deck.

```powershell
$case = "data/simulations/v2/00_controls/{CASE_ID}"
$cudaArgs = @()
if ($env:PGDA_CUDA_BIN) {{ $cudaArgs = @("--cuda-bin", $env:PGDA_CUDA_BIN) }}

python scripts/run_native_256_release_pilot.py $case --run-id formal02_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal02_smoke1 --trace-count 1 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal02_distributed32_stride8 --trace-count 32 --trace-stride 8 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal02_full256_pair --trace-count 256 --skip-air-reference @cudaArgs --execute
```

Do not run all {spec.trace_count} traces until the one-trace pair and a
distributed 32-trace pair pass. Run `air_reference.in` only after the source
proxy is accepted. The declared supervision-valid interval ends at
{spec.protected_window_end_ns:g} ns; 500-650 ns is boundary diagnostics only.
""",
        encoding="utf-8",
    )
    preview(case_dir, spec, indices, profile, full_rows, control_rows)
    write_checksums(case_dir)
    return case_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate(args.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
