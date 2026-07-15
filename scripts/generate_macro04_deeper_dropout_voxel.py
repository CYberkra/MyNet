#!/usr/bin/env python3
"""Generate MACRO04 with deeper gentle relief and transition-driven dropout."""

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
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
CASE_ID = "MACRO04_DEEPER_GENTLE_DROPOUT_DIAGNOSTIC"
C0 = 299_792_458.0
Q_OFFSETS = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
REGIONS = ("topsoil", "deep_cover", "transition_1", "transition_2", "transition_3", "bedrock")


@dataclass(frozen=True)
class Spec:
    domain_x_m: float = 480.1
    domain_y_m: float = 45.0
    dl_m: float = 0.05
    pml_cells: int = 60
    trace_count: int = 128
    trace_spacing_m: float = 1.7
    scan_start_x_m: float = 120.0
    source_y_m: float = 29.65
    tx_rx_offset_m: float = 0.2
    center_frequency_hz: float = 55e6
    solver_time_window_s: float = 701e-9
    field_seed: int = 2026071301
    profile_seed: int = 2026071302

    @property
    def nx(self) -> int:
        return int(round(self.domain_x_m / self.dl_m))

    @property
    def ny(self) -> int:
        return int(round(self.domain_y_m / self.dl_m))

    @property
    def scan_span_m(self) -> float:
        return (self.trace_count - 1) * self.trace_spacing_m


FULL_PROPERTIES = {
    "topsoil": (13.0, 1.20, 0.0043, 0.0009),
    "deep_cover": (11.8, 0.95, 0.0022, 0.0005),
    "transition_1": (11.4, 0.80, 0.0020, 0.0004),
    "transition_2": (10.7, 0.70, 0.0018, 0.00035),
    "transition_3": (10.0, 0.60, 0.0015, 0.00025),
    "bedrock": (9.1, 0.60, 0.0010, 0.0002),
}
LENSES = (
    {"name": "weak_moist_lens", "x0": 152.0, "x1": 214.0, "depth": 6.8, "thickness": 0.28, "eps": 13.8, "sigma": 0.0038},
    {"name": "dry_ribbon", "x0": 260.0, "x1": 323.0, "depth": 8.7, "thickness": 0.24, "eps": 10.2, "sigma": 0.0017},
    {"name": "short_silt_lens", "x0": 348.0, "x1": 392.0, "depth": 5.8, "thickness": 0.22, "eps": 13.0, "sigma": 0.0032},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    scale = float(np.std(field))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot normalise a constant/non-finite field")
    return (field - float(np.mean(field))) / scale


def correlated_field(spec: Spec) -> tuple[np.ndarray, list[float]]:
    """Create a multi-scale 2D field without cell-scale white noise."""
    factor = 4
    coarse_shape = (math.ceil(spec.nx / factor), math.ceil(spec.ny / factor))
    rng = np.random.default_rng(spec.field_seed)
    components = (
        ((48.0, 6.0), 0.50),
        ((16.0, 3.0), 0.34),
        ((5.0, 1.2), 0.16),
    )
    field = np.zeros(coarse_shape, dtype=np.float32)
    for sigma, weight in components:
        noise = rng.standard_normal(coarse_shape).astype(np.float32)
        component = gaussian_filter(noise, sigma=sigma, mode="reflect")
        field += weight * normalise(component)
    field = normalise(field)
    field = zoom(field, (spec.nx / coarse_shape[0], spec.ny / coarse_shape[1]), order=1, mode="reflect")
    field = normalise(field[: spec.nx, : spec.ny])
    thresholds = np.quantile(field, [0.10, 0.30, 0.70, 0.90]).astype(np.float32)
    quantiles = np.digitize(field, thresholds).astype(np.int16)
    return quantiles, [float(value) for value in thresholds]


def smooth_noise_1d(rng: np.random.Generator, size: int, sigma: float, amplitude: float) -> np.ndarray:
    values = gaussian_filter1d(rng.standard_normal(size), sigma=sigma, mode="reflect")
    values = normalise(values)
    return amplitude * values


def profiles(spec: Spec) -> dict[str, np.ndarray]:
    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    rng = np.random.default_rng(spec.profile_seed)
    ground = (
        21.65
        + 0.15 * np.sin(2 * np.pi * x / 205.0 + 0.42)
        + 0.07 * np.sin(2 * np.pi * x / 87.0 + 1.22)
        + smooth_noise_1d(rng, spec.nx, 260.0, 0.08)
    )
    basal_depth = (
        15.20
        + 0.34 * np.sin(2 * np.pi * x / 188.0 + 0.61)
        + 0.18 * np.sin(2 * np.pi * x / 73.0 + 1.63)
        + 0.09 * np.sin(2 * np.pi * x / 33.0 + 0.24)
        + 0.22 * np.exp(-0.5 * ((x - 283.0) / 29.0) ** 2)
        - 0.14 * np.exp(-0.5 * ((x - 198.0) / 21.0) ** 2)
        + smooth_noise_1d(rng, spec.nx, 175.0, 0.11)
    )
    topsoil_depth = (
        3.30
        + 0.22 * np.sin(2 * np.pi * x / 133.0 + 2.02)
        + smooth_noise_1d(rng, spec.nx, 205.0, 0.08)
    )
    broad_weakening = 1.05 * np.exp(-0.5 * ((x - 191.0) / 25.0) ** 2)
    second_weakening = 0.72 * np.exp(-0.5 * ((x - 317.0) / 20.0) ** 2)
    transition_thickness = (
        1.55
        + 0.16 * np.sin(2 * np.pi * x / 91.0 + 0.74)
        + broad_weakening
        + second_weakening
        + smooth_noise_1d(rng, spec.nx, 235.0, 0.07)
    )
    transition_thickness = np.clip(transition_thickness, 1.2, 3.0)
    basal_depth = np.maximum(basal_depth, topsoil_depth + transition_thickness + 4.5)
    return {
        "x_m": x.astype(np.float32),
        "ground_y_m": ground.astype(np.float32),
        "topsoil_depth_m": topsoil_depth.astype(np.float32),
        "basal_depth_m": basal_depth.astype(np.float32),
        "transition_thickness_m": transition_thickness.astype(np.float32),
        "basal_y_m": (ground - basal_depth).astype(np.float32),
        "transition_top_y_m": (ground - basal_depth + transition_thickness).astype(np.float32),
    }


def material_rows(control: bool) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    deep = FULL_PROPERTIES["deep_cover"]
    for region_index, region in enumerate(REGIONS):
        base = FULL_PROPERTIES[region]
        if control and region.startswith(("transition_", "bedrock")):
            base = deep
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
    for lens_index, lens in enumerate(LENSES):
        rows.append(
            {
                "index": len(REGIONS) * len(Q_OFFSETS) + lens_index,
                "id": str(lens["name"]),
                "epsilon_r": float(lens["eps"]),
                "conductivity_s_per_m": float(lens["sigma"]),
            }
        )
    return rows


def build_indices(spec: Spec, quantiles: np.ndarray, profile: dict[str, np.ndarray]) -> np.ndarray:
    y = (np.arange(spec.ny, dtype=np.float32) + 0.5) * spec.dl_m
    depth = profile["ground_y_m"][:, None] - y[None, :]
    data = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    sub = depth >= 0
    top = depth < profile["topsoil_depth_m"][:, None]
    basal = profile["basal_depth_m"][:, None]
    transition = profile["transition_thickness_m"][:, None]
    transition_top = basal - transition
    transition_fraction = np.clip((depth - transition_top) / np.maximum(transition, spec.dl_m), 0.0, 1.0)

    region = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    region[sub & top] = 0
    region[sub & ~top & (depth < transition_top)] = 1
    transition_mask = sub & (depth >= transition_top) & (depth < basal)
    region[transition_mask & (transition_fraction < 1 / 3)] = 2
    region[transition_mask & (transition_fraction >= 1 / 3) & (transition_fraction < 2 / 3)] = 3
    region[transition_mask & (transition_fraction >= 2 / 3)] = 4
    region[sub & (depth >= basal)] = 5
    data[sub] = region[sub] * len(Q_OFFSETS) + quantiles[sub]

    x = profile["x_m"]
    for lens_index, lens in enumerate(LENSES):
        x0, x1 = float(lens["x0"]), float(lens["x1"])
        phase = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
        taper = np.sin(np.pi * phase) ** 2
        thickness = float(lens["thickness"]) * taper
        centre = float(lens["depth"]) + 0.16 * np.sin(2 * np.pi * phase + 0.4 * lens_index)
        lens_mask = (
            (x[:, None] >= x0)
            & (x[:, None] <= x1)
            & (np.abs(depth - centre[:, None]) <= thickness[:, None] / 2)
            & (depth > profile["topsoil_depth_m"][:, None] + 0.4)
            & (depth < transition_top - 0.5)
        )
        data[lens_mask] = len(REGIONS) * len(Q_OFFSETS) + lens_index
    return data[:, :, None]


def write_materials(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    lines = [
        f"#material: {row['epsilon_r']} {row['conductivity_s_per_m']} 1 0 {row['id']}"
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def base_input(
    spec: Spec,
    title: str,
    material_file: str | None,
    geometry_view: str | None = None,
    output_dir: str | None = None,
) -> str:
    lines = [
        f"#title: {title}",
        f"#domain: {spec.domain_x_m:g} {spec.domain_y_m:g} {spec.dl_m:g}",
        f"#dx_dy_dz: {spec.dl_m:g} {spec.dl_m:g} {spec.dl_m:g}",
        f"#time_window: {spec.solver_time_window_s:g}",
        f"#pml_cells: {spec.pml_cells} {spec.pml_cells} 0 {spec.pml_cells} {spec.pml_cells} 0",
        "#messages: y",
        f"#waveform: ricker 1 {spec.center_frequency_hz:g} macro04_wavelet",
        f"#hertzian_dipole: z {spec.scan_start_x_m:g} {spec.source_y_m:g} 0 macro04_wavelet",
        f"#rx: {spec.scan_start_x_m + spec.tx_rx_offset_m:g} {spec.source_y_m:g} 0 rx1 Ez",
        f"#src_steps: {spec.trace_spacing_m:g} 0 0",
        f"#rx_steps: {spec.trace_spacing_m:g} 0 0",
    ]
    if output_dir:
        lines.insert(6, f"#output_dir: {output_dir}")
    if material_file:
        lines.append(f"#geometry_objects_read: 0 0 0 geology_indices.h5 {material_file}")
    if geometry_view:
        lines.append(
            f"#geometry_view: 0 0 0 {spec.domain_x_m:g} {spec.domain_y_m:g} {spec.dl_m:g} "
            f"{spec.dl_m:g} {spec.dl_m:g} {spec.dl_m:g} {geometry_view} n"
        )
    return "\n".join(lines) + "\n"


def reference_arrival(spec: Spec, profile: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    source_x = spec.scan_start_x_m + np.arange(spec.trace_count) * spec.trace_spacing_m
    receiver_x = source_x + spec.tx_rx_offset_m
    midpoint = (source_x + receiver_x) / 2
    ground = np.interp(midpoint, profile["x_m"], profile["ground_y_m"])
    top = np.interp(midpoint, profile["x_m"], profile["topsoil_depth_m"])
    basal = np.interp(midpoint, profile["x_m"], profile["basal_depth_m"])
    transition = np.interp(midpoint, profile["x_m"], profile["transition_thickness_m"])
    air = np.maximum(spec.source_y_m - ground, 0.0)
    deep = np.maximum(basal - transition - top, 0.0)
    eps = [FULL_PROPERTIES[name][0] for name in ("topsoil", "deep_cover", "transition_1", "transition_2", "transition_3")]
    one_way = air + top * math.sqrt(eps[0]) + deep * math.sqrt(eps[1])
    one_way += (transition / 3) * sum(math.sqrt(value) for value in eps[2:])
    return {
        "source_x_m": source_x.astype(np.float32),
        "receiver_x_m": receiver_x.astype(np.float32),
        "trace_midpoint_x_m": midpoint.astype(np.float32),
        "antenna_y_m": np.full(spec.trace_count, spec.source_y_m, dtype=np.float32),
        "ground_y_m": ground.astype(np.float32),
        "flight_height_m": air.astype(np.float32),
        "basal_interface_depth_m": basal.astype(np.float32),
        "transition_thickness_m": transition.astype(np.float32),
        "geometric_reference_arrival_time_ns": (2e9 * one_way / C0).astype(np.float32),
    }


def _colour_map(values: np.ndarray, anchors: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    positions = values * (len(anchors) - 1)
    lower = np.floor(positions).astype(np.int16)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (positions - lower)[..., None]
    colours = np.asarray(anchors, dtype=np.float32)
    return np.rint(colours[lower] * (1 - weight) + colours[upper] * weight).astype(np.uint8)


def _polyline(draw: ImageDraw.ImageDraw, x: np.ndarray, y: np.ndarray, box: tuple[int, int, int, int], colour: str, width: int = 2) -> None:
    left, top, right, bottom = box
    px = left + x * (right - left)
    py = bottom - y * (bottom - top)
    draw.line(list(zip(px.tolist(), py.tolist())), fill=colour, width=width)


def preview(case_dir: Path, spec: Spec, data: np.ndarray, profile: dict[str, np.ndarray], full_rows: list[dict], control_rows: list[dict]) -> None:
    stride_x, stride_y = 8, 2
    sampled = data[::stride_x, ::stride_y, 0].T
    full_eps = np.asarray([float(row["epsilon_r"]) for row in full_rows])
    control_eps = np.asarray([float(row["epsilon_r"]) for row in control_rows])
    safe = np.clip(sampled, 0, len(full_eps) - 1)
    full_map = np.where(sampled >= 0, full_eps[safe], 1.0)
    delta_map = np.where(sampled >= 0, full_eps[safe] - control_eps[safe], 0.0)

    width, height = 1800, 1220
    margin_x = 90
    plot_right = width - 40
    panel_boxes = ((margin_x, 70, plot_right, 525), (margin_x, 600, plot_right, 930), (margin_x, 1000, plot_right, 1170))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()

    viridis = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))
    coolwarm = ((49, 54, 149), (116, 173, 209), (247, 247, 247), (244, 109, 67), (165, 0, 38))
    full_rgb = _colour_map((full_map - 1.0) / 14.0, viridis)
    delta_rgb = _colour_map((delta_map + 4.0) / 8.0, coolwarm)

    for rgb, box in ((full_rgb, panel_boxes[0]), (delta_rgb, panel_boxes[1])):
        panel = Image.fromarray(np.flipud(rgb), mode="RGB")
        panel = panel.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
        canvas.paste(panel, (box[0], box[1]))
        draw.rectangle(box, outline="black", width=2)

    x_norm = profile["x_m"] / spec.domain_x_m
    y_norm = profile["ground_y_m"] / spec.domain_y_m
    _polyline(draw, x_norm, y_norm, panel_boxes[0], "white", 3)
    _polyline(draw, x_norm, profile["transition_top_y_m"] / spec.domain_y_m, panel_boxes[0], "cyan", 3)
    _polyline(draw, x_norm, profile["basal_y_m"] / spec.domain_y_m, panel_boxes[0], "red", 3)
    _polyline(draw, x_norm, profile["transition_top_y_m"] / spec.domain_y_m, panel_boxes[1], "black", 2)
    _polyline(draw, x_norm, profile["basal_y_m"] / spec.domain_y_m, panel_boxes[1], "black", 2)

    for box in panel_boxes[:2]:
        scan_left = box[0] + spec.scan_start_x_m / spec.domain_x_m * (box[2] - box[0])
        scan_right = box[0] + (spec.scan_start_x_m + spec.scan_span_m) / spec.domain_x_m * (box[2] - box[0])
        draw.rectangle((scan_left, box[1], scan_right, box[3]), outline=(255, 255, 255, 150), width=2)

    profile_box = panel_boxes[2]
    draw.rectangle(profile_box, outline="black", width=2)
    max_depth = 16.5
    for values, colour in (
        (profile["basal_depth_m"], "#c51b7d"),
        (profile["transition_thickness_m"], "#2c7fb8"),
        (profile["topsoil_depth_m"], "#41ab5d"),
    ):
        _polyline(draw, x_norm, 1.0 - np.clip(values / max_depth, 0, 1), profile_box, colour, 3)

    draw.text((margin_x, 30), "MACRO04 deeper gentle-relief correlated voxel geology", fill="black", font=font)
    draw.text((margin_x, 535), "Full-scene epsilon_r; white=ground, cyan=transition top, red=basal; box=scan span", fill="black", font=font)
    draw.text((margin_x, 565), "Constitutive delta: full minus no-basal (same HDF5 indices)", fill="black", font=font)
    draw.text((margin_x, 945), "Profiles: magenta=basal depth, blue=transition thickness, green=topsoil depth", fill="black", font=font)
    draw.text((10, 280), "y (m)", fill="black", font=font)
    draw.text((10, 750), "y (m)", fill="black", font=font)
    draw.text((margin_x, 1185), "x (m), full 480.1 m domain; scan is 215.9 m", fill="black", font=font)
    canvas.save(case_dir / "preview_geometry_and_material_contrast.png")


def exact_gprmax_waveforms(spec: Spec) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reproduce the reviewed local gprMax 3.1.7 waveform formulas."""
    time_s = np.linspace(0.0, 90e-9, 3601, dtype=np.float64)
    frequency = spec.center_frequency_hz

    gaussian_chi = 1.0 / frequency
    gaussian_zeta = 2.0 * np.pi**2 * frequency**2
    gaussian_delay = time_s - gaussian_chi
    gaussian = np.exp(-gaussian_zeta * gaussian_delay**2)
    gaussian_dot = -2.0 * gaussian_zeta * gaussian_delay * gaussian
    gaussian_dot *= np.sqrt(np.e / (2.0 * gaussian_zeta))

    ricker_chi = np.sqrt(2.0) / frequency
    ricker_zeta = np.pi**2 * frequency**2
    ricker_delay = time_s - ricker_chi
    ricker = (1.0 - 2.0 * ricker_zeta * ricker_delay**2) * np.exp(-ricker_zeta * ricker_delay**2)
    return time_s, {
        "Plain Gaussian": gaussian,
        "Gaussian 1st derivative (normalised)": gaussian_dot,
        "Ricker (selected)": ricker,
    }


def preview_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def map_points(
    x: np.ndarray,
    y: np.ndarray,
    box: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    y_down: bool = False,
) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    x_norm = (np.asarray(x, dtype=np.float64) - xlim[0]) / max(xlim[1] - xlim[0], 1e-30)
    y_norm = (np.asarray(y, dtype=np.float64) - ylim[0]) / max(ylim[1] - ylim[0], 1e-30)
    px = left + np.clip(x_norm, 0.0, 1.0) * (right - left)
    if y_down:
        py = top + np.clip(y_norm, 0.0, 1.0) * (bottom - top)
    else:
        py = bottom - np.clip(y_norm, 0.0, 1.0) * (bottom - top)
    return [(int(round(xx)), int(round(yy))) for xx, yy in zip(px, py)]


def draw_chart_frame(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    for fraction in (0.25, 0.5, 0.75):
        x = int(left + fraction * (right - left))
        y = int(top + fraction * (bottom - top))
        draw.line((x, top, x, bottom), fill="#E3E5E7", width=1)
        draw.line((left, y, right, y), fill="#E3E5E7", width=1)
    draw.rectangle(box, outline="#555555", width=2)
    draw.text((left, top - 38), title, fill="#111111", font=label_font)
    draw.text((left, bottom + 8), x_label, fill="#333333", font=small_font)
    draw.text((left + 8, top + 8), y_label, fill="#555555", font=small_font)


def draw_series(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    colour: str,
    width: int,
) -> None:
    points = map_points(x, y, box, xlim, ylim)
    if len(points) >= 2:
        draw.line(points, fill=colour, width=width)


def source_waveform_preview(case_dir: Path, spec: Spec) -> None:
    time_s, waveforms = exact_gprmax_waveforms(spec)
    dt = float(time_s[1] - time_s[0])
    frequency_hz = np.fft.rfftfreq(time_s.size, dt)
    canvas = Image.new("RGB", (1900, 780), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = preview_font(30, bold=True)
    label_font = preview_font(20)
    small_font = preview_font(17)
    draw.text(
        (70, 24),
        "MACRO04 source decision: retain 55 MHz Ricker; plain Gaussian is not selected",
        fill="#111111",
        font=title_font,
    )
    left_box = (90, 120, 900, 660)
    right_box = (1010, 120, 1820, 660)
    draw_chart_frame(draw, left_box, "Exact local gprMax 3.1.7 time-domain formulas", "Time (ns)", "Normalised amplitude", label_font, small_font)
    draw_chart_frame(draw, right_box, "Magnitude spectrum", "Frequency (MHz)", "Normalised magnitude", label_font, small_font)
    colours = ("#C44E52", "#2A9D8F", "#3569A8")
    dc_ratios: dict[str, float] = {}
    for (name, values), colour in zip(waveforms.items(), colours):
        normalised = values / max(float(np.max(np.abs(values))), 1e-30)
        draw_series(draw, left_box, time_s * 1e9, normalised, (0.0, 70.0), (-1.1, 1.1), colour, width=3)
        spectrum = np.abs(np.fft.rfft(normalised))
        spectrum /= max(float(np.max(spectrum)), 1e-30)
        dc_ratios[name] = float(spectrum[0])
        draw_series(draw, right_box, frequency_hz / 1e6, spectrum, (0.0, 220.0), (0.0, 1.05), colour, width=3)
    draw_series(
        draw,
        right_box,
        np.asarray([spec.center_frequency_hz / 1e6] * 2),
        np.asarray([0.0, 1.0]),
        (0.0, 220.0),
        (0.0, 1.05),
        "#777777",
        width=2,
    )
    legend_y = 680
    for index, ((name, _), colour) in enumerate(zip(waveforms.items(), colours)):
        x = 95 + index * 590
        draw.line((x, legend_y + 10, x + 38, legend_y + 10), fill=colour, width=5)
        draw.text(
            (x + 48, legend_y),
            f"{name}; DC/max={dc_ratios[name]:.3g}",
            fill="#222222",
            font=small_font,
        )
    draw.text(
        (90, 735),
        "Plain Gaussian is unipolar and DC-rich. Ricker is the zero-mean normalised second Gaussian derivative used for this controlled comparison.",
        fill="#333333",
        font=small_font,
    )
    canvas.save(case_dir / "preview_source_waveform_choice.png")


def design_review_preview(case_dir: Path, spec: Spec, profile: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> None:
    midpoint = labels["trace_midpoint_x_m"]
    source_relative_x = midpoint - midpoint[0]
    basal = labels["basal_interface_depth_m"]
    transition = labels["transition_thickness_m"]
    arrival = labels["geometric_reference_arrival_time_ns"]
    flight = labels["flight_height_m"]
    weakening_proxy = np.exp(-0.9 * (transition - float(np.min(transition))))

    canvas = Image.new("RGB", (1900, 1250), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = preview_font(31, bold=True)
    label_font = preview_font(20)
    small_font = preview_font(17)
    summary_font = preview_font(18)
    draw.text((70, 24), "MACRO04 pre-solver design review", fill="#111111", font=title_font)
    depth_box = (90, 115, 900, 555)
    arrival_box = (1010, 115, 1820, 555)
    transition_box = (90, 690, 900, 1130)
    summary_box = (1010, 690, 1820, 1130)

    draw_chart_frame(draw, depth_box, "Geometry inside the 215.9 m scan", "Scan distance (m)", "Depth below ground (m)", label_font, small_font)
    depth_min = float(np.min(basal - transition)) - 0.25
    depth_max = float(np.max(basal)) + 0.25
    upper = map_points(source_relative_x, basal - transition, depth_box, (0.0, spec.scan_span_m), (depth_min, depth_max), y_down=True)
    lower = map_points(source_relative_x, basal, depth_box, (0.0, spec.scan_span_m), (depth_min, depth_max), y_down=True)
    draw.polygon(upper + list(reversed(lower)), fill="#CBE3DF")
    draw.line(lower, fill="#A23B72", width=4)
    draw.line(upper, fill="#2A9D8F", width=3)
    draw.text((110, 575), "Magenta: basal surface; green: top of weathered transition", fill="#333333", font=small_font)

    draw_chart_frame(draw, arrival_box, "Independent geometric reference, not a visible-phase label", "Scan distance (m)", "Two-way time (ns)", label_font, small_font)
    arrival_min = float(np.min(arrival)) - 12.0
    arrival_max = float(np.max(arrival)) + 12.0
    band_top = map_points(source_relative_x, arrival - 10.0, arrival_box, (0.0, spec.scan_span_m), (arrival_min, arrival_max))
    band_bottom = map_points(source_relative_x, arrival + 10.0, arrival_box, (0.0, spec.scan_span_m), (arrival_min, arrival_max))
    draw.polygon(band_top + list(reversed(band_bottom)), fill="#D9E5F4")
    draw.line(
        map_points(source_relative_x, arrival, arrival_box, (0.0, spec.scan_span_m), (arrival_min, arrival_max)),
        fill="#3569A8",
        width=4,
    )
    draw.text((1030, 575), "Blue band: +/-10 ns review context only", fill="#333333", font=small_font)

    draw_chart_frame(draw, transition_box, "Smooth transition-driven local weakening", "Scan distance (m)", "Scaled design value", label_font, small_font)
    draw.line(
        map_points(source_relative_x, transition, transition_box, (0.0, spec.scan_span_m), (0.0, 3.2)),
        fill="#2A9D8F",
        width=4,
    )
    draw.line(
        map_points(source_relative_x, weakening_proxy * 3.0, transition_box, (0.0, spec.scan_span_m), (0.0, 3.2)),
        fill="#E76F51",
        width=3,
    )
    draw.text((110, 1150), "Green: transition thickness (m); orange: qualitative weakening proxy x3", fill="#333333", font=small_font)

    draw.rounded_rectangle(summary_box, radius=8, fill="#F5F6F7", outline="#777777", width=2)
    summary = (
        f"Source: 55 MHz Ricker, z-polarised 2D line source\n"
        f"Basal depth: {float(np.min(basal)):.2f}-{float(np.max(basal)):.2f} m "
        f"(range {float(np.ptp(basal)):.2f} m)\n"
        f"Transition: {float(np.min(transition)):.2f}-{float(np.max(transition)):.2f} m\n"
        f"Geometric time: {float(np.min(arrival)):.1f}-{float(np.max(arrival)):.1f} ns "
        f"(range {float(np.ptp(arrival)):.1f} ns)\n"
        f"Flight height: {float(np.min(flight)):.2f}-{float(np.max(flight)):.2f} m\n\n"
        "Design constraints\n"
        "- No Line9 waveform, label curve, geometry, or timing was read.\n"
        "- Geometry is gentler than MACRO03; local weakening comes mainly from transition thickness.\n"
        "- This is a pre-solver design preview, not a simulated B-scan.\n"
        "- formal_training_allowed remains false."
    )
    draw.multiline_text((1045, 725), summary, fill="#222222", font=summary_font, spacing=12)
    draw.text(
        (90, 1205),
        "Preview status: geometry/material package generated; gprMax GPU solver has not been started.",
        fill="#7A1F1F",
        font=label_font,
    )
    canvas.save(case_dir / "preview_pre_solver_design_review.png")


def write_checksums(case_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file() and item.name != "FILE_SHA256.csv"):
        rows.append((path.relative_to(case_dir).as_posix(), sha256(path), path.stat().st_size))
    with (case_dir / "FILE_SHA256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        writer.writerows(rows)


def generate(output_root: Path) -> Path:
    spec = Spec()
    case_dir = output_root / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    quantiles, thresholds = correlated_field(spec)
    profile = profiles(spec)
    data = build_indices(spec, quantiles, profile)
    h5_path = case_dir / "geology_indices.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = "scripts/generate_macro04_deeper_dropout_voxel.py"
        handle.attrs["field_seed"] = spec.field_seed
        handle.attrs["profile_seed"] = spec.profile_seed
        handle.create_dataset("data", data=data, dtype=np.int16)

    full_rows = material_rows(control=False)
    control_rows = material_rows(control=True)
    write_materials(case_dir / "materials_full.txt", full_rows)
    write_materials(case_dir / "materials_no_basal.txt", control_rows)
    (case_dir / "full_scene.in").write_text(base_input(spec, f"{CASE_ID} full", "materials_full.txt"), encoding="ascii")
    (case_dir / "no_basal_contrast_control.in").write_text(
        base_input(spec, f"{CASE_ID} no basal", "materials_no_basal.txt"), encoding="ascii"
    )
    (case_dir / "air_reference.in").write_text(base_input(spec, f"{CASE_ID} air", None), encoding="ascii")
    (case_dir / "geometry_check_full.in").write_text(
        base_input(spec, f"{CASE_ID} geometry full", "materials_full.txt", "geometry_full"), encoding="ascii"
    )
    (case_dir / "geometry_check_control.in").write_text(
        base_input(spec, f"{CASE_ID} geometry control", "materials_no_basal.txt", "geometry_control"), encoding="ascii"
    )
    (case_dir / "smoke_full_scene.in").write_text(
        base_input(spec, f"{CASE_ID} smoke full", "materials_full.txt", output_dir="smoke"), encoding="ascii"
    )
    (case_dir / "smoke_no_basal.in").write_text(
        base_input(spec, f"{CASE_ID} smoke control", "materials_no_basal.txt", output_dir="smoke"), encoding="ascii"
    )

    label_arrays = reference_arrival(spec, profile)
    for name, values in label_arrays.items():
        np.save(labels_dir / f"{name}.npy", values)
    np.save(labels_dir / "reference_arrival_time_ns.npy", label_arrays["geometric_reference_arrival_time_ns"])
    np.save(labels_dir / "geometric_arrival_time_ns.npy", label_arrays["geometric_reference_arrival_time_ns"])
    np.save(labels_dir / "full_x_m.npy", profile["x_m"])
    for name in ("ground_y_m", "topsoil_depth_m", "basal_depth_m", "transition_thickness_m", "basal_y_m", "transition_top_y_m"):
        np.save(labels_dir / f"full_{name}.npy", profile[name])

    pair_same = []
    pair_changed = []
    for full, control in zip(full_rows, control_rows):
        fields = ("epsilon_r", "conductivity_s_per_m")
        changed = any(full[field] != control[field] for field in fields)
        (pair_changed if changed else pair_same).append(int(full["index"]))
    if pair_changed != list(range(2 * len(Q_OFFSETS), 6 * len(Q_OFFSETS))):
        raise RuntimeError(f"unexpected pair-change indices: {pair_changed}")

    source_root = ROOT.parent / "gprMax-master"
    source_hashes = {}
    for relative in (
        "gprMax/input_cmds_geometry.py",
        "gprMax/fractals.py",
        "gprMax/materials.py",
        "gprMax/pml.py",
        "gprMax/model_build_run.py",
    ):
        path = source_root / relative
        source_hashes[relative] = sha256(path) if path.is_file() else None

    max_eps = max(float(row["epsilon_r"]) for row in full_rows)
    cells_per_lambda = C0 / (2 * spec.center_frequency_hz * math.sqrt(max_eps) * spec.dl_m)
    manifest = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "case_id": CASE_ID,
        "family": "macro_deeper_gentle_dropout_diagnostic_only",
        "purpose": "Independent deeper long-line diagnostic with gentle relief, variable transition thickness, and strict shared-geometry control",
        "generator_path": "scripts/generate_macro04_deeper_dropout_voxel.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "material_revision": "v1_deeper_balanced_attenuation_preview",
        "target_presence": True,
        "formal_training_allowed": False,
        "training_block_reason": "preview-only candidate; requires user approval, solver pair, segment-level visible-phase review, and formal promotion",
        "line9_conditioned": False,
        "reference_line": None,
        "design_prior": {
            "depth_origin": "independent project engineering envelope; no Line9 waveform, label curve, or timing distribution was read",
            "target_depth_intent_m": [14.3, 16.2],
            "change_from_macro03": "deeper mean target with about half the broad geometric amplitude; transition thickness drives smooth local weakening"
        },
        "gprmax": {
            "reviewed_version": "3.1.7 Big Smoke",
            "source_hashes": source_hashes,
            "official_manual_reviewed_utc": "2026-07-12",
            "peplinski_used": False,
            "peplinski_guard_reason": "documented validity is 0.3-1.3 GHz, outside this 55 MHz case",
            "external_geometry_dielectric_smoothing": False,
        },
        "spec": asdict(spec),
        "source": {
            "waveform": "ricker",
            "center_frequency_hz": spec.center_frequency_hz,
            "plain_gaussian_selected": False,
            "rationale": "retain the audited zero-mean normalised second-Gaussian-derivative source while changing geology only",
            "future_preferred_upgrade": "measured system pulse via #excitation_file after acquisition-chain calibration"
        },
        "grid": {
            "dimension": "2D_x_y_with_one_z_cell",
            "nx_ny_nz": [spec.nx, spec.ny, 1],
            "dl_m": spec.dl_m,
            "pml_cells": [spec.pml_cells, spec.pml_cells, 0, spec.pml_cells, spec.pml_cells, 0],
            "trace_count": spec.trace_count,
            "trace_spacing_m": spec.trace_spacing_m,
            "solver_time_window_ns": spec.solver_time_window_s * 1e9,
            "gprmax_time_window_ns": spec.solver_time_window_s * 1e9,
            "canonical_time_window_ns": 700.0,
            "canonical_output_samples": 501,
            "canonical_output_dt_ns": 1.4,
            "cells_per_min_wavelength_at_2fc": cells_per_lambda,
            "trace_midpoint_span_m": spec.scan_span_m,
            "left_scan_margin_m": spec.scan_start_x_m,
            "right_scan_margin_m": spec.domain_x_m - (spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m),
        },
        "geometry": {
            "index_file": "geology_indices.h5",
            "index_file_sha256": sha256(h5_path),
            "index_shape": list(data.shape),
            "index_min_max": [int(data.min()), int(data.max())],
            "quantile_thresholds": thresholds,
            "field_seed": spec.field_seed,
            "profile_seed": spec.profile_seed,
            "correlation_design": "three-scale Gaussian-correlated field, generated at 0.2 m and linearly resampled to 0.05 m",
            "interface_origin": "independent engineering-depth envelope plus seeded analytic smooth noise; no Line9 labels, waveform, geometry, or timing used",
            "scan_design_stats": {
                "basal_depth_min_m": float(np.min(label_arrays["basal_interface_depth_m"])),
                "basal_depth_max_m": float(np.max(label_arrays["basal_interface_depth_m"])),
                "basal_depth_range_m": float(np.ptp(label_arrays["basal_interface_depth_m"])),
                "transition_thickness_min_m": float(np.min(label_arrays["transition_thickness_m"])),
                "transition_thickness_max_m": float(np.max(label_arrays["transition_thickness_m"])),
                "geometric_arrival_min_ns": float(np.min(label_arrays["geometric_reference_arrival_time_ns"])),
                "geometric_arrival_max_ns": float(np.max(label_arrays["geometric_reference_arrival_time_ns"])),
                "geometric_arrival_range_ns": float(np.ptp(label_arrays["geometric_reference_arrival_time_ns"])),
            },
            "finite_lenses": LENSES,
        },
        "strict_pair": {
            "shared_geometry_hdf5": True,
            "shared_geometry_sha256": sha256(h5_path),
            "unchanged_material_indices": pair_same,
            "changed_material_indices": pair_changed,
            "only_transition_and_bedrock_changed": True,
            "full_materials_sha256": sha256(case_dir / "materials_full.txt"),
            "control_materials_sha256": sha256(case_dir / "materials_no_basal.txt"),
        },
        "labels": {
            "geometric_reference": "columnar layered reference only; not a training label",
            "visible_phase": "pending signed full-minus-control response",
            "visible_phase_search_half_width_ns": 50.0,
            "visible_phase_phase_half_width_ns": 10.0,
            "training_allowed": False,
        },
    }
    (case_dir / "scene_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run_text = f"""# MACRO04 Commands

Run from this case directory with `PYTHONPATH` pointing to the reviewed gprMax source.

```powershell
python -m gprMax geometry_check_full.in --geometry-only
python -m gprMax geometry_check_control.in --geometry-only
python -m gprMax smoke_full_scene.in -n 1 --geometry-fixed -gpu 0
python -m gprMax smoke_no_basal.in -n 1 --geometry-fixed -gpu 0
```

Full GPU commands are intentionally omitted from this preview package. After approval, use a guarded runner that synchronously captures all per-trace metadata before merge. `air_reference` remains deferred until the pair passes.
"""
    (case_dir / "RUN_COMMANDS.md").write_text(run_text, encoding="utf-8")
    preview(case_dir, spec, data, profile, full_rows, control_rows)
    design_review_preview(case_dir, spec, profile, label_arrays)
    source_waveform_preview(case_dir, spec)
    write_checksums(case_dir)
    return case_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    case_dir = generate(args.output_root.resolve())
    print(case_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
