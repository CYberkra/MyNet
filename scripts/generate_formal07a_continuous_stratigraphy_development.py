#!/usr/bin/env python3
"""Generate the FORMAL07A continuous-stratigraphy development candidate.

FORMAL07A is a one-factor-group successor to FORMAL06C. It keeps the source,
grid, acquisition, materials, and strict full/control mapping fixed. It changes
only geology morphology: the basal path is gentler over the acquisition crop,
and the cover field combines broad heterogeneity with continuous warped
stratigraphy. No isolated inclusion or point target is introduced.

The design choice was informed by comparison with held-out Line9 morphology,
so this case is permanently development-only even though no measured array is
read by this generator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

import generate_formal03_correlated_cover_source_ablation as formal03
import generate_formal06_interface_conditioned_development as formal06
import generate_formal06c_subtle_interface_development as formal06c


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL07A_CONTINUOUS_STRATIGRAPHY_DEVELOPMENT"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN


def default_spec() -> formal03.Spec:
    return formal06c.default_spec()


def _normalise32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("cannot normalise a constant or non-finite field")
    values -= mean
    values /= std
    return values


def build_profiles(
    spec: formal03.Spec,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Build a bounded, non-periodic long-wave basal profile."""

    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    crop_midpoint = spec.scan_start_x_m + 0.5 * (
        spec.scan_span_m + spec.tx_rx_offset_m
    )
    crop_coordinate = (x - crop_midpoint) / spec.scan_span_m
    basal = 13.25
    basal += 0.20 * np.tanh((x - crop_midpoint) / 16.0)
    basal += 0.20 * np.exp(-0.5 * ((x - (crop_midpoint + 3.0)) / 7.5) ** 2)
    basal -= 0.12 * np.exp(-0.5 * ((x - (crop_midpoint - 7.0)) / 9.0) ** 2)
    basal += 0.035 * np.tanh(4.0 * crop_coordinate) * np.exp(
        -0.5 * crop_coordinate**2
    )
    basal = np.clip(basal, 12.4, 14.2)

    transition = 1.02
    transition += 0.055 * np.tanh((x - crop_midpoint) / 11.0)
    transition += 0.035 * np.exp(
        -0.5 * ((x - (crop_midpoint - 4.0)) / 5.0) ** 2
    )
    transition = np.clip(transition, 0.78, 1.28)

    stats = formal03.crop_statistics(x, basal, spec)
    if not 0.35 <= float(stats["range_m"]) <= 0.80:
        raise ValueError("FORMAL07A basal crop range is outside the gentle-relief gate")
    if int(stats["smoothed_extrema_count"]) > 3:
        raise ValueError("FORMAL07A basal crop contains too many long-wave extrema")
    if float(stats["abs_slope_p95"]) >= 0.08:
        raise ValueError("FORMAL07A basal crop is too steep")

    ground = np.full(spec.nx, spec.ground_y_m, dtype=np.float64)
    profile = {
        "full_x_m": x.astype(np.float32),
        "full_ground_y_m": ground.astype(np.float32),
        "full_basal_depth_m": basal.astype(np.float32),
        "full_transition_thickness_m": transition.astype(np.float32),
        "full_basal_y_m": (ground - basal).astype(np.float32),
        "full_transition_top_y_m": (ground - basal + transition).astype(
            np.float32
        ),
    }
    return profile, {
        **stats,
        "profile_model": "bounded_nonperiodic_long_wave",
        "profile_seed": spec.profile_seed,
        "domain_invariant_acquisition_crop": True,
    }


def build_bulk_field(
    spec: formal03.Spec,
    *,
    design: formal06.MaterialDesign = DESIGN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create continuous warped stratigraphy plus weak multiscale geology."""

    rng = np.random.default_rng(spec.field_seed + 31)
    shape = (spec.nx, spec.ny)
    broad = formal03._upsampled_component(
        rng,
        shape,
        (
            max(2, int(round(3.6 / spec.dl_m))),
            max(2, int(round(1.5 / spec.dl_m))),
        ),
        (2.7, 2.4),
    )
    broad = _normalise32(broad)

    meso = formal03._upsampled_component(
        rng,
        shape,
        (
            max(2, int(round(1.2 / spec.dl_m))),
            max(2, int(round(0.54 / spec.dl_m))),
        ),
        (2.2, 2.0),
    )
    meso = _normalise32(meso)

    x_warp_rng = np.random.default_rng(spec.field_seed + 4031)
    warp = 0.36 * formal03._correlated_1d(
        x_warp_rng, spec.nx, 18.0, spec.dl_m
    )
    warp += 0.09 * formal03._correlated_1d(
        x_warp_rng, spec.nx, 5.5, spec.dl_m
    )
    depth = spec.ground_y_m - (
        np.arange(spec.ny, dtype=np.float32) + 0.5
    ) * spec.dl_m
    warped_depth = depth[None, :] + warp[:, None].astype(np.float32)
    stratigraphy = np.sin(2.0 * math.pi * warped_depth / 2.35)
    stratigraphy += 0.42 * np.sin(
        2.0 * math.pi * warped_depth / 0.92 + 0.65
    )
    stratigraphy = gaussian_filter(
        stratigraphy.astype(np.float32), sigma=(2.0, 1.2), mode="reflect"
    )
    stratigraphy = _normalise32(stratigraphy)

    lateral_rng = np.random.default_rng(spec.field_seed + 7031)
    lateral_amplitude = formal03._correlated_1d(
        lateral_rng, spec.nx, 14.0, spec.dl_m
    )
    lateral_amplitude = np.clip(0.82 + 0.16 * lateral_amplitude, 0.55, 1.10)
    stratigraphy *= lateral_amplitude[:, None].astype(np.float32)

    field = 0.52 * broad
    field += 0.68 * stratigraphy
    field += 0.20 * meso
    field = _normalise32(field)
    field = np.clip(field, -2.5, 2.5).astype(np.float32)
    unit = (field + 2.5) / 5.0
    bins = np.minimum(
        (unit * spec.cover_bins).astype(np.int16), spec.cover_bins - 1
    )

    stats = {
        "model": "continuous_warped_multiscale_stratigraphy",
        "used_bins": int(np.unique(bins).size),
        "horizontal_neighbor_bin_change_rate": float(
            np.mean(bins[1:, :] != bins[:-1, :])
        ),
        "vertical_neighbor_bin_change_rate": float(
            np.mean(bins[:, 1:] != bins[:, :-1])
        ),
        "horizontal_latent_step_p95": float(
            np.percentile(np.abs(np.diff(field, axis=0)), 95)
        ),
        "vertical_latent_step_p95": float(
            np.percentile(np.abs(np.diff(field, axis=1)), 95)
        ),
        "latent_min": float(np.min(field)),
        "latent_median": float(np.median(field)),
        "latent_max": float(np.max(field)),
        "component_weights": {
            "broad_two_dimensional": 0.52,
            "continuous_warped_stratigraphy": 0.68,
            "mesoscale_two_dimensional": 0.20,
        },
        "stratigraphic_wavelengths_m": [2.35, 0.92],
        "warp_correlation_scales_m": [18.0, 5.5],
        "isolated_inclusions": 0,
        "point_targets": 0,
    }
    return field, bins, stats


def _epsilon_crop(
    spec: formal03.Spec,
    indices: np.ndarray,
    rows: list[formal03.Material],
) -> np.ndarray:
    x0 = max(0, int(round((spec.scan_start_x_m - 1.5) / spec.dl_m)))
    x1 = min(
        spec.nx,
        int(
            round(
                (spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m + 1.5)
                / spec.dl_m
            )
        ),
    )
    y0 = max(0, int(round((spec.ground_y_m - 16.0) / spec.dl_m)))
    y1 = min(spec.ny, int(round((spec.ground_y_m + 0.6) / spec.dl_m)))
    sample = indices[x0:x1, y0:y1, 0].T
    epsilon = np.asarray([row.epsilon_r for row in rows], dtype=np.float32)
    valid = sample >= 0
    safe = np.clip(sample, 0, epsilon.size - 1)
    return np.where(valid, epsilon[safe], 1.0)


def preview_predecessor_comparison(
    path: Path,
    spec: formal03.Spec,
    successor_indices: np.ndarray,
    successor_profile: dict[str, np.ndarray],
) -> None:
    """Render a geometry-only comparison without importing measured data."""

    predecessor_profile, _ = formal03.build_profiles(spec)
    _, predecessor_bins, _ = formal06.build_bulk_field(spec, design=DESIGN)
    predecessor_indices = formal03.build_indices(
        spec, predecessor_profile, predecessor_bins
    )
    rows = formal06.material_rows(spec, control=False, design=DESIGN)
    predecessor = _epsilon_crop(spec, predecessor_indices, rows)
    successor = _epsilon_crop(spec, successor_indices, rows)

    lo = min(float(np.min(predecessor)), float(np.min(successor)))
    hi = max(float(np.max(predecessor)), float(np.max(successor)))

    def colour(values: np.ndarray) -> np.ndarray:
        unit = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return np.stack(
            (
                32 + 210 * unit,
                48 + 180 * (1.0 - np.abs(unit - 0.52) * 1.75),
                220 - 180 * unit,
            ),
            axis=-1,
        ).astype(np.uint8)

    canvas = Image.new("RGB", (1900, 1020), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panels = (
        (70, 95, 1830, 475, predecessor, "FORMAL06C: broad smooth bulk field"),
        (
            70,
            575,
            1830,
            955,
            successor,
            "FORMAL07A: gentle basal relief + continuous warped stratigraphy",
        ),
    )
    for left, top, right, bottom, values, label in panels:
        image = Image.fromarray(np.flipud(colour(values)), mode="RGB").resize(
            (right - left, bottom - top), Image.Resampling.BILINEAR
        )
        canvas.paste(image, (left, top))
        draw.rectangle((left, top, right, bottom), outline="black", width=2)
        draw.text((left, top - 24), label, fill="black", font=font)
    draw.text(
        (70, 25),
        "Geometry-only acquisition-crop comparison; identical GABOR80, grid, acquisition, and materials",
        fill="black",
        font=font,
    )
    draw.text(
        (70, 985),
        "Pre-solver development model. Colours encode epsilon_r; no measured array or visible-phase label is used.",
        fill="black",
        font=font,
    )
    canvas.save(path)


def preview_model_structure(
    path: Path,
    spec: formal03.Spec,
    indices: np.ndarray,
    profile: dict[str, np.ndarray],
) -> None:
    """Render weak geology and target contrast with audit-appropriate scales."""

    x_left = spec.scan_start_x_m - 1.5
    x_right = (
        spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m + 1.5
    )
    y_low = spec.ground_y_m - 16.0
    y_high = spec.ground_y_m + 0.6
    x0 = max(0, int(round(x_left / spec.dl_m)))
    x1 = min(spec.nx, int(round(x_right / spec.dl_m)))
    y0 = max(0, int(round(y_low / spec.dl_m)))
    y1 = min(spec.ny, int(round(y_high / spec.dl_m)))
    sample = indices[x0:x1, y0:y1, 0].T
    full_rows = formal06.material_rows(spec, control=False, design=DESIGN)
    control_rows = formal06.material_rows(spec, control=True, design=DESIGN)
    full_epsilon = np.asarray([row.epsilon_r for row in full_rows])
    control_epsilon = np.asarray([row.epsilon_r for row in control_rows])
    valid = sample >= 0
    safe = np.clip(sample, 0, full_epsilon.size - 1)
    full = np.where(valid, full_epsilon[safe], np.nan)
    control = np.where(valid, control_epsilon[safe], np.nan)
    contrast = np.where(valid, full - control, 0.0)

    def sequential(values: np.ndarray) -> np.ndarray:
        unit = np.clip((values - 12.0) / 0.8, 0.0, 1.0)
        rgb = np.stack(
            (
                25 + 220 * unit,
                55 + 175 * (1.0 - np.abs(unit - 0.5) * 1.65),
                235 - 195 * unit,
            ),
            axis=-1,
        )
        rgb[~np.isfinite(values)] = (245, 248, 252)
        return rgb.astype(np.uint8)

    contrast_limit = max(float(np.percentile(np.abs(contrast[valid]), 99)), 1e-4)

    def diverging(values: np.ndarray) -> np.ndarray:
        unit = np.clip(0.5 + 0.5 * values / contrast_limit, 0.0, 1.0)
        return np.stack(
            (
                35 + 220 * unit,
                245 - 350 * np.abs(unit - 0.5),
                255 - 220 * unit,
            ),
            axis=-1,
        ).astype(np.uint8)

    canvas = Image.new("RGB", (1900, 1120), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panel_boxes = (
        (70, 105, 1830, 380),
        (70, 465, 1830, 740),
        (70, 825, 1830, 1100),
    )
    images = (sequential(control), sequential(full), diverging(contrast))
    labels = (
        "A. Non-target geology only: epsilon_r 12.0-12.8, enhanced scale",
        "B. Full scene: same geology plus graded weathered cap and weak basal contrast",
        f"C. Full minus no-basal material contrast: dynamic +/-{contrast_limit:.3f} epsilon_r",
    )
    for box, values, label in zip(panel_boxes, images, labels):
        panel = Image.fromarray(np.flipud(values), mode="RGB").resize(
            (box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR
        )
        canvas.paste(panel, (box[0], box[1]))
        draw.rectangle(box, outline="black", width=2)
        draw.text((box[0], box[1] - 24), label, fill="black", font=font)

    profile_mask = (profile["full_x_m"] >= x_left) & (
        profile["full_x_m"] <= x_right
    )
    for key, colour_name in (
        ("full_transition_top_y_m", "cyan"),
        ("full_basal_y_m", "yellow"),
    ):
        points = [
            (
                panel_boxes[1][0]
                + float((px - x_left) / (x_right - x_left))
                * (panel_boxes[1][2] - panel_boxes[1][0]),
                panel_boxes[1][3]
                - float((py - y_low) / (y_high - y_low))
                * (panel_boxes[1][3] - panel_boxes[1][1]),
            )
            for px, py in zip(
                profile["full_x_m"][profile_mask], profile[key][profile_mask]
            )
        ]
        draw.line(points, fill=colour_name, width=3)

    draw.text(
        (70, 25),
        "FORMAL07A model structure: native 256-trace acquisition crop (22.95 m)",
        fill="black",
        font=font,
    )
    draw.text(
        (70, 52),
        "No isolated bodies, no point scatterers, no vertical partitions; flat ground and fixed 8.01 m AGL",
        fill="black",
        font=font,
    )
    canvas.save(path)


def generate(output_root: Path, spec: formal03.Spec | None = None) -> Path:
    spec = spec or default_spec()
    case_dir = formal06.generate_case(
        output_root,
        spec,
        design=DESIGN,
        source=SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL07A_CONTINUOUS_STRATIGRAPHY_POLICY.json",
        run_prefix="formal07a",
        purpose="continuous stratigraphy and gentle basal morphology development",
        predecessor_case_id=formal06c.CASE_ID,
        changed_factors=[
            "basal path morphology",
            "transition-thickness morphology",
            "non-target cover-field spatial organisation",
        ],
        generator_path=Path(__file__),
        preview_title=(
            f"{FAMILY_ID}: gentle relief and continuous multiscale stratigraphy; "
            "pre-solver only"
        ),
        profile_builder=build_profiles,
        bulk_field_builder=build_bulk_field,
        locked_factors=[
            "GABOR80 source waveform",
            "grid and PML",
            "acquisition and fixed flight height",
            "cover/cap/bedrock constitutive values",
            "strict full/no-basal local-cover mapping",
        ],
        geometry_description=(
            "gentle non-periodic basal relief with continuous warped multiscale "
            "stratigraphy; no isolated inclusions or point targets"
        ),
    )
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        indices = handle["data"][:]
    profile = {
        name.removesuffix(".npy"): np.load(path)
        for path in (case_dir / "labels").glob("full_*.npy")
        for name in [path.name]
    }
    preview_predecessor_comparison(
        case_dir / "preview_FORMAL06C_vs_FORMAL07A_geometry.png",
        spec,
        indices,
        profile,
    )
    preview_model_structure(
        case_dir / "preview_model_structure_enhanced.png",
        spec,
        indices,
        profile,
    )

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator_dependencies"] = {
        "generate_formal03_correlated_cover_source_ablation.py": formal03.sha256(
            Path(formal03.__file__).resolve()
        ),
        "generate_formal06_interface_conditioned_development.py": formal03.sha256(
            Path(formal06.__file__).resolve()
        ),
        "generate_formal06c_subtle_interface_development.py": formal03.sha256(
            Path(formal06c.__file__).resolve()
        ),
    }
    manifest["comparison_contract"] = {
        "predecessor": formal06c.CASE_ID,
        "identical_source": True,
        "identical_material_design": True,
        "identical_grid_and_acquisition": True,
        "changed_factor_group": "geology_morphology_only",
        "pre_solver_preview": "preview_FORMAL06C_vs_FORMAL07A_geometry.png",
        "enhanced_structure_preview": "preview_model_structure_enhanced.png",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    formal03.write_checksums(case_dir)
    return case_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate(args.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
