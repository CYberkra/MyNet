#!/usr/bin/env python3
"""Build independent dense-window cover-weathered-bedrock FDTD cases.

FORMAL01 is a pre-release physical family. It intentionally does not read any
measured trace, Line9 label, Line9 statistic, or MACRO07 material field. The
cases remain non-trainable until their runtime, pair, visible-phase, and human
review gates have passed.
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
from scipy.ndimage import gaussian_filter, gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL01_BEDROCK_DENSE_WINDOW"
LIFECYCLE_STATE = "archived_causal_regression"
C0 = 299_792_458.0
REGIONS = ("topsoil", "cover", "transition_1", "transition_2", "transition_3", "bedrock")


@dataclass(frozen=True)
class Spec:
    """A local 2-D window with dense physical acquisition sampling."""

    domain_x_m: float = 40.2
    domain_y_m: float = 50.0
    dl_m: float = 0.025
    pml_cells: int = 50
    trace_count: int = 256
    trace_spacing_m: float = 0.10
    scan_start_x_m: float = 7.25
    source_y_m: float = 38.0
    ground_y_m: float = 30.0
    tx_rx_offset_m: float = 0.20
    center_frequency_hz: float = 100e6
    solver_time_window_s: float = 650e-9
    profile_seed: int = 2026071501
    cover_seed: int = 2026071502

    @property
    def nx(self) -> int:
        return int(round(self.domain_x_m / self.dl_m))

    @property
    def ny(self) -> int:
        return int(round(self.domain_y_m / self.dl_m))

    @property
    def scan_span_m(self) -> float:
        return (self.trace_count - 1) * self.trace_spacing_m


@dataclass(frozen=True)
class Variant:
    tag: str
    description: str
    cover_heterogeneity: bool
    transition_variation: bool

    @property
    def case_id(self) -> str:
        return f"{FAMILY_ID}_{self.tag}"


VARIANTS = (
    Variant("F0_BASELINE", "homogeneous cover and finite smooth transition", False, False),
    Variant("F1_CORRELATED_COVER", "correlated cover heterogeneity only", True, False),
    Variant("F2_TRANSITION_VARIATION", "smooth weathered-transition variation only", False, True),
    Variant("F3_COMBINED", "correlated cover plus smooth transition variation", True, True),
)

# Broad effective ranges for a clayey/weathered cover over a competent rock.
# They are independent modelling priors, not fitted values from any held-out line.
BASE_PROPERTIES = {
    "topsoil": (14.0, 1.20, 0.00280, 0.00060),
    "cover": (12.5, 0.90, 0.00160, 0.00040),
    "transition_1": (10.8, 0.45, 0.00140, 0.00030),
    "transition_2": (9.0, 0.40, 0.00115, 0.00025),
    "transition_3": (7.5, 0.35, 0.00095, 0.00020),
    "bedrock": (6.0, 0.30, 0.00075, 0.00015),
}


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
        raise ValueError("cannot normalise a constant or non-finite array")
    return (values - float(np.mean(values))) / standard_deviation


def smooth_noise(rng: np.random.Generator, size: int, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * normalise(gaussian_filter1d(rng.standard_normal(size), sigma=sigma, mode="reflect"))


def profiles(spec: Spec, transition_variation: bool) -> dict[str, np.ndarray]:
    """Generate one independent continuous basal centre and transition field."""
    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    rng = np.random.default_rng(spec.profile_seed)
    ground = np.full(spec.nx, spec.ground_y_m, dtype=np.float64)
    topsoil = 2.40 + 0.10 * np.sin(2.0 * np.pi * x / 11.0 + 0.2)
    topsoil += smooth_noise(rng, spec.nx, 28.0, 0.05)
    topsoil = np.clip(topsoil, 2.20, 2.60)

    basal = 13.8 + 0.48 * np.sin(2.0 * np.pi * x / 18.0 + 0.7)
    basal += 0.22 * np.sin(2.0 * np.pi * x / 5.8 + 1.4)
    basal += smooth_noise(rng, spec.nx, 46.0, 0.16)
    basal += smooth_noise(rng, spec.nx, 12.0, 0.05)
    basal = np.clip(basal, 12.2, 15.6)

    transition = np.full(spec.nx, 0.90, dtype=np.float64)
    if transition_variation:
        transition += 0.28 * np.sin(2.0 * np.pi * x / 7.5 + 0.5)
        transition += 0.46 * np.exp(-0.5 * ((x - 10.4) / 1.7) ** 2)
        transition += smooth_noise(rng, spec.nx, 20.0, 0.10)
    transition = np.clip(transition, 0.55, 1.75)
    basal = np.maximum(basal, topsoil + transition + 6.0)
    return {
        "full_x_m": x.astype(np.float32),
        "full_ground_y_m": ground.astype(np.float32),
        "full_topsoil_depth_m": topsoil.astype(np.float32),
        "full_basal_depth_m": basal.astype(np.float32),
        "full_transition_thickness_m": transition.astype(np.float32),
        "full_basal_y_m": (ground - basal).astype(np.float32),
        "full_transition_top_y_m": (ground - basal + transition).astype(np.float32),
    }


def cover_quantiles(spec: Spec, enabled: bool, levels: int) -> tuple[np.ndarray, list[float]]:
    """Return a correlated cover-only material field on the FDTD grid."""
    midpoint = (levels - 1) // 2
    if not enabled:
        return np.full((spec.nx, spec.ny), midpoint, dtype=np.int16), []
    rng = np.random.default_rng(spec.cover_seed)
    coarse = rng.standard_normal((math.ceil(spec.nx / 4), math.ceil(spec.ny / 4)))
    regional = gaussian_filter(coarse, sigma=(16.0, 6.0), mode="reflect")
    local = gaussian_filter(coarse, sigma=(4.0, 2.0), mode="reflect")
    field = normalise(0.72 * normalise(regional) + 0.28 * normalise(local))
    field = np.repeat(np.repeat(field, 4, axis=0), 4, axis=1)[: spec.nx, : spec.ny]
    field = normalise(field)
    thresholds = np.quantile(field, np.linspace(1 / levels, (levels - 1) / levels, levels - 1))
    return np.digitize(field, thresholds).astype(np.int16), [float(value) for value in thresholds]


def material_rows(levels: int, control: bool) -> list[dict[str, float | int | str]]:
    offsets = np.linspace(-1.0, 1.0, levels)
    rows: list[dict[str, float | int | str]] = []
    for region_index, region in enumerate(REGIONS):
        source_region = "cover" if control and region.startswith(("transition", "bedrock")) else region
        epsilon, epsilon_scale, conductivity, conductivity_scale = BASE_PROPERTIES[source_region]
        for quantile, offset in enumerate(offsets):
            rows.append(
                {
                    "index": region_index * levels + quantile,
                    "id": f"{region}_q{quantile}",
                    "epsilon_r": round(float(epsilon + epsilon_scale * offset), 6),
                    "conductivity_s_per_m": round(float(max(0.0, conductivity + conductivity_scale * offset)), 8),
                }
            )
    return rows


def build_indices(spec: Spec, profile: dict[str, np.ndarray], cover_field: np.ndarray, levels: int) -> np.ndarray:
    y = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    depth = profile["full_ground_y_m"][:, None] - y[None, :]
    topsoil = profile["full_topsoil_depth_m"][:, None]
    basal = profile["full_basal_depth_m"][:, None]
    transition = profile["full_transition_thickness_m"][:, None]
    transition_top = basal - transition
    fraction = np.clip((depth - transition_top) / np.maximum(transition, spec.dl_m), 0.0, 1.0)
    sub = depth >= 0.0
    region = np.full((spec.nx, spec.ny), -1, dtype=np.int16)
    region[sub & (depth < topsoil)] = 0
    region[sub & (depth >= topsoil) & (depth < transition_top)] = 1
    transition_mask = sub & (depth >= transition_top) & (depth < basal)
    region[transition_mask & (fraction < 1 / 3)] = 2
    region[transition_mask & (fraction >= 1 / 3) & (fraction < 2 / 3)] = 3
    region[transition_mask & (fraction >= 2 / 3)] = 4
    region[sub & (depth >= basal)] = 5

    midpoint = (levels - 1) // 2
    quantile = np.full_like(region, midpoint)
    cover_regions = (region == 0) | (region == 1)
    quantile[cover_regions] = cover_field[cover_regions]
    data = np.full_like(region, -1)
    data[sub] = region[sub] * levels + quantile[sub]
    return data[:, :, None]


def write_materials(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.write_text(
        "\n".join(
            f"#material: {row['epsilon_r']} {row['conductivity_s_per_m']} 1 0 {row['id']}"
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
        f"#waveform: ricker 1 {spec.center_frequency_hz:g} formal01_ricker100",
        f"#hertzian_dipole: z {spec.scan_start_x_m:g} {spec.source_y_m:g} 0 formal01_ricker100",
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


def reference_arrival(spec: Spec, profile: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    source_x = spec.scan_start_x_m + np.arange(spec.trace_count) * spec.trace_spacing_m
    receiver_x = source_x + spec.tx_rx_offset_m
    midpoint = (source_x + receiver_x) / 2.0
    topsoil = np.interp(midpoint, profile["full_x_m"], profile["full_topsoil_depth_m"])
    basal = np.interp(midpoint, profile["full_x_m"], profile["full_basal_depth_m"])
    transition = np.interp(midpoint, profile["full_x_m"], profile["full_transition_thickness_m"])
    cover = np.maximum(basal - topsoil - transition, 0.0)
    mean_eps = {name: BASE_PROPERTIES[name][0] for name in REGIONS}
    one_way = spec.source_y_m - spec.ground_y_m
    one_way += topsoil * math.sqrt(mean_eps["topsoil"])
    one_way += cover * math.sqrt(mean_eps["cover"])
    one_way += transition / 3.0 * sum(math.sqrt(mean_eps[f"transition_{index}"]) for index in (1, 2, 3))
    return {
        "source_x_m": source_x.astype(np.float32),
        "receiver_x_m": receiver_x.astype(np.float32),
        "trace_midpoint_x_m": midpoint.astype(np.float32),
        "flight_height_m": np.full(spec.trace_count, spec.source_y_m - spec.ground_y_m, dtype=np.float32),
        "basal_interface_depth_m": basal.astype(np.float32),
        "transition_thickness_m": transition.astype(np.float32),
        "geometric_reference_arrival_time_ns": (2e9 * one_way / C0).astype(np.float32),
    }


def preview(case_dir: Path, spec: Spec, indices: np.ndarray, profile: dict[str, np.ndarray], full_rows: list[dict[str, float | int | str]], control_rows: list[dict[str, float | int | str]], variant: Variant) -> None:
    sample = indices[::3, ::2, 0].T
    full_epsilon = np.asarray([float(row["epsilon_r"]) for row in full_rows])
    control_epsilon = np.asarray([float(row["epsilon_r"]) for row in control_rows])
    valid = sample >= 0
    safe = np.clip(sample, 0, len(full_epsilon) - 1)
    full = np.where(valid, full_epsilon[safe], 1.0)
    contrast = np.where(valid, full_epsilon[safe] - control_epsilon[safe], 0.0)

    def colour(values: np.ndarray, low: float, high: float, diverging: bool = False) -> np.ndarray:
        unit = np.clip((values - low) / (high - low), 0.0, 1.0)
        if diverging:
            red = np.rint(255 * unit)
            blue = np.rint(255 * (1.0 - unit))
            green = np.rint(255 * (1.0 - np.abs(unit - 0.5) * 1.4))
            return np.stack((red, green, blue), axis=-1).astype(np.uint8)
        return np.stack((np.rint(255 * unit), np.rint(255 * (1.0 - np.abs(unit - 0.5))), np.rint(255 * (1.0 - unit))), axis=-1).astype(np.uint8)

    width, height = 1840, 980
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    boxes = ((70, 90, 1770, 445), (70, 560, 1770, 915))
    for box, image in zip(boxes, (colour(full, 1.0, 16.0), colour(contrast, -8.0, 8.0, True))):
        panel = Image.fromarray(np.flipud(image), mode="RGB").resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
        canvas.paste(panel, (box[0], box[1]))
        draw.rectangle(box, outline="black", width=2)
    for key, colour_name in (("full_transition_top_y_m", "cyan"), ("full_basal_y_m", "gold")):
        points = [
            (boxes[0][0] + float(x / spec.domain_x_m) * (boxes[0][2] - boxes[0][0]), boxes[0][3] - float(y / spec.domain_y_m) * (boxes[0][3] - boxes[0][1]))
            for x, y in zip(profile["full_x_m"][::3], profile[key][::3])
        ]
        draw.line(points, fill=colour_name, width=3)
    scan_left = boxes[0][0] + spec.scan_start_x_m / spec.domain_x_m * (boxes[0][2] - boxes[0][0])
    scan_right = boxes[0][0] + (spec.scan_start_x_m + spec.scan_span_m) / spec.domain_x_m * (boxes[0][2] - boxes[0][0])
    for box in boxes:
        draw.rectangle((scan_left, box[1], scan_right, box[3]), outline="white", width=2)
    draw.text((70, 25), f"{variant.case_id}: independent dense-window geometry (not yet trainable)", fill="black", font=font)
    draw.text((70, 465), "Effective relative permittivity; cyan=transition top, gold=basal centre, white=dense scan", fill="black", font=font)
    draw.text((70, 935), "Full minus no-basal constitutive contrast; cover and acquisition are identical", fill="black", font=font)
    canvas.save(case_dir / "preview_geometry_and_strict_pair_contrast.png")


def write_checksums(case_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file() and item.name != "FILE_SHA256.csv"):
        rows.append((path.relative_to(case_dir).as_posix(), sha256(path), path.stat().st_size))
    with (case_dir / "FILE_SHA256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        writer.writerows(rows)


def generate_variant(output_root: Path, spec: Spec, variant: Variant) -> Path:
    case_dir = output_root / variant.case_id
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    levels = 7
    profile = profiles(spec, variant.transition_variation)
    field, thresholds = cover_quantiles(spec, variant.cover_heterogeneity, levels)
    indices = build_indices(spec, profile, field, levels)
    geometry_path = case_dir / "geology_indices.h5"
    with h5py.File(geometry_path, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = "scripts/generate_formal01_bedrock_dense_window.py"
        handle.attrs["variant"] = variant.tag
        handle.create_dataset("data", data=indices, dtype=np.int16, compression="gzip", compression_opts=4)

    full_rows = material_rows(levels, control=False)
    control_rows = material_rows(levels, control=True)
    write_materials(case_dir / "materials_full.txt", full_rows)
    write_materials(case_dir / "materials_no_basal.txt", control_rows)
    for filename, title, materials, view in (
        ("full_scene.in", "full scene", "materials_full.txt", None),
        ("no_basal_contrast_control.in", "no-basal contrast control", "materials_no_basal.txt", None),
        ("air_reference.in", "air reference", None, None),
        ("geometry_check_full.in", "geometry full", "materials_full.txt", "geometry_check_full"),
        ("geometry_check_control.in", "geometry control", "materials_no_basal.txt", "geometry_check_control"),
    ):
        (case_dir / filename).write_text(input_text(spec, f"{variant.case_id} {title}", materials, view), encoding="ascii")

    arrival = reference_arrival(spec, profile)
    for name, values in {**profile, **arrival}.items():
        np.save(labels_dir / f"{name}.npy", values)

    changed = [full["index"] for full, control in zip(full_rows, control_rows) if full["epsilon_r"] != control["epsilon_r"] or full["conductivity_s_per_m"] != control["conductivity_s_per_m"]]
    pml_m = spec.pml_cells * spec.dl_m
    left_guard = spec.scan_start_x_m - pml_m
    right_guard = spec.domain_x_m - pml_m - (spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m)
    max_epsilon = max(float(row["epsilon_r"]) for row in full_rows)
    cells_per_wavelength = C0 / (2.8 * spec.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m)
    manifest = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "family_id": FAMILY_ID,
        "case_id": variant.case_id,
        "variant": asdict(variant),
        "purpose": "archived cover-weathered-bedrock causal-control regression family",
        "lifecycle_state": LIFECYCLE_STATE,
        "promotion_allowed": False,
        "permitted_use": "strict-pair and audit-tool regression only",
        "generator_path": "scripts/generate_formal01_bedrock_dense_window.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "formal_training_allowed": False,
        "strict_line9_holdout_allowed": False,
        "line9_conditioned": False,
        "training_block_reason": "permanently blocked after morphology review: layered wavelet combs, over-regular target response, unresolved guard convergence, and invalid visible-phase candidates",
        "morphology_review": {
            "decision": "rejected_for_realism_and_training",
            "family_scale_up_allowed": False,
            "review_report": "reports/formal01_bedrock_dense_window_20260714/FORMAL01_F0_SMOKE_STAGE_REPORT.md",
        },
        "parameter_provenance": "independent generic lithology priors plus project-wide borehole depth range; no Line9-derived geometry, labels, or statistics",
        "source_proxy": "100 MHz Ricker pulse proxy; not an instrument-faithful SFCW synthesis",
        "terrain_stage": "flat ground and fixed 8 m height while subsurface physics are isolated",
        "spec": asdict(spec),
        "acquisition": {"trace_count": spec.trace_count, "spacing_m": spec.trace_spacing_m, "span_m": spec.scan_span_m},
        "grid": {
            "nx_ny_nz": [spec.nx, spec.ny, 1],
            "pml_thickness_m": pml_m,
            "left_physical_guard_m": left_guard,
            "right_physical_guard_m": right_guard,
            "cells_per_min_wavelength_at_2_8fc": cells_per_wavelength,
        },
        "geometry": {
            "shared_index_file": "geology_indices.h5",
            "shared_index_sha256": sha256(geometry_path),
            "index_shape": list(indices.shape),
            "flat_ground": True,
            "fixed_flight_height_m": 8.0,
            "discrete_anomaly_bodies": 0,
            "cover_field_levels": levels,
            "cover_field_thresholds": thresholds,
        },
        "strict_pair": {
            "shared_geometry_hdf5": True,
            "changed_material_indices": changed,
            "only_transition_and_bedrock_changed": True,
            "full_materials_sha256": sha256(case_dir / "materials_full.txt"),
            "control_materials_sha256": sha256(case_dir / "materials_no_basal.txt"),
        },
        "reference_statistics": {
            "basal_depth_m": {"min": float(arrival["basal_interface_depth_m"].min()), "median": float(np.median(arrival["basal_interface_depth_m"])), "max": float(arrival["basal_interface_depth_m"].max())},
            "transition_thickness_m": {"min": float(arrival["transition_thickness_m"].min()), "median": float(np.median(arrival["transition_thickness_m"])), "max": float(arrival["transition_thickness_m"].max())},
            "geometric_arrival_time_ns": {"min": float(arrival["geometric_reference_arrival_time_ns"].min()), "median": float(np.median(arrival["geometric_reference_arrival_time_ns"])), "max": float(arrival["geometric_reference_arrival_time_ns"].max())},
        },
        "labels": {
            "geometric_reference": "material-interface estimate only; not a visible-phase label",
            "visible_phase": "must be extracted after a successful signed full-minus-control pair",
            "training_allowed": False,
        },
    }
    (case_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (case_dir / "RUN_COMMANDS.md").write_text(
        f"""# {variant.case_id}\n\nThis family is permanently archived as a causal-control regression. It must not be promoted, scaled to training data, or used to generate visible-phase labels.\n\nAllowed regression checks:\n\n```powershell\npython -m gprMax geometry_check_full.in --geometry-only\npython -m gprMax geometry_check_control.in --geometry-only\npython -m gprMax full_scene.in -n 8 --geometry-fixed -gpu 0\npython -m gprMax no_basal_contrast_control.in -n 8 --geometry-fixed -gpu 0\n```\n\nDo not start another {spec.trace_count}-trace F-series run. New realism work belongs to FORMAL02 or a later family.\n""",
        encoding="utf-8",
    )
    preview(case_dir, spec, indices, profile, full_rows, control_rows, variant)
    write_checksums(case_dir)
    return case_dir


def generate(output_root: Path, selected: set[str] | None = None) -> list[Path]:
    spec = Spec()
    coordinates = (spec.domain_x_m, spec.domain_y_m, spec.scan_start_x_m, spec.source_y_m, spec.ground_y_m, spec.tx_rx_offset_m, spec.trace_spacing_m)
    if any(abs(round(value / spec.dl_m) - value / spec.dl_m) > 1e-8 for value in coordinates):
        raise ValueError("all domain and acquisition coordinates must align to the FDTD grid")
    if spec.scan_start_x_m <= spec.pml_cells * spec.dl_m:
        raise ValueError("left physical guard must exceed the PML")
    if spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m >= spec.domain_x_m - spec.pml_cells * spec.dl_m:
        raise ValueError("scan plus receiver must remain inside the right physical guard")
    return [generate_variant(output_root, spec, variant) for variant in VARIANTS if not selected or variant.tag in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", action="append", choices=[variant.tag for variant in VARIANTS])
    args = parser.parse_args()
    for path in generate(args.output_root.resolve(), set(args.variant) if args.variant else None):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
