#!/usr/bin/env python3
"""Generate the FORMAL04 fixed-source geology factorial.

FORMAL04 keeps the accepted FORMAL03 GABOR80 source, grid, acquisition,
basal path, transition thickness, stochastic field, and index geometry fixed.
It changes only two constitutive axes: basal contrast and the amplitude of the
correlated cover-material field.  The existing FORMAL03 GABOR80 case is the
low-texture/strong-basal reference; the three cases here complete the 2x2
factorial without reading measured data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_formal03_correlated_cover_source_ablation as formal03


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "00_controls"
FAMILY_ID = "FORMAL04_GEOLOGY_FACTORIAL_GABOR80"
REFERENCE_CASE_ID = "FORMAL03_CORRELATED_COVER_GABOR80"
SOURCE = formal03.SourceVariant(
    "FORMAL04_SHARED_GABOR80",
    "gaussian_modulated_zero_mean",
    80e6,
    "formal04_shared_gabor80",
    50.0,
    11.5,
)


@dataclass(frozen=True)
class GeologyVariant:
    case_id: str
    texture_level: str
    basal_contrast_level: str
    cover_epsilon_min: float
    cover_epsilon_max: float
    cover_conductivity_min_s_per_m: float
    cover_conductivity_max_s_per_m: float
    bedrock_epsilon_r: float
    bedrock_conductivity_s_per_m: float


VARIANTS = (
    GeologyVariant(
        "FORMAL04_A_WEAK_BASAL",
        "baseline",
        "weak",
        11.2,
        13.8,
        0.0017,
        0.0033,
        11.2,
        0.0023,
    ),
    GeologyVariant(
        "FORMAL04_B_STRONG_TEXTURE",
        "strong",
        "strong",
        9.8,
        15.2,
        0.0012,
        0.0044,
        9.0,
        0.0018,
    ),
    GeologyVariant(
        "FORMAL04_C_COMBINED",
        "strong",
        "weak",
        9.8,
        15.2,
        0.0012,
        0.0044,
        11.2,
        0.0023,
    ),
)


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def base_materials(spec: formal03.Spec, variant: GeologyVariant) -> list[formal03.Material]:
    fractions = (np.arange(spec.cover_bins, dtype=np.float64) + 0.5) / spec.cover_bins
    epsilon = variant.cover_epsilon_min + (
        variant.cover_epsilon_max - variant.cover_epsilon_min
    ) * fractions
    conductivity = variant.cover_conductivity_min_s_per_m + (
        variant.cover_conductivity_max_s_per_m
        - variant.cover_conductivity_min_s_per_m
    ) * fractions
    return [
        formal03.Material(
            f"cover_{index:02d}",
            float(epsilon[index]),
            float(conductivity[index]),
        )
        for index in range(spec.cover_bins)
    ]


def material_rows(
    spec: formal03.Spec,
    variant: GeologyVariant,
    control: bool,
) -> list[formal03.Material]:
    """Use the same index geometry and restore target voxels locally in control."""

    bases = base_materials(spec, variant)
    rows = list(bases)
    for level in range(spec.transition_levels):
        fraction = (level + 0.5) / spec.transition_levels
        for base_index, base in enumerate(bases):
            if control:
                epsilon = base.epsilon_r
                conductivity = base.conductivity_s_per_m
            else:
                epsilon = base.epsilon_r + fraction * (
                    variant.bedrock_epsilon_r - base.epsilon_r
                )
                conductivity = base.conductivity_s_per_m + fraction * (
                    variant.bedrock_conductivity_s_per_m
                    - base.conductivity_s_per_m
                )
            rows.append(
                formal03.Material(
                    f"transition_{level:02d}_cover_{base_index:02d}",
                    float(epsilon),
                    float(conductivity),
                )
            )
    for base_index, base in enumerate(bases):
        rows.append(
            formal03.Material(
                f"bedrock_cover_{base_index:02d}",
                base.epsilon_r if control else variant.bedrock_epsilon_r,
                (
                    base.conductivity_s_per_m
                    if control
                    else variant.bedrock_conductivity_s_per_m
                ),
            )
        )
    return rows


def reflection_proxy(spec: formal03.Spec, variant: GeologyVariant) -> dict[str, float]:
    cover = np.asarray([row.epsilon_r for row in base_materials(spec, variant)])
    impedance_term = np.sqrt(cover)
    bedrock_term = math.sqrt(variant.bedrock_epsilon_r)
    coefficient = np.abs((bedrock_term - impedance_term) / (bedrock_term + impedance_term))
    return {
        "min": float(np.min(coefficient)),
        "median": float(np.median(coefficient)),
        "max": float(np.max(coefficient)),
    }


def validate_spec(
    spec: formal03.Spec,
    variants: tuple[GeologyVariant, ...] = VARIANTS,
) -> None:
    formal03.validate_spec(spec, variants=(SOURCE,))
    max_epsilon = max(
        max(variant.cover_epsilon_max, variant.bedrock_epsilon_r)
        for variant in variants
    )
    cells = formal03.C0 / (
        2.8 * SOURCE.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m
    )
    if cells < 10.0:
        raise ValueError("FORMAL04 fails the 2.8fc ten-cell wavelength gate")
    cells_in_transition = min(
        np.min(formal03.build_profiles(spec)[0]["full_transition_thickness_m"])
        / spec.dl_m,
        spec.transition_levels,
    )
    if cells_in_transition < 4:
        raise ValueError("transition is too thin for the declared material levels")


def _preview_factorial(
    path: Path,
    spec: formal03.Spec,
    indices: np.ndarray,
    variants: tuple[GeologyVariant, ...],
) -> None:
    width, height = 1900, 480 * len(variants) + 90
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    valid = indices[:, :, 0] >= 0
    source_index = np.clip(indices[:, :, 0], 0, spec.cover_bins * (spec.transition_levels + 2) - 1)
    x0 = int(round((spec.scan_start_x_m - 2.0) / spec.dl_m))
    x1 = int(round((spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m + 2.0) / spec.dl_m))
    y0 = int(round((spec.ground_y_m - 18.0) / spec.dl_m))
    y1 = int(round((spec.ground_y_m + 1.0) / spec.dl_m))
    draw.text((45, 22), f"{FAMILY_ID}: exact shared geometry and source; constitutive factors only", fill="black", font=font)
    for row_index, variant in enumerate(variants):
        full = material_rows(spec, variant, control=False)
        control = material_rows(spec, variant, control=True)
        full_eps = np.asarray([item.epsilon_r for item in full])
        control_eps = np.asarray([item.epsilon_r for item in control])
        sampled = source_index[x0:x1, y0:y1].T
        sampled_valid = valid[x0:x1, y0:y1].T
        epsilon = np.where(sampled_valid, full_eps[sampled], 1.0)
        contrast = np.where(sampled_valid, full_eps[sampled] - control_eps[sampled], 0.0)

        epsilon_unit = np.clip((epsilon - 1.0) / 14.2, 0.0, 1.0)
        epsilon_rgb = np.stack(
            (255 * epsilon_unit, 255 * (1.0 - np.abs(epsilon_unit - 0.5)), 255 * (1.0 - epsilon_unit)),
            axis=-1,
        ).astype(np.uint8)
        contrast_unit = np.clip((contrast + 6.2) / 12.4, 0.0, 1.0)
        contrast_rgb = np.stack(
            (255 * contrast_unit, 255 * (1.0 - np.abs(contrast_unit - 0.5) * 1.6), 255 * (1.0 - contrast_unit)),
            axis=-1,
        ).astype(np.uint8)
        top = 70 + row_index * 480
        for left, image, title in (
            (45, epsilon_rgb, "full epsilon"),
            (970, contrast_rgb, "full minus local-cover control"),
        ):
            panel = Image.fromarray(np.flipud(image), mode="RGB").resize((870, 390), Image.Resampling.BILINEAR)
            canvas.paste(panel, (left, top + 45))
            draw.rectangle((left, top + 45, left + 870, top + 435), outline="black", width=2)
            draw.text((left, top + 12), f"{variant.case_id}: {title}", fill="black", font=font)
    canvas.save(path)


def generate(
    output_root: Path,
    spec: formal03.Spec | None = None,
    variants: tuple[GeologyVariant, ...] = VARIANTS,
) -> list[Path]:
    spec = spec or formal03.Spec()
    validate_spec(spec, variants)
    output_root.mkdir(parents=True, exist_ok=True)

    profile, crop_stats = formal03.build_profiles(spec)
    latent, cover_bins, field_stats = formal03.build_cover_field(spec)
    indices = formal03.build_indices(spec, profile, cover_bins)
    index_array_hash = _array_sha256(indices)

    if not variants:
        raise ValueError("at least one geology variant is required")
    first_dir = output_root / variants[0].case_id
    first_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = first_dir / "geology_indices.h5"
    with h5py.File(geometry_path, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = "scripts/generate_formal04_geology_factorial.py"
        handle.attrs["family_id"] = FAMILY_ID
        handle.create_dataset(
            "data", data=indices, dtype=np.int16, compression="gzip", compression_opts=4
        )
    geometry_file_hash = formal03.sha256(geometry_path)

    case_dirs: list[Path] = []
    source_hashes: set[str] = set()
    for variant in variants:
        case_dir = output_root / variant.case_id
        labels_dir = case_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        if case_dir != first_dir:
            shutil.copy2(geometry_path, case_dir / "geology_indices.h5")

        full_rows = material_rows(spec, variant, control=False)
        control_rows = material_rows(spec, variant, control=True)
        formal03.write_materials(case_dir / "materials_full.txt", full_rows)
        formal03.write_materials(case_dir / "materials_no_basal.txt", control_rows)
        waveform_stats = formal03.write_custom_waveform(
            case_dir / "source_waveform.txt", SOURCE, spec
        )
        source_hashes.add(formal03.sha256(case_dir / "source_waveform.txt"))

        for filename, title, materials, view in (
            ("full_scene.in", "full scene", "materials_full.txt", None),
            ("no_basal_contrast_control.in", "no-basal contrast control", "materials_no_basal.txt", None),
            ("air_reference.in", "air reference", None, None),
            ("geometry_check_full.in", "geometry full", "materials_full.txt", "geometry_check_full"),
            ("geometry_check_control.in", "geometry control", "materials_no_basal.txt", "geometry_check_control"),
        ):
            (case_dir / filename).write_text(
                formal03.input_text(
                    spec,
                    SOURCE,
                    f"{variant.case_id} {title}",
                    materials,
                    view,
                ),
                encoding="ascii",
            )

        arrival = formal03.reference_arrival(spec, profile, indices, full_rows)
        source_reference = (
            arrival["geometric_reference_arrival_time_ns"] + SOURCE.reference_delay_ns
        )
        for name, values in {**profile, **arrival}.items():
            np.save(labels_dir / f"{name}.npy", values)
        np.save(
            labels_dir / "source_referenced_arrival_time_ns.npy",
            source_reference.astype(np.float32),
        )

        max_epsilon = max(row.epsilon_r for row in full_rows)
        cells = formal03.C0 / (
            2.8 * SOURCE.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m
        )
        max_epsilon_step, max_conductivity_step = formal03._physical_transition_step(
            spec, full_rows
        )
        manifest = {
            "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
            "family_id": FAMILY_ID,
            "case_id": variant.case_id,
            "lifecycle_state": "pre_solver_review",
            "purpose": "fixed-source geology factorial for basal strength and correlated cover texture",
            "generator_path": "scripts/generate_formal04_geology_factorial.py",
            "generator_sha256": formal03.sha256(Path(__file__).resolve()),
            "formal_training_allowed": False,
            "promotion_allowed": False,
            "target_presence": True,
            "strict_line9_holdout_allowed": False,
            "line9_conditioned": False,
            "training_block_reason": "requires static, geometry, one-trace causal pair, sparse spatial, and human morphology gates",
            "parameter_provenance": "FORMAL03 generic deterministic priors with declared constitutive factorial; no measured data are read",
            "terrain_stage": "flat ground and fixed height",
            "spec": asdict(spec),
            "source": {
                **asdict(SOURCE),
                "shared_across_factorial": True,
                "proxy_only": True,
                "not_sfcw": True,
                "custom_waveform_statistics": waveform_stats,
            },
            "ablation": {
                "reference_case_id": REFERENCE_CASE_ID,
                "factorial_cell": {
                    "texture_level": variant.texture_level,
                    "basal_contrast_level": variant.basal_contrast_level,
                },
                "locked": [
                    "source waveform",
                    "grid and PML",
                    "acquisition and flight height",
                    "basal path and transition thickness",
                    "correlated latent field and material indices",
                ],
                "changed": "constitutive material mapping only",
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
                "shared_index_file_sha256": geometry_file_hash,
                "shared_index_array_sha256": index_array_hash,
                "index_shape": list(indices.shape),
                "flat_ground": True,
                "fixed_flight_height_m": spec.flight_height_m,
                "discrete_anomaly_bodies": 0,
                "cover_field_statistics": field_stats,
                "crop_shape_gate": crop_stats,
            },
            "materials": {
                **asdict(variant),
                "cover_bins": spec.cover_bins,
                "transition_levels": spec.transition_levels,
                "normal_incidence_dielectric_reflection_proxy": reflection_proxy(spec, variant),
                "maximum_physical_transition_epsilon_step": max_epsilon_step,
                "maximum_physical_transition_conductivity_step_s_per_m": max_conductivity_step,
                "full_material_count": len(full_rows),
            },
            "strict_pair": {
                "shared_geometry_hdf5": True,
                "control_restores_each_target_voxel_to_its_local_cover_bin": True,
                "full_materials_sha256": formal03.sha256(case_dir / "materials_full.txt"),
                "control_materials_sha256": formal03.sha256(case_dir / "materials_no_basal.txt"),
            },
            "reference_statistics": {
                "source_referenced_arrival_time_ns": {
                    "min": float(np.min(source_reference)),
                    "median": float(np.median(source_reference)),
                    "max": float(np.max(source_reference)),
                }
            },
            "labels": {
                "geometric_reference": "material-interface estimate only",
                "source_referenced_arrival": "geometric estimate plus explicit source peak delay; not a visible-phase label",
                "visible_phase_search_half_width_ns": 55.0,
                "visible_phase": "absent until a signed full-minus-control runtime pair is reviewed",
                "training_allowed": False,
            },
            "next_gate": "static and geometry audits, then one-trace full/control smoke",
        }
        (case_dir / "scene_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (case_dir / "RUN_COMMANDS.md").write_text(
            f"""# {variant.case_id}

This case is blocked from training. Run from the repository root.

```powershell
$case = "data/PGDA_SYNTH_DATASET_V2/00_controls/{variant.case_id}"
$cudaArgs = @()
if ($env:PGDA_CUDA_BIN) {{ $cudaArgs = @("--cuda-bin", $env:PGDA_CUDA_BIN) }}

python scripts/run_native_256_release_pilot.py $case --run-id formal04_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal04_smoke1 --trace-count 1 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal04_distributed8_stride36 --trace-count 8 --trace-stride 36 --skip-air-reference @cudaArgs --execute
```

Do not run the sparse pair until the one-trace factorial is audited. Do not
run 256 traces until the selected geology passes a human morphology review.
""",
            encoding="utf-8",
        )
        formal03.preview_geometry(
            case_dir / "preview_geometry_and_strict_pair_contrast.png",
            spec,
            indices,
            profile,
            full_rows,
            control_rows,
        )
        formal03.write_checksums(case_dir)
        case_dirs.append(case_dir)

    if len(source_hashes) != 1:
        raise RuntimeError("FORMAL04 source waveforms are not byte-identical")
    summary_preview = output_root / "FORMAL04_GEOLOGY_FACTORIAL_PREVIEW.png"
    _preview_factorial(summary_preview, spec, indices, variants)
    policy = {
        "family_id": FAMILY_ID,
        "formal_training_allowed": False,
        "line9_conditioned": False,
        "reference_case_id": REFERENCE_CASE_ID,
        "shared_index_array_sha256": index_array_hash,
        "shared_source_waveform_sha256": next(iter(source_hashes)),
        "case_ids": [variant.case_id for variant in variants],
        "factorial_definition": {
            "reference": "baseline texture, strong basal contrast",
            "A": "baseline texture, weak basal contrast",
            "B": "strong texture, strong basal contrast",
            "C": "strong texture, weak basal contrast",
        },
        "selection_rule": "use one-trace pairs only for causal detectability; use a sparse paired B-scan to assess target/background balance and coherence",
        "sparse_morphology_gates": {
            "target_to_adjacent_background_rms_preferred_range": [1.5, 4.5],
            "path_envelope_cv_range": [0.30, 0.85],
            "path_dropout_fraction_max": 0.10,
            "unexplained_periodic_or_crossing_branches_allowed": False,
        },
        "no_full_256_before_pilot_pass": True,
    }
    (output_root / "FORMAL04_GEOLOGY_FACTORIAL_POLICY.json").write_text(
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
