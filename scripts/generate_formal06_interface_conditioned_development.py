#!/usr/bin/env python3
"""Generate the FORMAL06 interface-conditioned development candidate.

FORMAL06 changes the failed FORMAL05 geology mechanism rather than
interpolating another constitutive endpoint. It uses smooth, weak bulk
heterogeneity and a finite weathered cap that approaches a controlled
cover-side impedance before a stable bedrock contrast. The qualitative design
was informed by a held-out measured-line morphology gap, so this family is
permanently development-only even though the generator reads no measured data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

try:
    import generate_formal03_correlated_cover_source_ablation as formal03
    import generate_formal04_geology_factorial as formal04
except ModuleNotFoundError:  # Package import used by tests and successor generators.
    from scripts import generate_formal03_correlated_cover_source_ablation as formal03
    from scripts import generate_formal04_geology_factorial as formal04


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL06_INTERFACE_CONDITIONED_DEVELOPMENT"
CASE_ID = "FORMAL06_INTERFACE_CONDITIONED_DEVELOPMENT"
SOURCE = formal04.SOURCE


@dataclass(frozen=True)
class MaterialDesign:
    cover_epsilon_min: float = 12.0
    cover_epsilon_max: float = 12.8
    cover_conductivity_min_s_per_m: float = 0.0018
    cover_conductivity_max_s_per_m: float = 0.0025
    weathered_cap_epsilon_r: float = 13.4
    weathered_cap_conductivity_s_per_m: float = 0.0027
    bedrock_epsilon_r: float = 9.6
    bedrock_conductivity_s_per_m: float = 0.0017
    bulk_long_x_scale_m: float = 6.7
    bulk_long_y_scale_m: float = 3.0
    bulk_meso_x_scale_m: float = 1.8
    bulk_meso_y_scale_m: float = 0.96
    bulk_meso_weight: float = 0.18


DESIGN = MaterialDesign()


def default_spec() -> formal03.Spec:
    return replace(
        formal03.Spec(),
        cover_bins=32,
        transition_levels=8,
        field_seed=2026071526,
    )


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def validate_spec(
    spec: formal03.Spec,
    *,
    design: MaterialDesign = DESIGN,
    source: formal03.SourceVariant = SOURCE,
) -> None:
    formal03.validate_spec(spec, variants=(source,))
    max_epsilon = max(design.cover_epsilon_max, design.weathered_cap_epsilon_r)
    cells = formal03.C0 / (
        2.8 * source.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m
    )
    if cells < 10.0:
        raise ValueError("FORMAL06 fails the 2.8fc ten-cell wavelength gate")
    thinnest_transition_cells = int(math.floor(0.66 / spec.dl_m))
    if spec.transition_levels > thinnest_transition_cells:
        raise ValueError("FORMAL06 transition levels exceed the thinnest physical cap")


def build_bulk_field(
    spec: formal03.Spec,
    *,
    design: MaterialDesign = DESIGN,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | dict[str, float]]]:
    """Create broad weak heterogeneity without dense stratigraphic contours."""

    rng = np.random.default_rng(spec.field_seed)
    shape = (spec.nx, spec.ny)
    long_field = formal03._upsampled_component(
        rng,
        shape,
        (max(2, int(round(2.4 / spec.dl_m))), max(2, int(round(1.2 / spec.dl_m)))),
        (2.8, 2.5),
    )
    meso = formal03._upsampled_component(
        rng,
        shape,
        (max(2, int(round(0.9 / spec.dl_m))), max(2, int(round(0.48 / spec.dl_m)))),
        (2.0, 2.0),
    )
    field = formal03.normalise(long_field + design.bulk_meso_weight * meso)
    field = np.clip(field, -2.5, 2.5).astype(np.float32)
    unit = (field + 2.5) / 5.0
    bins = np.minimum((unit * spec.cover_bins).astype(np.int16), spec.cover_bins - 1)
    stats: dict[str, float | int | dict[str, float]] = {
        "used_bins": int(np.unique(bins).size),
        "horizontal_neighbor_bin_change_rate": float(np.mean(bins[1:, :] != bins[:-1, :])),
        "vertical_neighbor_bin_change_rate": float(np.mean(bins[:, 1:] != bins[:, :-1])),
        "horizontal_latent_step_p95": float(np.percentile(np.abs(np.diff(field, axis=0)), 95)),
        "vertical_latent_step_p95": float(np.percentile(np.abs(np.diff(field, axis=1)), 95)),
        "latent_min": float(np.min(field)),
        "latent_median": float(np.median(field)),
        "latent_max": float(np.max(field)),
        "correlation_design_m": {
            "long_x": design.bulk_long_x_scale_m,
            "long_y": design.bulk_long_y_scale_m,
            "meso_x": design.bulk_meso_x_scale_m,
            "meso_y": design.bulk_meso_y_scale_m,
        },
    }
    return field, bins, stats


def base_materials(
    spec: formal03.Spec,
    *,
    design: MaterialDesign = DESIGN,
) -> list[formal03.Material]:
    fractions = (np.arange(spec.cover_bins, dtype=np.float64) + 0.5) / spec.cover_bins
    epsilon = design.cover_epsilon_min + (
        design.cover_epsilon_max - design.cover_epsilon_min
    ) * fractions
    conductivity = design.cover_conductivity_min_s_per_m + (
        design.cover_conductivity_max_s_per_m
        - design.cover_conductivity_min_s_per_m
    ) * fractions
    return [
        formal03.Material(
            f"cover_{index:02d}",
            float(epsilon[index]),
            float(conductivity[index]),
        )
        for index in range(spec.cover_bins)
    ]


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def material_rows(
    spec: formal03.Spec,
    control: bool,
    *,
    design: MaterialDesign = DESIGN,
) -> list[formal03.Material]:
    """Map the complete cap/bedrock target state back to local cover in control."""

    bases = base_materials(spec, design=design)
    rows = list(bases)
    for level in range(spec.transition_levels):
        blend = _smoothstep((level + 0.5) / spec.transition_levels)
        for base in bases:
            if control:
                epsilon = base.epsilon_r
                conductivity = base.conductivity_s_per_m
            else:
                epsilon = base.epsilon_r + blend * (
                    design.weathered_cap_epsilon_r - base.epsilon_r
                )
                conductivity = base.conductivity_s_per_m + blend * (
                    design.weathered_cap_conductivity_s_per_m
                    - base.conductivity_s_per_m
                )
            rows.append(
                formal03.Material(
                    f"weathered_cap_{level:02d}_{base.material_id}",
                    float(epsilon),
                    float(conductivity),
                )
            )
    for base in bases:
        rows.append(
            formal03.Material(
                f"bedrock_{base.material_id}",
                base.epsilon_r if control else design.bedrock_epsilon_r,
                base.conductivity_s_per_m
                if control
                else design.bedrock_conductivity_s_per_m,
            )
        )
    return rows


def generate_case(
    output_root: Path,
    spec: formal03.Spec,
    *,
    design: MaterialDesign,
    source: formal03.SourceVariant,
    family_id: str,
    case_id: str,
    policy_filename: str,
    run_prefix: str,
    purpose: str,
    predecessor_case_id: str,
    changed_factors: list[str],
    generator_path: Path,
    preview_title: str,
    profile_builder: Callable[
        [formal03.Spec], tuple[dict[str, np.ndarray], dict[str, float | int]]
    ] = formal03.build_profiles,
    bulk_field_builder: Callable[
        ..., tuple[np.ndarray, np.ndarray, dict]
    ] = build_bulk_field,
    material_rows_builder: Callable[..., list[formal03.Material]] = material_rows,
    locked_factors: list[str] | None = None,
    geometry_description: str = "smooth weak two-dimensional bulk heterogeneity",
    flat_ground: bool = True,
    terrain_stage: str = "flat ground and fixed height",
) -> Path:
    validate_spec(spec, design=design, source=source)
    output_root.mkdir(parents=True, exist_ok=True)
    case_dir = output_root / case_id
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    profile, crop_stats = profile_builder(spec)
    ground_y_m = np.asarray(profile["full_ground_y_m"], dtype=np.float64)
    flight_height_agl_m = np.asarray(spec.source_y_m - ground_y_m, dtype=np.float64)
    _, cover_bins, field_stats = bulk_field_builder(spec, design=design)
    indices = formal03.build_indices(spec, profile, cover_bins)
    index_array_hash = _array_sha256(indices)
    geometry_path = case_dir / "geology_indices.h5"
    with h5py.File(geometry_path, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = generator_path.name
        handle.attrs["family_id"] = family_id
        handle.create_dataset(
            "data",
            data=indices,
            dtype=np.int16,
            compression="gzip",
            compression_opts=4,
        )

    full_rows = material_rows_builder(spec, control=False, design=design)
    control_rows = material_rows_builder(spec, control=True, design=design)
    formal03.write_materials(case_dir / "materials_full.txt", full_rows)
    formal03.write_materials(case_dir / "materials_no_basal.txt", control_rows)
    waveform_stats: dict[str, float] | None = None
    if source.kind != "ricker":
        waveform_stats = formal03.write_custom_waveform(
            case_dir / "source_waveform.txt", source, spec
        )
    for filename, title, materials, view in (
        ("full_scene.in", "full scene", "materials_full.txt", None),
        (
            "no_basal_contrast_control.in",
            "no-basal contrast control",
            "materials_no_basal.txt",
            None,
        ),
        ("air_reference.in", "air reference", None, None),
        (
            "geometry_check_full.in",
            "geometry full",
            "materials_full.txt",
            "geometry_check_full",
        ),
        (
            "geometry_check_control.in",
            "geometry control",
            "materials_no_basal.txt",
            "geometry_check_control",
        ),
    ):
        (case_dir / filename).write_text(
            formal03.input_text(spec, source, f"{case_id} {title}", materials, view),
            encoding="ascii",
        )

    arrival = formal03.reference_arrival(spec, profile, indices, full_rows)
    source_reference = (
        arrival["geometric_reference_arrival_time_ns"] + source.reference_delay_ns
    )
    for name, values in {**profile, **arrival}.items():
        np.save(labels_dir / f"{name}.npy", values)
    np.save(
        labels_dir / "source_referenced_arrival_time_ns.npy",
        source_reference.astype(np.float32),
    )

    max_epsilon = max(row.epsilon_r for row in full_rows)
    cells = formal03.C0 / (
        2.8 * source.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m
    )
    epsilon_step, conductivity_step = formal03._physical_transition_step(
        spec, full_rows
    )
    manifest = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "family_id": family_id,
        "case_id": case_id,
        "lifecycle_state": "pre_solver_review",
        "purpose": purpose,
        "generator_path": f"scripts/{generator_path.name}",
        "generator_sha256": formal03.sha256(generator_path.resolve()),
        "formal_training_allowed": False,
        "promotion_allowed": False,
        "target_presence": True,
        "strict_line9_holdout_allowed": False,
        "line9_conditioned": True,
        "training_block_reason": (
            "qualitative mechanism was selected after held-out morphology diagnosis; "
            "requires complete staged runtime and independent formal regeneration"
        ),
        "parameter_provenance": (
            "bounded custom materials and generic seeded profiles; no measured file is "
            "read, but the mechanism choice is development-only"
        ),
        "terrain_stage": "flat ground and fixed height",
        "spec": asdict(spec),
        "source": {
            **asdict(source),
            "locked_from_formal03_source_ablation": True,
            "proxy_only": True,
            "not_sfcw": True,
            "custom_waveform_statistics": waveform_stats,
        },
        "ablation": {
            "predecessor_case_id": predecessor_case_id,
            "locked": locked_factors
            or [
                "source waveform",
                "grid and PML",
                "acquisition and flight height",
                "basal path family and transition-thickness family",
            ],
            "changed": changed_factors,
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
            "shared_index_array_sha256": index_array_hash,
            "index_shape": list(indices.shape),
            "flat_ground": flat_ground,
            "terrain_stage": terrain_stage,
            "fixed_source_elevation_m": spec.source_y_m,
            "flight_height_agl_m": {
                "min": float(np.min(flight_height_agl_m)),
                "median": float(np.median(flight_height_agl_m)),
                "max": float(np.max(flight_height_agl_m)),
            },
            "discrete_anomaly_bodies": 0,
            "description": geometry_description,
            "bulk_field_statistics": field_stats,
            "crop_shape_gate": crop_stats,
        },
        "materials": {
            **asdict(design),
            "cover_bins": spec.cover_bins,
            "weathered_cap_levels": spec.transition_levels,
            "maximum_physical_epsilon_step": epsilon_step,
            "maximum_physical_conductivity_step_s_per_m": conductivity_step,
            "full_material_count": len(full_rows),
            "basal_reflection_proxy": float(
                (math.sqrt(design.bedrock_epsilon_r) - math.sqrt(design.weathered_cap_epsilon_r))
                / (math.sqrt(design.bedrock_epsilon_r) + math.sqrt(design.weathered_cap_epsilon_r))
            ),
        },
        "strict_pair": {
            "shared_geometry_hdf5": True,
            "control_restores_cap_and_bedrock_to_each_local_cover_bin": True,
            "full_materials_sha256": formal03.sha256(case_dir / "materials_full.txt"),
            "control_materials_sha256": formal03.sha256(
                case_dir / "materials_no_basal.txt"
            ),
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
            "source_referenced_arrival": (
                "geometric estimate plus source delay; not a visible-phase label"
            ),
            "visible_phase_search_half_width_ns": 55.0,
            "visible_phase": "absent until a complete signed runtime pair is reviewed",
            "training_allowed": False,
        },
        "visibility_gate": {
            "full_scene_target_to_local_background_rms_min": 1.0,
            "full_scene_target_to_local_background_rms_max": 5.0,
            "full_scene_target_to_background_rms_min": 0.35,
            "raw_target_to_early_rms": "diagnostic_only_cross_domain_uncalibrated",
            "difference_only_visibility_is_sufficient": False,
            "requires_shared_full_window_display": True,
            "requires_human_visible_multicycle_interface": True,
            "requires_blind_review_before_label_overlay": True,
        },
        "next_gate": "static, geometry, attenuation, then one-trace full/control smoke",
    }
    (case_dir / "scene_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (case_dir / "RUN_COMMANDS.md").write_text(
        f"""# {case_id}

This case is development-only and blocked from training.

```powershell
$case = "data/simulations/v2/00_controls/{case_id}"
$cudaArgs = @()
if ($env:PGDA_CUDA_BIN) {{ $cudaArgs = @("--cuda-bin", $env:PGDA_CUDA_BIN) }}

python scripts/run_native_256_release_pilot.py $case --run-id {run_prefix}_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id {run_prefix}_smoke1 --trace-count 1 --skip-air-reference @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id {run_prefix}_blind8_full --trace-count 8 --skip-air-reference --full-scene-only @cudaArgs --execute
python scripts/run_native_256_release_pilot.py $case --run-id {run_prefix}_distributed32_full --trace-count 32 --trace-stride 8 --skip-air-reference --full-scene-only @cudaArgs --execute
```

Do not start the distributed 32-trace full scene until the one-trace causal
audit and blind local 8-trace amplitude review pass. Do not start a matched
32-trace pair until the distributed full scene passes morphology review. The
unlabelled full-scene preview is reviewed before any target path is overlaid.
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
        title=preview_title,
    )
    formal03.write_checksums(case_dir)

    policy = {
        "family_id": family_id,
        "case_id": case_id,
        "formal_training_allowed": False,
        "line9_conditioned": True,
        "development_only": True,
        "locked_source": {
            "kind": source.kind,
            "center_frequency_hz": source.center_frequency_hz,
            "waveform_id": source.waveform_id,
            "custom_waveform_sha256": (
                formal03.sha256(case_dir / "source_waveform.txt")
                if (case_dir / "source_waveform.txt").is_file()
                else None
            ),
        },
        "locked_index_array_sha256": index_array_hash,
        "material_design": asdict(design),
        "release_order": [
            "static_geometry_attenuation",
            "one_trace_strict_pair",
            "blind_local_8_trace_full_scene",
            "distributed_span_32_trace_full_scene_only_after_blind_pass",
            "distributed_span_32_trace_strict_pair_only_after_morphology_pass",
            "dense_full_256_trace_pair_only_after_sparse_pair_pass",
        ],
        "full_scene_visibility_gate": manifest["visibility_gate"],
    }
    (output_root / policy_filename).write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )
    return case_dir


def generate(output_root: Path, spec: formal03.Spec | None = None) -> Path:
    spec = spec or default_spec()
    return generate_case(
        output_root,
        spec,
        design=DESIGN,
        source=SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL06_INTERFACE_CONDITIONED_POLICY.json",
        run_prefix="formal06",
        purpose="interface-conditioned morphology mechanism development",
        predecessor_case_id="FORMAL05_MODERATE_TEXTURE_BALANCED_BASAL",
        changed_factors=[
            "bulk heterogeneity correlation and amplitude",
            "cover material span",
            "weathered-cap constitutive trajectory",
            "stable cap-to-bedrock contrast",
        ],
        generator_path=Path(__file__),
        preview_title=(
            f"{FAMILY_ID}: smooth bulk field and interface-conditioned cap; "
            "pre-solver only"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate(args.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
