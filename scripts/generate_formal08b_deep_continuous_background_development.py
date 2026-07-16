#!/usr/bin/env python3
"""Generate the FORMAL08B deep continuous-background candidate.

FORMAL08B is a direct one-factor successor to the project-owner accepted
FORMAL06C morphology. Source, constitutive materials, grid, acquisition,
basal path, and transition geometry are locked. Only a stronger multiscale
continuous cover field is added. Unlike FORMAL08A's fixed-depth taper, this
field follows the transition-top geometry and remains active through the deep
cover while preserving an exact guard above the transition.

This is explicitly Line9-conditioned realism development. It may use Line9 as
a visual calibration reference, but it cannot support an unseen-Line9 claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import generate_formal03_correlated_cover_source_ablation as formal03
import generate_formal06_interface_conditioned_development as formal06
import generate_formal06c_subtle_interface_development as formal06c
import generate_formal07b_weak_aperiodic_background_development as formal07b
import generate_formal08a_line9_realism_background_development as formal08a


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL08B_LINE9_REALISM_DEEP_BACKGROUND_DEVELOPMENT"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN

TEXTURE_COMPONENTS = (
    {
        "name": "broad_deep_aperiodic_2d",
        "weight": 0.43,
        "correlation_xy_m": (5.40, 1.80),
        "smoothing_sigma": (2.8, 2.3),
    },
    {
        "name": "mesoscale_deep_aperiodic_2d",
        "weight": 0.50,
        "correlation_xy_m": (2.10, 0.72),
        "smoothing_sigma": (2.2, 2.0),
    },
    {
        "name": "fine_deep_aperiodic_2d",
        "weight": 0.24,
        "correlation_xy_m": (0.75, 0.30),
        "smoothing_sigma": (1.8, 1.8),
    },
)

ENVELOPE_CONTRACT_M = {
    "surface_zero_end": 0.45,
    "surface_full_start": 1.25,
    "transition_zero_gap": 0.45,
    "transition_full_gap": 1.35,
}


def default_spec() -> formal03.Spec:
    return formal06c.default_spec()


def _dynamic_depth_envelope(
    spec: formal03.Spec,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    profile, _ = formal03.build_profiles(spec)
    y = (np.arange(spec.ny, dtype=np.float32) + 0.5) * spec.dl_m
    depth = np.float32(spec.ground_y_m) - y
    transition_top_depth = (
        profile["full_basal_depth_m"]
        - profile["full_transition_thickness_m"]
    ).astype(np.float32)
    gap = transition_top_depth[:, None] - depth[None, :]

    surface = formal08a._smoothstep(
        (depth - ENVELOPE_CONTRACT_M["surface_zero_end"])
        / (
            ENVELOPE_CONTRACT_M["surface_full_start"]
            - ENVELOPE_CONTRACT_M["surface_zero_end"]
        )
    )
    transition = formal08a._smoothstep(
        (gap - ENVELOPE_CONTRACT_M["transition_zero_gap"])
        / (
            ENVELOPE_CONTRACT_M["transition_full_gap"]
            - ENVELOPE_CONTRACT_M["transition_zero_gap"]
        )
    )
    envelope = surface[None, :] * transition
    subsurface = depth[None, :] >= 0.0
    envelope[~subsurface.repeat(spec.nx, axis=0)] = 0.0
    envelope[gap <= ENVELOPE_CONTRACT_M["transition_zero_gap"]] = 0.0

    protected = (depth[None, :] <= ENVELOPE_CONTRACT_M["surface_zero_end"]) | (
        gap <= ENVELOPE_CONTRACT_M["transition_zero_gap"]
    )
    stats = {
        "transition_top_depth_min_m": float(np.min(transition_top_depth)),
        "transition_top_depth_median_m": float(np.median(transition_top_depth)),
        "transition_top_depth_max_m": float(np.max(transition_top_depth)),
        "active_fraction": float(np.mean(envelope > 0.0)),
        "full_strength_fraction": float(np.mean(envelope >= 0.999)),
    }
    return envelope.astype(np.float32), protected, stats


def build_bulk_field(
    spec: formal03.Spec,
    *,
    design: formal06.MaterialDesign = DESIGN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Add stronger continuous deep-cover texture with an exact target guard."""

    predecessor, predecessor_bins, predecessor_stats = formal06.build_bulk_field(
        spec, design=design
    )
    shape = (spec.nx, spec.ny)
    rng = np.random.default_rng(spec.field_seed + 8207)
    envelope, protected, envelope_stats = _dynamic_depth_envelope(spec)
    perturbation = np.zeros(shape, dtype=np.float32)
    component_records: list[dict[str, object]] = []

    for component in TEXTURE_COMPONENTS:
        correlation = component["correlation_xy_m"]
        values = formal03._upsampled_component(
            rng,
            shape,
            (
                max(2, int(round(correlation[0] / spec.dl_m))),
                max(2, int(round(correlation[1] / spec.dl_m))),
            ),
            component["smoothing_sigma"],
        )
        values = formal07b._normalise(values)
        weight = float(component["weight"])
        perturbation += weight * values * envelope
        component_records.append(
            {
                "name": component["name"],
                "weight": weight,
                "correlation_xy_m": list(correlation),
                "smoothing_sigma": list(component["smoothing_sigma"]),
            }
        )
        del values

    candidate = np.clip(predecessor + perturbation, -2.5, 2.5).astype(np.float32)
    unit = (candidate + 2.5) / 5.0
    bins = np.minimum(
        (unit * spec.cover_bins).astype(np.int16), spec.cover_bins - 1
    )

    baseline_crop = formal07b._cover_audit_crop(spec, predecessor)
    candidate_crop = formal07b._cover_audit_crop(spec, candidate)
    delta_crop = candidate_crop - baseline_crop
    baseline_metrics = formal07b._texture_metrics(baseline_crop)
    candidate_metrics = formal07b._texture_metrics(candidate_crop)

    baseline_centered = baseline_crop.astype(np.float64)
    baseline_centered -= float(np.mean(baseline_centered))
    candidate_centered = candidate_crop.astype(np.float64)
    candidate_centered -= float(np.mean(candidate_centered))
    correlation = float(
        np.sum(baseline_centered * candidate_centered)
        / max(
            float(
                np.sqrt(
                    np.sum(baseline_centered**2)
                    * np.sum(candidate_centered**2)
                )
            ),
            1e-12,
        )
    )
    delta_rms = float(np.sqrt(np.mean(delta_crop.astype(np.float64) ** 2)))
    bin_delta = np.abs(bins.astype(np.int32) - predecessor_bins.astype(np.int32))
    bin_delta_p99 = float(np.percentile(bin_delta, 99.0))
    changed_fraction = float(np.mean(bins != predecessor_bins))
    protected_bins_exact = bool(
        np.array_equal(bins[protected], predecessor_bins[protected])
    )

    gates = {
        "predecessor_latent_correlation_min": 0.72,
        "perturbation_rms_min": 0.50,
        "perturbation_rms_max": 0.82,
        "changed_cover_bin_fraction_min": 0.20,
        "changed_cover_bin_fraction_max": 0.48,
        "laterally_coherent_depth_variance_ratio_max": 0.60,
        "vertical_spectral_peak_fraction_max": 0.97,
        "bin_delta_p99_max": 9.0,
        "protected_surface_and_transition_bins_exact": True,
    }
    gate_results = {
        "predecessor_latent_correlation": correlation
        >= gates["predecessor_latent_correlation_min"],
        "perturbation_rms": gates["perturbation_rms_min"]
        <= delta_rms
        <= gates["perturbation_rms_max"],
        "changed_cover_bin_fraction": gates["changed_cover_bin_fraction_min"]
        <= changed_fraction
        <= gates["changed_cover_bin_fraction_max"],
        "coherent_depth_variance": candidate_metrics[
            "laterally_coherent_depth_variance_ratio"
        ]
        <= gates["laterally_coherent_depth_variance_ratio_max"],
        "vertical_spectral_peak": candidate_metrics[
            "vertical_spectral_peak_fraction"
        ]
        <= gates["vertical_spectral_peak_fraction_max"],
        "bin_delta_p99": bin_delta_p99 <= gates["bin_delta_p99_max"],
        "protected_surface_and_transition_bins_exact": protected_bins_exact,
    }
    if not all(gate_results.values()):
        failed = [name for name, passed in gate_results.items() if not passed]
        diagnostics = {
            "correlation": correlation,
            "perturbation_rms": delta_rms,
            "changed_fraction": changed_fraction,
            "bin_delta_p99": bin_delta_p99,
            "baseline_texture": baseline_metrics,
            "candidate_texture": candidate_metrics,
        }
        raise ValueError(
            f"FORMAL08B background gate failed: {failed}; {diagnostics}"
        )

    stats = {
        "model": "formal06c_plus_deep_transition_following_multiscale_cover",
        "predecessor_model": "FORMAL06C smooth weak two-dimensional bulk field",
        "line9_calibration_scope": (
            "background continuity, depth distribution, target prominence, and "
            "non-target texture; no measured array is read by the generator"
        ),
        "envelope_contract_m": ENVELOPE_CONTRACT_M,
        "envelope_statistics": envelope_stats,
        "texture_components": component_records,
        "used_bins": int(np.unique(bins).size),
        "horizontal_neighbor_bin_change_rate": float(
            np.mean(bins[1:, :] != bins[:-1, :])
        ),
        "vertical_neighbor_bin_change_rate": float(
            np.mean(bins[:, 1:] != bins[:, :-1])
        ),
        "horizontal_latent_step_p95": float(
            np.percentile(np.abs(np.diff(candidate, axis=0)), 95)
        ),
        "vertical_latent_step_p95": float(
            np.percentile(np.abs(np.diff(candidate, axis=1)), 95)
        ),
        "latent_min": float(np.min(candidate)),
        "latent_median": float(np.median(candidate)),
        "latent_max": float(np.max(candidate)),
        "predecessor_statistics": predecessor_stats,
        "predecessor_latent_correlation": correlation,
        "perturbation_rms": delta_rms,
        "changed_cover_bin_fraction": changed_fraction,
        "cover_bin_delta_p99": bin_delta_p99,
        "protected_surface_and_transition_bins_exact": protected_bins_exact,
        "baseline_cover_texture_metrics": baseline_metrics,
        "candidate_cover_texture_metrics": candidate_metrics,
        "gates": gates,
        "gate_results": gate_results,
        "sinusoidal_stratigraphy": False,
        "isolated_inclusions": 0,
        "point_targets": 0,
        "vertical_partitions": 0,
    }
    return candidate, bins, stats


def generate(output_root: Path, spec: formal03.Spec | None = None) -> Path:
    spec = spec or default_spec()
    case_dir = formal06.generate_case(
        output_root,
        spec,
        design=DESIGN,
        source=SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL08B_LINE9_REALISM_DEEP_BACKGROUND_POLICY.json",
        run_prefix="formal08b",
        purpose="Line9-calibrated stronger continuous deep-background development",
        predecessor_case_id=formal06c.CASE_ID,
        changed_factors=[
            "transition-following multiscale continuous non-target cover field only"
        ],
        generator_path=Path(__file__),
        preview_title=(
            f"{FAMILY_ID}: FORMAL06C-locked packet with stronger continuous "
            "deep-cover texture; pre-solver only"
        ),
        profile_builder=formal03.build_profiles,
        bulk_field_builder=build_bulk_field,
        locked_factors=[
            "FORMAL06C basal path arrays",
            "FORMAL06C transition-thickness arrays",
            "GABOR80 source waveform",
            "grid, PML, and acquisition",
            "cover/cap/bedrock constitutive values",
            "strict full/no-basal local-cover mapping",
            "surface and transition-neighbour cover bins",
        ],
        geometry_description=(
            "FORMAL06C basal morphology plus stronger transition-following "
            "multiscale continuous non-target cover texture; no slabs or "
            "discrete bodies"
        ),
    )
    lock_report = formal07b._verify_locked_predecessor(case_dir, spec)
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        indices = handle["data"][:]
    preview_name = "preview_FORMAL06C_vs_FORMAL08B_geometry.png"
    formal08a.preview_predecessor_comparison(
        case_dir / preview_name,
        spec,
        indices,
        candidate_label=(
            "B. FORMAL08B: same basal packet + stronger continuous deep background"
        ),
        title="FORMAL06C -> FORMAL08B one-factor pre-solver comparison",
        subtitle=(
            "Source, materials, grid, acquisition, basal path and transition are "
            "locked; only transition-following deep-cover texture changes"
        ),
    )

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "line9_conditioned": True,
            "strict_line9_holdout_allowed": False,
            "formal_training_allowed": False,
            "promotion_allowed": False,
            "training_block_reason": (
                "Line9 is explicitly used as the measured-realism calibration "
                "target; runtime and human gates are pending"
            ),
            "parameter_provenance": (
                "FORMAL06C locked physics plus a Line9-conditioned decision to "
                "strengthen continuous deep non-target background; no measured "
                "array is read"
            ),
            "predecessor_lock": lock_report,
            "comparison_contract": {
                "predecessor": formal06c.CASE_ID,
                "identical_source": True,
                "identical_material_design": True,
                "identical_grid_and_acquisition": True,
                "identical_basal_and_transition_profiles": True,
                "protected_surface_and_transition_bins_exact": True,
                "changed_factor_group": (
                    "transition_following_multiscale_continuous_deep_background_only"
                ),
                "pre_solver_preview": preview_name,
                "runtime_visual_references": [formal06c.CASE_ID, "Line9"],
                "common_trace_runtime_review_required": True,
            },
            "next_gate": (
                "static and geometry audit, then consecutive-eight full-only "
                "morphology checkpoint against FORMAL06C and Line9 before any "
                "distributed run"
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    policy_path = output_root / "FORMAL08B_LINE9_REALISM_DEEP_BACKGROUND_POLICY.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    candidate_hash = policy.pop("locked_index_array_sha256", None)
    policy.update(
        {
            "candidate_index_array_sha256": candidate_hash,
            "predecessor_case_id": formal06c.CASE_ID,
            "strict_line9_holdout_allowed": False,
            "promotion_allowed": False,
            "conditioning_scope": (
                "Line9 morphology, spectrum, target prominence, depth-distributed "
                "background, and continuous texture are visual calibration "
                "references; no measured array is read by the generator"
            ),
            "locked_factor_groups": [
                "source waveform",
                "constitutive material deck",
                "grid, PML, acquisition, and time window",
                "basal path and transition profiles",
                "surface and transition-neighbour cover bins",
            ],
            "changed_factor_group": (
                "transition_following_multiscale_continuous_deep_background_only"
            ),
            "runtime_state": "not_started",
            "next_gate": (
                "consecutive-eight full-scene traces, reviewed blind against "
                "FORMAL06C and Line9 before any distributed solve"
            ),
        }
    )
    policy_path.write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
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
