#!/usr/bin/env python3
"""Generate the FORMAL08A Line9-calibrated background candidate.

FORMAL08A inherits the exact project-owner accepted FORMAL06C source,
materials, grid, acquisition, basal profile, and transition profile. It changes
only continuous non-target cover geology inside a depth-tapered middle-cover
zone. The added field is aperiodic, anisotropic, and contains no isolated body,
point target, vertical partition, or regular stratigraphic slab.

This is explicitly Line9-conditioned development evidence. It is designed for
measured-realism calibration and is permanently blocked from a strict unseen-
Line9 claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_formal03_correlated_cover_source_ablation as formal03
import generate_formal06_interface_conditioned_development as formal06
import generate_formal06c_subtle_interface_development as formal06c
import generate_formal07b_weak_aperiodic_background_development as formal07b


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
PREDECESSOR_DIR = DEFAULT_OUTPUT / formal06c.CASE_ID
FAMILY_ID = "FORMAL08A_LINE9_REALISM_BACKGROUND_DEVELOPMENT"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN

TEXTURE_COMPONENTS = (
    {
        "name": "broad_aperiodic_2d",
        "weight": 0.22,
        "correlation_xy_m": (4.50, 1.80),
        "smoothing_sigma": (2.8, 2.3),
    },
    {
        "name": "mesoscale_aperiodic_2d",
        "weight": 0.22,
        "correlation_xy_m": (1.80, 0.75),
        "smoothing_sigma": (2.2, 2.0),
    },
    {
        "name": "weak_fine_aperiodic_2d",
        "weight": 0.08,
        "correlation_xy_m": (0.72, 0.36),
        "smoothing_sigma": (1.8, 1.8),
    },
)

ACTIVE_DEPTH_M = {
    "surface_zero_end": 0.75,
    "surface_full_start": 2.00,
    "deep_full_end": 9.50,
    "deep_zero_start": 11.50,
}


def default_spec() -> formal03.Spec:
    return formal06c.default_spec()


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _depth_envelope(spec: formal03.Spec) -> np.ndarray:
    y = (np.arange(spec.ny, dtype=np.float32) + 0.5) * spec.dl_m
    depth = np.float32(spec.ground_y_m) - y
    surface = _smoothstep(
        (depth - ACTIVE_DEPTH_M["surface_zero_end"])
        / (
            ACTIVE_DEPTH_M["surface_full_start"]
            - ACTIVE_DEPTH_M["surface_zero_end"]
        )
    )
    deep = _smoothstep(
        (ACTIVE_DEPTH_M["deep_zero_start"] - depth)
        / (
            ACTIVE_DEPTH_M["deep_zero_start"]
            - ACTIVE_DEPTH_M["deep_full_end"]
        )
    )
    envelope = surface * deep
    envelope[(depth < 0.0) | (depth > ACTIVE_DEPTH_M["deep_zero_start"])] = 0.0
    return envelope.astype(np.float32)


def build_bulk_field(
    spec: formal03.Spec,
    *,
    design: formal06.MaterialDesign = DESIGN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Add moderate continuous texture while preserving the basal neighbourhood."""

    predecessor, predecessor_bins, predecessor_stats = formal06.build_bulk_field(
        spec, design=design
    )
    shape = (spec.nx, spec.ny)
    rng = np.random.default_rng(spec.field_seed + 8103)
    envelope = _depth_envelope(spec)
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
        perturbation += weight * values * envelope[None, :]
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

    y = (np.arange(spec.ny, dtype=np.float32) + 0.5) * spec.dl_m
    depth = np.float32(spec.ground_y_m) - y
    protected = (depth <= ACTIVE_DEPTH_M["surface_zero_end"]) | (
        depth >= ACTIVE_DEPTH_M["deep_zero_start"]
    )
    protected_bins_exact = bool(
        np.array_equal(bins[:, protected], predecessor_bins[:, protected])
    )

    gates = {
        "predecessor_latent_correlation_min": 0.90,
        "perturbation_rms_min": 0.20,
        "perturbation_rms_max": 0.42,
        "laterally_coherent_depth_variance_ratio_max": 0.60,
        "vertical_spectral_peak_fraction_max": 0.97,
        "bin_delta_p99_max": 5.0,
        "protected_surface_and_basal_bins_exact": True,
    }
    bin_delta_p99 = float(np.percentile(bin_delta, 99.0))
    gate_results = {
        "predecessor_latent_correlation": correlation
        >= gates["predecessor_latent_correlation_min"],
        "perturbation_rms": gates["perturbation_rms_min"]
        <= delta_rms
        <= gates["perturbation_rms_max"],
        "coherent_depth_variance": candidate_metrics[
            "laterally_coherent_depth_variance_ratio"
        ]
        <= gates["laterally_coherent_depth_variance_ratio_max"],
        "vertical_spectral_peak": candidate_metrics[
            "vertical_spectral_peak_fraction"
        ]
        <= gates["vertical_spectral_peak_fraction_max"],
        "bin_delta_p99": bin_delta_p99 <= gates["bin_delta_p99_max"],
        "protected_surface_and_basal_bins_exact": protected_bins_exact,
    }
    if not all(gate_results.values()):
        failed = [name for name, passed in gate_results.items() if not passed]
        diagnostics = {
            "correlation": correlation,
            "perturbation_rms": delta_rms,
            "bin_delta_p99": bin_delta_p99,
            "baseline_texture": baseline_metrics,
            "candidate_texture": candidate_metrics,
        }
        raise ValueError(
            f"FORMAL08A background gate failed: {failed}; {diagnostics}"
        )

    stats = {
        "model": "formal06c_plus_depth_tapered_multiscale_aperiodic_cover",
        "predecessor_model": "FORMAL06C smooth weak two-dimensional bulk field",
        "line9_calibration_scope": (
            "background continuity, target prominence and non-target texture; "
            "no measured array is read by the generator"
        ),
        "active_depth_m": ACTIVE_DEPTH_M,
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
        "changed_cover_bin_fraction": float(np.mean(bins != predecessor_bins)),
        "cover_bin_delta_p99": bin_delta_p99,
        "protected_surface_and_basal_bins_exact": protected_bins_exact,
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


def preview_predecessor_comparison(
    path: Path,
    spec: formal03.Spec,
    candidate_indices: np.ndarray,
) -> None:
    with h5py.File(PREDECESSOR_DIR / "geology_indices.h5", "r") as handle:
        predecessor_indices = handle["data"][:]
    rows = formal06.material_rows(spec, control=False, design=DESIGN)
    predecessor = formal07b._epsilon_crop(spec, predecessor_indices, rows)
    candidate = formal07b._epsilon_crop(spec, candidate_indices, rows)
    difference = np.nan_to_num(candidate - predecessor, nan=0.0)
    finite = np.isfinite(predecessor) | np.isfinite(candidate)
    lo = min(float(np.nanmin(predecessor)), float(np.nanmin(candidate)))
    hi = max(float(np.nanmax(predecessor)), float(np.nanmax(candidate)))
    delta_limit = max(float(np.percentile(np.abs(difference[finite]), 99)), 1e-4)

    def colour(values: np.ndarray) -> np.ndarray:
        unit = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        rgb = np.stack(
            (
                35 + 210 * unit,
                55 + 175 * (1.0 - np.abs(unit - 0.5) * 1.7),
                230 - 185 * unit,
            ),
            axis=-1,
        )
        rgb[~np.isfinite(values)] = (246, 248, 251)
        return rgb.astype(np.uint8)

    def diverging(values: np.ndarray) -> np.ndarray:
        unit = np.clip(0.5 + 0.5 * values / delta_limit, 0.0, 1.0)
        return np.stack(
            (
                35 + 220 * unit,
                245 - 350 * np.abs(unit - 0.5),
                255 - 220 * unit,
            ),
            axis=-1,
        ).astype(np.uint8)

    canvas = Image.new("RGB", (1900, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panels = (
        (70, 105, 1830, 475, colour(predecessor), "A. FORMAL06C locked baseline"),
        (
            70,
            565,
            1830,
            935,
            colour(candidate),
            "B. FORMAL08A: same basal packet + moderate continuous cover texture",
        ),
        (
            70,
            1025,
            1830,
            1395,
            diverging(difference),
            f"C. Enhanced material delta (+/-{delta_limit:.3f} epsilon_r)",
        ),
    )
    for left, top, right, bottom, values, label in panels:
        image = Image.fromarray(np.flipud(values), mode="RGB").resize(
            (right - left, bottom - top), Image.Resampling.BILINEAR
        )
        canvas.paste(image, (left, top))
        draw.rectangle((left, top, right, bottom), outline="black", width=2)
        draw.text((left, top - 24), label, fill="black", font=font)
    draw.text(
        (70, 25),
        "FORMAL06C -> FORMAL08A one-factor pre-solver comparison",
        fill="black",
        font=font,
    )
    draw.text(
        (70, 52),
        "Source, materials, grid, acquisition, basal path and transition are locked; only middle-cover texture changes",
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
        policy_filename="FORMAL08A_LINE9_REALISM_BACKGROUND_POLICY.json",
        run_prefix="formal08a",
        purpose="Line9-calibrated continuous non-target background development",
        predecessor_case_id=formal06c.CASE_ID,
        changed_factors=[
            "depth-tapered multiscale aperiodic non-target cover texture only"
        ],
        generator_path=Path(__file__),
        preview_title=(
            f"{FAMILY_ID}: FORMAL06C-locked basal packet with moderate "
            "continuous cover texture; pre-solver only"
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
            "surface and basal-neighbour cover bins",
        ],
        geometry_description=(
            "FORMAL06C basal morphology plus depth-tapered continuous "
            "multiscale non-target cover texture; no slabs or discrete bodies"
        ),
    )
    lock_report = formal07b._verify_locked_predecessor(case_dir, spec)
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        indices = handle["data"][:]
    preview_name = "preview_FORMAL06C_vs_FORMAL08A_geometry.png"
    preview_predecessor_comparison(case_dir / preview_name, spec, indices)

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["line9_conditioned"] = True
    manifest["strict_line9_holdout_allowed"] = False
    manifest["formal_training_allowed"] = False
    manifest["promotion_allowed"] = False
    manifest["training_block_reason"] = (
        "Line9 is explicitly used as the measured-realism calibration target; "
        "runtime and human gates are also pending"
    )
    manifest["parameter_provenance"] = (
        "FORMAL06C locked physics plus a Line9-conditioned decision to increase "
        "continuous non-target background; no measured array is read"
    )
    manifest["predecessor_lock"] = lock_report
    manifest["comparison_contract"] = {
        "predecessor": formal06c.CASE_ID,
        "identical_source": True,
        "identical_material_design": True,
        "identical_grid_and_acquisition": True,
        "identical_basal_and_transition_profiles": True,
        "protected_surface_and_basal_bins_exact": True,
        "changed_factor_group": (
            "depth_tapered_multiscale_aperiodic_non_target_background_only"
        ),
        "pre_solver_preview": preview_name,
        "runtime_visual_references": [formal06c.CASE_ID, "Line9"],
        "common_trace_runtime_review_required": True,
    }
    manifest["next_gate"] = (
        "static and geometry audit, then consecutive-eight full-only morphology "
        "checkpoint against FORMAL06C and Line9 before any distributed run"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    policy_path = output_root / "FORMAL08A_LINE9_REALISM_BACKGROUND_POLICY.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    candidate_hash = policy.pop("locked_index_array_sha256", None)
    policy.update(
        {
            "candidate_index_array_sha256": candidate_hash,
            "predecessor_case_id": formal06c.CASE_ID,
            "strict_line9_holdout_allowed": False,
            "promotion_allowed": False,
            "conditioning_scope": (
                "Line9 morphology, spectrum, target prominence, and continuous "
                "background character are visual calibration references; no "
                "measured array is read by the generator"
            ),
            "locked_factor_groups": [
                "source waveform",
                "constitutive material deck",
                "grid, PML, acquisition, and time window",
                "basal path and transition profiles",
                "surface and basal-neighbour cover bins",
            ],
            "changed_factor_group": (
                "depth_tapered_multiscale_aperiodic_non_target_background_only"
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
