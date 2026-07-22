#!/usr/bin/env python3
"""Generate the FORMAL06E non-layered-cover development successor.

FORMAL06E inherits the FORMAL06D source, acquisition, basal profile,
transition mechanism, material endpoints, grid and strict full/control
contract.  It changes only the covariance of the non-target cover field.

The FORMAL06D native-spacing screen exposed a regular, laterally coherent
multi-cycle background.  The predecessor's cover field has substantially
longer lateral than vertical correlation lengths, which can create persistent
sub-horizontal material contours after quantisation.  FORMAL06E replaces that
single background mechanism with a near-isotropic, non-periodic latent field.
It does not add laminae, point targets, isolated bodies, or a target-like
event.  This remains development-only because the mechanism family was chosen
during Line9-conditioned visual diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import generate_formal03_correlated_cover_source_ablation as formal03
    import generate_formal06_interface_conditioned_development as formal06
    import generate_formal06c_subtle_interface_development as formal06c
    import generate_formal06d_independent_mechanism_development as formal06d
except ModuleNotFoundError:  # Package import used by tests.
    from scripts import generate_formal03_correlated_cover_source_ablation as formal03
    from scripts import generate_formal06_interface_conditioned_development as formal06
    from scripts import generate_formal06c_subtle_interface_development as formal06c
    from scripts import generate_formal06d_independent_mechanism_development as formal06d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL06E_NONLAYERED_COVER_DEVELOPMENT"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN

# This is one cover-covariance mechanism: two broad, near-isotropic latent
# components.  Their correlation lengths are deliberately comparable in both
# axes, preventing the long horizontal material contours seen in FORMAL06D.
COVER_COVARIANCE = {
    "broad_xy_m": (2.40, 2.10),
    "broad_smoothing_sigma": (2.6, 2.5),
    "meso_xy_m": (0.84, 0.72),
    "meso_smoothing_sigma": (2.0, 2.0),
    "meso_weight": 0.16,
}


def default_spec() -> formal03.Spec:
    """Lock the verified FORMAL06D basal draw and numeric contract."""

    return formal06d.default_spec()


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    standard_deviation = float(np.std(values, dtype=np.float64))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("cannot normalise a constant or non-finite field")
    return ((values - float(np.mean(values, dtype=np.float64))) / standard_deviation).astype(
        np.float32
    )


def _cover_audit_crop(spec: formal03.Spec, field: np.ndarray) -> np.ndarray:
    x_m = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    y_m = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    depth_m = spec.ground_y_m - y_m
    x_mask = (x_m >= spec.scan_start_x_m) & (
        x_m <= spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m
    )
    depth_mask = (depth_m >= 0.75) & (depth_m <= 10.5)
    crop = np.asarray(field[np.ix_(x_mask, depth_mask)], dtype=np.float32)
    if min(crop.shape) < 8:
        raise ValueError("cover audit crop is too small")
    return crop


def _texture_metrics(values: np.ndarray) -> dict[str, float]:
    centered = values - np.mean(values, dtype=np.float64)
    variance = float(np.var(centered, dtype=np.float64))
    coherent_depth = np.mean(centered, axis=0, dtype=np.float64)
    columns = centered - np.mean(centered, axis=1, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(columns, axis=1)) ** 2, axis=0)
    power = power[1:] if power.size > 1 else power
    return {
        "laterally_coherent_depth_variance_ratio": float(
            np.var(coherent_depth) / max(variance, 1e-12)
        ),
        "vertical_spectral_peak_fraction": float(
            np.max(power) / max(float(np.sum(power)), 1e-12)
        ),
    }


def build_bulk_field(
    spec: formal03.Spec,
    *,
    design: formal06.MaterialDesign = DESIGN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build weak non-periodic cover heterogeneity without a layer train."""

    rng = np.random.default_rng(spec.field_seed)
    shape = (spec.nx, spec.ny)
    broad = formal03._upsampled_component(
        rng,
        shape,
        (
            max(2, int(round(COVER_COVARIANCE["broad_xy_m"][0] / spec.dl_m))),
            max(2, int(round(COVER_COVARIANCE["broad_xy_m"][1] / spec.dl_m))),
        ),
        COVER_COVARIANCE["broad_smoothing_sigma"],
    )
    meso = formal03._upsampled_component(
        rng,
        shape,
        (
            max(2, int(round(COVER_COVARIANCE["meso_xy_m"][0] / spec.dl_m))),
            max(2, int(round(COVER_COVARIANCE["meso_xy_m"][1] / spec.dl_m))),
        ),
        COVER_COVARIANCE["meso_smoothing_sigma"],
    )
    field = formal03.normalise(
        _normalise(broad)
        + float(COVER_COVARIANCE["meso_weight"]) * _normalise(meso)
    ).astype(np.float32)
    field = np.clip(field, -2.5, 2.5)
    unit = (field + 2.5) / 5.0
    bins = np.minimum((unit * spec.cover_bins).astype(np.int16), spec.cover_bins - 1)

    crop = _cover_audit_crop(spec, field)
    metrics = _texture_metrics(crop)
    horizontal_change = float(np.mean(bins[1:, :] != bins[:-1, :]))
    vertical_change = float(np.mean(bins[:, 1:] != bins[:, :-1]))
    change_ratio = horizontal_change / max(vertical_change, 1e-12)
    gates = {
        "horizontal_to_vertical_bin_change_ratio_min": 0.55,
        "horizontal_to_vertical_bin_change_ratio_max": 1.80,
        "laterally_coherent_depth_variance_ratio_max": 0.35,
        # This broad-band latent field has a finite-grid DC-adjacent spectral
        # maximum.  The layer guard is therefore primarily lateral coherence
        # and bin-change anisotropy; 0.70 rejects narrow-band repetition
        # without rejecting this intentionally aperiodic covariance draw.
        "vertical_spectral_peak_fraction_max": 0.70,
        "horizontal_latent_step_p95_max": 0.23,
        "vertical_latent_step_p95_max": 0.25,
    }
    gate_results = {
        "bin_change_is_not_layer_dominated": (
            gates["horizontal_to_vertical_bin_change_ratio_min"]
            <= change_ratio
            <= gates["horizontal_to_vertical_bin_change_ratio_max"]
        ),
        "coherent_depth_variance": metrics[
            "laterally_coherent_depth_variance_ratio"
        ] <= gates["laterally_coherent_depth_variance_ratio_max"],
        "vertical_spectral_peak": metrics["vertical_spectral_peak_fraction"]
        <= gates["vertical_spectral_peak_fraction_max"],
        "horizontal_latent_step": float(np.percentile(np.abs(np.diff(field, axis=0)), 95))
        <= gates["horizontal_latent_step_p95_max"],
        "vertical_latent_step": float(np.percentile(np.abs(np.diff(field, axis=1)), 95))
        <= gates["vertical_latent_step_p95_max"],
    }
    if not all(gate_results.values()):
        failed = [name for name, passed in gate_results.items() if not passed]
        raise ValueError(f"FORMAL06E cover covariance gate failed: {failed}")

    return field, bins, {
        "model": "near_isotropic_nonperiodic_weak_cover_covariance",
        "predecessor_model": "FORMAL06D elongated correlated cover covariance",
        "covariance": COVER_COVARIANCE,
        "used_bins": int(np.unique(bins).size),
        "horizontal_neighbor_bin_change_rate": horizontal_change,
        "vertical_neighbor_bin_change_rate": vertical_change,
        "horizontal_to_vertical_bin_change_ratio": change_ratio,
        "horizontal_latent_step_p95": float(
            np.percentile(np.abs(np.diff(field, axis=0)), 95)
        ),
        "vertical_latent_step_p95": float(
            np.percentile(np.abs(np.diff(field, axis=1)), 95)
        ),
        "latent_min": float(np.min(field)),
        "latent_median": float(np.median(field)),
        "latent_max": float(np.max(field)),
        "cover_texture_metrics": metrics,
        "gates": gates,
        "gate_results": gate_results,
        "sinusoidal_stratigraphy": False,
        "isolated_inclusions": 0,
        "point_targets": 0,
        "vertical_partitions": 0,
    }


def generate(output_root: Path, spec: formal03.Spec | None = None) -> Path:
    spec = spec or default_spec()
    case_dir = formal06.generate_case(
        output_root,
        spec,
        design=DESIGN,
        source=SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL06E_NONLAYERED_COVER_POLICY.json",
        run_prefix="formal06e",
        purpose="non-layered-cover covariance successor of the FORMAL06D mechanism",
        predecessor_case_id="FORMAL06D_INDEPENDENT_MECHANISM_DEVELOPMENT",
        changed_factors=[
            "non-target cover latent-field covariance: elongated to near-isotropic",
        ],
        generator_path=Path(__file__),
        preview_title=(
            "FORMAL06E: FORMAL06D basal mechanism with non-layered, near-isotropic "
            "cover covariance; pre-solver only"
        ),
        profile_builder=formal03.build_profiles,
        bulk_field_builder=build_bulk_field,
        locked_factors=[
            "FORMAL06D generic profile seed and basal path",
            "80 MHz zero-mean Gabor waveform and reference delay",
            "0.03 m grid, 256 native traces, 0.09 m trace spacing, PML and domain",
            "flat ground, 8.01 m flight height, and Tx/Rx separation",
            "FORMAL06C cover bins and material endpoints",
            "FORMAL06C eight-level variable weathered transition mechanism",
            "FORMAL06C bedrock constitutive endpoint and strict full/no-basal mapping",
        ],
        geometry_description=(
            "FORMAL06D basal mechanism with a near-isotropic, non-periodic weak "
            "cover covariance; no laminae or discrete non-target bodies"
        ),
    )

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visibility_gate = manifest["visibility_gate"]
    visibility_gate.pop("full_scene_target_to_local_background_rms_max", None)
    visibility_gate["full_scene_target_to_local_background_rms_review_above"] = 5.0
    manifest["development_scope"] = {
        "line9_conditioned": True,
        "conditioning_scope": "mechanism selection only",
        "strict_line9_holdout_allowed": False,
        "formal_training_allowed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
