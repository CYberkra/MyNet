#!/usr/bin/env python3
"""Generate the FORMAL06G terrain/acquisition development diagnosis.

FORMAL06G inherits FORMAL06F's one-cap transition, FORMAL06D's cover and
basal mechanisms, and the same source/material/grid contract.  Its only
changed physical factor is coupled ground/acquisition geometry: a bounded,
non-periodic terrain profile is introduced while the aerial transmitter and
receiver remain at a fixed absolute elevation.  Thus basal depth is preserved
relative to local ground and the AGL variation is explicit and auditable.
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
    import generate_formal06f_single_cap_transition_development as formal06f
except ModuleNotFoundError:  # Package import used by tests.
    from scripts import generate_formal03_correlated_cover_source_ablation as formal03
    from scripts import generate_formal06_interface_conditioned_development as formal06
    from scripts import generate_formal06c_subtle_interface_development as formal06c
    from scripts import generate_formal06d_independent_mechanism_development as formal06d
    from scripts import generate_formal06f_single_cap_transition_development as formal06f


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL06G_TERRAIN_ACQUISITION_DEVELOPMENT"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN
TERRAIN = {
    "broad_correlation_m": 13.0,
    "meso_correlation_m": 3.4,
    "broad_amplitude_m": 0.36,
    "meso_amplitude_m": 0.12,
    "agl_min_m": 7.15,
    "agl_max_m": 8.85,
}


def default_spec() -> formal03.Spec:
    return formal06f.default_spec()


def build_terrain_profiles(
    spec: formal03.Spec,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Keep local basal depth fixed while varying local ground and AGL."""

    profile, basal_stats = formal03.build_profiles(spec)
    x_m = np.asarray(profile["full_x_m"], dtype=np.float64)
    rng = np.random.default_rng(spec.profile_seed + 6067)
    terrain = (
        TERRAIN["broad_amplitude_m"]
        * formal03._correlated_1d(
            rng, spec.nx, TERRAIN["broad_correlation_m"], spec.dl_m
        )
        + TERRAIN["meso_amplitude_m"]
        * formal03._correlated_1d(
            rng, spec.nx, TERRAIN["meso_correlation_m"], spec.dl_m
        )
    )
    terrain -= float(np.median(terrain))
    terrain = np.clip(terrain, -0.72, 0.72)
    ground = np.asarray(spec.ground_y_m + terrain, dtype=np.float32)
    agl = np.asarray(spec.source_y_m - ground, dtype=np.float64)
    terrain_stats = formal03.crop_statistics(x_m, terrain, spec)
    if not (TERRAIN["agl_min_m"] <= float(np.min(agl)) <= float(np.max(agl)) <= TERRAIN["agl_max_m"]):
        raise ValueError("FORMAL06G AGL range falls outside its acquisition contract")
    if float(terrain_stats["abs_slope_p95"]) > 0.16:
        raise ValueError("FORMAL06G terrain slope exceeds the bounded geometry contract")

    basal_depth = np.asarray(profile["full_basal_depth_m"], dtype=np.float32)
    transition = np.asarray(profile["full_transition_thickness_m"], dtype=np.float32)
    profile["full_ground_y_m"] = ground
    profile["full_basal_y_m"] = ground - basal_depth
    profile["full_transition_top_y_m"] = ground - basal_depth + transition
    return profile, {
        **basal_stats,
        "terrain_seed": spec.profile_seed + 6067,
        "terrain_range_m": float(np.ptp(terrain)),
        "terrain_abs_slope_p95": float(terrain_stats["abs_slope_p95"]),
        "flight_height_agl_min_m": float(np.min(agl)),
        "flight_height_agl_median_m": float(np.median(agl)),
        "flight_height_agl_max_m": float(np.max(agl)),
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
        policy_filename="FORMAL06G_TERRAIN_ACQUISITION_POLICY.json",
        run_prefix="formal06g",
        purpose="terrain/acquisition geometry diagnosis of full-window coherence",
        predecessor_case_id="FORMAL06F_SINGLE_CAP_TRANSITION_DEVELOPMENT",
        changed_factors=[
            "coupled ground/acquisition geometry: flat ground to bounded non-periodic terrain under fixed absolute aerial traverse",
        ],
        generator_path=Path(__file__),
        preview_title=(
            "FORMAL06G: FORMAL06F mechanism with bounded terrain and fixed absolute "
            "aerial traverse; pre-solver only"
        ),
        profile_builder=build_terrain_profiles,
        bulk_field_builder=formal06.build_bulk_field,
        material_rows_builder=formal06f.one_cap_material_rows,
        locked_factors=[
            "FORMAL06D generic basal profile and cover-field seeds",
            "FORMAL06D cover covariance and material endpoints",
            "FORMAL06F single weathered-cap material mapping",
            "80 MHz zero-mean Gabor waveform and reference delay",
            "0.03 m grid, 256 native traces, 0.09 m trace spacing, PML and domain",
            "fixed absolute Tx/Rx elevation and separation",
            "basal depth and transition thickness relative to local ground",
            "strict full/no-basal shared indexed geometry and local-cover control mapping",
        ],
        geometry_description=(
            "FORMAL06F basal mechanism under a bounded non-periodic terrain profile "
            "and fixed absolute aerial traverse; no discrete non-target bodies"
        ),
        flat_ground=False,
        terrain_stage="bounded non-periodic terrain with fixed absolute aerial traverse",
    )

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visibility_gate = manifest["visibility_gate"]
    visibility_gate.pop("full_scene_target_to_local_background_rms_max", None)
    visibility_gate["full_scene_target_to_local_background_rms_review_above"] = 5.0
    manifest["development_scope"] = {
        "line9_conditioned": True,
        "conditioning_scope": "mechanism diagnosis only",
        "strict_line9_holdout_allowed": False,
        "formal_training_allowed": False,
    }
    manifest["terrain_acquisition_diagnosis"] = {
        "predecessor_flat_ground": True,
        "candidate_fixed_absolute_source_elevation_m": spec.source_y_m,
        "terrain_contract": TERRAIN,
        "basal_depth_reference": "local ground",
        "hypothesis": (
            "perfect flat-ground/fixed-height geometry preserves unrealistically "
            "coherent ground and multiple events across all traces"
        ),
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
