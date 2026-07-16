#!/usr/bin/env python3
"""Generate a FORMAL06C-locked finite-laminae physical ablation."""

from __future__ import annotations

import argparse
import json
import math
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
FAMILY_ID = "FORMAL09C_P1_DENSE_PHYSICAL_FINITE_LAMINAE"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN

# Fold-local statistics: Line3, Line7, and LineL1 only. Coordinates and
# measured waveform patches are deliberately absent.
EVENT_PRIOR = {
    "fit_lines": ["Line3", "Line7", "LineL1"],
    "held_out_lines": ["Line9"],
    "events_per_25m": 8.08116521107172,
    "length_q10_q50_q90_m": [1.7904300403592928, 2.97126647989365, 11.855676351563504],
    "time_slope_q10_q50_q90_ns_per_m": [-5.494420731664983, -0.21031678179812055, 5.280414975286209],
}

ACTIVE_DEPTH_M = (2.5, 10.5)
LATERAL_CONTEXT_M = 5.0
THICKNESS_M = (0.18, 0.30, 0.48)
BIN_DELTA = (4, 6, 9)
MAX_LENS_OVERLAP_FRACTION = 0.10


def default_spec() -> formal03.Spec:
    return formal06c.default_spec()


def _sample_quantile_span(
    rng: np.random.Generator, quantiles: list[float]
) -> float:
    probability = float(rng.uniform(0.10, 0.90))
    return float(
        np.interp(probability, (0.10, 0.50, 0.90), tuple(quantiles))
    )


def _event_count_in_interval(
    records: list[dict[str, float]], low_m: float, high_m: float
) -> int:
    return sum(
        record["support_x_max_m"] >= low_m
        and record["support_x_min_m"] <= high_m
        for record in records
    )


def add_finite_laminae(
    spec: formal03.Spec,
    predecessor_bins: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], dict[str, float]]:
    """Add finite, tapered, low-contrast ribbons to middle-cover bins."""

    x_m = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    y_m = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    depth_m = spec.ground_y_m - y_m
    active_x_min = max(
        spec.pml_m + spec.dl_m,
        spec.scan_start_x_m - LATERAL_CONTEXT_M,
    )
    active_x_max = min(
        spec.domain_x_m - spec.pml_m - spec.dl_m,
        spec.scan_start_x_m + spec.scan_span_m + LATERAL_CONTEXT_M,
    )
    active_span_m = active_x_max - active_x_min
    requested_count = max(
        3,
        int(round(EVENT_PRIOR["events_per_25m"] * active_span_m / 25.0)),
    )
    cover_velocity_m_per_ns = formal03.C0 * 1e-9 / math.sqrt(
        0.5 * (DESIGN.cover_epsilon_min + DESIGN.cover_epsilon_max)
    )
    first64_end_m = spec.scan_start_x_m + 63 * spec.trace_spacing_m

    for scene_attempt in range(64):
        rng = np.random.default_rng(seed + scene_attempt)
        candidate = predecessor_bins.copy()
        occupancy = np.zeros(candidate.shape, dtype=bool)
        signed_delta = np.zeros(candidate.shape, dtype=np.int16)
        records: list[dict[str, float]] = []
        attempts = 0
        while len(records) < requested_count and attempts < requested_count * 100:
            attempts += 1
            length_m = float(
                np.clip(
                    _sample_quantile_span(
                        rng, EVENT_PRIOR["length_q10_q50_q90_m"]
                    ),
                    1.8,
                    12.0,
                )
            )
            center_x_m = float(
                rng.uniform(
                    active_x_min + length_m / 2.0,
                    active_x_max - length_m / 2.0,
                )
            )
            time_slope_ns_per_m = float(
                np.clip(
                    _sample_quantile_span(
                        rng, EVENT_PRIOR["time_slope_q10_q50_q90_ns_per_m"]
                    ),
                    -5.5,
                    5.5,
                )
            )
            depth_slope_m_per_m = (
                0.5 * cover_velocity_m_per_ns * time_slope_ns_per_m
            )
            depth_curvature_per_m = float(
                np.clip(rng.normal(0.0, 0.004), -0.010, 0.010)
            )
            thickness_m = float(rng.triangular(*THICKNESS_M))
            center_depth_m = float(rng.uniform(3.0, 9.7))
            bin_delta = int(
                round(rng.triangular(BIN_DELTA[0], BIN_DELTA[1], BIN_DELTA[2]))
            )
            bin_delta *= int(rng.choice((-1, 1)))

            lateral = np.abs(x_m - center_x_m) <= length_m / 2.0
            if np.count_nonzero(lateral) < 20:
                continue
            local_x = (x_m[lateral] - center_x_m) / (length_m / 2.0)
            taper = np.sqrt(np.clip(np.cos(0.5 * math.pi * local_x), 0.0, 1.0))
            center_depth = (
                center_depth_m
                + depth_slope_m_per_m * (x_m[lateral] - center_x_m)
                + depth_curvature_per_m * np.square(x_m[lateral] - center_x_m)
            )
            half_thickness = 0.5 * thickness_m * taper
            if (
                np.min(center_depth - half_thickness) < ACTIVE_DEPTH_M[0]
                or np.max(center_depth + half_thickness) > ACTIVE_DEPTH_M[1]
            ):
                continue

            local_mask = np.abs(
                depth_m[None, :] - center_depth[:, None]
            ) <= half_thickness[:, None]
            event_mask = np.zeros_like(occupancy)
            event_mask[np.flatnonzero(lateral)] = local_mask
            event_cells = int(np.count_nonzero(event_mask))
            if event_cells == 0:
                continue
            overlap_fraction = float(
                np.count_nonzero(event_mask & occupancy) / event_cells
            )
            if overlap_fraction > MAX_LENS_OVERLAP_FRACTION:
                continue

            lateral_delta = np.maximum(
                1,
                np.rint(np.abs(bin_delta) * np.square(taper)).astype(np.int16),
            ) * np.sign(bin_delta)
            delta_field = np.zeros_like(candidate, dtype=np.int16)
            lateral_indices = np.flatnonzero(lateral)
            for local_index, column in enumerate(lateral_indices):
                delta_field[column, local_mask[local_index]] = lateral_delta[
                    local_index
                ]
            candidate[event_mask] = np.clip(
                candidate[event_mask].astype(np.int32)
                + delta_field[event_mask].astype(np.int32),
                0,
                spec.cover_bins - 1,
            ).astype(np.int16)
            signed_delta[event_mask] += delta_field[event_mask]
            occupancy |= event_mask
            records.append(
                {
                    "center_x_m": center_x_m,
                    "support_x_min_m": center_x_m - length_m / 2.0,
                    "support_x_max_m": center_x_m + length_m / 2.0,
                    "center_depth_m": center_depth_m,
                    "length_m": length_m,
                    "thickness_m": thickness_m,
                    "time_slope_ns_per_m": time_slope_ns_per_m,
                    "depth_slope_m_per_m": depth_slope_m_per_m,
                    "depth_curvature_per_m": depth_curvature_per_m,
                    "peak_cover_bin_delta": bin_delta,
                    "overlap_fraction": overlap_fraction,
                }
            )

        full_scan_count = _event_count_in_interval(
            records,
            spec.scan_start_x_m,
            spec.scan_start_x_m + spec.scan_span_m,
        )
        first64_count = _event_count_in_interval(
            records, spec.scan_start_x_m, first64_end_m
        )
        if (
            len(records) == requested_count
            and full_scan_count >= 7
            and 2 <= first64_count <= 4
        ):
            changed = candidate != predecessor_bins
            stats = {
                "accepted_scene_seed": seed + scene_attempt,
                "scene_attempt": scene_attempt,
                "requested_event_count": requested_count,
                "generated_event_count": len(records),
                "events_intersecting_native_scan": full_scan_count,
                "events_intersecting_first64_native_traces": first64_count,
                "first64_native_event_count_gate": [2, 4],
                "changed_cover_cell_fraction": float(np.mean(changed)),
                "changed_cover_bin_delta_p50": float(
                    np.percentile(np.abs(signed_delta[changed]), 50)
                ),
                "changed_cover_bin_delta_p99": float(
                    np.percentile(np.abs(signed_delta[changed]), 99)
                ),
                "cover_velocity_m_per_ns_for_dip_conversion": cover_velocity_m_per_ns,
                "active_x_m": [active_x_min, active_x_max],
                "active_depth_m": list(ACTIVE_DEPTH_M),
            }
            return candidate, signed_delta, records, stats
    raise RuntimeError("could not generate a finite-laminae scene that passes coverage gates")


def build_bulk_field(
    spec: formal03.Spec,
    *,
    design: formal06.MaterialDesign = DESIGN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    predecessor, predecessor_bins, predecessor_stats = formal06.build_bulk_field(
        spec, design=design
    )
    candidate_bins, signed_delta, records, event_stats = add_finite_laminae(
        spec, predecessor_bins, seed=spec.field_seed + 9103
    )
    candidate = predecessor + signed_delta.astype(np.float32) / max(
        spec.cover_bins, 1
    )
    stats = {
        "model": "formal06c_plus_finite_tapered_mid_cover_laminae",
        "predecessor_model": "FORMAL06C smooth weak two-dimensional bulk field",
        "predecessor_statistics": predecessor_stats,
        "event_prior": EVENT_PRIOR,
        "event_records": records,
        "event_statistics": event_stats,
        "thickness_prior_m": list(THICKNESS_M),
        "peak_cover_bin_delta_prior": list(BIN_DELTA),
        "target_corridor_protected": True,
        "surface_corridor_protected": True,
        "point_targets": 0,
        "vertical_partitions": 0,
        "periodic_slabs": 0,
        "compact_anomaly_bodies": 0,
        "finite_non_target_laminae": len(records),
    }
    return candidate.astype(np.float32), candidate_bins, stats


def generate(output_root: Path, spec: formal03.Spec | None = None) -> Path:
    spec = spec or default_spec()
    case_dir = formal06.generate_case(
        output_root,
        spec,
        design=DESIGN,
        source=SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL09C_P1_FINITE_LAMINAE_POLICY.json",
        run_prefix="formal09c_p1",
        purpose="native-trace physical finite-laminae morphology ablation",
        predecessor_case_id=formal06c.CASE_ID,
        changed_factors=["finite tapered low-contrast mid-cover laminae only"],
        generator_path=Path(__file__),
        preview_title=(
            f"{FAMILY_ID}: FORMAL06C-locked packet with finite mid-cover laminae; pre-solver only"
        ),
        profile_builder=formal03.build_profiles,
        bulk_field_builder=build_bulk_field,
        locked_factors=[
            "FORMAL06C basal path arrays",
            "FORMAL06C transition-thickness arrays",
            "GABOR80 source waveform",
            "grid, PML, acquisition, and time window",
            "cover/cap/bedrock constitutive material deck",
        ],
        geometry_description=(
            "FORMAL06C basal morphology plus finite, tapered, gently dipping low-contrast mid-cover laminae"
        ),
    )
    lock_report = formal07b._verify_locked_predecessor(case_dir, spec)
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        indices = handle["data"][:]
    preview_name = "preview_FORMAL06C_vs_FORMAL09C_P1_geometry.png"
    formal08a.preview_predecessor_comparison(
        case_dir / preview_name,
        spec,
        indices,
        candidate_label="B. FORMAL09C-P1: same basal packet + finite tapered mid-cover laminae",
        title="FORMAL06C -> FORMAL09C-P1 physical finite-laminae ablation",
        subtitle=(
            "Source, materials, grid, acquisition, basal path and transition are locked; enhanced delta is not amplitude evidence"
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
                "FORMAL06C source/material mechanism remains Line9-conditioned; runtime and human morphology gates are pending"
            ),
            "parameter_provenance": (
                "FORMAL06C locked physics; finite-event length/slope distributions fitted on Line3/Line7/LineL1 only; no measured coordinates or waveform patches copied"
            ),
            "predecessor_lock": lock_report,
            "comparison_contract": {
                "predecessor": formal06c.CASE_ID,
                "identical_source_and_material_decks": True,
                "identical_grid_acquisition_basal_and_transition": True,
                "changed_factor_group": "finite_tapered_low_contrast_mid_cover_laminae_only",
                "pre_solver_preview": preview_name,
                "first_runtime_gate": "native-spacing consecutive 64-trace full-scene blind morphology",
                "sparse_stride8_topology_gate_allowed": False,
            },
            "next_gate": (
                "static and geometry audit, then native-spacing consecutive 64-trace full-scene blind morphology before a matched pair"
            ),
        }
    )
    manifest["geometry"]["finite_non_target_laminae_count"] = manifest[
        "geometry"
    ]["bulk_field_statistics"]["finite_non_target_laminae"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    policy_path = output_root / "FORMAL09C_P1_FINITE_LAMINAE_POLICY.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy.update(
        {
            "runtime_state": "not_started",
            "formal_training_allowed": False,
            "strict_line9_holdout_allowed": False,
            "source_case_id": CASE_ID,
            "predecessor_case_id": formal06c.CASE_ID,
            "next_gate": (
                "native-spacing consecutive 64-trace full-scene blind morphology; no stride-8 interpolation"
            ),
        }
    )
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    (case_dir / "RUN_COMMANDS.md").write_text(
        f"""# {CASE_ID}

This case is development-only and blocked from training.

```powershell
$case = "data/simulations/v2/00_controls/{CASE_ID}"
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p1_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p1_smoke1 --trace-count 1 --skip-air-reference --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p1_native64_full --trace-count 64 --skip-air-reference --full-scene-only --execute
```

Do not use a stride-8 subset to judge finite-event topology. Run the exact
matched pair only after the native 64-trace blind image passes.
""",
        encoding="utf-8",
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
