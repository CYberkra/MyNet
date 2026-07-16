#!/usr/bin/env python3
"""Generate a FORMAL06C-locked sparse irregular finite-laminae ablation."""

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
FAMILY_ID = "FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN

EVENT_PRIOR = {
    "fit_lines": ["Line3", "Line7", "LineL1"],
    "held_out_lines": ["Line9"],
    "observed_signed_events_per_25m": 8.08116521107172,
    "assumed_signed_lobes_per_physical_lamina": 3.4,
    "physical_laminae_per_25m": 2.376813297374035,
    "provenance_note": (
        "Physical-lamina density is a conservative deconvolution assumption, not a "
        "measured geological count. One thin lamina can generate several signed lobes."
    ),
}

ACTIVE_DEPTH_M = (3.0, 9.4)
LATERAL_CONTEXT_M = 5.0
LENGTH_M = (1.35, 2.45, 4.50)
THICKNESS_M = (0.12, 0.20, 0.33)
BIN_DELTA = (2, 3, 4)
MIN_CENTERLINE_SEPARATION_M = 0.90


def default_spec() -> formal03.Spec:
    return formal06c.default_spec()


def _correlated_roughness(
    rng: np.random.Generator,
    local_x_m: np.ndarray,
    *,
    amplitude_m: float,
) -> np.ndarray:
    span = float(np.ptp(local_x_m))
    knot_spacing = float(rng.uniform(0.55, 1.10))
    knot_count = max(4, int(math.ceil(span / knot_spacing)) + 2)
    knots_x = np.linspace(local_x_m[0], local_x_m[-1], knot_count)
    knots_y = rng.normal(0.0, 1.0, size=knot_count)
    knots_y[0] = 0.0
    knots_y[-1] = 0.0
    rough = np.interp(local_x_m, knots_x, knots_y)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= np.sum(kernel)
    rough = np.convolve(rough, kernel, mode="same")
    rough -= np.mean(rough)
    peak = float(np.max(np.abs(rough)))
    return rough * (amplitude_m / peak) if peak > 0 else rough


def _count_intersections(
    records: list[dict[str, float]], low_m: float, high_m: float
) -> int:
    return sum(
        record["effective_support_x_max_m"] >= low_m
        and record["effective_support_x_min_m"] <= high_m
        for record in records
    )


def add_sparse_irregular_laminae(
    spec: formal03.Spec,
    predecessor_bins: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], dict[str, float]]:
    """Add sparse, non-crossing, endpoint-visible weak physical laminae."""

    x_m = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    y_m = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    depth_m = spec.ground_y_m - y_m
    active_x_min = max(spec.pml_m + spec.dl_m, spec.scan_start_x_m - LATERAL_CONTEXT_M)
    active_x_max = min(
        spec.domain_x_m - spec.pml_m - spec.dl_m,
        spec.scan_start_x_m + spec.scan_span_m + LATERAL_CONTEXT_M,
    )
    requested_count = max(
        3,
        int(
            round(
                EVENT_PRIOR["physical_laminae_per_25m"]
                * (active_x_max - active_x_min)
                / 25.0
            )
        ),
    )
    cover_velocity_m_per_ns = formal03.C0 * 1e-9 / math.sqrt(
        0.5 * (DESIGN.cover_epsilon_min + DESIGN.cover_epsilon_max)
    )
    scan_low = spec.scan_start_x_m
    scan_high = spec.scan_start_x_m + spec.scan_span_m
    first64_high = spec.scan_start_x_m + 63 * spec.trace_spacing_m

    for scene_attempt in range(256):
        rng = np.random.default_rng(seed + scene_attempt)
        candidate = predecessor_bins.copy()
        signed_delta = np.zeros_like(candidate, dtype=np.int16)
        records: list[dict[str, float]] = []
        accepted_centerlines: list[tuple[np.ndarray, np.ndarray]] = []
        attempts = 0
        while len(records) < requested_count and attempts < requested_count * 160:
            attempts += 1
            length_m = float(rng.triangular(*LENGTH_M))
            center_x_m = float(
                rng.uniform(
                    active_x_min + length_m / 2.0,
                    active_x_max - length_m / 2.0,
                )
            )
            lateral = np.abs(x_m - center_x_m) <= length_m / 2.0
            lateral_indices = np.flatnonzero(lateral)
            if lateral_indices.size < 30:
                continue
            local_x = x_m[lateral]
            unit = np.clip((local_x - local_x[0]) / max(local_x[-1] - local_x[0], 1e-12), 0, 1)
            taper = np.square(np.sin(math.pi * unit))
            time_slope_ns_per_m = float(rng.uniform(-3.2, 3.2))
            depth_slope = 0.5 * cover_velocity_m_per_ns * time_slope_ns_per_m
            roughness_amplitude_m = float(rng.triangular(0.03, 0.065, 0.12))
            roughness = _correlated_roughness(
                rng, local_x, amplitude_m=roughness_amplitude_m
            )
            center_depth_m = float(rng.uniform(3.4, 9.0))
            center_depth = (
                center_depth_m
                + depth_slope * (local_x - center_x_m)
                + roughness
            )
            thickness_m = float(rng.triangular(*THICKNESS_M))
            half_thickness = 0.5 * thickness_m * np.power(taper, 0.65)
            if (
                np.min(center_depth - half_thickness) < ACTIVE_DEPTH_M[0]
                or np.max(center_depth + half_thickness) > ACTIVE_DEPTH_M[1]
            ):
                continue
            crossing = False
            for previous_columns, previous_depth in accepted_centerlines:
                common, left_index, right_index = np.intersect1d(
                    lateral_indices,
                    previous_columns,
                    assume_unique=True,
                    return_indices=True,
                )
                if common.size and np.min(
                    np.abs(center_depth[left_index] - previous_depth[right_index])
                ) < MIN_CENTERLINE_SEPARATION_M:
                    crossing = True
                    break
            if crossing:
                continue

            peak_delta = int(round(rng.triangular(*BIN_DELTA)))
            peak_delta *= int(rng.choice((-1, 1)))
            local_delta = np.rint(np.abs(peak_delta) * taper).astype(np.int16)
            local_delta *= np.sign(peak_delta)
            active_columns = local_delta != 0
            if np.count_nonzero(active_columns) < 18:
                continue
            local_mask = np.abs(
                depth_m[None, :] - center_depth[:, None]
            ) <= half_thickness[:, None]
            local_mask &= active_columns[:, None]
            event_cells = int(np.count_nonzero(local_mask))
            if event_cells == 0:
                continue
            delta_field = np.zeros_like(candidate, dtype=np.int16)
            for local_index, column in enumerate(lateral_indices):
                delta_field[column, local_mask[local_index]] = local_delta[local_index]
            event_mask = delta_field != 0
            candidate[event_mask] = np.clip(
                candidate[event_mask].astype(np.int32)
                + delta_field[event_mask].astype(np.int32),
                0,
                spec.cover_bins - 1,
            ).astype(np.int16)
            signed_delta[event_mask] += delta_field[event_mask]
            accepted_centerlines.append((lateral_indices, center_depth))
            effective_columns = lateral_indices[active_columns]
            records.append(
                {
                    "center_x_m": center_x_m,
                    "design_support_x_min_m": center_x_m - length_m / 2.0,
                    "design_support_x_max_m": center_x_m + length_m / 2.0,
                    "effective_support_x_min_m": float(x_m[effective_columns[0]]),
                    "effective_support_x_max_m": float(x_m[effective_columns[-1]]),
                    "center_depth_m": center_depth_m,
                    "length_m": length_m,
                    "thickness_m": thickness_m,
                    "time_slope_ns_per_m": time_slope_ns_per_m,
                    "roughness_amplitude_m": roughness_amplitude_m,
                    "peak_cover_bin_delta": peak_delta,
                    "minimum_centerline_separation_m": MIN_CENTERLINE_SEPARATION_M,
                }
            )

        scan_count = _count_intersections(records, scan_low, scan_high)
        first64_count = _count_intersections(records, scan_low, first64_high)
        first64_endpoints = sum(
            scan_low + 0.30 <= record["effective_support_x_min_m"] <= first64_high - 0.30
            or scan_low + 0.30 <= record["effective_support_x_max_m"] <= first64_high - 0.30
            for record in records
        )
        if (
            len(records) == requested_count
            and scan_count >= 2
            and 1 <= first64_count <= 2
            and first64_endpoints >= 1
        ):
            changed = candidate != predecessor_bins
            stats = {
                "accepted_scene_seed": seed + scene_attempt,
                "scene_attempt": scene_attempt,
                "requested_physical_lamina_count": requested_count,
                "generated_physical_lamina_count": len(records),
                "laminae_intersecting_native_scan": scan_count,
                "laminae_intersecting_first64_native_traces": first64_count,
                "first64_visible_endpoint_count": first64_endpoints,
                "changed_cover_cell_fraction": float(np.mean(changed)),
                "changed_cover_bin_delta_p50": float(
                    np.percentile(np.abs(signed_delta[changed]), 50)
                ),
                "changed_cover_bin_delta_p99": float(
                    np.percentile(np.abs(signed_delta[changed]), 99)
                ),
                "active_x_m": [active_x_min, active_x_max],
                "active_depth_m": list(ACTIVE_DEPTH_M),
            }
            return candidate, signed_delta, records, stats
    raise RuntimeError("could not generate sparse irregular laminae that pass topology gates")


def build_bulk_field(
    spec: formal03.Spec,
    *,
    design: formal06.MaterialDesign = DESIGN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    predecessor, predecessor_bins, predecessor_stats = formal06.build_bulk_field(
        spec, design=design
    )
    candidate_bins, signed_delta, records, event_stats = add_sparse_irregular_laminae(
        spec, predecessor_bins, seed=spec.field_seed + 9203
    )
    candidate = predecessor + signed_delta.astype(np.float32) / max(spec.cover_bins, 1)
    return candidate.astype(np.float32), candidate_bins, {
        "model": "formal06c_plus_sparse_non_crossing_irregular_finite_laminae",
        "predecessor_model": "FORMAL06C smooth weak two-dimensional bulk field",
        "predecessor_statistics": predecessor_stats,
        "event_prior": EVENT_PRIOR,
        "event_records": records,
        "event_statistics": event_stats,
        "length_prior_m": list(LENGTH_M),
        "thickness_prior_m": list(THICKNESS_M),
        "peak_cover_bin_delta_prior": list(BIN_DELTA),
        "minimum_centerline_separation_m": MIN_CENTERLINE_SEPARATION_M,
        "point_targets": 0,
        "vertical_partitions": 0,
        "periodic_slabs": 0,
        "finite_non_target_laminae": len(records),
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
        policy_filename="FORMAL09C_P2_SPARSE_IRREGULAR_LAMINAE_POLICY.json",
        run_prefix="formal09c_p2",
        purpose="native-trace sparse irregular finite-laminae physical ablation",
        predecessor_case_id=formal06c.CASE_ID,
        changed_factors=["sparse non-crossing tapered irregular low-contrast mid-cover laminae only"],
        generator_path=Path(__file__),
        preview_title=f"{FAMILY_ID}: FORMAL06C-locked sparse irregular laminae; pre-solver only",
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
            "FORMAL06C basal morphology plus sparse, tapered, non-crossing, irregular weak mid-cover laminae"
        ),
    )
    lock_report = formal07b._verify_locked_predecessor(case_dir, spec)
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        indices = handle["data"][:]
    preview_name = "preview_FORMAL06C_vs_FORMAL09C_P2_geometry.png"
    formal08a.preview_predecessor_comparison(
        case_dir / preview_name,
        spec,
        indices,
        candidate_label="B. FORMAL09C-P2: sparse non-crossing irregular finite laminae",
        title="FORMAL06C -> FORMAL09C-P2 sparse physical laminae ablation",
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
                "FORMAL06C mechanism remains Line9-conditioned; P2 runtime and human morphology gates are pending"
            ),
            "parameter_provenance": (
                "FORMAL06C locked physics; fold-safe signed-event density is conservatively deconvolved into a physical-lamina prior; no measured coordinates or waveform patches copied"
            ),
            "predecessor_lock": lock_report,
            "comparison_contract": {
                "predecessor": formal06c.CASE_ID,
                "identical_source_and_material_decks": True,
                "identical_grid_acquisition_basal_and_transition": True,
                "changed_factor_group": "sparse_non_crossing_irregular_weak_mid_cover_laminae_only",
                "pre_solver_preview": preview_name,
                "first_runtime_gate": "native-spacing consecutive 64-trace full-scene blind morphology",
                "exact_common_predecessor_run_available": True,
            },
            "next_gate": (
                "static and geometry audit, one-trace smoke, then exact native-spacing 64-trace predecessor comparison"
            ),
        }
    )
    manifest["geometry"]["finite_non_target_laminae_count"] = manifest["geometry"][
        "bulk_field_statistics"
    ]["finite_non_target_laminae"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    policy_path = output_root / "FORMAL09C_P2_SPARSE_IRREGULAR_LAMINAE_POLICY.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy.update(
        {
            "runtime_state": "not_started",
            "formal_training_allowed": False,
            "strict_line9_holdout_allowed": False,
            "source_case_id": CASE_ID,
            "predecessor_case_id": formal06c.CASE_ID,
            "next_gate": "native-spacing consecutive 64-trace exact predecessor comparison",
        }
    )
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    (case_dir / "RUN_COMMANDS.md").write_text(
        f"""# {CASE_ID}

This case is development-only and blocked from training.

```powershell
$case = "data/simulations/v2/00_controls/{CASE_ID}"
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p2_geometry --trace-count 256 --skip-air-reference --geometry-only --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p2_smoke1 --trace-count 1 --skip-air-reference --execute
python scripts/run_native_256_release_pilot.py $case --run-id formal09c_p2_native64_full --trace-count 64 --skip-air-reference --full-scene-only --execute
```

Compare the native 64 full scene against the exact FORMAL06C native 64 run.
Do not start a matched 64-trace control until the blind morphology passes.
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
