#!/usr/bin/env python3
"""Generate the FORMAL07B weak-aperiodic-background development candidate.

FORMAL07B is a strict one-factor-group successor to the human-accepted
FORMAL06C morphology baseline. It preserves the exact source, grid,
acquisition, materials, basal path, and transition thickness. The only change
is a weak two-dimensional aperiodic perturbation of the non-target cover field.

No sinusoidal stratigraphy, isolated body, point target, or vertical partition
is introduced. This family remains Line9-conditioned development evidence and
is permanently blocked from formal training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_formal03_correlated_cover_source_ablation as formal03
import generate_formal06_interface_conditioned_development as formal06
import generate_formal06c_subtle_interface_development as formal06c


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
PREDECESSOR_DIR = DEFAULT_OUTPUT / formal06c.CASE_ID
FAMILY_ID = "FORMAL07B_WEAK_APERIODIC_BACKGROUND_DEVELOPMENT"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN
TEXTURE_WEIGHTS = {
    "weak_mesoscale_2d": 0.105,
    "weak_fine_2d": 0.035,
}


def default_spec() -> formal03.Spec:
    return formal06c.default_spec()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _normalised_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="ascii").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("cannot normalise a constant or non-finite field")
    return ((values - mean) / std).astype(np.float32)


def _cover_audit_crop(spec: formal03.Spec, field: np.ndarray) -> np.ndarray:
    """Return an acquisition-aligned rectangular crop safely above the target."""

    x = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    y = (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dl_m
    depth = spec.ground_y_m - y
    x_mask = (x >= spec.scan_start_x_m) & (
        x <= spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m
    )
    depth_mask = (depth >= 0.75) & (depth <= 10.5)
    crop = field[np.ix_(x_mask, depth_mask)]
    if min(crop.shape) < 8:
        raise ValueError("cover audit crop is too small")
    return np.asarray(crop, dtype=np.float32)


def _texture_metrics(values: np.ndarray) -> dict[str, float]:
    centered = values - np.mean(values, dtype=np.float64)
    variance = float(np.var(centered, dtype=np.float64))
    coherent_depth = np.mean(centered, axis=0, dtype=np.float64)
    coherence = float(np.var(coherent_depth) / max(variance, 1e-12))

    columns = centered - np.mean(centered, axis=1, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(columns, axis=1)) ** 2, axis=0)
    if power.size > 1:
        power = power[1:]
    periodic_peak = float(np.max(power) / max(float(np.sum(power)), 1e-12))
    return {
        "laterally_coherent_depth_variance_ratio": coherence,
        "vertical_spectral_peak_fraction": periodic_peak,
    }


def build_bulk_field(
    spec: formal03.Spec,
    *,
    design: formal06.MaterialDesign = DESIGN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Add weak non-periodic 2-D texture to the exact FORMAL06C bulk field."""

    predecessor, predecessor_bins, predecessor_stats = formal06.build_bulk_field(
        spec, design=design
    )
    shape = (spec.nx, spec.ny)
    rng = np.random.default_rng(spec.field_seed + 7102)
    mesoscale = formal03._upsampled_component(
        rng,
        shape,
        (
            max(2, int(round(1.20 / spec.dl_m))),
            max(2, int(round(0.90 / spec.dl_m))),
        ),
        (2.3, 2.2),
    )
    fine = formal03._upsampled_component(
        rng,
        shape,
        (
            max(2, int(round(0.54 / spec.dl_m))),
            max(2, int(round(0.45 / spec.dl_m))),
        ),
        (1.9, 1.9),
    )
    mesoscale = _normalise(mesoscale)
    fine = _normalise(fine)
    perturbation = (
        TEXTURE_WEIGHTS["weak_mesoscale_2d"] * mesoscale
        + TEXTURE_WEIGHTS["weak_fine_2d"] * fine
    )
    candidate = np.clip(predecessor + perturbation, -2.5, 2.5).astype(np.float32)
    unit = (candidate + 2.5) / 5.0
    bins = np.minimum(
        (unit * spec.cover_bins).astype(np.int16), spec.cover_bins - 1
    )

    baseline_crop = _cover_audit_crop(spec, predecessor)
    candidate_crop = _cover_audit_crop(spec, candidate)
    delta_crop = candidate_crop - baseline_crop
    baseline_metrics = _texture_metrics(baseline_crop)
    candidate_metrics = _texture_metrics(candidate_crop)
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

    gates = {
        "predecessor_latent_correlation_min": 0.985,
        "perturbation_rms_min": 0.07,
        "perturbation_rms_max": 0.16,
        "coherent_depth_variance_increase_max": 0.02,
        "vertical_spectral_peak_increase_max": 0.04,
        "bin_delta_p99_max": 3.0,
    }
    gate_results = {
        "predecessor_latent_correlation": correlation >= gates[
            "predecessor_latent_correlation_min"
        ],
        "perturbation_rms": gates["perturbation_rms_min"]
        <= delta_rms
        <= gates["perturbation_rms_max"],
        "coherent_depth_variance": candidate_metrics[
            "laterally_coherent_depth_variance_ratio"
        ]
        <= baseline_metrics["laterally_coherent_depth_variance_ratio"]
        + gates["coherent_depth_variance_increase_max"],
        "vertical_spectral_peak": candidate_metrics[
            "vertical_spectral_peak_fraction"
        ]
        <= baseline_metrics["vertical_spectral_peak_fraction"]
        + gates["vertical_spectral_peak_increase_max"],
        "bin_delta_p99": bin_delta_p99 <= gates["bin_delta_p99_max"],
    }
    if not all(gate_results.values()):
        failed = [name for name, passed in gate_results.items() if not passed]
        raise ValueError(f"FORMAL07B weak-background gate failed: {failed}")

    stats = {
        "model": "formal06c_plus_weak_aperiodic_two_dimensional_texture",
        "predecessor_model": "FORMAL06C smooth weak two-dimensional bulk field",
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
        "component_weights": TEXTURE_WEIGHTS,
        "component_correlation_scales_m": {
            "mesoscale_xy": [1.20, 0.90],
            "fine_xy": [0.54, 0.45],
        },
        "predecessor_latent_correlation": correlation,
        "perturbation_rms": delta_rms,
        "changed_cover_bin_fraction": changed_fraction,
        "cover_bin_delta_p99": bin_delta_p99,
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
                (
                    spec.scan_start_x_m
                    + spec.scan_span_m
                    + spec.tx_rx_offset_m
                    + 1.5
                )
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
    return np.where(valid, epsilon[safe], np.nan)


def preview_predecessor_comparison(
    path: Path,
    spec: formal03.Spec,
    candidate_indices: np.ndarray,
) -> None:
    with h5py.File(PREDECESSOR_DIR / "geology_indices.h5", "r") as handle:
        predecessor_indices = handle["data"][:]
    rows = formal06.material_rows(spec, control=False, design=DESIGN)
    predecessor = _epsilon_crop(spec, predecessor_indices, rows)
    candidate = _epsilon_crop(spec, candidate_indices, rows)
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
            "B. FORMAL07B: same basal morphology + weak aperiodic 2-D background",
        ),
        (
            70,
            1025,
            1830,
            1395,
            diverging(difference),
            f"C. FORMAL07B - FORMAL06C enhanced material delta (+/-{delta_limit:.3f} epsilon_r)",
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
        "FORMAL06C -> FORMAL07B one-factor geometry comparison; shared epsilon scale",
        fill="black",
        font=font,
    )
    draw.text(
        (70, 52),
        "Basal path, transition, source, grid, acquisition, and material values are locked byte/array exact",
        fill="black",
        font=font,
    )
    canvas.save(path)


def _verify_locked_predecessor(case_dir: Path, spec: formal03.Spec) -> dict:
    if not PREDECESSOR_DIR.is_dir():
        raise FileNotFoundError(f"missing FORMAL06C predecessor assets: {PREDECESSOR_DIR}")
    native_spec = asdict(spec) == asdict(default_spec())
    file_locks = {}
    for name in ("source_waveform.txt", "materials_full.txt", "materials_no_basal.txt"):
        candidate_hash = formal03.sha256(case_dir / name)
        candidate_normalised_hash = _normalised_text_sha256(case_dir / name)
        byte_comparable = native_spec or name == "source_waveform.txt"
        predecessor_path = PREDECESSOR_DIR / name
        predecessor_hash = (
            formal03.sha256(predecessor_path)
            if byte_comparable
            else candidate_hash
        )
        predecessor_normalised_hash = (
            _normalised_text_sha256(predecessor_path)
            if byte_comparable
            else candidate_normalised_hash
        )
        file_locks[name] = {
            "predecessor_sha256": predecessor_hash,
            "candidate_sha256": candidate_hash,
            "predecessor_normalised_text_sha256": predecessor_normalised_hash,
            "candidate_normalised_text_sha256": candidate_normalised_hash,
            "exact": predecessor_normalised_hash == candidate_normalised_hash,
            "comparison_mode": (
                "normalised_text_content"
                if byte_comparable
                else "same_design_rebuilt_for_reduced_test_spec"
            ),
        }
    if not all(item["exact"] for item in file_locks.values()):
        raise ValueError("FORMAL07B source/material lock differs from FORMAL06C")

    profile_locks = {}
    for name in (
        "full_x_m.npy",
        "full_ground_y_m.npy",
        "full_basal_depth_m.npy",
        "full_transition_thickness_m.npy",
        "full_basal_y_m.npy",
        "full_transition_top_y_m.npy",
    ):
        candidate = np.load(case_dir / "labels" / name)
        if native_spec:
            predecessor = np.load(PREDECESSOR_DIR / "labels" / name)
        else:
            profile, _ = formal03.build_profiles(spec)
            predecessor = profile[name.removesuffix(".npy")]
        exact = np.array_equal(candidate, predecessor)
        profile_locks[name] = {
            "candidate_array_sha256": _array_sha256(candidate),
            "reference_array_sha256": _array_sha256(predecessor),
            "exact": exact,
        }
    if not all(item["exact"] for item in profile_locks.values()):
        raise ValueError("FORMAL07B profile lock differs from FORMAL06C")
    return {
        "predecessor_case_id": formal06c.CASE_ID,
        "native_predecessor_assets_used": native_spec,
        "locked_files": file_locks,
        "locked_profile_arrays": profile_locks,
        "all_exact": True,
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
        policy_filename="FORMAL07B_WEAK_APERIODIC_BACKGROUND_POLICY.json",
        run_prefix="formal07b",
        purpose="weak aperiodic non-target background development",
        predecessor_case_id=formal06c.CASE_ID,
        changed_factors=["non-target cover-field weak aperiodic texture only"],
        generator_path=Path(__file__),
        preview_title=(
            f"{FAMILY_ID}: FORMAL06C-locked basal morphology with weak "
            "aperiodic cover texture; pre-solver only"
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
        ],
        geometry_description=(
            "FORMAL06C basal morphology plus weak aperiodic two-dimensional "
            "non-target cover texture; no stratigraphic slabs or discrete bodies"
        ),
    )
    lock_report = _verify_locked_predecessor(case_dir, spec)
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        indices = handle["data"][:]
    preview_predecessor_comparison(
        case_dir / "preview_FORMAL06C_vs_FORMAL07B_geometry.png",
        spec,
        indices,
    )

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predecessor_lock"] = lock_report
    manifest["comparison_contract"] = {
        "predecessor": formal06c.CASE_ID,
        "identical_source": True,
        "identical_material_design": True,
        "identical_grid_and_acquisition": True,
        "identical_basal_and_transition_profiles": True,
        "changed_factor_group": "weak_aperiodic_non_target_background_only",
        "pre_solver_preview": "preview_FORMAL06C_vs_FORMAL07B_geometry.png",
        "common_trace_runtime_review_required": True,
    }
    manifest["next_gate"] = (
        "static and geometry audit, then one-trace strict pair and blind common-trace "
        "sparse full-scene comparison against FORMAL06C"
    )
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
