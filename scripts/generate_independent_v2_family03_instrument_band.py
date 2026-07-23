#!/usr/bin/env python3
"""Generate the independent instrument-band V2 Family 03 pilot.

The generator reads only its committed contract and generic seeded priors. The
custom pulse is derived from project-wide hardware band metadata, not from a
measured line or a development-case waveform.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import generate_independent_v2_family01 as f01
except ModuleNotFoundError:  # Package import used by pytest and downstream tools.
    from scripts import generate_independent_v2_family01 as f01


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT
    / "data"
    / "contracts"
    / "simulation_v2"
    / "independent_v2_family03_instrument_band_pilot.json"
)
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
CONTRACT_ID = "PGDA_IV2_FAMILY03_INSTRUMENT_BAND_PILOT_V1"
GENERATOR_PATH = "scripts/generate_independent_v2_family03_instrument_band.py"
C0 = f01.C0
COVER_BINS = f01.COVER_BINS
TRANSITION_LEVELS = f01.TRANSITION_LEVELS
TRANSITION_START = f01.TRANSITION_START
BEDROCK_START = f01.BEDROCK_START
MATERIAL_COUNT = f01.MATERIAL_COUNT
WAVEFORM_ID = "iv2_f03_instrument_band100"
REFERENCE_DELAY_NS = 50.0
SUPPORT_LOW_HZ = 20e6
NOMINAL_LOW_HZ = 50e6
ANTENNA_CENTER_HZ = 100e6
NOMINAL_HIGH_HZ = 150e6
SUPPORT_HIGH_HZ = 170e6


@dataclass(frozen=True)
class Spec(f01.Spec):
    center_frequency_hz: float = ANTENNA_CENTER_HZ

    @property
    def source_reference_delay_ns(self) -> float:
        return REFERENCE_DELAY_NS


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _cover_properties(local_bin: int) -> tuple[float, float]:
    fraction = (local_bin + 0.5) / COVER_BINS
    return 12.0 + 1.2 * fraction, 0.0018 + 0.0016 * fraction


def material_rows(*, control: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_bin in range(COVER_BINS):
        epsilon, conductivity = _cover_properties(local_bin)
        rows.append(
            {
                "index": local_bin,
                "region": "cover",
                "local_bin": local_bin,
                "epsilon_r": epsilon,
                "conductivity_s_per_m": conductivity,
                "id": f"cover_{local_bin:02d}",
            }
        )
    for level in range(TRANSITION_LEVELS):
        blend = _smoothstep((level + 0.5) / TRANSITION_LEVELS)
        for local_bin in range(COVER_BINS):
            cover_epsilon, cover_conductivity = _cover_properties(local_bin)
            if control:
                epsilon, conductivity = cover_epsilon, cover_conductivity
            else:
                epsilon = cover_epsilon + blend * (13.3 - cover_epsilon)
                conductivity = cover_conductivity + blend * (0.0030 - cover_conductivity)
            index = TRANSITION_START + level * COVER_BINS + local_bin
            rows.append(
                {
                    "index": index,
                    "region": f"transition_{level + 1}",
                    "local_bin": local_bin,
                    "epsilon_r": epsilon,
                    "conductivity_s_per_m": conductivity,
                    "id": f"transition_{level + 1}_{local_bin:02d}",
                }
            )
    for local_bin in range(COVER_BINS):
        cover_epsilon, cover_conductivity = _cover_properties(local_bin)
        epsilon = cover_epsilon if control else 12.7
        conductivity = cover_conductivity if control else 0.0020
        index = BEDROCK_START + local_bin
        rows.append(
            {
                "index": index,
                "region": "bedrock",
                "local_bin": local_bin,
                "epsilon_r": epsilon,
                "conductivity_s_per_m": conductivity,
                "id": f"bedrock_{local_bin:02d}",
            }
        )
    if [row["index"] for row in rows] != list(range(MATERIAL_COUNT)):
        raise RuntimeError("material rows are not contiguous and index-aligned")
    return rows


def instrument_band_waveform(
    *,
    sample_interval_s: float = 0.1e-9,
    sample_count: int = 8192,
    reference_delay_ns: float = REFERENCE_DELAY_NS,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Create a real zero-phase amplitude-only 20-170 MHz pulse proxy."""
    frequency = np.fft.rfftfreq(sample_count, sample_interval_s)
    response = np.zeros_like(frequency)
    lower = (frequency >= SUPPORT_LOW_HZ) & (frequency < NOMINAL_LOW_HZ)
    middle = (frequency >= NOMINAL_LOW_HZ) & (frequency <= NOMINAL_HIGH_HZ)
    upper = (frequency > NOMINAL_HIGH_HZ) & (frequency <= SUPPORT_HIGH_HZ)
    response[lower] = 0.5 * (
        1.0
        - np.cos(
            math.pi
            * (frequency[lower] - SUPPORT_LOW_HZ)
            / (NOMINAL_LOW_HZ - SUPPORT_LOW_HZ)
        )
    )
    response[middle] = 1.0
    response[upper] = 0.5 * (
        1.0
        + np.cos(
            math.pi
            * (frequency[upper] - NOMINAL_HIGH_HZ)
            / (SUPPORT_HIGH_HZ - NOMINAL_HIGH_HZ)
        )
    )
    antenna_envelope = np.exp(-0.5 * ((frequency - ANTENNA_CENTER_HZ) / 55e6) ** 2)
    response *= antenna_envelope
    values = np.fft.irfft(response, n=sample_count)
    delay_samples = int(round(reference_delay_ns * 1e-9 / sample_interval_s))
    values = np.roll(values, delay_samples)
    values /= float(np.max(np.abs(values)))
    time_s = np.arange(sample_count, dtype=np.float64) * sample_interval_s
    measured = np.abs(np.fft.rfft(values))
    peak_index = int(np.argmax(measured[1:])) + 1
    positive = frequency > 0
    centroid = float(np.sum(frequency[positive] * measured[positive]) / np.sum(measured[positive]))
    outside = (frequency < SUPPORT_LOW_HZ) | (frequency > SUPPORT_HIGH_HZ)
    total_power = float(np.sum(measured * measured))
    outside_fraction = float(np.sum(measured[outside] ** 2) / total_power)
    stats = {
        "sample_interval_ns": sample_interval_s * 1e9,
        "sample_count": sample_count,
        "reference_delay_ns": reference_delay_ns,
        "discrete_time_integral": float(np.sum(values) * sample_interval_s),
        "peak_frequency_hz": float(frequency[peak_index]),
        "spectral_centroid_hz": centroid,
        "power_fraction_outside_declared_support": outside_fraction,
        "maximum_absolute_amplitude": float(np.max(np.abs(values))),
    }
    return time_s, values, stats


def write_instrument_band_waveform(path: Path) -> dict[str, float]:
    time_s, values, stats = instrument_band_waveform()
    with path.open("w", encoding="ascii", newline="") as handle:
        handle.write(f"time {WAVEFORM_ID}\n")
        for time_value, amplitude in zip(time_s, values):
            handle.write(f"{time_value:.12e} {amplitude:.12e}\n")
    return stats


def input_text(
    spec: Spec,
    title: str,
    material_file: str | None,
    *,
    geometry_view: str | None = None,
) -> str:
    def number(value: float) -> str:
        return f"{value:.12g}"

    lines = [
        f"#title: {title}",
        f"#domain: {number(spec.domain_x_m)} {number(spec.domain_y_m)} {number(spec.dl_m)}",
        f"#dx_dy_dz: {number(spec.dl_m)} {number(spec.dl_m)} {number(spec.dl_m)}",
        f"#time_window: {number(spec.solver_time_window_s)}",
        f"#pml_cells: {spec.pml_cells} {spec.pml_cells} 0 {spec.pml_cells} {spec.pml_cells} 0",
        "#messages: y",
        "#excitation_file: source_waveform.txt",
        f"#hertzian_dipole: z {number(spec.scan_start_x_m)} {number(spec.source_y_m)} 0 {WAVEFORM_ID}",
        f"#rx: {number(spec.scan_start_x_m + spec.tx_rx_offset_m)} {number(spec.source_y_m)} 0 rx1 Ez",
        f"#src_steps: {number(spec.trace_spacing_m)} 0 0",
        f"#rx_steps: {number(spec.trace_spacing_m)} 0 0",
    ]
    if material_file:
        lines.append(f"#geometry_objects_read: 0 0 0 geology_indices.h5 {material_file}")
    if geometry_view:
        lines.append(
            f"#geometry_view: 0 0 0 {number(spec.domain_x_m)} {number(spec.domain_y_m)} {number(spec.dl_m)} "
            f"{number(spec.dl_m)} {number(spec.dl_m)} {number(spec.dl_m)} {geometry_view} n"
        )
    return "\n".join(lines) + "\n"


def _write_case(
    case_dir: Path,
    case: dict[str, Any],
    contract: dict[str, Any],
    spec: Spec,
    data: np.ndarray,
    profiles: dict[str, np.ndarray],
    shape_metrics: dict[str, Any],
    field_stats: dict[str, Any],
    full_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    geometry_source: Path,
    contract_path: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True)
    labels_dir = case_dir / "labels"
    labels_dir.mkdir()
    geometry_path = case_dir / "geology_indices.h5"
    shutil.copy2(geometry_source, geometry_path)
    target_presence = bool(case["target_presence"])
    physical_rows = full_rows if target_presence else control_rows
    f01.write_materials(case_dir / "materials_full.txt", physical_rows)
    waveform_stats = write_instrument_band_waveform(case_dir / "source_waveform.txt")
    f01.write_canonical_text(
        case_dir / "full_scene.in",
        input_text(spec, f"{case['case_id']} full", "materials_full.txt"),
        encoding="ascii",
    )
    f01.write_canonical_text(
        case_dir / "geometry_check_full.in",
        input_text(
            spec,
            f"{case['case_id']} geometry",
            "materials_full.txt",
            geometry_view="geometry_full",
        ),
        encoding="ascii",
    )
    if target_presence:
        f01.write_materials(case_dir / "materials_no_basal.txt", control_rows)
        f01.write_canonical_text(
            case_dir / "no_basal_contrast_control.in",
            input_text(spec, f"{case['case_id']} no basal", "materials_no_basal.txt"),
            encoding="ascii",
        )
        f01.write_canonical_text(
            case_dir / "geometry_check_control.in",
            input_text(
                spec,
                f"{case['case_id']} control geometry",
                "materials_no_basal.txt",
                geometry_view="geometry_control",
            ),
            encoding="ascii",
        )
    f01.write_canonical_text(
        case_dir / "air_reference.in",
        input_text(spec, f"{case['case_id']} air", None),
        encoding="ascii",
    )

    np.save(labels_dir / "target_presence.npy", np.asarray(target_presence, dtype=np.bool_))
    np.save(labels_dir / "valid_trace_mask.npy", np.ones(spec.trace_count, dtype=np.bool_))
    for name, values in f01.acquisition_arrays(spec).items():
        np.save(labels_dir / f"{name}.npy", values)
    if target_presence:
        arrays = f01.scan_arrays(spec, profiles, full_rows)
        for name, values in arrays.items():
            np.save(labels_dir / f"{name}.npy", values)
        np.save(labels_dir / "reference_arrival_time_ns.npy", arrays["geometric_reference_arrival_time_ns"])
        np.save(labels_dir / "geometric_arrival_time_ns.npy", arrays["geometric_reference_arrival_time_ns"])
        np.save(labels_dir / "target_mask.npy", np.zeros((501, spec.trace_count), dtype=np.float32))
    else:
        np.save(labels_dir / "trace_state.npy", np.zeros(spec.trace_count, dtype=np.int8))
        np.save(labels_dir / "target_mask.npy", np.zeros((501, spec.trace_count), dtype=np.float32))

    pml_m = spec.pml_cells * spec.dl_m
    right_guard = spec.domain_x_m - pml_m - (
        spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m
    )
    max_epsilon = max(float(row["epsilon_r"]) for row in full_rows)
    cells_per_min_wavelength = C0 / (
        SUPPORT_HIGH_HZ * math.sqrt(max_epsilon) * spec.dl_m
    )
    manifest: dict[str, Any] = {
        "contract_id": contract["contract_id"],
        "case_id": case["case_id"],
        "scene_family_id": contract["scene_family_id"],
        "lifecycle_state": "pre_solver_independent_pilot",
        "formal_training_allowed": False,
        "promotion_allowed": False,
        "target_presence": target_presence,
        "negative_semantics": case["negative_semantics"],
        "line9_conditioned": False,
        "decision_conditioned_on_held_out_morphology": False,
        "strict_line9_holdout_allowed": False,
        "generator_path": GENERATOR_PATH,
        "generator_sha256": f01.sha256(Path(__file__)),
        "contract_path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
        "contract_sha256": f01.sha256(contract_path),
        "training_block_reason": "independent runtime, native-256 and human release gates are pending",
        "parameter_provenance": contract["provenance"],
        "grid": {
            "trace_count": spec.trace_count,
            "trace_spacing_m": spec.trace_spacing_m,
            "dl_m": spec.dl_m,
            "domain_x_m": spec.domain_x_m,
            "domain_y_m": spec.domain_y_m,
            "nx_ny_nz": [spec.nx, spec.ny, 1],
            "pml_cells": [spec.pml_cells, spec.pml_cells, 0, spec.pml_cells, spec.pml_cells, 0],
            "left_physical_guard_m": spec.scan_start_x_m - pml_m,
            "right_physical_guard_m": right_guard,
            "earliest_lateral_boundary_round_trip_ns": 2e9
            * min(spec.scan_start_x_m - pml_m, right_guard)
            / C0,
            "solver_time_window_ns": spec.solver_time_window_s * 1e9,
            "protected_time_window_ns": spec.protected_time_window_ns,
            "canonical_output_samples": 501,
            "canonical_output_dt_ns": 1.4,
            "declared_source_support_high_hz": SUPPORT_HIGH_HZ,
            "cells_per_min_wavelength_at_declared_support": cells_per_min_wavelength,
        },
        "source": {**contract["source"], "custom_waveform_statistics": waveform_stats},
        "geometry": {
            "index_file": "geology_indices.h5",
            "index_file_sha256": f01.sha256(geometry_path),
            "index_array_sha256": f01.array_sha256(data),
            "index_shape": list(data.shape),
            "geometry_lineage_id": contract["geology"]["geometry_lineage_id"],
            "profile_shape_gate": shape_metrics,
            "cover_field_statistics": field_stats,
            "flat_ground": True,
            "fixed_flight_height_m": spec.source_y_m - spec.ground_y_m,
            "discrete_anomaly_bodies": 0,
            "visible_phase_geometry_written": False,
            "development_arrays_reused": False,
            "geometry_regenerated_from_generic_seeds": True,
        },
        "materials": {
            "full_materials_sha256": f01.sha256(case_dir / "materials_full.txt"),
            "material_count": MATERIAL_COUNT,
            "cover_bins": COVER_BINS,
            "transition_levels": TRANSITION_LEVELS,
            "only_positive_full_has_basal_contrast": True,
        },
        "strict_pair": {
            "required": target_presence,
            "shared_geometry_hdf5": target_presence,
            "control_restores_transition_and_bedrock_to_each_local_cover_bin": True,
            "positive_control_equals_family_negative_full": True,
        },
        "labels": {
            "geometric_reference": "audit prior only" if target_presence else "absent for true negative",
            "source_referenced_arrival": "audit prior only" if target_presence else "absent for true negative",
            "visible_phase": "absent until solved signed pair review",
            "target_mask_training_allowed": False,
        },
        "next_gate": "static and geometry-only audit, then one-trace positive pair and negative-full equality",
    }
    if target_presence:
        manifest["materials"]["no_basal_materials_sha256"] = f01.sha256(
            case_dir / "materials_no_basal.txt"
        )
    f01.write_canonical_text(
        case_dir / "scene_manifest.json",
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    f01.write_canonical_text(
        case_dir / "RUN_COMMANDS.md",
        "# Runtime commands\n\n"
        "Run through `scripts/run_native_256_release_pilot.py`; never execute or overwrite this source deck in place.\n",
        encoding="utf-8",
    )
    f01.write_checksums(case_dir)
    return manifest


def generate_family(
    contract_path: Path,
    output_root: Path,
    *,
    overwrite: bool,
    spec: Spec | None = None,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError("unexpected F03 instrument-band contract")
    if contract.get("formal_training_allowed") is not False:
        raise ValueError("pre-solver pilot must remain blocked")
    provenance = contract["provenance"]
    if provenance.get("line9_conditioned") is not False:
        raise ValueError("F03 must be non-Line9-conditioned")
    if provenance.get("measured_files_read_by_generator") != []:
        raise ValueError("F03 generator must not read measured files")
    if provenance.get("development_case_arrays_reused") is not False:
        raise ValueError("F03 must regenerate generic geometry")
    spec = spec or Spec()
    family_dir = output_root / contract["scene_family_id"]
    if family_dir.exists():
        if not overwrite:
            raise FileExistsError(f"family exists; pass --overwrite: {family_dir}")
        shutil.rmtree(family_dir)
    family_dir.mkdir(parents=True)

    positive = next(case for case in contract["cases"] if case["target_presence"])
    negative = next(case for case in contract["cases"] if not case["target_presence"])
    if positive["profile_seed_base"] != negative["profile_seed_base"]:
        raise ValueError("matched cases must share profile seed")
    if positive["field_seed"] != negative["field_seed"]:
        raise ValueError("matched cases must share cover-field seed")

    profiles, shape_metrics = f01.build_profiles(spec, int(positive["profile_seed_base"]))
    bins, field_stats = f01.correlated_cover_bins(spec, int(positive["field_seed"]))
    data = f01.build_indices(spec, bins, profiles)
    full_rows = material_rows(control=False)
    control_rows = material_rows(control=True)

    common_geometry = family_dir / "_shared_geology_indices.h5"
    with h5py.File(common_geometry, "w") as handle:
        handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
        handle.attrs["generator"] = GENERATOR_PATH
        handle.attrs["scene_family_id"] = contract["scene_family_id"]
        handle.create_dataset(
            "data",
            data=data,
            dtype=np.int16,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(256, spec.nx), min(256, spec.ny), 1),
        )

    manifests = []
    for case in (positive, negative):
        manifests.append(
            _write_case(
                family_dir / case["case_id"],
                case,
                contract,
                spec,
                data,
                profiles,
                shape_metrics,
                field_stats,
                full_rows,
                control_rows,
                common_geometry,
                contract_path,
            )
        )

    positive_dir = family_dir / positive["case_id"]
    negative_dir = family_dir / negative["case_id"]
    if f01.sha256(positive_dir / "geology_indices.h5") != f01.sha256(
        negative_dir / "geology_indices.h5"
    ):
        raise RuntimeError("matched family geometry files are not byte-identical")
    if f01.sha256(positive_dir / "materials_no_basal.txt") != f01.sha256(
        negative_dir / "materials_full.txt"
    ):
        raise RuntimeError("positive control and negative full materials are not byte-identical")
    preview_path = f01.render_preview(
        family_dir,
        spec,
        data,
        profiles,
        full_rows,
        control_rows,
        shape_metrics,
        output_name="IV2_F03_FAMILY_PRE_SOLVER_PREVIEW.png",
        heading="IV2 Family 03: independent instrument-band weak-interface pilot",
        positive_title="Positive full: generic geometry + project-wide instrument-band pulse proxy",
    )
    common_geometry.unlink()

    family_manifest = {
        "contract_id": contract["contract_id"],
        "scene_family_id": contract["scene_family_id"],
        "lifecycle_state": "pre_solver_independent_pilot",
        "formal_training_allowed": False,
        "line9_conditioned": False,
        "decision_conditioned_on_held_out_morphology": False,
        "split_group_indivisible": True,
        "geometry_lineage_id": contract["geology"]["geometry_lineage_id"],
        "case_ids": [case["case_id"] for case in contract["cases"]],
        "positive_case_id": positive["case_id"],
        "true_negative_case_id": negative["case_id"],
        "shared_geometry_sha256": f01.sha256(positive_dir / "geology_indices.h5"),
        "shared_geometry_array_sha256": f01.array_sha256(data),
        "positive_control_materials_sha256": f01.sha256(
            positive_dir / "materials_no_basal.txt"
        ),
        "negative_full_materials_sha256": f01.sha256(negative_dir / "materials_full.txt"),
        "exact_negative_equivalence": True,
        "profile_shape_gate": shape_metrics,
        "cover_field_statistics": field_stats,
        "preview": preview_path.name,
        "next_gate": "static, geometry-only and one-trace runtime audit; no training export",
    }
    f01.write_canonical_text(
        family_dir / "family_manifest.json",
        json.dumps(family_manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    f01.write_checksums(family_dir)
    return family_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = generate_family(
        args.contract.resolve(), args.output_root.resolve(), overwrite=args.overwrite
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
