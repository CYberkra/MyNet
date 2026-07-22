from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from scripts.generate_independent_v2_family01 import (
    BEDROCK_START,
    COVER_BINS,
    DEFAULT_CONTRACT,
    Spec,
    TRANSITION_START,
    build_indices,
    build_profiles,
    correlated_cover_bins,
    generate_family,
    material_rows,
    sha256,
)


R2_CONTRACT = DEFAULT_CONTRACT.with_name("independent_v2_family01_r2_pilot.json")


def _assert_lf_only(path: Path) -> None:
    content = path.read_bytes()
    assert b"\r" not in content, path


def _tiny_spec() -> Spec:
    return Spec(
        domain_x_m=18.0,
        domain_y_m=24.0,
        dl_m=0.12,
        pml_cells=4,
        physical_side_guard_m=4.2,
        trace_count=32,
        trace_spacing_m=0.18,
        scan_start_x_m=5.04,
        tx_rx_offset_m=0.24,
        ground_y_m=17.4,
        source_y_m=20.64,
        solver_time_window_s=300e-9,
        protected_time_window_ns=250.0,
    )


def test_contract_is_independent_and_blocked() -> None:
    contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    assert contract["formal_training_allowed"] is False
    assert contract["provenance"]["line9_conditioned"] is False
    assert contract["provenance"]["measured_files_read_by_generator"] == []
    assert contract["pair_contract"]["positive_control_equals_negative_full_material_map"] is True
    assert {case["target_presence"] for case in contract["cases"]} == {True, False}


def test_r2_contract_has_a_new_immutable_lineage() -> None:
    contract = json.loads(R2_CONTRACT.read_text(encoding="utf-8"))
    assert contract["formal_training_allowed"] is False
    assert contract["supersedes"] == "PGDA_INDEPENDENT_V2_FAMILY01_PILOT_V1"
    assert contract["provenance"]["line9_conditioned"] is False
    assert contract["provenance"]["measured_files_read_by_generator"] == []
    assert contract["scene_family_id"] != "IV2_F01_GENTLE_APERIODIC_COVER_BEDROCK"


def test_default_grid_passes_resolution_and_boundary_contract() -> None:
    spec = Spec()
    maximum_epsilon = max(float(row["epsilon_r"]) for row in material_rows(control=False))
    cells = 299_792_458.0 / (2.8 * spec.center_frequency_hz * np.sqrt(maximum_epsilon)) / spec.dl_m
    earliest_boundary_ns = 2e9 * spec.physical_side_guard_m / 299_792_458.0
    assert cells >= 10.0
    assert earliest_boundary_ns > spec.protected_time_window_ns
    assert spec.scan_start_x_m - spec.pml_cells * spec.dl_m == spec.physical_side_guard_m


def test_seeded_profiles_and_cover_are_deterministic() -> None:
    spec = _tiny_spec()
    first_profiles, first_metrics = build_profiles(spec, 2026071501)
    second_profiles, second_metrics = build_profiles(spec, 2026071501)
    assert first_metrics == second_metrics
    assert np.array_equal(first_profiles["basal_depth_m"], second_profiles["basal_depth_m"])
    first_bins, first_stats = correlated_cover_bins(spec, 2026071511)
    second_bins, second_stats = correlated_cover_bins(spec, 2026071511)
    assert first_stats == second_stats
    assert np.array_equal(first_bins, second_bins)


def test_control_maps_every_target_partition_back_to_local_cover() -> None:
    full = material_rows(control=False)
    control = material_rows(control=True)
    assert full[:COVER_BINS] == control[:COVER_BINS]
    for index in range(TRANSITION_START, len(control)):
        local_bin = int(control[index]["local_bin"])
        assert control[index]["epsilon_r"] == control[local_bin]["epsilon_r"]
        assert control[index]["conductivity_s_per_m"] == control[local_bin]["conductivity_s_per_m"]
    assert any(
        full[index]["epsilon_r"] != control[index]["epsilon_r"]
        for index in range(TRANSITION_START, BEDROCK_START)
    )


def test_tiny_family_has_exact_positive_control_negative_equivalence(tmp_path: Path) -> None:
    manifest = generate_family(DEFAULT_CONTRACT, tmp_path, overwrite=False, spec=_tiny_spec())
    family = tmp_path / manifest["scene_family_id"]
    positive = family / manifest["positive_case_id"]
    negative = family / manifest["true_negative_case_id"]
    assert sha256(positive / "geology_indices.h5") == sha256(negative / "geology_indices.h5")
    assert sha256(positive / "materials_no_basal.txt") == sha256(negative / "materials_full.txt")
    assert not (negative / "no_basal_contrast_control.in").exists()
    assert not (negative / "labels" / "source_referenced_arrival_time_ns.npy").exists()
    assert (negative / "labels" / "source_x_m.npy").is_file()
    assert (negative / "labels" / "receiver_x_m.npy").is_file()
    assert np.count_nonzero(np.load(negative / "labels" / "target_mask.npy")) == 0
    assert np.all(np.load(negative / "labels" / "trace_state.npy") == 0)
    with h5py.File(positive / "geology_indices.h5", "r") as handle:
        data = handle["data"]
        assert data.compression == "gzip"
        assert tuple(handle.attrs["dx_dy_dz"]) == (_tiny_spec().dl_m,) * 3


def test_generated_hash_protected_text_is_canonical_lf(tmp_path: Path) -> None:
    manifest = generate_family(DEFAULT_CONTRACT, tmp_path, overwrite=False, spec=_tiny_spec())
    family = tmp_path / manifest["scene_family_id"]
    for text_path in family.rglob("*"):
        if text_path.suffix in {".csv", ".in", ".json", ".md", ".txt"}:
            _assert_lf_only(text_path)


def test_r2_tiny_family_is_generated_under_the_new_contract(tmp_path: Path) -> None:
    manifest = generate_family(R2_CONTRACT, tmp_path, overwrite=False, spec=_tiny_spec())
    assert manifest["contract_id"] == "PGDA_INDEPENDENT_V2_FAMILY01_R2_PILOT_V1"
    assert manifest["formal_training_allowed"] is False
    assert (tmp_path / manifest["scene_family_id"] / "family_manifest.json").is_file()


def test_index_geometry_contains_cover_transition_and_bedrock() -> None:
    spec = _tiny_spec()
    profiles, _ = build_profiles(spec, 2026071501)
    bins, _ = correlated_cover_bins(spec, 2026071511)
    data = build_indices(spec, bins, profiles)
    unique = set(np.unique(data))
    assert -1 in unique
    assert any(0 <= value < COVER_BINS for value in unique)
    assert any(TRANSITION_START <= value < BEDROCK_START for value in unique)
    assert any(value >= BEDROCK_START for value in unique)
