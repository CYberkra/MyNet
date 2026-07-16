from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.generate_independent_v2_family01 import sha256
from scripts.generate_independent_v2_family03_instrument_band import (
    ANTENNA_CENTER_HZ,
    BEDROCK_START,
    CONTRACT_ID,
    DEFAULT_CONTRACT,
    NOMINAL_HIGH_HZ,
    NOMINAL_LOW_HZ,
    SUPPORT_HIGH_HZ,
    SUPPORT_LOW_HZ,
    Spec,
    TRANSITION_START,
    generate_family,
    instrument_band_waveform,
    material_rows,
)


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


def test_contract_is_independent_and_hardware_derived() -> None:
    contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["formal_training_allowed"] is False
    provenance = contract["provenance"]
    assert provenance["line9_conditioned"] is False
    assert provenance["decision_conditioned_on_held_out_morphology"] is False
    assert provenance["measured_files_read_by_generator"] == []
    assert provenance["development_case_arrays_reused"] is False
    source = contract["source"]
    assert source["antenna_center_frequency_hz"] == ANTENNA_CENTER_HZ
    assert source["nominal_band_hz"] == [NOMINAL_LOW_HZ, NOMINAL_HIGH_HZ]
    assert source["support_band_hz"] == [SUPPORT_LOW_HZ, SUPPORT_HIGH_HZ]


def test_instrument_band_pulse_is_zero_dc_and_spectrally_bounded() -> None:
    time_s, values, stats = instrument_band_waveform()
    assert abs(float(np.sum(values) * (time_s[1] - time_s[0]))) < 1e-15
    assert 95e6 <= stats["peak_frequency_hz"] <= 105e6
    assert stats["power_fraction_outside_declared_support"] < 1e-20
    assert 70e6 <= stats["spectral_centroid_hz"] <= 120e6
    assert np.isclose(np.max(np.abs(values)), 1.0)


def test_material_indices_and_exact_target_absent_mapping() -> None:
    full = material_rows(control=False)
    control = material_rows(control=True)
    assert [row["index"] for row in full] == list(range(len(full)))
    assert len(full) == len(control)
    assert all(row["epsilon_r"] == 12.7 for row in full[BEDROCK_START:])
    assert all(row["conductivity_s_per_m"] == 0.002 for row in full[BEDROCK_START:])
    for index in range(TRANSITION_START, len(control)):
        local_bin = int(control[index]["local_bin"])
        assert control[index]["epsilon_r"] == control[local_bin]["epsilon_r"]
        assert control[index]["conductivity_s_per_m"] == control[local_bin]["conductivity_s_per_m"]


def test_default_grid_resolves_declared_support_and_protects_time_window() -> None:
    spec = Spec()
    maximum_epsilon = max(float(row["epsilon_r"]) for row in material_rows(control=False))
    cells = 299_792_458.0 / (SUPPORT_HIGH_HZ * np.sqrt(maximum_epsilon)) / spec.dl_m
    earliest_boundary_ns = 2e9 * spec.physical_side_guard_m / 299_792_458.0
    assert cells >= 10.0
    assert earliest_boundary_ns > spec.protected_time_window_ns


def test_tiny_family_preserves_independent_exact_negative(tmp_path: Path) -> None:
    manifest = generate_family(DEFAULT_CONTRACT, tmp_path, overwrite=False, spec=_tiny_spec())
    family = tmp_path / manifest["scene_family_id"]
    positive = family / manifest["positive_case_id"]
    negative = family / manifest["true_negative_case_id"]
    assert sha256(positive / "geology_indices.h5") == sha256(negative / "geology_indices.h5")
    assert sha256(positive / "materials_no_basal.txt") == sha256(negative / "materials_full.txt")
    assert (positive / "source_waveform.txt").is_file()
    assert "#excitation_file: source_waveform.txt" in (positive / "full_scene.in").read_text(
        encoding="ascii"
    )
    assert not (negative / "no_basal_contrast_control.in").exists()
    assert manifest["line9_conditioned"] is False
    assert manifest["decision_conditioned_on_held_out_morphology"] is False
    assert manifest["formal_training_allowed"] is False
