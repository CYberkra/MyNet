from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.generate_independent_v2_family02_formal06c_mechanism import (
    BEDROCK_START,
    CONTRACT_ID,
    COVER_BINS,
    DEFAULT_CONTRACT,
    SOURCE,
    Spec,
    TRANSITION_START,
    generate_family,
    material_rows,
)
from scripts.generate_independent_v2_family01 import sha256
from scripts.generate_formal03_correlated_cover_source_ablation import custom_waveform


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


def test_contract_records_mechanism_conditioning_without_array_reuse() -> None:
    contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["formal_training_allowed"] is False
    assert contract["provenance"]["line9_conditioned"] is True
    assert contract["provenance"]["measured_files_read_by_generator"] == []
    assert contract["provenance"]["development_case_arrays_reused"] is False


def test_formal06c_material_mechanism_and_exact_control_mapping() -> None:
    full = material_rows(control=False)
    control = material_rows(control=True)
    assert len(full) == len(control)
    assert full[0]["epsilon_r"] == 12.0125
    assert full[COVER_BINS - 1]["epsilon_r"] == 12.7875
    assert all(row["epsilon_r"] == 12.55 for row in full[BEDROCK_START:])
    assert all(row["conductivity_s_per_m"] == 0.00225 for row in full[BEDROCK_START:])
    for index in range(TRANSITION_START, len(control)):
        local_bin = int(control[index]["local_bin"])
        assert control[index]["epsilon_r"] == control[local_bin]["epsilon_r"]
        assert control[index]["conductivity_s_per_m"] == control[local_bin]["conductivity_s_per_m"]


def test_custom_source_is_zero_mean_and_peaks_near_80_mhz() -> None:
    time_s, values = custom_waveform(SOURCE, Spec())
    spectrum = np.abs(np.fft.rfft(values))
    frequency = np.fft.rfftfreq(values.size, float(time_s[1] - time_s[0]))
    peak_hz = float(frequency[int(np.argmax(spectrum[1:])) + 1])
    assert abs(float(np.trapz(values, time_s))) < 1e-15
    assert 75e6 <= peak_hz <= 85e6


def test_default_resolution_and_boundary_contract() -> None:
    spec = Spec()
    maximum_epsilon = max(float(row["epsilon_r"]) for row in material_rows(control=False))
    cells = 299_792_458.0 / (2.8 * spec.center_frequency_hz * np.sqrt(maximum_epsilon)) / spec.dl_m
    earliest_boundary_ns = 2e9 * spec.physical_side_guard_m / 299_792_458.0
    assert cells >= 10.0
    assert earliest_boundary_ns > spec.protected_time_window_ns


def test_tiny_family_preserves_exact_negative_and_custom_source(tmp_path: Path) -> None:
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
    assert manifest["line9_conditioned"] is True
    assert manifest["formal_training_allowed"] is False
