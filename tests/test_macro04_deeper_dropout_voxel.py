from __future__ import annotations

import numpy as np

from scripts.generate_macro04_deeper_dropout_voxel import (
    LENSES,
    Q_OFFSETS,
    REGIONS,
    Spec,
    build_indices,
    correlated_field,
    exact_gprmax_waveforms,
    material_rows,
    profiles,
    reference_arrival,
)


def small_spec() -> Spec:
    return Spec(
        domain_x_m=24.0,
        domain_y_m=24.0,
        dl_m=0.1,
        pml_cells=10,
        trace_count=4,
        trace_spacing_m=0.5,
        scan_start_x_m=2.0,
        source_y_m=23.0,
    )


def test_correlated_field_is_seeded_and_bounded() -> None:
    spec = small_spec()
    first, thresholds_first = correlated_field(spec)
    second, thresholds_second = correlated_field(spec)
    assert np.array_equal(first, second)
    assert thresholds_first == thresholds_second
    assert first.shape == (spec.nx, spec.ny)
    assert set(np.unique(first)) <= set(range(len(Q_OFFSETS)))


def test_strict_pair_changes_only_transition_and_bedrock() -> None:
    full = material_rows(control=False)
    control = material_rows(control=True)
    changed = [
        index
        for index, (left, right) in enumerate(zip(full, control))
        if (left["epsilon_r"], left["conductivity_s_per_m"])
        != (right["epsilon_r"], right["conductivity_s_per_m"])
    ]
    assert changed == list(range(2 * len(Q_OFFSETS), len(REGIONS) * len(Q_OFFSETS)))
    assert len(full) == len(REGIONS) * len(Q_OFFSETS) + len(LENSES)
    assert all(full[index] == control[index] for index in range(2 * len(Q_OFFSETS)))
    assert all(full[index] == control[index] for index in range(len(REGIONS) * len(Q_OFFSETS), len(full)))


def test_voxel_indices_preserve_air_and_material_contract() -> None:
    spec = small_spec()
    quantiles, _ = correlated_field(spec)
    profile = profiles(spec)
    data = build_indices(spec, quantiles, profile)
    assert data.shape == (spec.nx, spec.ny, 1)
    assert data.dtype == np.int16
    assert int(data.min()) == -1
    assert int(data.max()) < len(material_rows(control=False))
    assert np.any(data[:, :, 0] < 0)
    assert np.any(data[:, :, 0] >= 0)


def test_default_scan_has_deep_gentle_interface_and_finite_arrival() -> None:
    spec = Spec()
    profile = profiles(spec)
    labels = reference_arrival(spec, profile)
    midpoint = labels["trace_midpoint_x_m"]
    basal = np.interp(midpoint, profile["x_m"], profile["basal_depth_m"])
    transition = np.interp(midpoint, profile["x_m"], profile["transition_thickness_m"])
    arrival = labels["geometric_reference_arrival_time_ns"]

    assert 14.3 < float(np.min(basal)) < float(np.max(basal)) < 16.2
    assert float(np.ptp(basal)) < 1.2
    assert 1.2 <= float(np.min(transition)) < float(np.max(transition)) <= 3.0
    assert np.all(np.isfinite(arrival))
    assert 10.0 < float(np.ptp(arrival)) < 25.0


def test_selected_ricker_is_bipolar_and_nearly_dc_free() -> None:
    _, waveforms = exact_gprmax_waveforms(Spec())
    gaussian = waveforms["Plain Gaussian"]
    gaussian_dot = waveforms["Gaussian 1st derivative (normalised)"]
    ricker = waveforms["Ricker (selected)"]

    def relative_dc(values: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(values))
        return float(spectrum[0] / np.max(spectrum))

    assert np.all(gaussian >= 0.0)
    assert relative_dc(gaussian) > 0.9
    assert float(np.min(gaussian_dot)) < 0.0 < float(np.max(gaussian_dot))
    assert float(np.min(ricker)) < 0.0 < float(np.max(ricker))
    assert relative_dc(gaussian_dot) < 1e-6
    assert relative_dc(ricker) < 1e-6
