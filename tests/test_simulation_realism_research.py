from __future__ import annotations

import numpy as np

from pgdacsnet.simulation_realism_research import (
    SpectrumEstimate,
    apply_trace_gain_delay,
    apply_zero_phase_response,
    bounded_system_response_db,
    packet_metrics,
    pool_spectra,
    smooth_unit_noise,
)
from scripts.research_round03_coherent_residual import _coherent_energy_fractions


def test_pool_spectra_weights_lines_equally() -> None:
    frequency = np.asarray([0.0, 1.0, 2.0])
    first = SpectrumEstimate(frequency, np.asarray([0.0, 1.0, 2.0]), 1000)
    second = SpectrumEstimate(frequency, np.asarray([2.0, 3.0, 4.0]), 10)
    pooled = pool_spectra([first, second])
    np.testing.assert_allclose(pooled.log_amplitude, [1.0, 2.0, 3.0])
    assert pooled.trace_count == 1010


def test_bounded_system_response_respects_gain_contract() -> None:
    frequency = np.linspace(0.0, 250e6, 64)
    simulated = SpectrumEstimate(frequency, np.zeros(64), 10)
    measured = SpectrumEstimate(frequency, np.linspace(-4.0, 4.0, 64), 10)
    response = bounded_system_response_db(simulated, measured, max_abs_db=5.0)
    assert response[0] == 0.0
    assert float(np.max(np.abs(response))) <= 5.0 + 1e-12
    assert np.all(np.isfinite(response))


def test_zero_phase_response_does_not_move_centered_impulse() -> None:
    values = np.zeros((257, 3), dtype=np.float64)
    values[128] = 1.0
    frequency = np.fft.rfftfreq(values.shape[0], d=1.4e-9)
    response = -3.0 * np.square(frequency / max(float(frequency[-1]), 1.0))
    filtered = apply_zero_phase_response(
        values, 1.4, frequency, response, strength=1.0
    )
    np.testing.assert_array_equal(np.argmax(np.abs(filtered), axis=0), [128, 128, 128])


def test_packet_metrics_are_finite() -> None:
    time_ns = np.arange(0.0, 501.0, 1.4)
    path_ns = np.asarray([280.0, 282.0, 284.0])
    values = np.column_stack(
        [
            np.exp(-0.5 * np.square((time_ns - center) / 12.0))
            * np.cos(2.0 * np.pi * 80e6 * time_ns * 1e-9)
            for center in path_ns
        ]
    )
    metrics = packet_metrics(values, time_ns, path_ns)
    assert metrics["target_to_background"] > 1.0
    assert 0.0 <= metrics["target_dropout_fraction"] <= 1.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_smooth_unit_noise_has_declared_moments() -> None:
    values = smooth_unit_noise(256, 4.0, np.random.default_rng(123))
    assert abs(float(np.mean(values))) < 1e-12
    assert abs(float(np.std(values)) - 1.0) < 1e-12


def test_trace_gain_delay_moves_each_trace_continuously() -> None:
    time_ns = np.arange(0.0, 20.0, 1.0)
    values = np.zeros((time_ns.size, 2), dtype=np.float64)
    values[8, :] = 1.0
    shifted = apply_trace_gain_delay(
        values,
        time_ns,
        np.log(np.asarray([1.0, 2.0])),
        np.asarray([2.0, -1.0]),
    )
    np.testing.assert_array_equal(np.argmax(shifted, axis=0), [10, 7])
    np.testing.assert_allclose(np.max(shifted, axis=0), [1.0, 2.0])


def test_coherent_energy_fraction_detects_rank_one_matrix() -> None:
    time_ns = np.arange(0.0, 501.0, 1.4)
    temporal = np.sin(2.0 * np.pi * time_ns / 35.0)
    spatial = np.linspace(-1.0, 1.0, 16)
    values = temporal[:, None] * spatial[None, :]
    fractions = _coherent_energy_fractions(values, time_ns)
    assert fractions["rank_1_energy_fraction"] > 0.999999
