from pathlib import Path

import numpy as np

from scripts.generate_formal09b1_empirical_spectrum import PAPER_FIT_LINES, SpectrumFit
from scripts.generate_formal09b2_lateral_covariance import (
    SpatialFit,
    build_lateral_covariance_basis,
    fit_line_spatial_statistics,
)


def dummy_fits() -> tuple[SpectrumFit, SpatialFit]:
    temporal_frequency = np.fft.rfftfreq(512, d=1.4e-9)
    temporal_amplitude = np.exp(
        -0.5 * ((temporal_frequency - 90e6) / 35e6) ** 2
    )
    temporal = SpectrumFit(
        "temporal",
        ("Line3",),
        temporal_frequency,
        temporal_amplitude,
        {"Line3": temporal_amplitude},
        {"Line3": 10},
    )
    spatial_frequency = np.linspace(0.0, 0.7, 141)
    spatial_amplitude = np.exp(-spatial_frequency / 0.18)
    spatial = SpatialFit(
        "spatial",
        ("Line3",),
        spatial_frequency,
        spatial_amplitude,
        {"Line3": spatial_amplitude},
        {"Line3": 10},
        {"Line3": 0.3},
        {"Line3": 4.0},
        0.3,
        4.0,
    )
    return temporal, spatial


def test_lateral_basis_is_deterministic() -> None:
    temporal, spatial = dummy_fits()
    time_ns = np.arange(501, dtype=np.float64) * 1.4
    left = build_lateral_covariance_basis(
        (501, 32), 1.4e-9, time_ns, temporal, spatial, 0.72, 11, nonstationary=True
    )
    right = build_lateral_covariance_basis(
        (501, 32), 1.4e-9, time_ns, temporal, spatial, 0.72, 11, nonstationary=True
    )
    assert all(np.array_equal(a, b) for a, b in zip(left, right))


def test_nonstationary_candidate_has_bounded_varying_envelope() -> None:
    temporal, spatial = dummy_fits()
    time_ns = np.arange(501, dtype=np.float64) * 1.4
    _, _, _, envelope = build_lateral_covariance_basis(
        (501, 32), 1.4e-9, time_ns, temporal, spatial, 0.72, 13, nonstationary=True
    )
    assert np.std(envelope) > 0
    assert np.min(envelope) >= 0.45
    assert np.max(envelope) <= 2.2


def test_paper_lateral_fit_excludes_line9_and_line6() -> None:
    assert "Line9" not in PAPER_FIT_LINES
    assert "Line6" not in PAPER_FIT_LINES


def test_real_line_spatial_fit_uses_physical_frequency() -> None:
    frequency, amplitude, count, cv, correlation_m = fit_line_spatial_statistics(
        Path("data/measured/yingshan_v15/lines/Line3.npz"), [(21, 1791)]
    )
    assert frequency[-1] == 0.7
    assert frequency.shape == amplitude.shape
    assert count > 100
    assert np.all(np.isfinite(amplitude))
    assert cv > 0
    assert correlation_m > 0
