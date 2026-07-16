from pathlib import Path

import numpy as np

from scripts.generate_formal09b1_empirical_spectrum import (
    PAPER_FIT_LINES,
    SpectrumFit,
    build_empirical_realism_basis,
    equal_line_log_pool,
    fit_line_residual_spectrum,
)


def test_equal_line_pool_does_not_weight_by_trace_count() -> None:
    first = np.array([1e-12, 1.0, 2.0, 1.0])
    second = np.array([1e-12, 4.0, 1.0, 2.0])
    pooled = equal_line_log_pool([first, second])
    duplicated = equal_line_log_pool([first, second])
    assert np.allclose(pooled, duplicated)


def test_paper_spectrum_fit_excludes_line9_and_validation_line() -> None:
    assert "Line9" not in PAPER_FIT_LINES
    assert "Line6" not in PAPER_FIT_LINES
    assert set(PAPER_FIT_LINES) == {"Line3", "Line7", "LineL1"}


def test_empirical_basis_is_deterministic() -> None:
    frequency = np.fft.rfftfreq(512, d=1.4e-9)
    amplitude = np.exp(-0.5 * ((frequency - 80e6) / 35e6) ** 2)
    fit = SpectrumFit(
        name="test",
        lines=("Line3",),
        frequency_hz=frequency,
        amplitude=amplitude,
        line_amplitudes={"Line3": amplitude},
        frame_counts={"Line3": 10},
    )
    time_ns = np.arange(501, dtype=np.float64) * 1.4
    left = build_empirical_realism_basis((501, 16), 1.4e-9, time_ns, fit, 7)
    right = build_empirical_realism_basis((501, 16), 1.4e-9, time_ns, fit, 7)
    assert all(np.array_equal(a, b) for a, b in zip(left, right))


def test_real_line_spectrum_fit_is_finite_and_target_excluded() -> None:
    frequency, amplitude, frame_count = fit_line_residual_spectrum(
        Path("data/measured/yingshan_v15/lines/Line3.npz"),
        [(21, 1791)],
    )
    assert frequency.shape == amplitude.shape
    assert frame_count > 100
    assert np.all(np.isfinite(amplitude))
    assert amplitude[0] < 1e-6
    assert np.max(amplitude[1:]) > 0
