from pathlib import Path

import numpy as np

from scripts.generate_formal09c_sparse_event_field import (
    EventFit,
    Variant,
    build_candidate_basis,
    detect_line_events,
    render_event_field,
)
from scripts.generate_formal09b1_empirical_spectrum import SpectrumFit


def dummy_temporal_fit() -> SpectrumFit:
    frequency = np.fft.rfftfreq(512, d=1.4e-9)
    amplitude = np.exp(-0.5 * ((frequency - 90e6) / 35e6) ** 2)
    return SpectrumFit(
        "test", ("Line3",), frequency, amplitude, {"Line3": amplitude}, {"Line3": 10}
    )


def dummy_event_fit() -> EventFit:
    return EventFit(
        lines=("Line3",),
        line_event_counts={"Line3": 8},
        line_lengths_m={"Line3": 25.0},
        events_per_25m={"Line3": 8.0},
        pooled_quantiles={
            "length_m": (2.0, 4.0, 8.0),
            "slope_ns_per_m": (-4.0, 0.0, 4.0),
            "curvature_ns_per_m2": (-0.1, 0.0, 0.1),
            "amplitude_p99_fraction": (0.2, 0.3, 0.45),
            "center_time_ns": (270.0, 330.0, 450.0),
        },
        pooled_events_per_25m=8.0,
    )


def test_event_renderer_is_deterministic_and_avoids_target() -> None:
    time_ns = np.arange(501, dtype=np.float64) * 1.4
    target = np.full(32, 410.0)
    args = (
        (501, 32),
        time_ns,
        target,
        0.72,
        dummy_temporal_fit(),
        dummy_event_fit(),
        Variant("test", 0.35, 1.0),
        21,
    )
    left = render_event_field(*args)
    right = render_event_field(*args)
    assert np.array_equal(left[0], right[0])
    assert left[1] == right[1]
    assert all(record["target_overlap_fraction"] <= 0.20 for record in left[1])
    assert all(abs(record["curvature_ns_per_m2"]) <= 0.12 for record in left[1])


def test_candidate_basis_is_finite_and_normalised() -> None:
    rng = np.random.default_rng(3)
    basis = build_candidate_basis(
        rng.standard_normal((101, 12)), rng.standard_normal((101, 12)), 0.35
    )
    assert np.all(np.isfinite(basis))
    assert abs(float(np.mean(basis))) < 1e-12
    assert abs(float(np.std(basis)) - 1.0) < 1e-12


def test_real_line_detector_returns_finite_events() -> None:
    events, length_m = detect_line_events(
        Path("data/measured/yingshan_v15/lines/Line3.npz"), [(21, 1791)]
    )
    assert length_m > 100.0
    assert len(events) > 10
    assert all(np.isfinite(event.length_m) for event in events)
    assert all(event.length_m >= 1.5 for event in events)
