from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_full_control_continuity as continuity


def test_response_metrics_reports_continuous_strict_pair_event() -> None:
    time_ns = np.linspace(0.0, 600.0, 6001)
    reference = np.linspace(350.0, 360.0, 8)
    difference = np.zeros((time_ns.size, reference.size))
    for trace, centre in enumerate(reference):
        visible = centre + 21.0 + trace * 0.08
        difference[:, trace] = np.exp(-0.5 * ((time_ns - visible) / 7.0) ** 2)
    metrics = continuity.response_metrics(difference, time_ns, reference)
    assert metrics["dropout_fraction_below_25pct_median"] == 0.0
    assert metrics["adjacent_wavelet_correlation_median"] > 0.99
    assert metrics["adjacent_peak_time_step_abs_ns_p95"] < 2.0
    assert 20.0 < metrics["geometric_to_visible_offset_ns_median"] < 23.0
    path = np.asarray(metrics["path_constrained_signed_difference_peak_ns"])
    assert np.max(np.abs(path - (reference + 21.0))) < 0.5
    assert metrics["path_adjacent_wavelet_correlation_median"] > 0.99
