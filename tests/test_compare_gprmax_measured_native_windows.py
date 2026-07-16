from __future__ import annotations

import numpy as np
import pytest

from scripts.compare_gprmax_measured_native_windows import (
    RESEARCH_ROLES,
    _centered_window,
    _longest_true_run,
    _short_case_label,
)


def test_longest_true_run_includes_terminal_value() -> None:
    mask = np.asarray([False, True, True, False, True, True, True])
    assert _longest_true_run(mask) == (4, 7)


def test_centered_window_stays_inside_eligible_extent_when_possible() -> None:
    assert _centered_window(100, 300, 64, 500) == (168, 232)


def test_centered_window_rejects_short_eligible_run() -> None:
    with pytest.raises(ValueError, match="fewer than 64"):
        _centered_window(10, 50, 64, 100)


def test_comparison_source_has_no_case_specific_panel_label() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "compare_gprmax_measured_native_windows.py").read_text(
        encoding="utf-8"
    )
    assert "scene['case_id']" in source
    assert "FORMAL09C-P1 (development)" not in source


def test_case_label_and_research_roles_are_explicit() -> None:
    assert _short_case_label("FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE") == "FORMAL09C_P2_SPARSE"
    assert RESEARCH_ROLES["Line3"] == "fit"
    assert RESEARCH_ROLES["Line6"] == "validation"
    assert RESEARCH_ROLES["Line9"] == "heldout"
    assert RESEARCH_ROLES["LineX1"] == "review-only"
