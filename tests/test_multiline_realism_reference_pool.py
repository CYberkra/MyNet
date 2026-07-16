from pathlib import Path

from scripts.build_multiline_realism_reference_pool import (
    PRIMARY_FORMAL_ROLES,
    build_segments,
    write_outputs,
)


LINES_DIR = Path("data/measured/yingshan_v15/lines")


def test_multiline_reference_pool_excludes_x1_from_calibration() -> None:
    segments = build_segments(LINES_DIR, min_traces=64)
    x1_purposes = {segment.purpose for segment in segments if segment.line == "LineX1"}
    assert x1_purposes == {"review_only"}


def test_multiline_reference_pool_keeps_line7_height_exception_as_stress() -> None:
    segments = build_segments(LINES_DIR, min_traces=64)
    stress = [
        segment
        for segment in segments
        if segment.line == "Line7" and segment.purpose == "stress_only"
    ]
    assert [(segment.start, segment.end) for segment in stress] == [(475, 708)]


def test_primary_formal_fold_holds_out_line9() -> None:
    assert PRIMARY_FORMAL_ROLES["Line9"] == "held_out_test"
    assert PRIMARY_FORMAL_ROLES["Line6"] == "validation"
    assert PRIMARY_FORMAL_ROLES["LineX1"] == "review_only"


def test_stress_segments_are_not_nominal_calibration_evidence(tmp_path: Path) -> None:
    segments = build_segments(LINES_DIR, min_traces=64)
    write_outputs(segments, tmp_path, LINES_DIR, min_traces=64)
    text = (tmp_path / "measured_reference_segments.csv").read_text(
        encoding="utf-8"
    )
    line7_stress = next(
        line for line in text.splitlines() if "Line7,stress_only" in line
    )
    assert ",false,stress_only," in line7_stress
