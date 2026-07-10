from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "yingshan_direction_profile_audit"


def read_rows(name: str):
    with (REPORT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_crossings_are_geometrically_coincident():
    rows = read_rows("line_intersections.csv")
    assert len(rows) == 8
    assert max(float(row["nearest_separation_m"]) for row in rows) < 0.05


def test_profile_crossing_orders_match_orientation_contract():
    rows = {row["line"]: row for row in read_rows("profile_alignment_checks.csv")}
    assert rows["Line3"]["crossings_left_to_right"].startswith("Line7@")
    assert "LineL1@" in rows["Line3"]["crossings_left_to_right"]
    assert rows["Line6"]["crossings_left_to_right"].startswith("Line7@")
    assert rows["Line7"]["crossings_left_to_right"].startswith("Line6@")
    assert rows["LineL1"]["crossings_left_to_right"].startswith("Line3@")
    assert rows["LineX1"]["crossings_left_to_right"].startswith("LineL1@")


def test_crossing_time_mismatches_are_not_silently_accepted():
    rows = {row["crossing"]: row for row in read_rows("line_intersections.csv")}
    assert rows["Line3-Line9"]["crossing_qc_grade"].startswith("critical_mismatch")
    assert rows["Line6-Line9"]["crossing_qc_grade"].startswith("high_risk")
    assert rows["Line3-Line7"]["crossing_qc_grade"] == "pass"
