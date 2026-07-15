from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "measured" / "yingshan_v15"
LINES = ("Line3", "Line6", "Line7", "Line9", "LineL1", "LineX1")
VERSION = "YINGSHAN_V15_FINAL_20260710"


@pytest.fixture(scope="session")
def line_data() -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for line in LINES:
        with np.load(DATA / "lines" / f"{line}.npz", allow_pickle=False) as archive:
            result[line] = {name: archive[name] for name in archive.files}
    return result


def test_release_manifest_and_sources_are_self_contained():
    manifest = json.loads((DATA / "manifests" / "v15_final_manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((DATA / "dataset_policy.json").read_text(encoding="utf-8"))
    assert manifest["version"] == VERSION
    assert manifest["formal_training_allowed"] is False
    assert manifest["crossing_supervision_resolved"] is True
    assert policy["training_allowed"] is False
    assert (DATA / "source" / "ying_shan_measurement_lines_original.zip").is_file()
    assert (DATA / "source" / "ying_shan_profiles_and_boreholes.zip").is_file()


def test_v14_rollback_arrays_are_embedded(line_data):
    for line in LINES:
        archive = line_data[line]
        assert archive["soft_mask_v14_original"].shape == archive["soft_mask_train"].shape
        assert archive["status_code_v14_original"].shape == archive["status_code"].shape
        assert archive["label_weight_v14_original"].shape == archive["label_weight"].shape
        assert str(archive["v15_final_version"].item()) == VERSION


def test_line9_remains_the_unchanged_test_anchor(line_data):
    line9 = line_data["Line9"]
    assert np.array_equal(line9["soft_mask_review_v15_final"], line9["soft_mask_v14_original"])
    assert not line9["v15_final_ignore_trace"].any()
    assert str(line9["split"].item()) == "test"


def test_accepted_weak_relabels_are_present(line_data):
    line3 = line_data["Line3"]
    assert 450.0 <= float(line3["v15_final_center_time_ns"][167]) <= 455.0
    assert int(line3["status_code"][167]) == 2
    assert not bool(line3["v15_final_ignore_trace"][167])
    assert str(line3["v15_final_decision_code"][167]) == "RELABEL_WEAK_LINE9_ANCHORED"

    x1 = line_data["LineX1"]
    assert 326.0 <= float(x1["v15_final_center_time_ns"][659]) <= 330.0
    assert int(x1["status_code"][659]) == 2
    assert not bool(x1["v15_final_ignore_trace"][659])
    assert str(x1["v15_final_decision_code"][659]) == "RELABEL_WEAK_L1_ANCHORED"
    assert str(x1["split"].item()) == "exclude"


def test_ambiguous_crossings_are_excluded(line_data):
    assert bool(line_data["Line6"]["v15_final_ignore_trace"][523])
    assert bool(line_data["LineX1"]["v15_final_ignore_trace"][197])
    for archive in line_data.values():
        ignored = archive["v15_final_ignore_trace"].astype(bool)
        if ignored.any():
            assert np.all(archive["soft_mask_train"][:, ignored] == 0.0)
            assert np.all(archive["label_weight"][ignored] == 0.0)
            assert np.all(archive["status_code"][ignored] == 2)


def test_all_78_windows_match_the_full_line_release(line_data):
    rows = list(csv.DictReader((DATA / "window_index.csv").open(encoding="utf-8")))
    assert len(rows) == 78
    for row in rows:
        start, end = int(row["start"]), int(row["end"]) + 1
        full = line_data[row["line"]]
        with np.load(DATA / "windows" / f"{row['sample_id']}.npz", allow_pickle=False) as window:
            assert np.array_equal(window["x_raw"], full["raw_full_normalized"][:, start:end])
            assert np.array_equal(window["y_mask"], full["soft_mask_train"][:, start:end])
            assert np.array_equal(window["ignore_mask"], full["ignore_mask"][:, start:end])
            assert str(window["label_version"].item()) == VERSION
