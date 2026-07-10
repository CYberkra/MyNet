from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.train_raw_only import DS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"


def test_canonical_line_contains_original_spatial_metadata():
    with np.load(DATA_ROOT / "lines" / "Line9.npz", allow_pickle=False) as data:
        assert str(data["canonical_source"]) == "original_yingshan_csv"
        assert data["longitude"].shape == (2378,)
        assert data["ground_elevation_m"].shape == (2378,)
        assert data["flight_height_agl_m"].shape == (2378,)
        assert np.allclose(
            data["antenna_elevation_m"],
            data["ground_elevation_m"] + data["flight_height_agl_m"],
            atol=2e-4,
        )
        assert np.all(np.diff(data["gnss_cumulative_distance_m"]) >= -1e-8)


def test_dataset_exposes_tracewise_measured_flight_height():
    cfg = {
        "data_root": str(DATA_ROOT),
        "height_resize": 64,
        "width_resize": 64,
        "input_log_scale": 1e-3,
        "per_trace_robust_norm": False,
        "use_terrain_features": False,
        "val_lines": ["Line7"],
        "val_sample_ids": ["Line7_tr0512_0767"],
        "loss": {},
        "augment": {"enabled": False},
    }
    ds = DS("val", cfg)
    sample = ds[0]
    assert sample["altitude"].shape == (64,)
    assert sample["altitude_valid"].shape == (64,)
    assert bool(torch_all(sample["altitude_valid"] > 0.5))
    assert float(sample["altitude"].max()) > 20.0


def torch_all(value):
    return value.all().item()


def test_terrain_features_use_fixed_physical_scaling_without_line9_statistics():
    manifest = json.loads(
        (DATA_ROOT / "terrain_features" / "terrain_feature_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "terrain_features_original_csv_v2"
    assert "No statistics from Line9" in manifest["normalization_policy"]
    assert "main_lines_for_normalization" not in manifest


def test_dataset_policy_declares_tracewise_arrival_height():
    policy = json.loads((DATA_ROOT / "dataset_policy.json").read_text(encoding="utf-8"))
    height = policy["height_policy"]
    assert height["field"] == "flight_height_agl_m"
    assert height["arrival_prior_uses_tracewise_measured_height"] is True
    assert height["window_median_is_legacy_fallback_only"] is True
    assert "arrival_prior_uses_window_median" not in height


def test_canonical_line_contains_explicit_profile_orientation_contract():
    expected_flips = {
        "Line3": True,
        "Line6": True,
        "Line7": False,
        "Line9": True,
        "LineL1": False,
        "LineX1": True,
    }
    for line, expected_flip in expected_flips.items():
        with np.load(DATA_ROOT / "lines" / f"{line}.npz", allow_pickle=False) as data:
            assert str(data["orientation_contract"]) == "canonical arrays remain acquisition order; profile flip is display-only"
            assert bool(data["profile_display_flip"]) is expected_flip
            assert data["profile_chainage_m"].shape == data["gnss_cumulative_distance_m"].shape
            assert np.all(np.diff(data["profile_chainage_m"]) >= -1e-8)


def test_orientation_registry_and_profile_source_archive_are_present():
    registry = DATA_ROOT / "trace_direction_registry.csv"
    contract = DATA_ROOT / "orientation_contract.json"
    source = DATA_ROOT / "source" / "ying_shan_profiles_and_boreholes.zip"
    assert registry.is_file()
    assert contract.is_file()
    assert source.is_file()
    payload = json.loads(contract.read_text(encoding="utf-8"))
    assert payload["canonical_order"].startswith("All saved arrays")
    assert len(payload["lines"]) == 6
