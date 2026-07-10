from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pgdacsnet.spatial_orientation import (
    ORIENTATION_REGISTRY,
    align_array_for_display,
    display_distance_axis,
    orientation_metadata,
    profile_index_order,
)
from scripts.train_raw_only import flip_directional_terrain_channels


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"


def test_orientation_registry_has_all_survey_lines_and_expected_flips():
    assert set(ORIENTATION_REGISTRY) == {"Line3", "Line6", "Line7", "Line9", "LineL1", "LineX1"}
    expected = {
        "Line3": True,
        "Line6": True,
        "Line7": False,
        "Line9": True,
        "LineL1": False,
        "LineX1": True,
    }
    assert {line: item.profile_display_flip for line, item in ORIENTATION_REGISTRY.items()} == expected


def test_canonical_line_bearings_match_trace_order():
    expected_compass = {
        "Line3": "S",
        "Line6": "S",
        "Line7": "E",
        "Line9": "W",
        "LineL1": "W",
        "LineX1": "S",
    }
    for line, compass in expected_compass.items():
        with np.load(DATA_ROOT / "lines" / f"{line}.npz", allow_pickle=False) as data:
            meta = orientation_metadata(line, data["longitude"], data["latitude"])
            assert meta["acquisition_compass"] == compass
            assert str(data["canonical_source"]) == "original_yingshan_csv"


def test_profile_display_transform_is_view_only_and_reversible():
    values = np.arange(6)
    assert np.array_equal(align_array_for_display(values, "Line3", orientation="acquisition"), values)
    assert np.array_equal(align_array_for_display(values, "Line3", orientation="profile"), values[::-1])
    assert np.array_equal(align_array_for_display(values, "Line7", orientation="profile"), values)
    assert np.array_equal(profile_index_order(6, "Line9"), np.arange(6)[::-1])


def test_profile_display_distance_starts_at_zero_after_flip():
    acquisition = np.asarray([0.0, 2.0, 5.0, 9.0])
    assert np.allclose(display_distance_axis(acquisition, "Line7", orientation="profile"), acquisition)
    assert np.allclose(display_distance_axis(acquisition, "Line9", orientation="profile"), [0.0, 4.0, 7.0, 9.0])


def test_horizontal_flip_negates_directional_metadata_only():
    # raw, relative height, ground, altitude, terrain slope, trace position
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[10.0, 11.0, 12.0]],
            [[20.0, 21.0, 22.0]],
            [[30.0, 31.0, 32.0]],
            [[-2.0, 0.0, 2.0]],
            [[-1.0, 0.0, 1.0]],
        ]
    )
    flipped = torch.flip(x, dims=[-1])
    corrected = flip_directional_terrain_channels(
        flipped,
        {
            "terrain_feature_names": [
                "relative_height_z",
                "ground_elevation_z",
                "altitude_z",
                "terrain_slope_z",
                "trace_position",
            ]
        },
    )
    assert torch.equal(corrected[0], torch.tensor([[3.0, 2.0, 1.0]]))
    assert torch.equal(corrected[1], torch.tensor([[12.0, 11.0, 10.0]]))
    assert torch.equal(corrected[4], torch.tensor([[-2.0, -0.0, 2.0]]))
    assert torch.equal(corrected[5], torch.tensor([[-1.0, -0.0, 1.0]]))
