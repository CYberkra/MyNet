import json
from pathlib import Path

import numpy as np

from scripts.generate_formal06g_terrain_acquisition_development import (
    TERRAIN,
    build_terrain_profiles,
    default_spec,
    generate,
)


def test_formal06g_preserves_local_basal_depth_under_bounded_terrain() -> None:
    spec = default_spec()
    profile, stats = build_terrain_profiles(spec)
    local_depth = profile["full_ground_y_m"] - profile["full_basal_y_m"]
    assert np.allclose(local_depth, profile["full_basal_depth_m"])
    assert TERRAIN["agl_min_m"] <= stats["flight_height_agl_min_m"]
    assert stats["flight_height_agl_max_m"] <= TERRAIN["agl_max_m"]
    assert stats["terrain_range_m"] > 0.0


def test_formal06g_records_terrain_as_the_only_changed_factor(tmp_path: Path) -> None:
    source = generate(tmp_path)
    manifest = json.loads((source / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["geometry"]["flat_ground"] is False
    assert manifest["geometry"]["terrain_stage"].startswith("bounded non-periodic")
    assert len(manifest["ablation"]["changed"]) == 1
    assert manifest["formal_training_allowed"] is False
