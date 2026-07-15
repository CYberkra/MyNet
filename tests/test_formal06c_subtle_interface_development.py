from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_formal06_interface_conditioned_development as formal06a
import generate_formal06b_tempered_interface_development as formal06b
import generate_formal06c_subtle_interface_development as formal06c


def mini_spec():
    return formal06a.formal03.Spec(
        domain_x_m=28.44,
        domain_y_m=24.99,
        dl_m=0.03,
        pml_cells=10,
        physical_side_guard_m=2.01,
        trace_count=256,
        trace_spacing_m=0.09,
        scan_start_x_m=3.0,
        tx_rx_offset_m=0.18,
        ground_y_m=20.01,
        source_y_m=21.0,
        protected_window_end_ns=10.0,
        cover_bins=8,
        transition_levels=6,
    )


def reflection_proxy(design: formal06a.MaterialDesign) -> float:
    return (
        math.sqrt(design.bedrock_epsilon_r)
        - math.sqrt(design.weathered_cap_epsilon_r)
    ) / (
        math.sqrt(design.bedrock_epsilon_r)
        + math.sqrt(design.weathered_cap_epsilon_r)
    )


def test_formal06c_halves_only_the_remaining_interface_contrast() -> None:
    unchanged = (
        "cover_epsilon_min",
        "cover_epsilon_max",
        "cover_conductivity_min_s_per_m",
        "cover_conductivity_max_s_per_m",
        "weathered_cap_epsilon_r",
        "weathered_cap_conductivity_s_per_m",
        "bulk_long_x_scale_m",
        "bulk_long_y_scale_m",
        "bulk_meso_x_scale_m",
        "bulk_meso_y_scale_m",
        "bulk_meso_weight",
    )
    for field in unchanged:
        assert getattr(formal06c.DESIGN, field) == getattr(formal06b.DESIGN, field)
    assert formal06c.SOURCE == formal06b.SOURCE
    assert abs(reflection_proxy(formal06c.DESIGN)) < abs(
        reflection_proxy(formal06b.DESIGN)
    ) * 0.55
    assert -0.012 < reflection_proxy(formal06c.DESIGN) < -0.006


def test_formal06c_preserves_formal06_geometry(tmp_path: Path) -> None:
    spec = mini_spec()
    a_dir = formal06a.generate(tmp_path / "a", spec=spec)
    c_dir = formal06c.generate(tmp_path / "c", spec=spec)
    with h5py.File(a_dir / "geology_indices.h5", "r") as a_handle:
        a_indices = a_handle["data"][:]
    with h5py.File(c_dir / "geology_indices.h5", "r") as c_handle:
        c_indices = c_handle["data"][:]
    assert np.array_equal(a_indices, c_indices)
    manifest = json.loads((c_dir / "scene_manifest.json").read_text("utf-8"))
    assert manifest["ablation"]["predecessor_case_id"] == formal06b.CASE_ID
    assert manifest["formal_training_allowed"] is False
    assert manifest["line9_conditioned"] is True
    assert not list(c_dir.joinpath("labels").glob("*visible*"))
