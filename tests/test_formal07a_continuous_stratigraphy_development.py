from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_formal03_correlated_cover_source_ablation as formal03
import generate_formal06_interface_conditioned_development as formal06
import generate_formal06c_subtle_interface_development as formal06c
import generate_formal07a_continuous_stratigraphy_development as formal07a


def mini_spec() -> formal03.Spec:
    return formal03.Spec(
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


def test_formal07a_changes_only_geology_morphology_factor_group() -> None:
    assert formal07a.SOURCE == formal06c.SOURCE
    assert formal07a.DESIGN == formal06c.DESIGN

    spec = mini_spec()
    predecessor_profile, predecessor_stats = formal03.build_profiles(spec)
    successor_profile, successor_stats = formal07a.build_profiles(spec)
    assert float(successor_stats["range_m"]) < float(predecessor_stats["range_m"])
    assert float(successor_stats["abs_slope_p95"]) < float(
        predecessor_stats["abs_slope_p95"]
    )
    assert not np.array_equal(
        predecessor_profile["full_basal_depth_m"],
        successor_profile["full_basal_depth_m"],
    )


def test_formal07a_adds_continuous_stratigraphy_without_point_targets() -> None:
    spec = mini_spec()
    _, predecessor_bins, predecessor_stats = formal06.build_bulk_field(
        spec, design=formal07a.DESIGN
    )
    _, successor_bins, successor_stats = formal07a.build_bulk_field(
        spec, design=formal07a.DESIGN
    )
    assert successor_stats["model"] == "continuous_warped_multiscale_stratigraphy"
    assert successor_stats["isolated_inclusions"] == 0
    assert successor_stats["point_targets"] == 0
    assert successor_stats["vertical_neighbor_bin_change_rate"] > predecessor_stats[
        "vertical_neighbor_bin_change_rate"
    ]
    assert not np.array_equal(predecessor_bins, successor_bins)


def test_formal07a_generation_preserves_strict_pair_and_development_gate(
    tmp_path: Path,
) -> None:
    case_dir = formal07a.generate(tmp_path, spec=mini_spec())
    manifest = json.loads((case_dir / "scene_manifest.json").read_text("utf-8"))

    assert manifest["ablation"]["predecessor_case_id"] == formal06c.CASE_ID
    assert manifest["comparison_contract"]["changed_factor_group"] == (
        "geology_morphology_only"
    )
    assert manifest["formal_training_allowed"] is False
    assert manifest["line9_conditioned"] is True
    assert manifest["geometry"]["discrete_anomaly_bodies"] == 0
    assert manifest["strict_pair"]["shared_geometry_hdf5"] is True
    assert manifest["strict_pair"][
        "control_restores_cap_and_bedrock_to_each_local_cover_bin"
    ] is True
    assert not list(case_dir.joinpath("labels").glob("*visible*"))
    assert (case_dir / "preview_FORMAL06C_vs_FORMAL07A_geometry.png").is_file()
    assert (case_dir / "preview_model_structure_enhanced.png").is_file()

    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        assert handle["data"].shape == (mini_spec().nx, mini_spec().ny, 1)
