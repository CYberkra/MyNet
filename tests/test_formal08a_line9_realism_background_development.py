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
import generate_formal08a_line9_realism_background_development as formal08a


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


def test_formal08a_locks_formal06c_physics() -> None:
    assert formal08a.SOURCE == formal06c.SOURCE
    assert formal08a.DESIGN == formal06c.DESIGN
    assert formal08a.default_spec() == formal06c.default_spec()


def test_formal08a_changes_only_middle_cover_texture() -> None:
    spec = mini_spec()
    predecessor, predecessor_bins, _ = formal06.build_bulk_field(
        spec, design=formal08a.DESIGN
    )
    candidate, candidate_bins, stats = formal08a.build_bulk_field(
        spec, design=formal08a.DESIGN
    )

    assert stats["model"] == (
        "formal06c_plus_depth_tapered_multiscale_aperiodic_cover"
    )
    assert all(stats["gate_results"].values())
    assert stats["protected_surface_and_basal_bins_exact"] is True
    assert stats["sinusoidal_stratigraphy"] is False
    assert stats["isolated_inclusions"] == 0
    assert stats["point_targets"] == 0
    assert stats["vertical_partitions"] == 0
    assert not np.array_equal(predecessor, candidate)
    assert not np.array_equal(predecessor_bins, candidate_bins)

    y = (np.arange(spec.ny, dtype=np.float32) + 0.5) * spec.dl_m
    depth = np.float32(spec.ground_y_m) - y
    protected = (depth <= formal08a.ACTIVE_DEPTH_M["surface_zero_end"]) | (
        depth >= formal08a.ACTIVE_DEPTH_M["deep_zero_start"]
    )
    assert np.array_equal(
        predecessor_bins[:, protected], candidate_bins[:, protected]
    )


def test_formal08a_generation_records_line9_conditioning_and_locks(
    tmp_path: Path,
) -> None:
    case_dir = formal08a.generate(tmp_path, spec=mini_spec())
    manifest = json.loads((case_dir / "scene_manifest.json").read_text("utf-8"))
    policy = json.loads(
        (tmp_path / "FORMAL08A_LINE9_REALISM_BACKGROUND_POLICY.json").read_text(
            "utf-8"
        )
    )

    assert manifest["ablation"]["predecessor_case_id"] == formal06c.CASE_ID
    assert manifest["line9_conditioned"] is True
    assert manifest["strict_line9_holdout_allowed"] is False
    assert manifest["formal_training_allowed"] is False
    assert manifest["predecessor_lock"]["all_exact"] is True
    assert manifest["comparison_contract"][
        "protected_surface_and_basal_bins_exact"
    ] is True
    assert manifest["geometry"]["discrete_anomaly_bodies"] == 0
    assert not list(case_dir.joinpath("labels").glob("*visible*"))
    assert (case_dir / "preview_FORMAL06C_vs_FORMAL08A_geometry.png").is_file()
    assert "locked_index_array_sha256" not in policy
    assert policy["candidate_index_array_sha256"]
    assert policy["runtime_state"] == "not_started"
    assert policy["line9_conditioned"] is True
    assert policy["strict_line9_holdout_allowed"] is False

    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        assert handle["data"].shape == (mini_spec().nx, mini_spec().ny, 1)
