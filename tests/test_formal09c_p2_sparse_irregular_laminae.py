from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_formal03_correlated_cover_source_ablation as formal03
import generate_formal06_interface_conditioned_development as formal06
import generate_formal06c_subtle_interface_development as formal06c
import generate_formal09c_p2_sparse_irregular_laminae as formal09c_p2


def mini_spec() -> formal03.Spec:
    return formal03.Spec(
        domain_x_m=37.95,
        domain_y_m=24.99,
        dl_m=0.03,
        pml_cells=10,
        physical_side_guard_m=7.02,
        trace_count=256,
        trace_spacing_m=0.09,
        scan_start_x_m=7.5,
        tx_rx_offset_m=0.18,
        ground_y_m=20.01,
        source_y_m=21.0,
        protected_window_end_ns=10.0,
        cover_bins=16,
        transition_levels=6,
    )


def test_p2_locks_formal06c_source_materials_and_spec() -> None:
    assert formal09c_p2.SOURCE == formal06c.SOURCE
    assert formal09c_p2.DESIGN == formal06c.DESIGN
    assert formal09c_p2.default_spec() == formal06c.default_spec()


def test_sparse_laminae_are_deterministic_weak_and_endpoint_visible() -> None:
    spec = mini_spec()
    _, predecessor_bins, _ = formal06.build_bulk_field(spec, design=formal09c_p2.DESIGN)
    left = formal09c_p2.add_sparse_irregular_laminae(spec, predecessor_bins, seed=203)
    right = formal09c_p2.add_sparse_irregular_laminae(spec, predecessor_bins, seed=203)
    assert np.array_equal(left[0], right[0])
    assert left[2] == right[2]
    assert left[3] == right[3]
    assert 1 <= left[3]["laminae_intersecting_first64_native_traces"] <= 2
    assert left[3]["first64_visible_endpoint_count"] >= 1
    assert left[3]["laminae_intersecting_native_scan"] >= 2
    assert np.max(np.abs(left[1])) <= formal09c_p2.BIN_DELTA[2]
    assert all(record["length_m"] <= formal09c_p2.LENGTH_M[2] for record in left[2])
    assert all(
        record["minimum_centerline_separation_m"]
        == formal09c_p2.MIN_CENTERLINE_SEPARATION_M
        for record in left[2]
    )


def test_generation_records_blocked_exact_predecessor_gate(tmp_path: Path) -> None:
    case_dir = formal09c_p2.generate(tmp_path, spec=mini_spec())
    manifest = json.loads((case_dir / "scene_manifest.json").read_text("utf-8"))
    assert manifest["formal_training_allowed"] is False
    assert manifest["strict_line9_holdout_allowed"] is False
    assert manifest["predecessor_lock"]["all_exact"] is True
    assert manifest["comparison_contract"]["exact_common_predecessor_run_available"] is True
    assert manifest["geometry"]["finite_non_target_laminae_count"] >= 3
    assert (case_dir / "preview_FORMAL06C_vs_FORMAL09C_P2_geometry.png").is_file()
