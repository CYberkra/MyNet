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
import generate_formal09c_p1_dense_physical_finite_laminae as formal09c_p1


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


def test_formal09c_p1_locks_formal06c_physics() -> None:
    assert formal09c_p1.SOURCE == formal06c.SOURCE
    assert formal09c_p1.DESIGN == formal06c.DESIGN
    assert formal09c_p1.default_spec() == formal06c.default_spec()


def test_finite_laminae_are_deterministic_bounded_and_cover_first64() -> None:
    spec = mini_spec()
    _, predecessor_bins, _ = formal06.build_bulk_field(
        spec, design=formal09c_p1.DESIGN
    )
    left = formal09c_p1.add_finite_laminae(spec, predecessor_bins, seed=91)
    right = formal09c_p1.add_finite_laminae(spec, predecessor_bins, seed=91)
    assert np.array_equal(left[0], right[0])
    assert left[2] == right[2]
    assert left[3] == right[3]
    assert left[3]["events_intersecting_first64_native_traces"] >= 2
    assert left[3]["events_intersecting_first64_native_traces"] <= 4
    assert left[3]["events_intersecting_native_scan"] >= 7
    assert np.max(np.abs(left[1])) <= formal09c_p1.BIN_DELTA[2]
    assert all(record["length_m"] >= 1.8 for record in left[2])
    assert all(record["length_m"] <= 12.0 for record in left[2])
    assert all(record["overlap_fraction"] <= 0.10 for record in left[2])


def test_generation_records_blocked_native_topology_gate(tmp_path: Path) -> None:
    case_dir = formal09c_p1.generate(tmp_path, spec=mini_spec())
    manifest = json.loads((case_dir / "scene_manifest.json").read_text("utf-8"))
    assert manifest["formal_training_allowed"] is False
    assert manifest["strict_line9_holdout_allowed"] is False
    assert manifest["predecessor_lock"]["all_exact"] is True
    assert manifest["comparison_contract"]["sparse_stride8_topology_gate_allowed"] is False
    assert manifest["geometry"]["finite_non_target_laminae_count"] >= 7
    assert (case_dir / "preview_FORMAL06C_vs_FORMAL09C_P1_geometry.png").is_file()
    assert "visible_phase" not in " ".join(
        path.name for path in case_dir.joinpath("labels").glob("*.npy")
    )
