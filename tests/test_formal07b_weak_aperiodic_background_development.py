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
import generate_formal07b_weak_aperiodic_background_development as formal07b


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


def test_formal07b_locks_formal06c_contract() -> None:
    assert formal07b.SOURCE == formal06c.SOURCE
    assert formal07b.DESIGN == formal06c.DESIGN
    assert formal07b.default_spec() == formal06c.default_spec()

    spec = mini_spec()
    predecessor_profile, _ = formal03.build_profiles(spec)
    candidate_profile, _ = formal03.build_profiles(spec)
    for name in predecessor_profile:
        assert np.array_equal(predecessor_profile[name], candidate_profile[name])


def test_formal07b_adds_only_weak_aperiodic_two_dimensional_texture() -> None:
    spec = mini_spec()
    predecessor, predecessor_bins, _ = formal06.build_bulk_field(
        spec, design=formal07b.DESIGN
    )
    candidate, candidate_bins, stats = formal07b.build_bulk_field(
        spec, design=formal07b.DESIGN
    )

    assert stats["model"] == (
        "formal06c_plus_weak_aperiodic_two_dimensional_texture"
    )
    assert stats["sinusoidal_stratigraphy"] is False
    assert stats["isolated_inclusions"] == 0
    assert stats["point_targets"] == 0
    assert stats["vertical_partitions"] == 0
    assert all(stats["gate_results"].values())
    assert stats["predecessor_latent_correlation"] >= 0.985
    assert not np.array_equal(predecessor, candidate)
    assert not np.array_equal(predecessor_bins, candidate_bins)


def test_formal07b_generation_records_exact_locks_and_training_block(
    tmp_path: Path,
) -> None:
    case_dir = formal07b.generate(tmp_path, spec=mini_spec())
    manifest = json.loads((case_dir / "scene_manifest.json").read_text("utf-8"))

    assert manifest["ablation"]["predecessor_case_id"] == formal06c.CASE_ID
    assert manifest["ablation"]["changed"] == [
        "non-target cover-field weak aperiodic texture only"
    ]
    assert manifest["comparison_contract"][
        "identical_basal_and_transition_profiles"
    ] is True
    assert manifest["comparison_contract"]["changed_factor_group"] == (
        "weak_aperiodic_non_target_background_only"
    )
    assert manifest["predecessor_lock"]["all_exact"] is True
    assert manifest["formal_training_allowed"] is False
    assert manifest["line9_conditioned"] is True
    assert manifest["geometry"]["discrete_anomaly_bodies"] == 0
    assert not list(case_dir.joinpath("labels").glob("*visible*"))
    assert (case_dir / "preview_FORMAL06C_vs_FORMAL07B_geometry.png").is_file()

    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        assert handle["data"].shape == (mini_spec().nx, mini_spec().ny, 1)
