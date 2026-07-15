from __future__ import annotations

import inspect
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


def test_formal06b_is_a_single_factor_tempered_successor() -> None:
    assert list(inspect.signature(formal06b.generate).parameters) == [
        "output_root",
        "spec",
    ]
    unchanged = (
        "cover_epsilon_min",
        "cover_epsilon_max",
        "cover_conductivity_min_s_per_m",
        "cover_conductivity_max_s_per_m",
        "bulk_long_x_scale_m",
        "bulk_long_y_scale_m",
        "bulk_meso_x_scale_m",
        "bulk_meso_y_scale_m",
        "bulk_meso_weight",
    )
    for field in unchanged:
        assert getattr(formal06b.DESIGN, field) == getattr(formal06a.DESIGN, field)
    assert formal06b.SOURCE == formal06a.SOURCE
    assert abs(reflection_proxy(formal06b.DESIGN)) < abs(
        reflection_proxy(formal06a.DESIGN)
    ) / 3.5
    assert -0.025 < reflection_proxy(formal06b.DESIGN) < -0.012


def test_formal06b_preserves_formal06a_geometry_exactly(tmp_path: Path) -> None:
    spec = mini_spec()
    a_dir = formal06a.generate(tmp_path / "a", spec=spec)
    b_dir = formal06b.generate(tmp_path / "b", spec=spec)
    with h5py.File(a_dir / "geology_indices.h5", "r") as a_handle:
        a_indices = a_handle["data"][:]
    with h5py.File(b_dir / "geology_indices.h5", "r") as b_handle:
        b_indices = b_handle["data"][:]
    assert np.array_equal(a_indices, b_indices)
    material_dependent = {
        "geometric_reference_arrival_time_ns.npy",
        "source_referenced_arrival_time_ns.npy",
    }
    for label in a_dir.joinpath("labels").glob("*.npy"):
        if label.name not in material_dependent:
            assert np.array_equal(np.load(label), np.load(b_dir / "labels" / label.name))
    assert not np.array_equal(
        np.load(a_dir / "labels" / "geometric_reference_arrival_time_ns.npy"),
        np.load(b_dir / "labels" / "geometric_reference_arrival_time_ns.npy"),
    )


def test_formal06b_manifest_records_tempered_development_contract(
    tmp_path: Path,
) -> None:
    case_dir = formal06b.generate(tmp_path, spec=mini_spec())
    manifest = json.loads((case_dir / "scene_manifest.json").read_text("utf-8"))
    policy = json.loads(
        (tmp_path / "FORMAL06B_TEMPERED_INTERFACE_POLICY.json").read_text("utf-8")
    )
    assert manifest["formal_training_allowed"] is False
    assert manifest["line9_conditioned"] is True
    assert manifest["ablation"]["predecessor_case_id"] == formal06a.CASE_ID
    assert manifest["geometry"]["discrete_anomaly_bodies"] == 0
    assert manifest["materials"]["basal_reflection_proxy"] == reflection_proxy(
        formal06b.DESIGN
    )
    assert policy["release_order"][3] == "distributed_span_32_trace_full_scene_only_after_blind_pass"
    assert policy["release_order"][4] == "distributed_span_32_trace_strict_pair_only_after_morphology_pass"
    assert not list(case_dir.joinpath("labels").glob("*visible*"))
