from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import h5py
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_formal03_correlated_cover_source_ablation as formal03
import generate_formal06_interface_conditioned_development as formal06


def mini_spec() -> formal03.Spec:
    return formal03.Spec(
        domain_x_m=31.7,
        domain_y_m=25.0,
        dl_m=0.1,
        pml_cells=10,
        physical_side_guard_m=2.0,
        trace_count=256,
        trace_spacing_m=0.1,
        scan_start_x_m=3.0,
        tx_rx_offset_m=0.2,
        ground_y_m=20.0,
        source_y_m=21.0,
        protected_window_end_ns=10.0,
        cover_bins=8,
        transition_levels=6,
    )


def test_formal06_is_explicitly_development_only() -> None:
    assert list(inspect.signature(formal06.generate).parameters) == [
        "output_root",
        "spec",
    ]
    source = Path(formal06.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("data_yingshan", "line9.npz", "soft_mask_train"):
        assert forbidden not in source


def test_weathered_cap_ends_at_one_stable_bedrock_state() -> None:
    spec = mini_spec()
    full = formal06.material_rows(spec, control=False)
    control = formal06.material_rows(spec, control=True)
    bases = formal06.base_materials(spec)
    bedrock_offset = spec.cover_bins * (spec.transition_levels + 1)
    for base_index, base in enumerate(bases):
        full_bedrock = full[bedrock_offset + base_index]
        control_bedrock = control[bedrock_offset + base_index]
        assert full_bedrock.epsilon_r == formal06.DESIGN.bedrock_epsilon_r
        assert full_bedrock.conductivity_s_per_m == formal06.DESIGN.bedrock_conductivity_s_per_m
        assert control_bedrock.epsilon_r == base.epsilon_r
        assert control_bedrock.conductivity_s_per_m == base.conductivity_s_per_m


def test_generated_case_has_shared_geometry_and_no_visible_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = formal03.SourceVariant(
        "FORMAL06_TEST_GABOR8",
        "gaussian_modulated_zero_mean",
        8e6,
        "formal06_test_gabor8",
        50.0,
        11.5,
    )
    monkeypatch.setattr(formal06, "SOURCE", source)
    case_dir = formal06.generate(tmp_path, spec=mini_spec())
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    policy = json.loads(
        (tmp_path / "FORMAL06_INTERFACE_CONDITIONED_POLICY.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["formal_training_allowed"] is False
    assert manifest["line9_conditioned"] is True
    assert manifest["strict_line9_holdout_allowed"] is False
    assert manifest["strict_pair"]["shared_geometry_hdf5"] is True
    assert manifest["geometry"]["discrete_anomaly_bodies"] == 0
    assert manifest["visibility_gate"]["requires_blind_review_before_label_overlay"] is True
    assert policy["release_order"][3] == "distributed_span_32_trace_full_scene_only_after_blind_pass"
    assert policy["release_order"][4] == "distributed_span_32_trace_strict_pair_only_after_morphology_pass"
    assert not list((case_dir / "labels").glob("*visible*"))
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        assert handle["data"].dtype.name == "int16"
        assert handle["data"].shape[2] == 1


def test_bulk_material_span_is_deliberately_weak() -> None:
    assert (
        formal06.DESIGN.cover_epsilon_max - formal06.DESIGN.cover_epsilon_min
        <= 0.8 + 1e-12
    )
    assert (
        formal06.DESIGN.cover_conductivity_max_s_per_m
        - formal06.DESIGN.cover_conductivity_min_s_per_m
        <= 0.0007 + 1e-12
    )
