from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_native_256_release_pilot import build
from scripts.create_native_domain_equivalence import create_equivalence
from scripts.create_native_256_waveform_ablation import create_waveform_ablation
from scripts.capture_gprmax_trace_contract import trace_filename, trace_index_for_path
from scripts.postprocess_sfcw_band_equivalent import apply_sfcw_band_equivalent, raised_cosine_sfcw_window
from scripts.run_native_256_release_pilot import _remove_geometry_views, stage_case
from scripts.audit_native_256_spatial_pilot import _position_tolerance_m
from scripts.audit_native_domain_equivalence import _corr
from scripts.audit_native_256_family_spatial_pilot import (
    _audit_mode,
    _common_output_end_ns,
    _continuity_step_limit_ns,
    _full_scene_detectability_gate,
    _path_step_statistics,
    _portable_path as _family_portable_path,
    _required_output_stems,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data" / "simulation_contract_v2"


def test_recommended_standard_is_native_and_boundary_safe() -> None:
    standard = json.loads((CONTRACT / "recommended_native_256_v1.json").read_text(encoding="utf-8"))
    assert standard["formal_training_allowed"] is False
    assert standard["canonical_output"]["trace_count"] == 256
    assert standard["canonical_output"]["time_samples"] == 501
    assert standard["canonical_output"]["horizontal_resize_or_padding"] == "forbidden"
    assert standard["fdtd"]["minimum_side_guard_from_inner_pml_m"] >= 109.5
    assert standard["acquisition"]["waveform"] == "ricker"


def test_native_pilot_generation_matches_static_contract(tmp_path: Path) -> None:
    report = build(
        cases_path=CONTRACT / "recommended_native_256_cases_v1.json",
        materials_path=CONTRACT / "recommended_native_256_materials_v1.json",
        standard_path=CONTRACT / "recommended_native_256_v1.json",
        out_root=tmp_path / "native_pilot",
        selected=set(),
        overwrite=False,
    )
    assert report["static_validation_ok"] is True
    assert report["case_count"] == 6
    assert report["positive_case_count"] == 4
    assert report["negative_case_count"] == 2
    for case_id in report["case_ids"]:
        manifest = json.loads((tmp_path / "native_pilot" / case_id / "scene_manifest.json").read_text(encoding="utf-8"))
        grid = manifest["grid"]
        assert grid["trace_count"] == 256
        assert grid["canonical_output_samples"] == 501
        assert grid["trace_spacing_m"] == 0.09
        assert manifest["formal_training_allowed"] is False
        assert manifest["line9_conditioned"] is False


def test_native_runner_stages_source_and_captures_trace_names_before_merge(tmp_path: Path) -> None:
    runner = (ROOT / "scripts" / "run_native_256_release_pilot.py").read_text(encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "scene_manifest.json").write_text(json.dumps({"grid": {"trace_count": 256}}), encoding="utf-8")
    (source / "full_scene.in").write_text("#title: source deck\n", encoding="utf-8")
    staged = stage_case(source, tmp_path / "solver_runs" / "case", requested_trace_count=32, geometry_only=False)
    assert (staged / "full_scene.in").read_text(encoding="utf-8") == "#title: source deck\n"
    provenance = json.loads((staged / "run_manifest.json").read_text(encoding="utf-8"))
    assert provenance["source_deck_read_only"] is True
    assert provenance["mode"] == "smoke_subset"
    assert 'shutil.copytree' in runner
    assert '"--trace-count"' in runner
    assert 'capture_gprmax_trace_contract.py' in runner
    assert 'tools.outputfiles_merge' in runner
    assert "_geometry_input" in runner
    assert "_remove_geometry_views" in runner
    assert "_load_vcvars_environment" in runner
    assert "gprmax_vcvars" in runner
    assert "full_causal_pair_complete" in runner


def test_geometry_views_are_hashed_before_transient_cleanup(tmp_path: Path) -> None:
    view = tmp_path / "geometry_check_full.vti"
    payload = b"transient geometry evidence"
    view.write_bytes(payload)
    records = _remove_geometry_views(tmp_path)
    assert not view.exists()
    assert records == [
        {
            "name": "geometry_check_full.vti",
            "bytes": len(payload),
            "sha256": "ceba7f7d70d574edda8c24a6b7377a2cf5ef0435ccc944f7f5960d5ffc38bc85",
            "deleted": True,
        }
    ]


def test_native_runner_stages_distributed_audit_subset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "target_presence": True,
        "grid": {"trace_count": 256, "trace_spacing_m": 0.09},
    }
    (source / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    deck = "#src_steps: 0.09 0 0\n#rx_steps: 0.09 0 0\n"
    for name in ("full_scene.in", "no_basal_contrast_control.in", "air_reference.in"):
        (source / name).write_text(deck, encoding="utf-8")
    staged = stage_case(
        source,
        tmp_path / "run",
        requested_trace_count=32,
        geometry_only=False,
        trace_stride=8,
    )
    assert "#src_steps: 0.72 0 0" in (staged / "full_scene.in").read_text(encoding="utf-8")
    provenance = json.loads((staged / "run_manifest.json").read_text(encoding="utf-8"))
    assert provenance["mode"] == "distributed_smoke_subset"
    assert provenance["selected_trace_indices_zero_based"] == list(range(0, 249, 8))


def test_native_runner_can_stage_only_the_causal_pair(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "target_presence": True,
        "grid": {"trace_count": 256, "trace_spacing_m": 0.09},
    }
    (source / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    deck = "#src_steps: 0.09 0 0\n#rx_steps: 0.09 0 0\n"
    for name in ("full_scene.in", "no_basal_contrast_control.in", "air_reference.in"):
        (source / name).write_text(deck, encoding="utf-8")
    staged = stage_case(
        source,
        tmp_path / "run",
        requested_trace_count=32,
        geometry_only=False,
        trace_stride=8,
        include_air_reference=False,
    )
    provenance = json.loads((staged / "run_manifest.json").read_text(encoding="utf-8"))
    assert provenance["input_groups"] == ["full_scene", "no_basal_contrast_control"]
    assert "#src_steps: 0.72 0 0" in (staged / "full_scene.in").read_text(encoding="utf-8")


def test_native_runner_can_stage_full_scene_only_for_morphology(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "target_presence": True,
        "grid": {"trace_count": 256, "trace_spacing_m": 0.09},
    }
    (source / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    deck = "#src_steps: 0.09 0 0\n#rx_steps: 0.09 0 0\n"
    for name in ("full_scene.in", "no_basal_contrast_control.in", "air_reference.in"):
        (source / name).write_text(deck, encoding="utf-8")
    staged = stage_case(
        source,
        tmp_path / "run",
        requested_trace_count=24,
        geometry_only=False,
        trace_stride=11,
        include_air_reference=False,
        full_scene_only=True,
    )
    provenance = json.loads((staged / "run_manifest.json").read_text(encoding="utf-8"))
    assert provenance["input_groups"] == ["full_scene"]
    assert provenance["causal_pair_complete"] is False
    assert provenance["selected_trace_indices_zero_based"][-1] == 253
    assert "#src_steps: 0.99 0 0" in (staged / "full_scene.in").read_text(encoding="utf-8")


def test_cv01_domain_equivalence_is_an_exact_grid_crop(tmp_path: Path) -> None:
    source = (
        ROOT
        / "data"
        / "PGDA_SYNTH_DATASET_V2"
        / "01_native_256_correlated_voxel_batch_v1"
        / "N256_CV01_BALANCED_MULTISCALE_POS"
    )
    if not source.is_dir():
        pytest.skip("CV01 source deck is not present in this checkout")
    report = create_equivalence(source, tmp_path / "equivalence", crop_cells=1313, overwrite=False)
    manifest = json.loads((tmp_path / "equivalence" / "scene_manifest.json").read_text(encoding="utf-8"))
    assert report["inner_pml_guard_m"] >= 80.0
    assert manifest["grid"]["nx_ny_nz"] == [8180, 1600, 1]
    assert manifest["geometry"]["domain_equivalence"]["exact_voxel_crop"] is True
    assert manifest["geometry"]["domain_equivalence"]["crop_each_side_cells"] == 1313


def test_native_domain_crop_supports_true_negative_family(tmp_path: Path) -> None:
    source = (
        ROOT
        / "data"
        / "PGDA_SYNTH_DATASET_V2"
        / "01_native_256_correlated_voxel_batch_v1"
        / "N256_CV04_UPPER_CLUTTER_TRUE_NEG"
    )
    if not source.is_dir():
        pytest.skip("CV04 source deck is not present in this checkout")
    output = tmp_path / "negative_crop"
    report = create_equivalence(source, output, crop_cells=1313, overwrite=False)
    manifest = json.loads((output / "scene_manifest.json").read_text(encoding="utf-8"))
    assert report["inner_pml_guard_m"] >= 80.0
    assert manifest["case_id"] == "N256_CV04_UPPER_CLUTTER_TRUE_NEG_80M"
    assert manifest["target_presence"] is False
    assert (output / "full_scene.in").is_file()
    assert not (output / "no_basal_contrast_control.in").exists()
    assert not (output / "geometry_check_control.in").exists()


def test_native_runner_restores_manifest_geometry_hdf5(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "target_presence": False,
        "grid": {"trace_count": 256, "trace_spacing_m": 0.09},
        "geometry": {"index_file": "geology_indices.h5"},
    }
    (source / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "full_scene.in").write_text("#title: full\n", encoding="utf-8")
    (source / "air_reference.in").write_text("#title: air\n", encoding="utf-8")
    geometry = b"audited geometry source"
    (source / "geology_indices.h5").write_bytes(geometry)
    staged = stage_case(source, tmp_path / "run", requested_trace_count=1, geometry_only=False)
    assert (staged / "geology_indices.h5").read_bytes() == geometry


def test_gprmax_single_trace_contract_uses_unnumbered_output_name() -> None:
    assert trace_filename("full_scene", 1, 1) == "full_scene.out"
    assert trace_index_for_path(Path("full_scene.out"), "full_scene", 1) == 1
    assert trace_index_for_path(Path("full_scene1.out"), "full_scene", 1) is None
    assert trace_filename("full_scene", 1, 2) == "full_scene1.out"


def test_spatial_audit_position_tolerance_is_subcell() -> None:
    tolerance = _position_tolerance_m(0.0225)
    assert tolerance == pytest.approx(2.25e-5)
    assert 3.7e-6 < tolerance < 0.001 * 0.0225 + 1e-12


def test_domain_equivalence_correlation_avoids_numpy_corrcoef_runtime() -> None:
    reference = np.asarray([1.0, 3.0, 2.0, -1.0], dtype=np.float64)
    assert _corr(reference, reference) == pytest.approx(1.0)
    assert _corr(reference, -reference) == pytest.approx(-1.0)


def test_family_spatial_audit_keeps_positive_and_negative_semantics_separate() -> None:
    positive = {"target_presence": True}
    negative = {"target_presence": False}
    assert _audit_mode(positive) == "positive_pair"
    assert _required_output_stems(positive) == ("full_scene", "no_basal_contrast_control")
    assert _audit_mode(negative) == "true_negative_full"
    assert _required_output_stems(negative) == ("full_scene",)


def test_family_spatial_audit_scales_continuity_with_sparse_trace_spacing() -> None:
    assert _continuity_step_limit_ns(1) == pytest.approx(5.6)
    assert _continuity_step_limit_ns(8) == pytest.approx(44.8)
    with pytest.raises(ValueError, match="positive"):
        _continuity_step_limit_ns(0)


def test_one_trace_causal_smoke_has_defined_zero_path_steps() -> None:
    stats = _path_step_statistics(np.asarray([417.0]), trace_stride=1)
    assert stats == {
        "p95_ns": 0.0,
        "max_ns": 0.0,
        "per_canonical_trace_p95_ns": 0.0,
        "per_canonical_trace_max_ns": 0.0,
    }


def test_family_spatial_audit_uses_real_common_output_end() -> None:
    loaded = {
        "full_scene": (0.5e-9, np.zeros((1301, 2)), {}),
        "no_basal_contrast_control": (0.5e-9, np.zeros((1301, 2)), {}),
    }
    assert _common_output_end_ns(loaded) == pytest.approx(650.0)
    loaded["no_basal_contrast_control"] = (0.5e-9, np.zeros((1299, 2)), {})
    with pytest.raises(RuntimeError, match="end times"):
        _common_output_end_ns(loaded)


def test_full_scene_detectability_rejects_locally_strong_but_early_wave_buried_target() -> None:
    metrics = {
        "full_scene_target_to_local_background_rms": 3.316,
        "full_scene_target_to_background_rms": 0.628,
        "raw_target_to_early_rms": 1.81e-5,
    }
    manifest = {
        "visibility_gate": {
            "full_scene_target_to_local_background_rms_min": 1.8,
            "full_scene_target_to_local_background_rms_max": 4.5,
            "full_scene_target_to_background_rms_min": 0.35,
            "raw_target_to_early_rms_min": 0.001,
        }
    }
    gate = _full_scene_detectability_gate(metrics, manifest)
    assert gate["checks"]["full_scene_target_to_local_background_rms"] is True
    assert gate["checks"]["full_scene_target_to_background_rms"] is True
    assert gate["checks"]["raw_target_to_early_rms"] is False
    assert gate["passed"] is False


def test_full_scene_detectability_keeps_human_review_as_separate_hard_gate() -> None:
    metrics = {
        "full_scene_target_to_local_background_rms": 2.0,
        "full_scene_target_to_background_rms": 0.5,
        "raw_target_to_early_rms": 0.002,
    }
    gate = _full_scene_detectability_gate(metrics, {"visibility_gate": {}})
    assert gate["automatic_metrics_passed"] is True
    assert gate["human_blind_review_required"] is True
    assert gate["human_morphology_review"] == "pending"
    assert gate["raw_target_to_early_is_cross_domain_diagnostic_only"] is True


def test_family_spatial_audit_records_portable_reference_paths() -> None:
    path = ROOT / "data" / "case" / "labels" / "source_referenced_arrival_time_ns.npy"
    assert _family_portable_path(path) == "data/case/labels/source_referenced_arrival_time_ns.npy"


def test_ricker100_ablation_changes_only_waveform_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "case_id": "N256_TEST_POS_80M",
        "scene_family_id": "test_family",
        "target_presence": True,
        "formal_training_allowed": False,
        "source": {"waveform": "ricker", "center_frequency_hz": 55e6},
        "spec": {"center_frequency_hz": 55e6},
        "grid": {},
        "geometry": {"index_file": "geology_indices.h5"},
    }
    (source / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "geology_indices.h5").write_bytes(b"fixed geometry")
    deck = "#waveform: ricker 1 55000000 native_cv_wavelet\n#src_steps: 0.09 0 0\n"
    for name in ("full_scene.in", "no_basal_contrast_control.in", "air_reference.in", "geometry_check_full.in"):
        (source / name).write_text(deck, encoding="ascii")
    output = tmp_path / "ricker100"
    report = create_waveform_ablation(source, output, center_frequency_hz=100e6)
    staged = json.loads((output / "scene_manifest.json").read_text(encoding="utf-8"))
    assert report["center_frequency_hz"] == 100e6
    assert staged["source"]["center_frequency_hz"] == 100e6
    assert staged["geometry"]["index_file"] == "geology_indices.h5"
    assert (output / "geology_indices.h5").read_bytes() == b"fixed geometry"
    assert "#waveform: ricker 1 100000000 native_cv_wavelet" in (output / "full_scene.in").read_text(encoding="ascii")


def test_sfcw_band_window_and_filter_are_band_limited() -> None:
    frequency = np.asarray([0.0, 19e6, 20e6, 30e6, 100e6, 160e6, 170e6, 171e6])
    window = raised_cosine_sfcw_window(frequency, low_hz=20e6, high_hz=170e6, taper_hz=10e6)
    assert window[0] == 0.0
    assert window[2] == 0.0
    assert window[3] == pytest.approx(1.0)
    assert window[4] == pytest.approx(1.0)
    assert window[6] == 0.0
    samples = 4096
    dt_s = 1e-9
    time = np.arange(samples) * dt_s
    signal = (np.sin(2 * np.pi * 100e6 * time) + 0.5 * np.sin(2 * np.pi * 300e6 * time))[:, None]
    filtered, filtered_frequency, _, meta = apply_sfcw_band_equivalent(signal, dt_s)
    spectrum = np.abs(np.fft.rfft(filtered[:, 0]))
    bins = np.fft.rfftfreq(samples, dt_s)
    amp_100 = spectrum[np.argmin(np.abs(bins - 100e6))]
    amp_300 = spectrum[np.argmin(np.abs(bins - 300e6))]
    assert amp_100 > 100 * amp_300
    assert filtered_frequency[-1] >= 170e6
    assert meta["native_frequency_resolution_mhz"] > 0
