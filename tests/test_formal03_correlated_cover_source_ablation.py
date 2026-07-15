from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_formal03_correlated_cover_source_ablation as formal03
import audit_native_gprmax_smoke as smoke_audit
import run_native_256_release_pilot as native_runner


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


def mini_variants() -> tuple[formal03.SourceVariant, ...]:
    return (
        formal03.SourceVariant("FORMAL03_TEST_RICKER5", "ricker", 5e6, "test_ricker5", math.sqrt(2.0) / 5e6 * 1e9),
        formal03.SourceVariant("FORMAL03_TEST_RICKER8", "ricker", 8e6, "test_ricker8", math.sqrt(2.0) / 8e6 * 1e9),
        formal03.SourceVariant("FORMAL03_TEST_GABOR8", "gaussian_modulated_zero_mean", 8e6, "test_gabor8", 50.0, 11.5),
    )


def test_default_grid_resolves_every_source_and_protects_window() -> None:
    spec = formal03.Spec()
    formal03.validate_spec(spec)
    assert spec.trace_count == 256
    assert spec.trace_spacing_m / spec.dl_m == pytest.approx(3.0)
    assert spec.tx_rx_offset_m / spec.dl_m == pytest.approx(6.0)
    assert spec.right_guard_m == pytest.approx(spec.physical_side_guard_m)
    assert spec.boundary_round_trip_ns >= spec.protected_window_end_ns
    max_epsilon = max(material.epsilon_r for material in formal03.base_materials(spec))
    for variant in formal03.SOURCE_VARIANTS:
        cells = formal03.C0 / (
            2.8 * variant.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m
        )
        assert cells >= 10.0


def test_generator_contract_has_no_measured_data_input() -> None:
    assert list(inspect.signature(formal03.generate).parameters) == ["output_root", "spec", "variants"]
    source = Path(formal03.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("data_yingshan", "line9.npz", "line9_v", "soft_mask_train"):
        assert forbidden not in source


def test_cover_is_two_dimensional_and_uses_all_bins() -> None:
    spec = mini_spec()
    latent, bins, stats = formal03.build_cover_field(spec)
    assert latent.shape == (spec.nx, spec.ny)
    assert bins.shape == latent.shape
    assert stats["used_bins"] == spec.cover_bins
    assert stats["horizontal_neighbor_bin_change_rate"] > 0.01
    assert stats["vertical_neighbor_bin_change_rate"] > 0.01
    assert stats["columns_with_near_zero_vertical_variation"] == 0
    assert np.std(latent, axis=0).mean() > 0.01
    assert np.std(latent, axis=1).mean() > 0.01


def test_control_restores_every_target_index_to_its_local_cover_bin() -> None:
    spec = mini_spec()
    full = formal03.material_rows(spec, control=False)
    control = formal03.material_rows(spec, control=True)
    bases = formal03.base_materials(spec)
    expected_count = spec.cover_bins * (spec.transition_levels + 2)
    assert len(full) == expected_count
    assert len(control) == expected_count
    for index, row in enumerate(control):
        base = bases[index % spec.cover_bins]
        assert row.epsilon_r == pytest.approx(base.epsilon_r)
        assert row.conductivity_s_per_m == pytest.approx(base.conductivity_s_per_m)
    max_epsilon_step, _ = formal03._physical_transition_step(spec, full)
    assert max_epsilon_step < 1.0


def test_custom_transient_is_zero_mean_and_peaks_near_declared_frequency() -> None:
    variant = formal03.SOURCE_VARIANTS[2]
    time_s, values = formal03.custom_waveform(variant, formal03.Spec())
    assert abs(float(np.trapz(values, time_s))) < 1e-15
    assert float(np.max(np.abs(values))) == pytest.approx(1.0)
    frequency = np.fft.rfftfreq(values.size, float(time_s[1] - time_s[0]))
    spectrum = np.abs(np.fft.rfft(values))
    peak = float(frequency[int(np.argmax(spectrum[1:])) + 1])
    assert peak == pytest.approx(variant.center_frequency_hz, abs=2e6)
    active = time_s[np.abs(values) > 1e-3]
    assert float(active[-1] - active[0]) < 120e-9


def test_smoke_audit_prefers_explicit_source_delay_reference(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    np.save(labels / "geometric_reference_arrival_time_ns.npy", np.asarray([350.0]))
    selected, semantics = smoke_audit.select_arrival_reference(tmp_path)
    assert selected.name == "geometric_reference_arrival_time_ns.npy"
    assert semantics == "geometric_interface_only"
    np.save(labels / "source_referenced_arrival_time_ns.npy", np.asarray([400.0]))
    selected, semantics = smoke_audit.select_arrival_reference(tmp_path)
    assert selected.name == "source_referenced_arrival_time_ns.npy"
    assert semantics == "geometric_interface_plus_explicit_source_reference_delay"


def test_generated_mini_family_shares_exact_geometry_and_stages(tmp_path: Path) -> None:
    spec = mini_spec()
    variants = mini_variants()
    case_dirs = formal03.generate(tmp_path / "source", spec=spec, variants=variants)
    assert len(case_dirs) == 3
    geometry_hashes: set[str] = set()
    source_delays: list[float] = []
    for case_dir, variant in zip(case_dirs, variants):
        manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
        assert manifest["family_id"] == formal03.FAMILY_ID
        assert manifest["line9_conditioned"] is False
        assert manifest["formal_training_allowed"] is False
        assert manifest["promotion_allowed"] is False
        assert manifest["strict_pair"]["control_restores_each_target_voxel_to_its_local_cover_bin"] is True
        assert not list((case_dir / "labels").glob("*visible*"))
        geometry_hashes.add(formal03.sha256(case_dir / "geology_indices.h5"))
        source_delays.append(float(manifest["source"]["reference_delay_ns"]))
        with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
            assert handle["data"].shape == (spec.nx, spec.ny, 1)
            assert handle["data"].dtype == np.dtype("int16")
            assert int(handle["data"][:].min()) == -1
        geometric = np.load(case_dir / "labels" / "geometric_reference_arrival_time_ns.npy")
        source_reference = np.load(case_dir / "labels" / "source_referenced_arrival_time_ns.npy")
        assert np.median(source_reference - geometric) == pytest.approx(variant.reference_delay_ns, abs=1e-4)
    assert len(geometry_hashes) == 1
    assert len(set(source_delays)) == 3
    assert not (case_dirs[0] / "source_waveform.txt").exists()
    assert (case_dirs[2] / "source_waveform.txt").is_file()

    staged = native_runner.stage_case(
        case_dirs[2],
        tmp_path / "solver" / "smoke1",
        requested_trace_count=1,
        geometry_only=False,
        include_air_reference=False,
    )
    run_manifest = json.loads((staged / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["input_groups"] == ["full_scene", "no_basal_contrast_control"]
    assert (staged / "source_waveform.txt").is_file()

    full_only = native_runner.stage_case(
        case_dirs[2],
        tmp_path / "solver" / "full_only",
        requested_trace_count=3,
        geometry_only=False,
        trace_stride=2,
        include_air_reference=False,
        full_scene_only=True,
    )
    full_only_manifest = json.loads((full_only / "run_manifest.json").read_text(encoding="utf-8"))
    assert full_only_manifest["input_groups"] == ["full_scene"]
    assert full_only_manifest["causal_pair_complete"] is False
