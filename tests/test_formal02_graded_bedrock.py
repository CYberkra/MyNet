from __future__ import annotations

import json
import math
import inspect
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_formal02_graded_bedrock as formal02
import run_native_256_release_pilot as native_runner


def test_spec_resolves_source_and_protects_target_window() -> None:
    spec = formal02.Spec()
    formal02.validate_spec(spec)
    assert spec.trace_count == 256
    assert spec.trace_spacing_m == pytest.approx(0.09)
    assert spec.scan_span_m == pytest.approx(22.95)
    assert spec.trace_spacing_m / spec.dl_m == pytest.approx(2.0)
    assert spec.physical_side_guard_m == pytest.approx(80.01)
    assert spec.right_guard_m == pytest.approx(spec.physical_side_guard_m)
    assert spec.boundary_round_trip_ns >= spec.protected_window_end_ns
    cells_per_wavelength = formal02.C0 / (
        2.8
        * spec.center_frequency_hz
        * math.sqrt(formal02.COVER.epsilon_r)
        * spec.dl_m
    )
    assert cells_per_wavelength >= 10.0


def test_profile_crop_is_nonperiodic_and_not_one_quadratic() -> None:
    spec = formal02.Spec()
    profile, stats = formal02.build_profiles(spec)
    assert np.isfinite(profile["full_basal_depth_m"]).all()
    assert np.isfinite(profile["full_transition_thickness_m"]).all()
    assert 0.65 <= stats["range_m"] <= 2.4
    assert stats["smoothed_extrema_count"] >= 2
    assert stats["quadratic_fit_r2"] < 0.985
    assert stats["abs_slope_p95"] < 0.22
    assert float(profile["full_transition_thickness_m"].min()) >= 0.50
    assert float(profile["full_transition_thickness_m"].max()) <= 1.55


def test_generator_has_no_measured_data_input_contract() -> None:
    assert list(inspect.signature(formal02.generate).parameters) == ["output_root"]
    source = Path(formal02.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("data_yingshan", "line9.npz", "line9_v", "soft_mask_train"):
        assert forbidden not in source


def test_gradient_has_small_monotonic_steps_and_exact_control() -> None:
    spec = formal02.Spec()
    full = formal02.material_rows(spec, control=False)
    control = formal02.material_rows(spec, control=True)
    full_epsilon = np.asarray([row.epsilon_r for row in full])
    full_conductivity = np.asarray([row.conductivity_s_per_m for row in full])
    assert len(full) == spec.transition_levels + 2
    assert len({row.material_id for row in full}) == len(full)
    assert np.all(np.diff(full_epsilon) < 0.0)
    assert np.all(np.diff(full_conductivity) < 0.0)
    assert float(np.max(np.abs(np.diff(full_epsilon)))) <= 0.51
    assert all(row.epsilon_r == formal02.COVER.epsilon_r for row in control)
    assert all(row.conductivity_s_per_m == formal02.COVER.conductivity_s_per_m for row in control)
    assert [row.material_id for row in control] == [row.material_id for row in full]


def test_generated_case_is_strictly_paired_and_has_no_visible_label(tmp_path: Path) -> None:
    case_dir = formal02.generate(tmp_path)
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["family_id"] == formal02.FAMILY_ID
    assert manifest["line9_conditioned"] is False
    assert manifest["formal_training_allowed"] is False
    assert manifest["promotion_allowed"] is False
    assert manifest["target_presence"] is True
    assert manifest["grid"]["trace_count"] == 256
    assert manifest["grid"]["trace_spacing_m"] == pytest.approx(0.09)
    assert manifest["grid"]["dl_m"] == pytest.approx(0.045)
    assert manifest["geometry"]["index_file"] == "geology_indices.h5"
    assert manifest["geometry"]["separate_topsoil_interface"] is False
    assert manifest["geometry"]["explicit_periodic_basal_components"] is False
    assert manifest["geometry"]["transition_levels"] == 12
    assert manifest["grid"]["earliest_lateral_boundary_round_trip_ns"] >= 500.0
    assert manifest["strict_pair"]["shared_geometry_hdf5"] is True
    assert manifest["strict_pair"]["only_transition_and_bedrock_changed"] is True
    assert (case_dir / "FILE_SHA256.csv").is_file()
    assert not list((case_dir / "labels").glob("*visible*"))
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        assert handle["data"].shape == (formal02.Spec().nx, formal02.Spec().ny, 1)
        assert handle["data"].dtype == np.dtype("int16")
        assert int(handle["data"][:].min()) == -1
        assert int(handle["data"][:].max()) == formal02.Spec().transition_levels + 1
    arrival = np.load(case_dir / "labels" / "geometric_reference_arrival_time_ns.npy")
    assert arrival.shape == (256,)
    assert np.isfinite(arrival).all()


def test_generated_case_stages_with_the_shared_native_runner(tmp_path: Path) -> None:
    source = formal02.generate(tmp_path / "source")
    staged = native_runner.stage_case(
        source,
        tmp_path / "solver" / "smoke1",
        requested_trace_count=1,
        geometry_only=False,
        include_air_reference=False,
    )
    run_manifest = json.loads((staged / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["input_groups"] == ["full_scene", "no_basal_contrast_control"]
    assert run_manifest["declared_trace_count"] == 256
    assert (staged / "geology_indices.h5").is_file()
