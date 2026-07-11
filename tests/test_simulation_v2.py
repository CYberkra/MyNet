from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import (  # noqa: E402
    GridSpec,
    Material,
    SourceSpec,
    assert_grid_multiple,
    extract_visible_phase,
    layered_bistatic_twt_ns,
    make_scene_arrays,
    resample_time_axis,
)


def test_contract_grid_matches_real_window_and_gprmax_resolution_rule() -> None:
    grid = GridSpec()
    assert grid.trace_count == 256
    assert math.isclose(grid.trace_spacing_m, 0.09)
    assert math.isclose(grid.scan_span_m, 22.95)
    assert math.isclose(grid.output_dt_ns, 1.4)
    assert math.isclose(grid.canonical_time_window_ns, 700.0)
    assert math.isclose(grid.solver_time_window_ns, 701.0)
    assert grid.guard_cells == 20
    grid.validate(max_epsilon_r=11.2, center_frequency_hz=100e6)
    assert assert_grid_multiple(0.09, grid.dl_m, "trace") == 4
    assert assert_grid_multiple(0.18, grid.dl_m, "offset") == 8


def test_layered_vertical_twt_matches_closed_form() -> None:
    d = np.array([2.0, 3.0, 4.0])
    er = np.array([1.0, 9.0, 4.0])
    expected = 2e9 * np.sum(d / (299_792_458.0 / np.sqrt(er)))
    actual = layered_bistatic_twt_ns(d, er, 0.0)
    assert actual == pytest.approx(expected, rel=1e-12)


def test_layered_bistatic_offset_increases_twt() -> None:
    vertical = layered_bistatic_twt_ns([5.0, 7.0], [1.0, 9.0], 0.0)
    bistatic = layered_bistatic_twt_ns([5.0, 7.0], [1.0, 9.0], 0.18)
    assert bistatic > vertical
    assert bistatic - vertical < 0.2


def test_scene_arrays_enforce_antenna_agl_and_layer_order() -> None:
    grid = GridSpec()
    source = SourceSpec()
    cover = Material("cover", 9.0, 0.001)
    weathered = Material("weathered", 7.0, 0.001)
    arrays = make_scene_arrays(
        grid=grid,
        source=source,
        basal_depth_m=np.full(256, 7.0),
        flight_height_agl_m=np.full(256, 2.0),
        cover_fraction=0.5,
        ground_y_m=12.0,
        cover_material=cover,
        weathered_material=weathered,
    )
    assert arrays.trace_midpoint_x_m.shape == (256,)
    assert np.allclose(np.diff(arrays.trace_midpoint_x_m), 0.09)
    assert np.allclose(arrays.antenna_y_m - arrays.ground_y_m, 2.0025, atol=1e-6) or np.allclose(
        arrays.antenna_y_m - arrays.ground_y_m, 2.0, atol=0.012
    )
    assert np.all(arrays.ground_y_m > arrays.cover_bottom_y_m)
    assert np.all(arrays.cover_bottom_y_m > arrays.basal_interface_y_m)
    assert np.all(np.isfinite(arrays.geometric_arrival_time_ns))


def test_resample_time_axis_uses_cfl_dt_not_assumed_501_solver_steps() -> None:
    source_dt_s = 0.05e-9
    source = np.arange(14001, dtype=np.float64)[:, None]
    time_ns, out = resample_time_axis(source, source_dt_s)
    assert time_ns.shape == (501,)
    assert out.shape == (501, 1)
    assert time_ns[-1] == pytest.approx(700.0)
    assert out[-1, 0] == pytest.approx(14000.0)


def test_visible_phase_extraction_tracks_matched_contrast() -> None:
    time = np.linspace(0, 700, 501)
    geom = np.linspace(300, 330, 8)
    control = np.zeros((501, 8), dtype=np.float32)
    full = np.zeros_like(control)
    expected = geom + 4.2
    for j, center in enumerate(expected):
        full[:, j] = np.exp(-0.5 * ((time - center) / 5.0) ** 2)
    visible, support, contrast = extract_visible_phase(full, control, time, geom)
    assert contrast.shape == full.shape
    assert np.max(np.abs(visible - expected)) <= 1.4
    assert np.all(support > 1.0)


def test_control_generator_and_static_validator(tmp_path: Path) -> None:
    out = tmp_path / "controls"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    gen = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_physical_sim_v2.py"),
            "--out-root",
            str(out),
            "--case-id",
            "CTRL01_FLAT_SHALLOW_LOWLOSS_POS",
            "--overwrite",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert gen.returncode == 0, gen.stdout
    case = out / "CTRL01_FLAT_SHALLOW_LOWLOSS_POS"
    assert (case / "full_scene.in").is_file()
    assert (case / "no_basal_contrast_control.in").is_file()
    assert (case / "air_reference.in").is_file()
    assert (case / "preview_geometry_and_arrival.png").is_file()
    manifest = json.loads((case / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["line9_conditioned"] is False
    assert manifest["formal_training_allowed"] is False
    assert "line9" not in (case / "full_scene.in").read_text(encoding="utf-8").lower()

    val = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_physical_sim_v2.py"),
            "--root",
            str(out),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert val.returncode == 0, val.stdout
    report = json.loads((out / "preflight_validation.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["formal_training_allowed"] is False


def _write_fake_gprmax_output(path: Path, data: np.ndarray, *, dt_s: float = 1e-9) -> None:
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["dt"] = dt_s
        f.attrs["gprMax"] = "3.1.7"
        f.attrs["Iterations"] = data.shape[0]
        f.attrs["dx_dy_dz"] = np.array([0.0225, 0.0225, 0.0225])
        f.attrs["srcsteps"] = np.array([0.09, 0.0, 0.0])
        f.attrs["rxsteps"] = np.array([0.09, 0.0, 0.0])
        f.attrs["nrx"] = 1
        rx = f.create_group("rxs").create_group("rx1")
        rx.create_dataset("Ez", data=data)


def test_negative_postprocess_creates_confirmed_negative_mask(tmp_path: Path) -> None:
    case = tmp_path / "negative"
    labels = case / "labels"
    labels.mkdir(parents=True)
    manifest = {
        "case_id": "NEG_TEST",
        "target_presence": False,
        "grid": {
            "trace_count": 256,
            "trace_spacing_m": 0.09,
            "dl_m": 0.0225,
            "gprmax_time_window_ns": 700.0,
        },
    }
    (case / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    data = np.zeros((701, 256), dtype=np.float32)
    _write_fake_gprmax_output(case / "full_scene_merged.out", data)
    _write_fake_gprmax_output(case / "air_reference_merged.out", data)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "postprocess_physical_sim_v2.py"), str(case)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    mask = np.load(labels / "target_mask_confirmed_negative_501x256.npy", allow_pickle=False)
    assert mask.shape == (501, 256)
    assert np.count_nonzero(mask) == 0
    result = json.loads((case / "postprocess_validation.json").read_text(encoding="utf-8"))
    assert result["postprocess_validated"] is True
    assert result["metadata_trusted"] is False
    assert result["formal_training_allowed"] is False


def test_generated_run_commands_use_geometry_fixed(tmp_path: Path) -> None:
    out = tmp_path / "controls"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_physical_sim_v2.py"),
            "--out-root",
            str(out),
            "--case-id",
            "CTRL01_FLAT_SHALLOW_LOWLOSS_POS",
            "--overwrite",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    run_text = (out / "CTRL01_FLAT_SHALLOW_LOWLOSS_POS" / "RUN_COMMANDS.md").read_text()
    assert "-n 256 --geometry-fixed" in run_text


def test_hdf5_contract_rejects_short_canonical_coverage(tmp_path: Path) -> None:
    from scripts.postprocess_physical_sim_v2 import read_merged_bscan, validate_hdf5_contract

    path = tmp_path / "short.out"
    data = np.zeros((700, 256), dtype=np.float32)
    _write_fake_gprmax_output(path, data, dt_s=1e-9)
    dt_s, loaded, attrs = read_merged_bscan(path)
    manifest = {
        "grid": {
            "trace_count": 256,
            "trace_spacing_m": 0.09,
            "dl_m": 0.0225,
            "canonical_time_window_ns": 700.0,
            "solver_time_window_ns": 701.0,
        }
    }
    errors = validate_hdf5_contract(
        path=path, data=loaded, dt_s=dt_s, attrs=attrs, manifest=manifest
    )
    assert any("before canonical endpoint" in item for item in errors)
    assert any("shorter than requested solver window" in item for item in errors)


def test_hdf5_contract_rejects_iterations_mismatch(tmp_path: Path) -> None:
    import h5py
    from scripts.postprocess_physical_sim_v2 import read_merged_bscan, validate_hdf5_contract

    path = tmp_path / "iterations.out"
    data = np.zeros((701, 256), dtype=np.float32)
    _write_fake_gprmax_output(path, data, dt_s=1e-9)
    with h5py.File(path, "r+") as f:
        f.attrs["Iterations"] = 700
    dt_s, loaded, attrs = read_merged_bscan(path)
    manifest = {
        "grid": {
            "trace_count": 256,
            "trace_spacing_m": 0.09,
            "dl_m": 0.0225,
            "canonical_time_window_ns": 700.0,
            "solver_time_window_ns": 701.0,
        }
    }
    errors = validate_hdf5_contract(
        path=path, data=loaded, dt_s=dt_s, attrs=attrs, manifest=manifest
    )
    assert any("Iterations=700" in item for item in errors)
