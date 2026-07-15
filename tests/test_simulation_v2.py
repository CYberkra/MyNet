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
    control_point_interface_depth,
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


def test_control_point_basal_can_vary_independently_of_cover_boundary() -> None:
    grid = GridSpec()
    x = np.arange(grid.trace_count, dtype=np.float64) * grid.trace_spacing_m
    basal = control_point_interface_depth(
        x,
        base_depth_m=9.4,
        control_point_fractions=[0.0, 0.2, 0.45, 0.7, 1.0],
        depth_offsets_m=[-0.4, -0.2, 0.3, 0.7, 0.25],
        smoothing_length_m=1.2,
    )
    arrays = make_scene_arrays(
        grid=grid,
        source=SourceSpec(),
        basal_depth_m=basal,
        flight_height_agl_m=np.full(grid.trace_count, 5.0),
        cover_thickness_m=3.2,
        ground_y_m=15.0,
        cover_material=Material("cover", 8.8, 0.0012),
        weathered_material=Material("weathered", 8.4, 0.0010),
        arrival_model="columnar_layered_reference_not_specular_exact",
    )
    assert np.ptp(arrays.basal_depth_m) > 0.5
    assert np.ptp(arrays.cover_thickness_m) == pytest.approx(0.0)
    assert np.all(arrays.weathered_thickness_m > 5.0)


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


def test_continuous_visible_phase_rejects_an_isolated_stronger_lobe() -> None:
    time = np.linspace(0, 700, 501)
    geom = np.full(9, 320.0)
    full = np.zeros((501, 9), dtype=np.float32)
    for j in range(full.shape[1]):
        full[:, j] = np.exp(-0.5 * ((time - 320.0) / 4.0) ** 2)
    # A single trace has a stronger but physically discontinuous distractor.
    full[:, 4] += 3.0 * np.exp(-0.5 * ((time - 348.0) / 3.0) ** 2)
    independent, _, _ = extract_visible_phase(full, np.zeros_like(full), time, geom)
    continuous, _, _ = extract_visible_phase(
        full,
        np.zeros_like(full),
        time,
        geom,
        enforce_continuity=True,
        max_trace_step_ns=5.6,
    )
    assert independent[4] > 340.0
    assert continuous[4] < 330.0
    assert np.max(np.abs(np.diff(continuous))) <= 5.6


def test_continuous_visible_phase_returns_signed_lobe_not_envelope_center() -> None:
    time = np.linspace(0, 700, 501)
    geom = np.full(9, 320.0)
    full = np.zeros((501, 9), dtype=np.float32)
    for j in range(full.shape[1]):
        delta = time - (320.0 + 0.15 * j)
        gaussian = np.exp(-0.5 * (delta / 5.6) ** 2)
        # Asymmetric bipolar wavelet: its envelope centre and strongest signed
        # lobe are deliberately different.
        full[:, j] = (delta / 5.6) * gaussian + 0.18 * gaussian
    independent, _, _ = extract_visible_phase(full, np.zeros_like(full), time, geom)
    continuous, _, _ = extract_visible_phase(
        full,
        np.zeros_like(full),
        time,
        geom,
        enforce_continuity=True,
        max_trace_step_ns=5.6,
    )
    assert np.max(np.abs(continuous - independent)) <= 1.4 + 1e-9
    assert np.median(np.abs(continuous - geom)) >= 4.2
    assert np.max(np.abs(np.diff(continuous))) <= 5.6


def test_continuous_visible_phase_follows_reference_slope_not_flat_distractor() -> None:
    time = np.linspace(0, 700, 501)
    geom = np.linspace(270.0, 390.0, 9)
    full = np.zeros((501, geom.size), dtype=np.float32)
    expected = geom + 5.6
    for j, center in enumerate(expected):
        target_delta = time - center
        target = (target_delta / 5.0) * np.exp(-0.5 * (target_delta / 5.0) ** 2)
        distractor_delta = time - 330.0
        distractor = 1.15 * (distractor_delta / 4.0) * np.exp(
            -0.5 * (distractor_delta / 4.0) ** 2
        )
        full[:, j] = target + distractor

    continuous, _, _ = extract_visible_phase(
        full,
        np.zeros_like(full),
        time,
        geom,
        search_half_width_ns=38.0,
        enforce_continuity=True,
        max_trace_step_ns=5.6,
        geometric_anchor_weight=0.05,
    )

    residual = continuous - geom
    assert np.ptp(continuous) >= 100.0
    assert np.max(np.abs(np.diff(residual))) <= 5.6 + 1e-9
    # The extractor returns the strongest signed lobe, not the envelope
    # centre at ``expected``. In this synthetic bipolar wavelet that lobe is
    # deliberately about 5 ns earlier and therefore lies close to ``geom``.
    assert np.median(np.abs(continuous - geom)) <= 2.8


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


def test_static_validator_accepts_successful_postprocessed_control(tmp_path: Path) -> None:
    out = tmp_path / "controls"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    generated = subprocess.run(
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
    assert generated.returncode == 0, generated.stdout
    case = out / "CTRL01_FLAT_SHALLOW_LOWLOSS_POS"
    labels = case / "labels"
    shape = (501, 256)
    np.save(labels / "visible_phase_time_ns.npy", np.full(256, 350.0, dtype=np.float32))
    for name in (
        "full_scene_501x256.npy",
        "air_reference_501x256.npy",
        "no_basal_contrast_501x256.npy",
        "contrast_response_501x256.npy",
        "target_mask_visible_phase_501x256.npy",
    ):
        np.save(labels / name, np.zeros(shape, dtype=np.float32))
    np.save(labels / "visible_phase_support_ratio.npy", np.ones(256, dtype=np.float32))
    (case / "postprocess_validation.json").write_text(
        json.dumps(
            {
                "ok": True,
                "postprocess_validated": True,
                "formal_training_allowed": False,
                "output_shape_canonical": list(shape),
            }
        ),
        encoding="utf-8",
    )
    validated = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_physical_sim_v2.py"), "--root", str(out)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert validated.returncode == 0, validated.stdout
    report = json.loads((out / "preflight_validation.json").read_text(encoding="utf-8"))
    assert report["results"][0]["lifecycle_state"] == "postprocessed"


def test_static_validator_reports_incomplete_case_without_aborting_catalog(tmp_path: Path) -> None:
    out = tmp_path / "controls"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    generated = subprocess.run(
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
    assert generated.returncode == 0, generated.stdout
    (out / "CTRL01_FLAT_SHALLOW_LOWLOSS_POS" / "labels" / "flight_height_agl_m.npy").unlink()

    validated = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_physical_sim_v2.py"), "--root", str(out)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert validated.returncode == 1
    assert "Traceback" not in validated.stdout
    report = json.loads((out / "preflight_validation.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["results"][0]["lifecycle_state"] == "invalid_or_incomplete"
    assert "flight_height_agl_m.npy" in report["results"][0]["errors"][0]


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


def test_selected_control_regeneration_preserves_complete_index(tmp_path: Path) -> None:
    out = tmp_path / "controls"
    base_command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_physical_sim_v2.py"),
        "--out-root",
        str(out),
    ]
    first = subprocess.run(
        base_command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert first.returncode == 0, first.stdout
    regenerated = subprocess.run(
        base_command
        + ["--case-id", "CTRL03_SMOOTH_INTERFACE_POS", "--overwrite"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert regenerated.returncode == 0, regenerated.stdout
    index = json.loads((out / "control_index.json").read_text(encoding="utf-8"))
    contract = json.loads(
        (ROOT / "data" / "simulation_contract_v2" / "control_cases_v1.json").read_text(encoding="utf-8")
    )
    expected = {entry["case_id"] for entry in contract["cases"]}
    assert index["case_count"] == len(expected)
    assert {entry["case_id"] for entry in index["cases"]} == expected


def test_negative_control_run_plan_includes_postprocess(tmp_path: Path) -> None:
    from scripts.run_physical_sim_v2_controls import case_plan

    case = tmp_path / "CTRL04"
    case.mkdir()
    (case / "scene_manifest.json").write_text(
        json.dumps({"case_id": "CTRL04", "target_presence": False, "grid": {"trace_count": 256}}),
        encoding="utf-8",
    )
    plan = case_plan(case, gpu=0, geometry_only=False, python_executable=sys.executable)
    assert [entry["stage"] for entry in plan["commands"]][-1] == "postprocess"


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
