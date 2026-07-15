from __future__ import annotations

import inspect
import json
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
import generate_formal04_geology_factorial as formal04
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


def test_default_factorial_is_exact_and_resolved() -> None:
    spec = formal03.Spec()
    formal04.validate_spec(spec)
    cells = formal03.C0 / (
        2.8 * formal04.SOURCE.center_frequency_hz * np.sqrt(15.2) * spec.dl_m
    )
    assert cells >= 10.0
    assert [(item.texture_level, item.basal_contrast_level) for item in formal04.VARIANTS] == [
        ("baseline", "weak"),
        ("strong", "strong"),
        ("strong", "weak"),
    ]


def test_generator_has_no_measured_data_input() -> None:
    assert list(inspect.signature(formal04.generate).parameters) == [
        "output_root",
        "spec",
        "variants",
    ]
    source = Path(formal04.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("data_yingshan", "line9.npz", "soft_mask_train"):
        assert forbidden not in source


def test_constitutive_axes_have_intended_ordering() -> None:
    spec = mini_spec()
    weak_basal, strong_texture, combined = formal04.VARIANTS
    baseline_span = np.ptp([row.epsilon_r for row in formal04.base_materials(spec, weak_basal)])
    strong_span = np.ptp([row.epsilon_r for row in formal04.base_materials(spec, strong_texture)])
    assert strong_span > 2.0 * baseline_span - 1e-6
    assert formal04.reflection_proxy(spec, weak_basal)["median"] < formal04.reflection_proxy(spec, strong_texture)["median"]
    assert formal04.reflection_proxy(spec, combined)["median"] < formal04.reflection_proxy(spec, strong_texture)["median"]
    assert formal04.base_materials(spec, strong_texture) == formal04.base_materials(spec, combined)


def test_control_restores_every_target_state_to_local_cover() -> None:
    spec = mini_spec()
    for variant in formal04.VARIANTS:
        bases = formal04.base_materials(spec, variant)
        control = formal04.material_rows(spec, variant, control=True)
        full = formal04.material_rows(spec, variant, control=False)
        assert len(control) == spec.cover_bins * (spec.transition_levels + 2)
        for index, row in enumerate(control):
            base = bases[index % spec.cover_bins]
            assert row.epsilon_r == pytest.approx(base.epsilon_r)
            assert row.conductivity_s_per_m == pytest.approx(base.conductivity_s_per_m)
        epsilon_step, _ = formal03._physical_transition_step(spec, full)
        assert epsilon_step < 1.1


def test_generated_family_shares_geometry_source_and_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = mini_spec()
    mini_source = formal03.SourceVariant(
        "FORMAL04_TEST_GABOR8",
        "gaussian_modulated_zero_mean",
        8e6,
        "formal04_test_gabor8",
        50.0,
        11.5,
    )
    monkeypatch.setattr(formal04, "SOURCE", mini_source)
    case_dirs = formal04.generate(tmp_path / "source", spec=spec)
    geometry_arrays: list[np.ndarray] = []
    geometry_hashes: set[str] = set()
    source_hashes: set[str] = set()
    for case_dir in case_dirs:
        manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
        assert manifest["family_id"] == formal04.FAMILY_ID
        assert manifest["line9_conditioned"] is False
        assert manifest["formal_training_allowed"] is False
        assert manifest["ablation"]["changed"] == "constitutive material mapping only"
        assert not list((case_dir / "labels").glob("*visible*"))
        geometry_hashes.add(formal03.sha256(case_dir / "geology_indices.h5"))
        source_hashes.add(formal03.sha256(case_dir / "source_waveform.txt"))
        with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
            geometry_arrays.append(handle["data"][:])
        geometric = np.load(case_dir / "labels" / "geometric_reference_arrival_time_ns.npy")
        source_reference = np.load(case_dir / "labels" / "source_referenced_arrival_time_ns.npy")
        assert np.median(source_reference - geometric) == pytest.approx(50.0, abs=1e-4)
    assert len(geometry_hashes) == 1
    assert len(source_hashes) == 1
    assert all(np.array_equal(geometry_arrays[0], item) for item in geometry_arrays[1:])

    staged = native_runner.stage_case(
        case_dirs[-1],
        tmp_path / "solver" / "smoke1",
        requested_trace_count=1,
        geometry_only=False,
        include_air_reference=False,
    )
    run_manifest = json.loads((staged / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["input_groups"] == ["full_scene", "no_basal_contrast_control"]
    assert (staged / "source_waveform.txt").is_file()
