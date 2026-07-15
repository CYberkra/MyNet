from __future__ import annotations

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

import generate_formal01_bedrock_dense_window as formal01


def test_spec_has_dense_grid_aligned_acquisition() -> None:
    spec = formal01.Spec()
    assert spec.trace_count == 256
    assert spec.trace_spacing_m == 0.10
    assert spec.scan_span_m == pytest.approx(25.5)
    assert spec.trace_spacing_m / spec.dl_m == 4
    assert spec.scan_start_x_m - spec.pml_cells * spec.dl_m == pytest.approx(6.0)
    right_guard = spec.domain_x_m - spec.pml_cells * spec.dl_m - (
        spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m
    )
    assert right_guard == pytest.approx(6.0)
    max_epsilon = max(row[0] + row[1] for row in formal01.BASE_PROPERTIES.values())
    cells_per_min_wavelength = formal01.C0 / (
        2.8 * spec.center_frequency_hz * math.sqrt(max_epsilon) * spec.dl_m
    )
    assert cells_per_min_wavelength >= 10.0


def test_profiles_are_continuous_and_have_finite_transition() -> None:
    spec = formal01.Spec()
    for varied in (False, True):
        profile = formal01.profiles(spec, varied)
        basal = profile["full_basal_depth_m"]
        transition = profile["full_transition_thickness_m"]
        assert np.isfinite(basal).all()
        assert np.isfinite(transition).all()
        assert float(basal.min()) >= 12.2
        assert float(basal.max()) <= 15.6
        assert float(transition.min()) >= 0.55
        assert float(transition.max()) <= 1.75


def test_generated_cases_are_independent_and_strictly_paired(tmp_path: Path) -> None:
    paths = formal01.generate(tmp_path)
    assert len(paths) == 4
    for case_dir in paths:
        manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
        assert manifest["line9_conditioned"] is False
        assert manifest["formal_training_allowed"] is False
        assert manifest["promotion_allowed"] is False
        assert manifest["lifecycle_state"] == "archived_causal_regression"
        assert manifest["morphology_review"]["decision"] == "rejected_for_realism_and_training"
        assert manifest["geometry"]["discrete_anomaly_bodies"] == 0
        assert manifest["acquisition"]["trace_count"] == 256
        assert manifest["acquisition"]["spacing_m"] == 0.10
        assert (case_dir / "FILE_SHA256.csv").is_file()
        with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
            assert handle["data"].shape == (formal01.Spec().nx, formal01.Spec().ny, 1)
            assert handle["data"].dtype == np.dtype("int16")
        arrival = np.load(case_dir / "labels" / "geometric_reference_arrival_time_ns.npy")
        assert arrival.shape == (256,)
        assert np.isfinite(arrival).all()
