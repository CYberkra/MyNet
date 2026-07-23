from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "data" / "contracts" / "simulation_v2" / "basal_shape_batch_v2.json"
NONFOCUSING_CONTRACT_PATH = ROOT / "data" / "contracts" / "simulation_v2" / "basal_shape_batch_v3_shape03_nonfocusing.json"
MEANDER_CONTRACT_PATH = ROOT / "data" / "contracts" / "simulation_v2" / "basal_shape_batch_v4_shape04_low_curvature_meander.json"
BROAD_LOW_RELIEF_CONTRACT_PATH = ROOT / "data" / "contracts" / "simulation_v2" / "basal_shape_batch_v5_shape05_broad_low_relief.json"


def test_shape02_contract_has_matching_shape_implementation() -> None:
    from scripts.generate_basal_shape_batch_v2 import SHAPE_FUNCTIONS

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["factor_isolation"]["only_variable"] == "basal_interface_depth_profile"
    assert [row["id"] for row in contract["shape_bank"]] == list(SHAPE_FUNCTIONS)
    assert contract["factor_isolation"]["locked"]["trace_count"] == 256
    assert contract["factor_isolation"]["locked"]["trace_spacing_m"] == 0.09


def test_shape02_profiles_are_bounded_and_nonidentical() -> None:
    from scripts.generate_basal_shape_batch_v2 import BASE_DEPTH_M, SHAPE_FUNCTIONS

    u = np.linspace(-1.0, 1.0, 256)
    hashes = set()
    for function in SHAPE_FUNCTIONS.values():
        depth = (BASE_DEPTH_M + function(u)).astype(np.float32)
        assert depth.shape == (256,)
        assert np.isfinite(depth).all()
        assert float(depth.min()) >= 13.5
        assert float(depth.max()) <= 17.5
        hashes.add(depth.tobytes())
    assert len(hashes) == len(SHAPE_FUNCTIONS)


def test_shape02_full_span_sparse_indices_cover_aperture() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    stage = next(row for row in contract["execution_stages"] if row["stage"] == "C_full_span_sparse")
    indices = np.arange(0, 256, 8)
    assert len(indices) == 32
    assert indices[0] == 0
    assert indices[-1] == 248
    assert stage["effective_spacing_m"] == 0.72


def test_shape03_profiles_are_monotonic_and_have_no_internal_extrema() -> None:
    from scripts.generate_basal_shape_batch_v2 import BASE_DEPTH_M, NONFOCUSING_SHAPE_FUNCTIONS, geometry_metrics

    contract = json.loads(NONFOCUSING_CONTRACT_PATH.read_text(encoding="utf-8"))
    ids = [row["id"] for row in contract["shape_bank"]]
    assert ids == list(NONFOCUSING_SHAPE_FUNCTIONS)
    x_m = np.arange(256, dtype=np.float64) * 0.09
    u = np.linspace(-1.0, 1.0, 256)
    for row in contract["shape_bank"]:
        depth = BASE_DEPTH_M + NONFOCUSING_SHAPE_FUNCTIONS[row["id"]](u)
        assert np.all(np.diff(depth) > 0.0)
        metrics = geometry_metrics(row["id"], row["role"], x_m, depth, contract["geometry_gates"])
        assert metrics["no_internal_extremum_gate_ok"]
        assert metrics["geometry_gate_ok"]


def test_shape04_profiles_keep_low_curvature_changes_monotonic() -> None:
    from scripts.generate_basal_shape_batch_v2 import (
        BASE_DEPTH_M,
        LOW_CURVATURE_MEANDER_FUNCTIONS,
        geometry_metrics,
    )

    contract = json.loads(MEANDER_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert [row["id"] for row in contract["shape_bank"]] == list(LOW_CURVATURE_MEANDER_FUNCTIONS)
    x_m = np.arange(256, dtype=np.float64) * 0.09
    u = np.linspace(-1.0, 1.0, 256)
    for row in contract["shape_bank"]:
        depth = BASE_DEPTH_M + LOW_CURVATURE_MEANDER_FUNCTIONS[row["id"]](u)
        assert np.all(np.diff(depth) > 0.0)
        assert geometry_metrics(row["id"], row["role"], x_m, depth, contract["geometry_gates"])["geometry_gate_ok"]


def test_shape05_rejects_the_too_flat_broad_arch_before_solver() -> None:
    from scripts.generate_basal_shape_batch_v2 import (
        BASE_DEPTH_M,
        BROAD_LOW_RELIEF_FUNCTIONS,
        geometry_metrics,
    )

    contract = json.loads(BROAD_LOW_RELIEF_CONTRACT_PATH.read_text(encoding="utf-8"))
    x_m = np.arange(256, dtype=np.float64) * 0.09
    u = np.linspace(-1.0, 1.0, 256)
    metrics = {}
    for row in contract["shape_bank"]:
        depth = BASE_DEPTH_M + BROAD_LOW_RELIEF_FUNCTIONS[row["id"]](u)
        metrics[row["id"]] = geometry_metrics(row["id"], row["role"], x_m, depth, contract["geometry_gates"])
    assert not metrics["GEO16_BROAD_SHALLOW_SAG"]["geometry_gate_ok"]
    assert metrics["GEO17_BROAD_ASYMMETRIC_SHOULDER"]["geometry_gate_ok"]
