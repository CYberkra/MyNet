from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_native_256_release_pilot import build


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


def test_native_runner_captures_gprmax_trace_names_before_merge() -> None:
    runner = (ROOT / "scripts" / "run_native_256_release_pilot.py").read_text(encoding="utf-8")
    assert 'prefix = stem' in runner
    assert 'capture_gprmax_trace_contract.py' in runner
    assert 'tools.outputfiles_merge' in runner
