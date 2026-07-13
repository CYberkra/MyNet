from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_native_256_release_pilot import build
from scripts.capture_gprmax_trace_contract import trace_filename, trace_index_for_path
from scripts.run_native_256_release_pilot import stage_case


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


def test_gprmax_single_trace_contract_uses_unnumbered_output_name() -> None:
    assert trace_filename("full_scene", 1, 1) == "full_scene.out"
    assert trace_index_for_path(Path("full_scene.out"), "full_scene", 1) == 1
    assert trace_index_for_path(Path("full_scene1.out"), "full_scene", 1) is None
    assert trace_filename("full_scene", 1, 2) == "full_scene1.out"
