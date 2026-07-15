from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.audit_full_scene_morphology as morphology
import scripts.compare_gprmax_common_traces as comparison
import scripts.preview_gprmax_raw_and_agc as preview


def _write_run_contract(root: Path, trace_count: int) -> None:
    (root / "labels").mkdir(parents=True)
    (root / "run_logs").mkdir()
    scene = {
        "case_id": "SYNTHETIC_FULL_ONLY",
        "grid": {
            "trace_count": trace_count,
            "trace_spacing_m": 0.09,
            "protected_window_end_ns": 500.0,
        },
    }
    run = {
        "selected_trace_indices_zero_based": list(range(trace_count)),
        "trace_stride": 1,
    }
    (root / "scene_manifest.json").write_text(json.dumps(scene), encoding="utf-8")
    (root / "run_manifest.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "run_logs" / "full_scene_trace_contract.json").write_text(
        json.dumps(
            {
                "expected_trace_count": trace_count,
                "captured_trace_count": trace_count,
                "complete": True,
                "failures_tail": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "full_scene_merged.out").write_bytes(b"mocked")


def test_preview_uses_manifest_protected_window_without_extrapolation(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_contract(run_dir, 4)
    time = np.arange(651, dtype=np.float32)
    raw = np.sin(time[:, None] / 13.0) * np.ones((1, 4), dtype=np.float32)
    monkeypatch.setattr(preview, "read_merged_bscan", lambda *_args, **_kwargs: (1e-9, raw, {}))

    output = tmp_path / "preview.png"
    result = preview.render_preview(run_dir, output, processing="time-power", time_power=1.5)

    assert output.is_file()
    assert result["raw_shape"] == [358, 4]
    assert result["source_end_ns"] == 650.0
    assert result["protected_end_ns"] == 500.0


def test_preview_selects_canonical_reference_for_sparse_subset(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_contract(run_dir, 4)
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    scene["grid"]["trace_count"] = 16
    (run_dir / "scene_manifest.json").write_text(json.dumps(scene), encoding="utf-8")
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run["selected_trace_indices_zero_based"] = [0, 5, 10, 15]
    (run_dir / "run_manifest.json").write_text(json.dumps(run), encoding="utf-8")
    raw = np.ones((651, 4), dtype=np.float32)
    monkeypatch.setattr(preview, "read_merged_bscan", lambda *_args, **_kwargs: (1e-9, raw, {}))
    reference = tmp_path / "reference.npy"
    np.save(reference, np.linspace(300.0, 360.0, 16, dtype=np.float32))

    result = preview.render_preview(
        run_dir,
        tmp_path / "preview.png",
        visible_phase_path=reference,
        processing="time-power",
    )

    assert result["visible_phase_overlay"] is True


def test_common_trace_comparison_matches_canonical_positions() -> None:
    common, left_columns, right_columns = comparison.common_trace_columns(
        np.arange(0, 256, 8),
        np.arange(0, 256, 32),
    )
    assert common.tolist() == [0, 32, 64, 96, 128, 160, 192, 224]
    assert left_columns.tolist() == [0, 4, 8, 12, 16, 20, 24, 28]
    assert right_columns.tolist() == list(range(8))


def test_common_trace_comparison_uses_one_shared_scale() -> None:
    left = np.asarray([[-1.0, 0.5]])
    right = np.asarray([[3.0, -2.0]])
    assert comparison.shared_quantile_scale(left, right, quantile=1.0) == 3.0


def test_common_trace_comparison_script_has_no_case_specific_title() -> None:
    source = (
        comparison.ROOT / "scripts" / "compare_gprmax_common_traces.py"
    ).read_text(
        encoding="utf-8"
    )
    assert "FORMAL06C versus FORMAL07A" not in source
    assert "comparison_title" in source


def test_full_only_audit_records_partial_control_and_blocks_promotion(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace_count = 16
    _write_run_contract(run_dir, trace_count)
    for index in range(1, 4):
        (run_dir / f"no_basal_contrast_control{index}.out").write_bytes(b"partial")

    time_ns = np.arange(651, dtype=np.float64)
    reference = 360.0 + 5.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, trace_count))
    np.save(run_dir / "labels" / "geometric_reference_arrival_time_ns.npy", reference.astype(np.float32))
    raw = np.zeros((time_ns.size, trace_count), dtype=np.float64)
    for trace, center in enumerate(reference + 5.0):
        phase = (time_ns - center) / 7.0
        raw[:, trace] = (1.0 - 2.0 * phase * phase) * np.exp(-phase * phase)
        raw[:, trace] += 0.002 * np.sin(0.1 * time_ns + trace)
    monkeypatch.setattr(morphology, "read_merged_bscan", lambda *_args, **_kwargs: (1e-9, raw, {}))

    output_dir = tmp_path / "audit"
    result = morphology.audit(run_dir, output_dir, component="Ez", line9_contract_path=None)

    assert result["execution_state"]["full_scene_captured_trace_count"] == trace_count
    assert result["execution_state"]["control_completed_trace_files"] == 3
    assert result["execution_state"]["control_merged"] is False
    assert result["gates"]["full_output_contract_pass"] is True
    assert result["gates"]["full_resolution_causal_pair_pass"] is False
    assert result["gates"]["formal_promotion_pass"] is False
    assert morphology._portable_path(
        morphology.ROOT / "reports" / "audit.json"
    ) == "reports/audit.json"
    assert (output_dir / "full_only_morphology_audit.json").is_file()
    assert (output_dir / "full_only_morphology_path_not_for_training.npy").is_file()
