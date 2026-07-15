from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import scripts.package_gprmax_release as release


def _write_out(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["gprMax"] = "test"
        handle.attrs["Iterations"] = 4
        handle.attrs["dt"] = 1e-9
        receiver = handle.create_group("rxs").create_group("rx1")
        receiver.create_dataset("Ez", data=np.arange(4, dtype=np.float32))


def _spec(tmp_path: Path) -> Path:
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ("smoke_full_merged.out", "smoke_control_merged.out", "distributed_merged.out"):
        _write_out(sources / name)
    for name in ("audit.json", "trace.json", "morphology.json", "decision.json", "preview.png"):
        (sources / name).write_bytes(b"evidence")
    artifacts = [
        ("one_trace_full_merged", "smoke_full_merged.out"),
        ("one_trace_control_merged", "smoke_control_merged.out"),
        ("one_trace_audit", "audit.json"),
        ("distributed_full_merged", "distributed_merged.out"),
        ("distributed_trace_contract", "trace.json"),
        ("distributed_morphology_audit", "morphology.json"),
        ("human_review_decision", "decision.json"),
        ("human_preview", "preview.png"),
    ]
    spec = {
        "schema": "gprmax_release_spec_v1",
        "release_id": "TEST_RELEASE",
        "case_id": "TEST_CASE",
        "release_class": "development_evidence",
        "created_date": "2026-07-15",
        "formal_training_allowed": False,
        "line9_conditioned": True,
        "human_review": {"decision": "accepted", "reviewer": "test"},
        "output_dir": "data/PGDA_SYNTH_DATASET_V2/02_released_solver_evidence/test-release",
        "artifacts": [
            {
                "role": role,
                "source": path.relative_to(release.ROOT).as_posix(),
                "destination": f"evidence/{name}",
            }
            for role, name in artifacts
            for path in [sources / name]
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_package_and_verify_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(
        release,
        "RELEASE_ROOT",
        (tmp_path / "data/PGDA_SYNTH_DATASET_V2/02_released_solver_evidence").resolve(),
    )
    spec = _spec(tmp_path)
    output = release.package(spec)
    assert (output / "RELEASE_MANIFEST.json").is_file()
    payload = json.loads(spec.read_text(encoding="utf-8"))
    spec.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    assert release.verify(spec) == output
    manifest = json.loads((output / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["artifact_count"] == 8
    assert manifest["formal_training_allowed"] is False


def test_rejects_unmerged_trace_output(tmp_path: Path) -> None:
    path = tmp_path / "full_scene1.out"
    _write_out(path)
    with pytest.raises(ValueError, match="unmerged per-trace"):
        release._validate_destination(path.name)


def test_development_release_requires_accepted_human_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(
        release,
        "RELEASE_ROOT",
        (tmp_path / "data/PGDA_SYNTH_DATASET_V2/02_released_solver_evidence").resolve(),
    )
    spec = _spec(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["human_review"]["decision"] = "pending"
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted human review"):
        release.load_spec(spec)
