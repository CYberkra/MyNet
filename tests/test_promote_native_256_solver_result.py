from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.promote_native_256_solver_result import promote


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_promotion_copies_canonical_evidence_but_not_raw_solver_outputs(tmp_path: Path) -> None:
    case = tmp_path / "N256_F01"
    labels = case / "labels"
    logs = case / "run_logs"
    labels.mkdir(parents=True)
    logs.mkdir()
    _write_json(
        case / "scene_manifest.json",
        {"case_id": "N256_F01", "target_presence": True, "formal_training_allowed": False},
    )
    _write_json(
        case / "postprocess_validation.json",
        {"ok": True, "postprocess_validated": True, "output_shape_canonical": [501, 256]},
    )
    for name in (
        "full_scene_501x256.npy",
        "air_reference_501x256.npy",
        "no_basal_contrast_501x256.npy",
        "contrast_response_501x256.npy",
        "target_mask_visible_phase_501x256.npy",
        "visible_phase_support_ratio.npy",
    ):
        shape = (256,) if name == "visible_phase_support_ratio.npy" else (501, 256)
        np.save(labels / name, np.zeros(shape, dtype=np.float32))
    for stem in ("full_scene", "no_basal_contrast_control", "air_reference"):
        _write_json(logs / f"{stem}_trace_contract.json", {"complete": True})
    (case / "full_scene_merged.out").write_bytes(b"raw solver output")

    release = promote(case, tmp_path / "release")
    destination = tmp_path / "release" / "N256_F01"
    assert release["formal_training_allowed"] is False
    assert (destination / "labels" / "full_scene_501x256.npy").is_file()
    assert (destination / "trace_contracts" / "full_scene_trace_contract.json").is_file()
    assert not (destination / "full_scene_merged.out").exists()
    assert json.loads((destination / "release_manifest.json").read_text(encoding="utf-8"))["raw_solver_outputs_included"] is False
