from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from scripts.export_sim_training_npz import _case_mask, load_eligible_rows


def _load_promotion_module():
    path = Path(__file__).resolve().parents[1] / "data" / "PGDA_SYNTH_DATASET_V1" / "tools" / "promote_to_accepted.py"
    spec = importlib.util.spec_from_file_location("promote_to_accepted", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["case_id", "train_allowed", "line9_conditioned"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def test_line9_terrain_family_is_not_misclassified_as_generic():
    module = _load_promotion_module()
    assert module._detect_family("LINE9_TERRAIN_011") == "line9_style/terrain"
    assert module._detect_family("LINE9_STYLE_TERRAIN_999") == "line9_style/terrain"


def test_sim_export_requires_explicit_train_allowed(tmp_path: Path):
    manifest = tmp_path / "simulation_cases.csv"
    _write_manifest(manifest, [{"case_id": "A", "train_allowed": "false", "line9_conditioned": "false"}])
    assert load_eligible_rows(manifest) == []


def test_sim_export_blocks_line9_conditioned_formal_holdout(tmp_path: Path):
    manifest = tmp_path / "simulation_cases.csv"
    _write_manifest(manifest, [{"case_id": "LEAKED", "train_allowed": "true", "line9_conditioned": "true"}])
    with pytest.raises(RuntimeError, match="Line9-conditioned"):
        load_eligible_rows(manifest, formal_test_line="Line9")


def test_v2_negative_export_uses_confirmed_negative_mask(tmp_path: Path):
    case = tmp_path / "CTRL_NEG"
    labels = case / "labels"
    labels.mkdir(parents=True)
    negative = labels / "target_mask_confirmed_negative_501x256.npy"
    negative.write_bytes(b"test")
    row = {
        "case_id": "CTRL_NEG",
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "target_presence": "false",
        "label_path": str(negative),
    }
    assert _case_mask(case, row) == negative


def test_human_promotion_requires_matching_source_hash(tmp_path: Path):
    module = _load_promotion_module()
    manifest = tmp_path / "human.csv"
    fields = ["case_id", "decision", "auditor", "date", "method", "source_sha256"]
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerow({
            "case_id": "CASE_001", "decision": "promote", "auditor": "reviewer",
            "date": "2026-07-10", "method": "geometry_review", "source_sha256": "wrong",
        })
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        module._load_human_decision(manifest, "CASE_001", "correct")
