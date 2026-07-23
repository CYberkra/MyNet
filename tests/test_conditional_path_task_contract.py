"""Regression tests for the measured-path versus abstention evidence boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_project_contracts import ROOT, validate_task_eligibility


def _manifest() -> dict[str, object]:
    path = ROOT / "data" / "contracts" / "dataset_v2" / "dataset_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_v15_conditional_path_does_not_require_a_real_negative_window():
    errors, warnings, counts = validate_task_eligibility(_manifest())
    assert not errors
    assert counts["measured_conditional_path_training_allowed"] is True
    assert counts["measured_abstention_evaluation_allowed"] is False
    assert counts["simulated_abstention_training_allowed"] is False
    assert any("simulated abstention" in item for item in warnings)


def test_contract_rejects_enabling_measured_abstention_without_evidence():
    manifest = copy.deepcopy(_manifest())
    manifest["task_eligibility"]["measured_abstention"]["evaluation_evidence_available"] = True
    errors, _warnings, _counts = validate_task_eligibility(manifest)
    assert any("measured abstention evaluation cannot be enabled" in item for item in errors)


def test_contract_rejects_making_conditional_path_depend_on_real_negatives():
    manifest = copy.deepcopy(_manifest())
    manifest["task_eligibility"]["measured_conditional_path"]["requires_confirmed_real_negative"] = True
    errors, _warnings, _counts = validate_task_eligibility(manifest)
    assert any("conditional path task must not require" in item for item in errors)
