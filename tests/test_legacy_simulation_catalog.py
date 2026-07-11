import csv

import numpy as np

from scripts.build_governed_legacy_simulation_catalog import DEFAULT_AUDIT, build
from scripts.validate_governed_legacy_simulation_catalog import validate


def test_governed_legacy_catalog_is_deduplicated_and_formally_quarantined(tmp_path):
    summary = build(DEFAULT_AUDIT, tmp_path / "catalog")
    assert summary["unique_case_count"] == 33
    assert summary["physical_copy_count"] == 45
    assert summary["development_train_candidate_count"] == 20
    assert summary["diagnostic_only_count"] == 13
    assert summary["formal_training_allowed_count"] == 0

    with (tmp_path / "catalog" / "manifests" / "legacy_simulation_registry.csv").open(encoding="utf-8") as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    assert all(row["formal_training_allowed"] == "false" for row in rows.values())
    assert rows["LINE9_STYLE_001"]["catalog_role"] == "diagnostic_only_template_or_artifact"
    weak = np.load(tmp_path / "catalog" / "supervision" / "B003_SHALLOW_DISTRACTOR_011_status_code.npy")
    assert np.all(weak == 2)
    ignored = np.load(tmp_path / "catalog" / "supervision" / "B003_SHALLOW_DISTRACTOR_006_ignore_mask.npy")
    assert np.all(ignored[:, 32:36] == 1.0)
    validation = validate(tmp_path / "catalog")
    assert validation["result"] == "ok"
    assert validation["formal_training_allowed_count"] == 0
