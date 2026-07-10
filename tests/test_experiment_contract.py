import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pgdacsnet.experiment_contract import (
    ContractError,
    enforce_simulation_holdout_policy,
    inspect_full_line_dataset,
    inspect_window_dataset,
    validate_experiment_config,
)


def _write_index(root: Path, rows):
    (root / "windows").mkdir(parents=True)
    with (root / "window_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "line", "start", "end", "present", "weak", "no_pick"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        np.savez(root / "windows" / f"{row['sample_id']}.npz", x=np.zeros((2, 2), np.float32))


def test_mixed_dataset_contract_rejects_missing_index(tmp_path):
    root = tmp_path / "sim"
    root.mkdir()
    with pytest.raises(ContractError, match="window_index.csv"):
        inspect_window_dataset(root)


def test_window_dataset_contract_accepts_resolved_samples(tmp_path):
    root = tmp_path / "sim"
    _write_index(root, [{"sample_id": "a", "line": "sim_a", "start": 0, "end": 1, "present": 1, "weak": 0, "no_pick": 0}])
    summary = inspect_window_dataset(root, required_lines=["sim_a"])
    assert summary.row_count == 1
    assert summary.lines == ("sim_a",)


def test_full_line_contract_requires_canonical_arrays(tmp_path):
    lines = tmp_path / "lines"
    lines.mkdir()
    np.savez(
        lines / "Line9.npz",
        raw_full_normalized=np.zeros((4, 3), np.float32),
        soft_mask_train=np.zeros((4, 3), np.float32),
        status_code=np.ones(3, np.int16),
        label_weight=np.ones(3, np.float32),
        dt_ns=np.asarray(1.4, np.float32),
    )
    audit = inspect_full_line_dataset(tmp_path, ["Line9"])
    assert audit["lines"]["Line9"]["shape"] == (4, 3)


def test_config_contract_rejects_review_only_validation(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"review_only_lines": ["LineX1"]}), encoding="utf-8")
    cfg = {
        "run_type": "holdout_eval",
        "paper_split_file": str(split),
        "train_lines": ["Line3"],
        "val_lines": ["LineX1"],
        "test_lines": ["Line9"],
        "review_lines": [],
    }
    with pytest.raises(ContractError, match="review-only"):
        validate_experiment_config(cfg, tmp_path)


def test_config_contract_requires_run_type(tmp_path):
    cfg = {"train_lines": ["Line3"], "val_lines": ["Line7"], "test_lines": ["Line9"]}
    with pytest.raises(ContractError, match="run_type"):
        validate_experiment_config(cfg, tmp_path)


def test_line_conditioned_sim_is_rejected_for_formal_holdout(tmp_path):
    root = tmp_path / "sim"
    root.mkdir()
    (root / "DATA_USAGE_POLICY.json").write_text(
        json.dumps({"conditioned_on_lines": ["Line9"], "train_allowed": True}),
        encoding="utf-8",
    )
    cfg = {"run_type": "holdout_eval", "test_lines": ["Line9"]}
    with pytest.raises(ContractError, match="conditioned"):
        enforce_simulation_holdout_policy(cfg, root)
