from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.losses_gprmambasep import arrival_time_prior_loss, curve_distribution_loss, global_no_target_loss
from scripts.train_raw_only import audit_config, audit_sim_dataset_policy, build_split_audit, validate_training_dataset_root


def _formal_cfg():
    return {
        "run_type": "holdout_eval",
        "model_arch": "v2_1_curvegassist_lite",
        "train_lines": ["Line3", "Line6"],
        "val_lines": ["Line7"],
        "test_lines": ["Line9"],
        "review_lines": ["LineX1"],
        "loss": {},
    }


def test_config_audit_rejects_review_only_x1_as_validation():
    cfg = _formal_cfg(); cfg["val_lines"] = ["LineX1"]
    with pytest.raises(RuntimeError, match="review-only"):
        audit_config(cfg)


def test_config_audit_requires_run_type():
    cfg = _formal_cfg(); cfg.pop("run_type")
    with pytest.raises(RuntimeError, match="run_type"):
        audit_config(cfg)


def test_line9_conditioned_sim_is_blocked_for_formal_line9_holdout(tmp_path: Path):
    (tmp_path / "dataset_policy.json").write_text(
        '{"training_allowed": true, "line9_conditioned": true}', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="conditioned"):
        audit_sim_dataset_policy(_formal_cfg(), tmp_path)


def test_missing_window_index_is_fatal(tmp_path: Path):
    with pytest.raises(RuntimeError, match="window_index.csv"):
        validate_training_dataset_root(tmp_path, purpose="simulation")


def test_split_audit_detects_overlapping_source_traces():
    class Dummy:
        def __init__(self, rows): self.rows = rows
    train = Dummy([{"sample_id": "a", "line": "Line9", "start": "0", "end": "100"}])
    test = Dummy([{"sample_id": "b", "line": "Line9", "start": "80", "end": "140"}])
    audit = build_split_audit(train_ds=train, test_ds=test)
    assert "test_train" in audit["trace_interval_overlaps"]


def test_global_no_target_ignores_all_unknown_window():
    logits = torch.tensor([[0.7]], requires_grad=True)
    batch = {"presence": torch.tensor([[[0.5, 0.5]]]), "presence_valid": torch.zeros(1, 1, 2)}
    loss = global_no_target_loss(logits, batch, {"loss": {}})
    assert loss.item() == pytest.approx(0.0)
    loss.backward(); assert logits.grad is not None


def test_global_no_target_uses_only_valid_traces():
    logits = torch.tensor([[0.0]])
    batch = {
        "presence": torch.tensor([[[0.8, 0.0]]]),
        "presence_valid": torch.tensor([[[0.0, 1.0]]]),
    }
    loss = global_no_target_loss(logits, batch, {"loss": {}})
    assert loss.item() == pytest.approx(0.693147, rel=1e-5)


def test_arrival_prior_has_no_implicit_2p4m_default():
    g = torch.ones(1, 1, 8, 2)
    cfg = {"height_resize": 8, "time_window_ns": 700.0, "arrival_prior_missing_height_policy": "skip"}
    assert arrival_time_prior_loss(g, {}, cfg).item() == pytest.approx(0.0)


def test_arrival_prior_error_policy_rejects_missing_height():
    g = torch.ones(1, 1, 8, 2)
    cfg = {"height_resize": 8, "time_window_ns": 700.0, "arrival_prior_missing_height_policy": "error"}
    with pytest.raises(RuntimeError, match="without valid"):
        arrival_time_prior_loss(g, {}, cfg)


def test_arrival_prior_uses_valid_agl_height():
    g = torch.ones(1, 1, 16, 2)
    cfg = {"height_resize": 16, "time_window_ns": 700.0}
    batch = {"altitude": torch.tensor([11.5]), "altitude_valid": torch.tensor([1.0])}
    assert arrival_time_prior_loss(g, batch, cfg).item() > 0


def test_shallow_suppression_does_not_penalise_legitimate_shallow_target():
    logits = torch.full((1, 1, 8, 2), -10.0); logits[:, :, 1, :] = 10.0
    common = {
        "weight": torch.ones(1, 2),
        "valid_pix": torch.ones(1, 1, 8, 2),
        "presence_valid": torch.ones(1, 1, 2),
    }
    cfg = {"time_window_ns": 8.0, "loss": {"shallow_suppression_max_ns": 3.0}}
    shallow_target = torch.zeros(1, 1, 8, 2); shallow_target[:, :, 1, :] = 1.0
    parts = curve_distribution_loss(logits, {**common, "y": shallow_target}, cfg)
    assert parts["curve_shallow_suppression"].item() == pytest.approx(0.0, abs=1e-7)
    deep_target = torch.zeros(1, 1, 8, 2); deep_target[:, :, 6, :] = 1.0
    parts_deep = curve_distribution_loss(logits, {**common, "y": deep_target}, cfg)
    assert parts_deep["curve_shallow_suppression"].item() > 0.9
