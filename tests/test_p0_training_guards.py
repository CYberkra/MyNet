from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pgdacsnet.experiment_contract import ContractError, enforce_simulation_holdout_policy
from pgdacsnet.losses_pgda import compute_segmentation_losses, dice_loss_from_prob
from pgdacsnet.model_interfaces import PGDAOutput
from scripts.train_raw_only import compute_loss, dice_loss_from_prob as trainer_dice_loss
from scripts.train_raw_only import _finite_array
from pgdacsnet.model_mamba import SelectiveSSMCUDA


def _segmentation_batch(batch_size: int = 2) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    height, width = 3, 4
    logits = torch.zeros(batch_size, 1, height, width, requires_grad=True)
    outputs = {
        "mask_logits": logits,
        "presence_logits": torch.zeros(batch_size, 1, width, requires_grad=True),
        "center_logits": None,
    }
    batch = {
        "y": torch.zeros(batch_size, 1, height, width),
        "y_core": torch.zeros(batch_size, 1, height, width),
        "presence": torch.zeros(batch_size, 1, width),
        "presence_valid": torch.tensor(
            [[[1.0, 0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0, 1.0]]]
        ),
        "weight": torch.ones(batch_size, width),
        "valid_pix": torch.ones(batch_size, 1, height, width),
        "valid_denom": torch.tensor(float(batch_size * height * width)),
    }
    return outputs, batch


def test_presence_validity_never_broadcasts_across_batch_samples():
    outputs, batch = _segmentation_batch()
    losses = compute_segmentation_losses(outputs, batch, {"loss": {}})
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["presence_logits"], batch["presence"], reduction="none"
    )
    class_weight = torch.full_like(expected, 5.0)
    weights = (0.25 + batch["weight"][:, None, :]) * class_weight * batch["presence_valid"]
    assert torch.allclose(losses["presence_loss"], (expected * weights).sum() / weights.sum())


def test_empty_target_dice_has_false_positive_gradient():
    prediction = torch.full((1, 1, 3, 4), 0.5, requires_grad=True)
    loss = dice_loss_from_prob(prediction, torch.zeros_like(prediction), torch.ones_like(prediction))
    loss.backward()
    assert loss.item() > 0.0
    assert prediction.grad is not None
    assert prediction.grad.abs().sum().item() > 0.0


def test_trainer_copy_of_empty_target_dice_has_false_positive_gradient():
    prediction = torch.full((1, 1, 3, 4), 0.5, requires_grad=True)
    loss = trainer_dice_loss(prediction, torch.zeros_like(prediction), torch.ones_like(prediction))
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad.abs().sum().item() > 0.0


def test_trainer_presence_validity_never_broadcasts_across_batch_samples():
    class FixedOutputModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> PGDAOutput:
            batch_size, _, height, width = x.shape
            return PGDAOutput(
                torch.zeros(batch_size, 1, height, width),
                torch.tensor([[[2.0, -1.0, 0.5, -2.0]], [[-3.0, 1.5, -0.5, 3.0]]]),
                None,
            )

    outputs, batch = _segmentation_batch()
    del outputs
    batch["x"] = torch.zeros(2, 1, 3, 4)
    _, parts = compute_loss(FixedOutputModel(), batch, torch.device("cpu"), {"loss": {}})
    logits = torch.tensor([[[2.0, -1.0, 0.5, -2.0]], [[-3.0, 1.5, -0.5, 3.0]]])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["presence"], reduction="none")
    weights = (0.25 + batch["weight"][:, None, :]) * 5.0 * batch["presence_valid"]
    expected = (bce * weights).sum() / weights.sum()
    assert parts["presence_loss"] == pytest.approx(expected.item())


def test_simulation_training_rejects_missing_usage_policy(tmp_path: Path):
    with pytest.raises(ContractError, match="no explicit training-use policy"):
        enforce_simulation_holdout_policy({"run_type": "development", "test_lines": []}, tmp_path)


def test_simulation_training_requires_explicit_true_policy(tmp_path: Path):
    (tmp_path / "dataset_policy.json").write_text('{"training_allowed": false}', encoding="utf-8")
    with pytest.raises(ContractError, match="forbids training use"):
        enforce_simulation_holdout_policy({"run_type": "development", "test_lines": []}, tmp_path)

    (tmp_path / "dataset_policy.json").write_text('{"training_allowed": true}', encoding="utf-8")
    assert enforce_simulation_holdout_policy({"run_type": "development", "test_lines": []}, tmp_path)["training_allowed"] is True


def test_training_array_loader_rejects_nonfinite_values():
    with pytest.raises(ContractError, match="NaN/Inf"):
        _finite_array([0.0, float("nan")], name="x_raw")


def test_cuda_ssm_has_single_causal_padding_mechanism():
    # The optional CUDA kernel is unavailable on this Windows test host, but
    # this source-level contract ensures a future CUDA run cannot combine
    # Conv1d padding with the explicit left pad in forward().
    import inspect

    source = inspect.getsource(SelectiveSSMCUDA)
    assert "padding=0" in source
    assert "F.pad(x_res, (self.d_conv - 1, 0))" in source
