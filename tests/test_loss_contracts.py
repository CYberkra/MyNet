import pytest
import torch

from scripts.losses_gprmambasep import (
    arrival_time_prior_loss,
    curve_distribution_loss,
    global_no_target_loss,
)


def test_arrival_prior_rejects_missing_height_by_default():
    g = torch.ones(2, 1, 16, 4)
    with pytest.raises(RuntimeError, match="height metadata"):
        arrival_time_prior_loss(g, {}, {"height_resize": 16, "time_window_ns": 700.0})


def test_arrival_prior_uses_only_valid_height_samples():
    g = torch.ones(2, 1, 16, 4)
    batch = {
        "altitude": torch.tensor([12.0, float("nan")]),
        "altitude_valid": torch.tensor([1.0, 0.0]),
    }
    loss = arrival_time_prior_loss(g, batch, {"time_window_ns": 700.0, "g_min_depth_m": 3.0, "arrival_prior_missing_height_policy": "skip"})
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_global_no_target_ignores_weak_invalid_traces():
    logits = torch.tensor([0.0, 0.0])
    batch = {
        "presence": torch.tensor([[[0.5, 0.5]], [[0.0, 0.0]]]),
        "presence_valid": torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]]]),
    }
    # First window is all unknown and skipped; second is a true no-target window.
    loss = global_no_target_loss(logits, batch, {"loss": {}})
    assert torch.allclose(loss, torch.tensor(0.6931472), atol=1e-5)


def test_shallow_suppression_excludes_legitimate_shallow_target():
    curve_logits = torch.zeros(1, 1, 8, 2)
    y = torch.zeros(1, 1, 8, 2)
    y[:, :, 0, :] = 1.0
    batch = {
        "y": y,
        "valid_pix": torch.ones_like(y),
        "weight": torch.ones(1, 2),
        "presence": torch.ones(1, 1, 2),
        "presence_valid": torch.ones(1, 1, 2),
    }
    parts = curve_distribution_loss(
        curve_logits,
        batch,
        {
            "time_window_ns": 700.0,
            "loss": {"shallow_suppression_max_ns": 200.0},
        },
    )
    assert torch.allclose(parts["curve_shallow_suppression"], torch.tensor(0.0), atol=1e-7)


def test_arrival_prior_supports_tracewise_measured_height():
    g = torch.zeros(1, 1, 16, 2)
    g[0, 0, 2, 1] = 1.0
    batch = {
        "altitude": torch.tensor([[5.0, 25.0]]),
        "altitude_valid": torch.ones(1, 2),
    }
    loss = arrival_time_prior_loss(
        g,
        batch,
        {
            "time_window_ns": 700.0,
            "g_min_depth_m": 0.0,
            "arrival_prior_missing_height_policy": "error",
        },
    )
    assert loss.item() > 0.0
