import torch

from pgdacsnet.model_raw_unet import build_model
from scripts.losses_gprmambasep import compute_gprmambasep_loss


def test_curvegassist_forward_and_loss_smoke():
    cfg = {
        "model_arch": "v2_1_curvegassist_lite",
        "base_ch": 2,
        "mamba_state_dim": 0,
        "height_resize": 16,
        "width_resize": 8,
        "enable_curve_head": True,
        "enable_global_no_target_head": True,
        "enable_uncertainty_head": True,
        "loss": {
            "g_segmentation_weight": 0.1,
            "curve_distribution_weight": 1.0,
            "global_no_target_weight": 0.1,
            "uncertainty_weight": 0.05,
        },
    }
    model = build_model(cfg)
    x = torch.randn(1, 1, 16, 8)
    out = model(x)
    assert out.mask_logits.shape == (1, 1, 16, 8)
    assert out.curve_logits.shape == (1, 1, 16, 8)
    assert out.global_no_target_logits.shape == (1, 1)
    assert out.uncertainty_logits is not None
    assert out.uncertainty_logits.shape == (1, 1, 16, 8)
    batch = {
        "x": x,
        "y": torch.rand(1, 1, 16, 8),
        "y_core": torch.rand(1, 1, 16, 8),
        "presence": torch.ones(1, 1, 8),
        "presence_valid": torch.ones(1, 1, 8),
        "weight": torch.ones(1, 8),
        "valid_pix": torch.ones(1, 1, 16, 8),
        "valid_denom": torch.tensor(128.0),
        "ignore_mask": torch.zeros(1, 1, 16, 8),
    }
    loss, parts = compute_gprmambasep_loss(out, batch, cfg)
    assert torch.isfinite(loss)
    assert "curve_ce" in parts
    assert "global_no_target" in parts
    assert "uncertainty_nll" in parts
