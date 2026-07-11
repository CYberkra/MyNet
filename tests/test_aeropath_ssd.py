import pytest
import torch

from pgdacsnet.model_aeropath_ssd import AeroPathSSD, SoftPathInference
from pgdacsnet.model_raw_unet import build_model


def test_aeropath_ssd_smoke_shape_gradient_and_path_normalisation():
    model = AeroPathSSD(
        base_ch=8,
        input_channels=3,
        metadata_channels=2,
        ssm_impl="ssm_lite",
        mamba_state_dim=8,
        max_path_step=3,
    )
    x = torch.randn(1, 3, 64, 32, requires_grad=True)
    altitude = torch.full((1, 32), 8.0)
    output = model(x, altitude=altitude)
    assert output.mask_logits.shape == (1, 1, 64, 32)
    assert output.curve_logits.shape == (1, 1, 64, 32)
    assert output.presence_logits.shape == (1, 1, 32)
    assert output.no_pick_logits.shape == (1, 1)
    assert torch.allclose(output.path_marginals.sum(dim=2), torch.ones((1, 1, 32)), atol=1e-5)
    (output.mask_logits.mean() + output.path_marginals.mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_aeropath_path_prefers_a_consistent_high_energy_track():
    decoder = SoftPathInference(max_step=2)
    unary = torch.full((1, 1, 21, 16), -7.0)
    unary[:, :, 11, :] = 7.0
    path = decoder(unary)
    assert torch.equal(path.argmax(dim=2).squeeze(), torch.full((16,), 11, dtype=torch.long))


def test_aeropath_build_dispatch_uses_explicit_ssm_backend():
    model = build_model({
        "model_arch": "aeropath_ssd",
        "base_ch": 8,
        "input_channels": 1,
        "ssm_impl": "ssm_lite",
        "mamba_state_dim": 8,
        "mamba_d_conv": 4,
    })
    assert isinstance(model, AeroPathSSD)


def test_aeropath_official_mamba2_never_falls_back_silently():
    try:
        import mamba_ssm  # type: ignore # pragma: no cover
    except Exception:
        with pytest.raises(ImportError, match="official_mamba2"):
            AeroPathSSD(base_ch=8, ssm_impl="official_mamba2")
