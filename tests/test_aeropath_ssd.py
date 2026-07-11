import pytest
import torch
import csv
import numpy as np

from pgdacsnet.model_aeropath_ssd import AeroPathSSD, SoftPathInference
from pgdacsnet.model_raw_unet import build_model
from pgdacsnet.losses_aeropath import compute_aeropath_loss
from pgdacsnet.experiment_contract import ContractError, validate_experiment_config
from scripts.eval_full_line import stitch_one
from scripts.train_raw_only import compute_loss


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
    assert output.uncertainty_logits.shape == (1, 1, 64, 32)
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


def test_aeropath_structured_loss_supervises_path_and_abstention():
    model = AeroPathSSD(base_ch=8, ssm_impl="ssm_lite", mamba_state_dim=8, max_path_step=3)
    x = torch.randn(2, 1, 32, 16)
    output = model(x, altitude=torch.full((2, 16), 8.0))
    y = torch.zeros(2, 1, 32, 16)
    y[0, 0, 18, :] = 1.0
    y[1, 0, 12, :] = 1.0
    batch = {
        "x": x,
        "y": y,
        "y_core": y,
        "presence": torch.ones(2, 16),
        "presence_valid": torch.ones(2, 16),
        "weight": torch.ones(2, 16),
        "valid_pix": torch.ones_like(y),
        "valid_denom": torch.tensor(float(y.numel())),
        "ignore_mask": torch.zeros_like(y),
    }
    loss, parts = compute_aeropath_loss(output, batch, {"loss": {}})
    assert torch.isfinite(loss)
    assert parts["path_supervised_trace_count"] == 32.0
    assert parts["no_pick_supervised_window_count"] == 2.0
    loss.backward()
    assert model.energy_head.weight.grad is not None


def test_aeropath_missing_altitude_stays_finite():
    model = AeroPathSSD(base_ch=8, ssm_impl="ssm_lite", mamba_state_dim=8)
    output = model(torch.randn(1, 1, 32, 16), altitude=torch.tensor([[8.0, float("nan")] + [8.0] * 14]))
    assert torch.isfinite(output.air_reduced_input).all()
    assert torch.isfinite(output.path_marginals).all()


def test_aeropath_full_line_stitch_passes_tracewise_altitude(tmp_path):
    data_root = tmp_path / "data"; (data_root / "lines").mkdir(parents=True)
    raw = np.random.default_rng(0).normal(size=(32, 16)).astype(np.float32)
    np.savez_compressed(
        data_root / "lines" / "LineA.npz",
        raw_full_normalized=raw,
        flight_height_agl_m=np.linspace(7.0, 9.0, 16, dtype=np.float32),
    )
    with (data_root / "window_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "line", "start", "end", "present", "weak", "no_pick"])
        writer.writeheader(); writer.writerow({"sample_id": "a", "line": "LineA", "start": 0, "end": 15, "present": 16, "weak": 0, "no_pick": 0})
    cfg = {"model_arch": "aeropath_ssd", "base_ch": 8, "ssm_impl": "ssm_lite", "mamba_state_dim": 8,
           "mamba_d_conv": 4, "height_resize": 32, "width_resize": 16, "input_log_scale": 1e-3,
           "data_root": str(data_root)}
    model = build_model(cfg)
    run_dir = tmp_path / "run"; run_dir.mkdir()
    torch.save({"cfg": cfg, "model": model.state_dict()}, run_dir / "checkpoint_last.pt")
    result = stitch_one(run_dir, "LineA", "last", torch.device("cpu"), return_details=True)
    pred, presence, _center, _curve, _cfg, _root, details = result
    assert pred.shape == raw.shape and presence.shape == (16,)
    assert details["altitude_conditioning_used"] is True
    assert details["structured_path_prob"].shape == raw.shape
    assert details["uncertainty_log_variance"].shape == raw.shape


def test_train_entry_routes_aeropath_to_structured_objective():
    model = AeroPathSSD(base_ch=8, ssm_impl="ssm_lite", mamba_state_dim=8)
    x = torch.randn(1, 1, 32, 16)
    y = torch.zeros(1, 1, 32, 16); y[:, :, 15, :] = 1.0
    batch = {
        "x": x, "y": y, "y_core": y, "presence": torch.ones(1, 16),
        "presence_valid": torch.ones(1, 16), "weight": torch.ones(1, 16),
        "ignore_mask": torch.zeros_like(y), "altitude": torch.full((1, 16), 8.0),
    }
    loss, parts = compute_loss(model, batch, torch.device("cpu"), {"loss": {}})
    assert torch.isfinite(loss)
    assert "path_nll" in parts and "no_pick_bce" in parts


def test_formal_aeropath_lite_backend_is_rejected(tmp_path):
    with pytest.raises(ContractError, match="official_mamba2"):
        validate_experiment_config(
            {
                "run_type": "paper_train", "model_arch": "aeropath_ssd", "ssm_impl": "ssm_lite",
                "aeropath_enable_structured_loss": True,
                "train_lines": ["Line3"], "val_lines": ["LineL1"], "test_lines": ["Line9"],
            },
            tmp_path,
        )
