import pytest
import torch
import csv
import numpy as np
import itertools
from types import SimpleNamespace

from pgdacsnet.model_aeropath_ssd import AeroPathSSD, SoftPathInference
from pgdacsnet.model_mamba import OfficialMamba2Sequence
from pgdacsnet.model_raw_unet import build_model
from pgdacsnet.losses_aeropath import compute_aeropath_loss, structured_path_losses
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
    assert output.null_marginals.shape == (1, 1, 32)
    assert torch.allclose(
        output.path_marginals.sum(dim=2) + output.null_marginals,
        torch.ones((1, 1, 32)), atol=1e-5,
    )
    (output.mask_logits.mean() + output.path_marginals.mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_aeropath_path_prefers_a_consistent_high_energy_track():
    decoder = SoftPathInference(max_step=2)
    unary = torch.full((1, 1, 21, 16), -7.0)
    unary[:, :, 11, :] = 7.0
    path = decoder(unary)
    assert torch.equal(path.argmax(dim=2).squeeze(), torch.full((16,), 11, dtype=torch.long))


def test_soft_path_marginals_match_bruteforce_with_null_state():
    """Regression for the alpha/beta emission accounting at path boundaries."""
    decoder = SoftPathInference(max_step=1, initial_slope_penalty=0.7, temperature=1.0,
                                initial_start_penalty=0.4, initial_end_penalty=0.6)
    unary = torch.tensor([[[[0.3, -0.2, 0.1], [0.0, 0.7, -0.5]]]])
    null_logits = torch.tensor([[[0.2, -0.1, 0.4]]])
    result = decoder(unary, null_logits=null_logits, return_details=True)
    score = unary[0, 0]
    null_score = null_logits[0, 0]
    slope = torch.nn.functional.softplus(decoder.log_slope_penalty).item()
    start = torch.nn.functional.softplus(decoder.log_start_penalty).item()
    end = torch.nn.functional.softplus(decoder.log_end_penalty).item()
    paths, scores = [], []
    for states in itertools.product(range(3), repeat=3):  # 0/1 physical, 2=NULL
        total = null_score[0].item() if states[0] == 2 else score[states[0], 0].item() - start
        valid = True
        for col in range(1, 3):
            prev, state = states[col - 1], states[col]
            if prev < 2 and state < 2:
                if abs(prev - state) > 1:
                    valid = False; break
                total += score[state, col].item() - slope * abs(prev - state)
            elif prev == 2 and state < 2:
                total += score[state, col].item() - start
            elif prev < 2 and state == 2:
                total += null_score[col].item() - end
            else:
                total += null_score[col].item()
        if valid:
            paths.append(states); scores.append(total)
    scores = torch.tensor(scores)
    probs = torch.softmax(scores, dim=0)
    expected = torch.zeros(3, 3)
    for states, probability in zip(paths, probs):
        for col, state in enumerate(states):
            expected[state, col] += probability
    actual = torch.cat([result.path_marginals[0, 0], result.null_marginals[0]], dim=0)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_official_mamba2_headdim_contract_fails_before_optional_import():
    with pytest.raises(ValueError, match="divisible by headdim"):
        OfficialMamba2Sequence(d_model=24, expand=2, headdim=64)


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


def test_aeropath_probability_losses_are_autocast_safe():
    """NULL/start/end posteriors are probabilities, so their BCE runs in FP32."""
    model = AeroPathSSD(base_ch=8, ssm_impl="ssm_lite", mamba_state_dim=8, max_path_step=3)
    x = torch.randn(1, 1, 24, 8)
    y = torch.zeros(1, 1, 24, 8)
    y[:, :, 12, :] = 1.0
    batch = {
        "x": x, "y": y, "y_core": y,
        "presence": torch.ones(1, 8), "presence_valid": torch.ones(1, 8),
        "weight": torch.ones(1, 8), "valid_pix": torch.ones_like(y),
        "valid_denom": torch.tensor(float(y.numel())), "ignore_mask": torch.zeros_like(y),
        "trace_state": torch.ones(1, 8, dtype=torch.long),
        "valid_trace_mask": torch.ones(1, 8),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(x, altitude=torch.full((1, 8), 8.0))
        loss, _parts = compute_aeropath_loss(output, batch, {"loss": {}})
    assert torch.isfinite(loss)
    loss.backward()


def test_aeropath_no_pick_skips_mixed_weak_and_negative_windows():
    model = AeroPathSSD(base_ch=8, ssm_impl="ssm_lite", mamba_state_dim=8)
    x = torch.randn(1, 1, 24, 4)
    output = model(x, altitude=torch.full((1, 4), 8.0))
    y = torch.zeros(1, 1, 24, 4)
    batch = {
        "x": x, "y": y, "y_core": y, "presence": torch.zeros(1, 4),
        "presence_valid": torch.tensor([[1.0, 0.0, 1.0, 1.0]]), "weight": torch.ones(1, 4),
        "valid_pix": torch.ones_like(y), "valid_denom": torch.tensor(float(y.numel())),
        "ignore_mask": torch.zeros_like(y), "trace_state": torch.tensor([[0, 2, 0, 0]]),
        "valid_trace_mask": torch.ones(1, 4),
    }
    _loss, parts = compute_aeropath_loss(output, batch, {"loss": {}})
    assert parts["no_pick_supervised_window_count"] == 0.0


def test_aeropath_missing_altitude_stays_finite():
    model = AeroPathSSD(base_ch=8, ssm_impl="ssm_lite", mamba_state_dim=8)
    output = model(torch.randn(1, 1, 32, 16), altitude=torch.tensor([[8.0, float("nan")] + [8.0] * 14]))
    assert torch.isfinite(output.air_reduced_input).all()
    assert torch.isfinite(output.path_marginals).all()


def test_aeropath_rejects_missing_metadata_channels():
    model = AeroPathSSD(
        base_ch=8,
        input_channels=3,
        metadata_channels=2,
        ssm_impl="ssm_lite",
        mamba_state_dim=8,
    )
    with pytest.raises(ValueError, match="metadata channels"):
        model(torch.randn(1, 2, 32, 16), altitude=torch.full((1, 16), 8.0))


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


def test_train_entry_preserves_explicit_weak_trace_state():
    model = AeroPathSSD(base_ch=8, ssm_impl="ssm_lite", mamba_state_dim=8)
    x = torch.randn(1, 1, 24, 4)
    y = torch.zeros(1, 1, 24, 4)
    batch = {
        "x": x,
        "y": y,
        "y_core": y,
        "presence": torch.zeros(1, 4),
        "presence_valid": torch.ones(1, 4),
        "weight": torch.ones(1, 4),
        "ignore_mask": torch.zeros_like(y),
        "altitude": torch.full((1, 4), 8.0),
        "trace_state": torch.tensor([[0, 2, 0, 0]]),
        "valid_trace_mask": torch.ones(1, 4),
        "chainage_m": torch.tensor([[0.0, 0.8, 1.7, 2.6]]),
    }
    _loss, parts = compute_loss(model, batch, torch.device("cpu"), {"loss": {}})
    assert parts["no_pick_supervised_window_count"] == 0.0


def test_aeropath_smoothness_uses_physical_trace_spacing():
    path = torch.full((1, 1, 5, 4), 1e-6)
    path[0, 0, 1, :2] = 1.0
    path[0, 0, 3, 2:] = 1.0
    path = path / path.sum(dim=2, keepdim=True)
    target = path.clone()
    output = SimpleNamespace(
        path_marginals=path,
        uncertainty_logits=None,
        null_marginals=None,
        path_start_prob=None,
        path_end_prob=None,
        no_pick_logits=torch.zeros(1, 1),
    )
    common = {
        "y": target,
        "valid_pix": torch.ones_like(target),
        "weight": torch.ones(1, 4),
        "presence": torch.ones(1, 4),
        "presence_valid": torch.ones(1, 4),
        "trace_state": torch.ones(1, 4, dtype=torch.long),
        "valid_trace_mask": torch.ones(1, 4),
        "ignore_mask": torch.zeros_like(target),
    }
    uniform = structured_path_losses(
        output, {**common, "chainage_m": torch.tensor([[0.0, 1.0, 2.0, 3.0]])}, {}
    )
    physical_gap = structured_path_losses(
        output, {**common, "chainage_m": torch.tensor([[0.0, 1.0, 11.0, 12.0]])}, {}
    )
    assert physical_gap["path_smooth"] < uniform["path_smooth"] * 0.5


def test_aeropath_position_losses_condition_on_non_null_path_mass():
    # A NULL probability must affect abstention supervision, not shrink a
    # correctly localized physical path toward time zero.
    path = torch.zeros(1, 1, 5, 1)
    path[0, 0, 4, 0] = 0.25
    target = torch.zeros_like(path)
    target[0, 0, 4, 0] = 1.0
    output = SimpleNamespace(
        path_marginals=path,
        uncertainty_logits=torch.zeros_like(path),
        null_marginals=torch.tensor([[[0.75]]]),
        path_start_prob=None,
        path_end_prob=None,
        no_pick_logits=torch.zeros(1, 1),
    )
    batch = {
        "y": target,
        "valid_pix": torch.ones_like(target),
        "weight": torch.ones(1, 1),
        "presence": torch.ones(1, 1),
        "presence_valid": torch.ones(1, 1),
        "trace_state": torch.ones(1, 1, dtype=torch.long),
        "valid_trace_mask": torch.ones(1, 1),
        "ignore_mask": torch.zeros_like(target),
    }
    losses = structured_path_losses(output, batch, {})
    assert losses["path_center_l1"].item() == pytest.approx(0.0)
    assert losses["path_uncertainty_nll"].item() == pytest.approx(0.0)


def test_no_pick_skips_windows_with_invalid_trace_padding():
    path = torch.full((1, 1, 5, 4), 0.2)
    output = SimpleNamespace(
        path_marginals=path,
        uncertainty_logits=None,
        null_marginals=None,
        path_start_prob=None,
        path_end_prob=None,
        no_pick_logits=torch.zeros(1, 1),
    )
    target = torch.zeros_like(path)
    batch = {
        "y": target,
        "valid_pix": torch.ones_like(target),
        "weight": torch.ones(1, 4),
        "presence": torch.zeros(1, 4),
        "presence_valid": torch.ones(1, 4),
        "trace_state": torch.zeros(1, 4, dtype=torch.long),
        "valid_trace_mask": torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        "ignore_mask": torch.zeros_like(target),
    }
    losses = structured_path_losses(output, batch, {})
    assert losses["no_pick_supervised_window_count"] == 0.0


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


def test_formal_aeropath_requires_tracewise_metadata_contract(tmp_path):
    with pytest.raises(ContractError, match="tracewise acquisition/terrain conditioning"):
        validate_experiment_config(
            {
                "run_type": "paper_train",
                "model_arch": "aeropath_ssd",
                "ssm_impl": "official_mamba2",
                "base_ch": 24,
                "mamba_expand": 2,
                "mamba_headdim": 16,
                "aeropath_bidirectional_axial": True,
                "aeropath_enable_structured_loss": True,
                "train_lines": ["Line3"],
                "val_lines": ["LineL1"],
                "test_lines": ["Line9"],
            },
            tmp_path,
        )


def test_formal_aeropath_accepts_matching_tracewise_metadata_contract(tmp_path):
    audit = validate_experiment_config(
        {
            "run_type": "paper_train",
            "model_arch": "aeropath_ssd",
            "ssm_impl": "official_mamba2",
            "base_ch": 24,
            "mamba_expand": 2,
            "mamba_headdim": 16,
            "aeropath_bidirectional_axial": True,
            "aeropath_enable_structured_loss": True,
            "use_terrain_features": True,
            "terrain_feature_names": ["altitude_z", "terrain_slope_z"],
            "aeropath_metadata_channels": 2,
            "input_channels": 3,
            "train_lines": ["Line3"],
            "val_lines": ["LineL1"],
            "test_lines": ["Line9"],
        },
        tmp_path,
    )
    assert audit["errors"] == []
