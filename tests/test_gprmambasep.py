"""Tests for GprMambaSep — full architecture, output interface, gradients, and dispatch."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))

import torch
import pytest

from pgdacsnet.model_raw_unet import build_model
from pgdacsnet.model_gprmambasep import build_gprmambasep, GprMambaSep
from pgdacsnet.model_mamba import SelectiveSSMLite, Mamba2DBlock
from pgdacsnet.model_interfaces import GprMambaSepOutput, unpack_pgda_output


# ── SelectiveSSMLite tests ──

class TestSelectiveSSMLite:
    def test_forward_shape(self):
        m = SelectiveSSMLite(64, 64, 4, 2)
        x = torch.randn(2, 64, 512)
        y = m(x)
        assert y.shape == (2, 64, 512)

    def test_gradient_flow(self):
        m = SelectiveSSMLite(32, 32, 3, 2)
        x = torch.randn(2, 32, 128, requires_grad=True)
        y = m(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_independence(self):
        m = SelectiveSSMLite(16, 16, 4, 2)
        x = torch.randn(2, 16, 64)
        y = m(x)
        assert not torch.allclose(y[0], y[1])

    def test_variable_lengths(self):
        m = SelectiveSSMLite(32, 32, 4, 2)
        for L in [64, 128, 256]:
            y = m(torch.randn(1, 32, L))
            assert y.shape == (1, 32, L)

    def test_param_count(self):
        m = SelectiveSSMLite(128, 64, 4, 2)
        n = sum(p.numel() for p in m.parameters())
        assert 50000 < n < 200000, f"params={n}"


# ── Mamba2DBlock tests ──

class TestMamba2DBlock:
    def test_forward_shape(self):
        m = Mamba2DBlock(64, 32, 4, 2)
        x = torch.randn(2, 64, 32, 16)
        y = m(x)
        assert y.shape == (2, 64, 32, 16)

    def test_gradient_flow(self):
        m = Mamba2DBlock(32, 16, 4, 2)
        x = torch.randn(1, 32, 64, 32, requires_grad=True)
        y = m(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_non_square_input(self):
        m = Mamba2DBlock(16, 16, 4, 2)
        y = m(torch.randn(1, 16, 48, 24))
        assert y.shape == (1, 16, 48, 24)

    def test_multiple_stacked(self):
        m = torch.nn.Sequential(*[Mamba2DBlock(32, 16, 4, 2) for _ in range(3)])
        x = torch.randn(2, 32, 64, 32)
        y = m(x)
        assert y.shape == (2, 32, 64, 32)


# ── GprMambaSep architecture tests ──

class TestGprMambaSepArch:
    def test_build_gprmambasep(self):
        """build_gprmambasep produces correct output structure."""
        m = build_gprmambasep({'base_ch': 16, 'mamba_state_dim': 32})
        assert isinstance(m, GprMambaSep), f"wrong type: {type(m)}"

    def test_build_model_dispatch(self):
        """build_model dispatches correctly via v2_0_gprmambasep key."""
        m = build_model({'model_arch': 'v2_0_gprmambasep', 'base_ch': 16, 'mamba_state_dim': 32})
        assert isinstance(m, GprMambaSep)

    def test_forward_shapes(self):
        """All 6 output heads have correct shapes."""
        m = build_gprmambasep({'base_ch': 16, 'mamba_state_dim': 32})
        x = torch.randn(2, 1, 128, 64)
        out = m(x)
        assert out.mask_logits.shape == (2, 1, 128, 64), f"mask: {out.mask_logits.shape}"
        assert out.presence_logits.shape == (2, 1, 64), f"pres: {out.presence_logits.shape}"
        assert out.center_logits.shape == (2, 1, 128, 64), f"center: {out.center_logits.shape}"
        assert out.A_hat.shape == (2, 1, 128, 64), f"A: {out.A_hat.shape}"
        assert out.S_hat.shape == (2, 1, 128, 64), f"S: {out.S_hat.shape}"
        assert out.G_hat.shape == (2, 1, 128, 64), f"G: {out.G_hat.shape}"

    def test_forward_shapes_with_aux_channels(self):
        """Stem keeps auxiliary terrain/meta channels without abs duplication bugs."""
        m = build_model({
            'model_arch': 'v2_1_gprmambasep_lite',
            'base_ch': 8,
            'mamba_state_dim': 16,
            'input_channels': 3,
        })
        x = torch.randn(2, 3, 128, 64)
        out = m(x)
        assert out.mask_logits.shape == (2, 1, 128, 64)
        assert out.presence_logits.shape == (2, 1, 64)


    def test_gprmambasep_output_interface(self):
        """GprMambaSepOutput is both dict-like and tuple-unpackable."""
        m = build_gprmambasep({'base_ch': 16, 'mamba_state_dim': 32})
        out = m(torch.randn(2, 1, 128, 64))
        assert isinstance(out, GprMambaSepOutput)
        # Dict access
        assert out['mask_logits'] is out.mask_logits
        assert 'A_hat' in out
        assert 'G_hat' in out
        # unpack_pgda_output takes mask/pres/center for backward compat
        unpacked = unpack_pgda_output(out)
        assert unpacked[0] is out.mask_logits

    def test_gradient_through_full_model(self):
        """Full forward-backward pass works."""
        m = build_gprmambasep({'base_ch': 8, 'mamba_state_dim': 16})
        x = torch.randn(1, 1, 64, 32)
        out = m(x)
        loss = out.mask_logits.sum() + out.A_hat.sum() + out.G_hat.sum()
        loss.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        assert grad_count > 50, f"only {grad_count} grads"

    def test_build_model_legacy_still_works(self):
        """Existing architectures remain unaffected."""
        m = build_model({'model_arch': 'v1_9d_mambavision_hybrid', 'base_ch': 20})
        out = m(torch.randn(1, 1, 128, 64))
        assert len(out) in (2, 3), f"legacy output len={len(out)}"


# ── Loss integration tests ──

class TestGprMambaSepLoss:
    def test_loss_computation(self):
        """Full loss pipeline with all component heads enabled."""
        m = build_gprmambasep({'base_ch': 8, 'mamba_state_dim': 16})
        x = torch.randn(2, 1, 64, 32)
        out = m(x)
        batch = {
            'x': x,
            'y': torch.sigmoid(torch.randn(2, 1, 64, 32)),
            'y_core': torch.sigmoid(torch.randn(2, 1, 64, 32)),
            'presence': torch.randn(2, 1, 32),
            'presence_valid': torch.ones(2, 1, 32),
            'weight': torch.ones(2, 32),
            'valid_pix': torch.ones(2, 1, 64, 32),
            'valid_denom': torch.tensor(64 * 32.0),
            'X_clean': x.clone(),
        }
        cfg = {
            'loss': {
                'base_pixel_weight': 0.1, 'positive_pixel_boost': 4,
                'dice_weight': 0.5, 'core_weight': 0.25, 'outside_weight': 0.4,
                'hard_negative_weight': 0.35, 'presence_weight': 0.25,
                'self_consistency_weight': 2.0, 'sim_supervised_weight': 0.5,
                'contrastive_weight': 0.0, 'arrival_prior_weight': 0.1,
            }
        }
        from scripts.losses_gprmambasep import compute_gprmambasep_loss
        total_loss, parts = compute_gprmambasep_loss(out, batch, cfg, m)
        assert len(parts) > 0
        assert parts['seg_band_bce'] > 0.0
        assert parts['seg_presence_loss'] >= 0.0
        for k, v in parts.items():
            assert torch.isfinite(torch.tensor(v)).all(), f"{k} non-finite"

    def test_loss_gradients_flow(self):
        """Backward pass from combined loss works."""
        m = build_gprmambasep({'base_ch': 8, 'mamba_state_dim': 16})
        x = torch.randn(1, 1, 64, 32)
        out = m(x)
        batch = {
            'x': x,
            'y': torch.sigmoid(torch.randn(1, 1, 64, 32)),
            'y_core': torch.sigmoid(torch.randn(1, 1, 64, 32)),
            'presence': torch.randn(1, 1, 32),
            'presence_valid': torch.ones(1, 1, 32),
            'weight': torch.ones(1, 32),
            'valid_pix': torch.ones(1, 1, 64, 32),
            'valid_denom': torch.tensor(64 * 32.0),
        }
        cfg = {
            'loss': {
                'base_pixel_weight': 0.1, 'positive_pixel_boost': 4,
                'dice_weight': 0.5, 'core_weight': 0.25, 'outside_weight': 0.4,
                'hard_negative_weight': 0.35, 'presence_weight': 0.25,
                'self_consistency_weight': 2.0, 'sim_supervised_weight': 0.5,
                'contrastive_weight': 0.0,
            }
        }
        from scripts.losses_gprmambasep import compute_gprmambasep_loss
        total_loss, parts = compute_gprmambasep_loss(out, batch, cfg, m)
        total_loss.backward()
        grads = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads > 0, f"no non-zero gradients ({grads})"
