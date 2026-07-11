"""Tests for GprMambaSep — full architecture, output interface, gradients, and dispatch."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))

import torch
import pytest

from pgdacsnet.model_raw_unet import build_model
from pgdacsnet.model_gprmambasep import build_gprmambasep, GprMambaSep
from pgdacsnet.model_mamba import SelectiveSSMLite, AxialSSMLiteBlock, Mamba2DBlock
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


# ── AxialSSMLiteBlock tests ──

class TestAxialSSMLiteBlock:
    def test_forward_shape(self):
        m = AxialSSMLiteBlock(64, 32, 4, 2)
        x = torch.randn(2, 64, 32, 16)
        y = m(x)
        assert y.shape == (2, 64, 32, 16)

    def test_gradient_flow(self):
        m = AxialSSMLiteBlock(32, 16, 4, 2)
        x = torch.randn(1, 32, 64, 32, requires_grad=True)
        y = m(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_non_square_input(self):
        m = AxialSSMLiteBlock(16, 16, 4, 2)
        y = m(torch.randn(1, 16, 48, 24))
        assert y.shape == (1, 16, 48, 24)

    def test_multiple_stacked(self):
        m = torch.nn.Sequential(*[AxialSSMLiteBlock(32, 16, 4, 2) for _ in range(3)])
        x = torch.randn(2, 32, 64, 32)
        y = m(x)
        assert y.shape == (2, 32, 64, 32)



def test_legacy_mamba2dblock_alias_kept_for_checkpoint_compatibility():
    """Old imports keep working, but new code should use AxialSSMLiteBlock."""
    m = Mamba2DBlock(8, 8, 4, 2)
    assert isinstance(m, AxialSSMLiteBlock)

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
            'model_arch': 'v2_0_gprmambasep',
            'base_ch': 8,
            'mamba_state_dim': 16,
            'input_channels': 3,
        })
        x = torch.randn(2, 3, 128, 64)
        out = m(x)
        assert out.mask_logits.shape == (2, 1, 128, 64)
        assert out.presence_logits.shape == (2, 1, 64)



    def test_component_gates_shape_and_sum(self):
        """Soft A/S/G gates are exposed and sum to one per pixel."""
        m = build_gprmambasep({'base_ch': 8, 'mamba_state_dim': 16})
        out = m(torch.randn(2, 1, 64, 32))
        assert out.component_gates.shape == (2, 3, 64, 32)
        s = out.component_gates.sum(dim=1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)

    def test_official_mamba2_requires_dependency_when_requested(self):
        """official_mamba2 is explicit and never silently falls back to SSM-lite."""
        from pgdacsnet.model_mamba import make_axial_sequence_block
        try:
            import mamba_ssm  # noqa: F401
            has_dep = True
        except Exception:
            has_dep = False
        if has_dep:
            block = make_axial_sequence_block('official_mamba2', channels=8, d_state=8, d_conv=4)
            assert block is not None
        else:
            with pytest.raises(ImportError):
                make_axial_sequence_block('official_mamba2', channels=8, d_state=8, d_conv=4)

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
            # Arrival-time prior is defined only when measured per-trace AGL
            # metadata is present. Keep this integration test physically valid
            # rather than weakening the production contract.
            'altitude': torch.full((2, 32), 8.0),
            'altitude_valid': torch.ones(2, 32),
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


def test_component_separation_requires_persistent_discriminator():
    from scripts.losses_gprmambasep import compute_gprmambasep_loss
    m = build_gprmambasep({'base_ch': 8, 'mamba_state_dim': 16})
    x = torch.randn(1, 1, 32, 16)
    out = m(x)
    batch = {
        'x': x,
        'y': torch.zeros(1, 1, 32, 16),
        'y_core': torch.zeros(1, 1, 32, 16),
        'presence': torch.zeros(1, 1, 16),
        'presence_valid': torch.ones(1, 1, 16),
        'weight': torch.ones(1, 16),
        'valid_pix': torch.ones(1, 1, 32, 16),
        'valid_denom': torch.tensor(32 * 16.0),
    }
    cfg = {'loss': {'component_separation_weight': 0.1, 'self_consistency_weight': 0.0}}
    with pytest.raises(RuntimeError):
        compute_gprmambasep_loss(out, batch, cfg, m, discriminator=None)


def test_component_coverage_default_gate_when_supervision_enabled():
    from scripts.train_raw_only import _min_component_target_coverage
    assert _min_component_target_coverage({'loss': {'sim_supervised_weight': 0.5}}) > 0
    assert _min_component_target_coverage({'loss': {'sim_supervised_weight': 0.0}}) == 0


def test_sim_supervised_component_loss_honors_valid_flags():
    from scripts.losses_gprmambasep import sim_supervised_component_loss
    pred = torch.ones(2, 1, 8, 4)
    target = torch.zeros_like(pred)
    valid = torch.tensor([1.0, 0.0])
    parts = sim_supervised_component_loss(pred, pred, pred, Y_air=target, Y_air_valid=valid)
    assert abs(float(parts['a_l1']) - 1.0) < 1e-6
    parts_zero = sim_supervised_component_loss(pred, pred, pred, Y_air=target, Y_air_valid=torch.zeros(2))
    assert abs(float(parts_zero['a_l1'])) < 1e-8


def test_dataset_returns_component_placeholders_for_mixed_collation(tmp_path):
    import csv
    import numpy as np
    from torch.utils.data import DataLoader
    from scripts.train_raw_only import DS, OPTIONAL_COMPONENT_ARRAY_ALIASES
    root = tmp_path / 'data'
    (root / 'windows').mkdir(parents=True)
    rows = []
    for i, with_comp in enumerate([False, True]):
        sid = f's{i}'
        rows.append({'sample_id': sid, 'line': 'sim', 'start': '0', 'end': '7', 'present': '16', 'weak': '0', 'no_pick': '0'})
        arr = {
            'x_raw': np.random.randn(16, 8).astype('float32'),
            'y_mask': np.zeros((16, 8), dtype='float32'),
            'status_code': np.ones((8,), dtype='int64'),
            'label_weight': np.ones((8,), dtype='float32'),
        }
        if with_comp:
            arr['Y_air'] = np.random.randn(16, 8).astype('float32')
        np.savez(root / 'windows' / f'{sid}.npz', **arr)
    with open(root / 'window_index.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    cfg = {'data_root': str(root), 'train_lines': ['sim'], 'val_lines': [], 'test_lines': [], 'height_resize': 16, 'width_resize': 8, 'batch_size': 2}
    batch = next(iter(DataLoader(DS('train', cfg), batch_size=2, shuffle=False)))
    for key in OPTIONAL_COMPONENT_ARRAY_ALIASES:
        assert key in batch and f'{key}_valid' in batch
        assert batch[key].shape == (2, 1, 16, 8)
    assert float(batch['Y_air_valid'].sum()) == 1.0
