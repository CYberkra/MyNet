#!/usr/bin/env python3
"""
PGDA-CSNet — Unsupervised Domain Adaptation (UDA) Training

Usage:
    python scripts/train_uda.py configs/uda_config.json

Architecture:
  - Shared encoder (PGDA-CSNet backbone) processes both source (sim) and target (real)
  - Domain discriminator with Gradient Reversal Layer makes features domain-invariant
  - Supervised loss on source domain + adversarial domain loss on both

Config additions:
  "uda": {
    "enabled": true,
    "domain_loss_weight": 0.1,
    "disc_hidden": [64, 32],
    "target_data_root": "data_corrected_v1_4_terrain_direction",
    "target_lines": ["Line9"],
    "target_pretrain_epochs": 5
  }
"""
import sys, json, os, time, csv, random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_raw_only import DS, run_epoch, resolve_data_root, add_terrain_channels
from pgdacsnet.model_raw_unet import build_model, compress_raw

# ── Gradient Reversal Layer ──
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradReverse.apply(x, self.alpha)

# ── Domain Discriminator ──
class DomainDisc(nn.Module):
    """Predicts domain (0=source/sim, 1=target/real) from bottleneck features."""
    def __init__(self, in_channels=80, hidden=[64, 32]):
        super().__init__()
        layers = []
        c = in_channels
        for h in hidden:
            layers.extend([
                nn.Conv2d(c, h, 3, padding=1),
                nn.BatchNorm2d(h),
                nn.LeakyReLU(0.2),
            ])
            c = h
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(c, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x).squeeze(1)


def extract_features(model, x):
    """Extract bottleneck feature maps from model with gradients retained."""
    raw = x[:, :1]
    meta = x[:, 1:] if x.shape[1] > 1 else None
    e1 = model.e1(torch.cat((raw, torch.abs(raw)), dim=1))
    if hasattr(model, 'film1') and meta is not None:
        e1 = model.film1(e1, meta)
    e2 = model.e2(model.p(e1))
    if hasattr(model, 'film2') and meta is not None:
        e2 = model.film2(e2, meta)
    e3 = model.e3(model.p(e2))
    if hasattr(model, 'film3') and meta is not None:
        e3 = model.film3(e3, meta)
    bottleneck = model.bottleneck(model.p(e3))
    if hasattr(model, 'filmb') and meta is not None:
        bottleneck = model.filmb(bottleneck, meta)
    return bottleneck


def train_uda(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uda_cfg = cfg.get('uda', {})
    domain_weight = float(uda_cfg.get('domain_loss_weight', 0.1))
    pretrain_epochs = int(uda_cfg.get('target_pretrain_epochs', 5))

    # ── Load data ──
    source_loader = torch.utils.data.DataLoader(
        DS('train', cfg), batch_size=int(cfg.get('batch_size', 2)),
        shuffle=True, num_workers=int(cfg.get('num_workers', 0)),
        pin_memory=True, drop_last=True
    )

    # Target domain: load real data
    target_cfg = dict(cfg)
    target_cfg['data_root'] = uda_cfg.get('target_data_root', resolve_data_root(cfg))
    target_cfg['test_lines'] = uda_cfg.get('target_lines', ['Line9'])
    target_loader = torch.utils.data.DataLoader(
        DS('test', target_cfg), batch_size=int(cfg.get('batch_size', 2)),
        shuffle=True, num_workers=int(cfg.get('num_workers', 0)),
        pin_memory=True, drop_last=True
    )

    # ── Build model ──
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('lr', 5e-4)))

    # ── Domain discriminator ──
    base_ch = int(cfg.get('base_ch', 20))
    bottleneck_ch = base_ch * 4  # v1_8b has 4x channels at bottleneck
    disc = DomainDisc(in_channels=bottleneck_ch, hidden=uda_cfg.get('disc_hidden', [64, 32])).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=float(cfg.get('lr', 5e-4)) * 0.5)
    grl = GRL(alpha=1.0)
    domain_loss_fn = nn.BCEWithLogitsLoss()

    # ── Warm-start with supervised pretraining ──
    print(f'Phase 1: Supervised pretrain on sim ({pretrain_epochs} epochs)')
    for epoch in range(pretrain_epochs):
        metrics = run_epoch(model, source_loader, device, cfg, optimizer)
        loss = metrics.get('loss', 0)
        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{pretrain_epochs}: loss={loss:.4f}')

    # ── UDA training ──
    print(f'\nPhase 2: Domain-adversarial training')
    epochs = int(cfg.get('epochs', 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        disc.train()
        total_sup, total_dom, count = 0, 0, 0
        t_iter = iter(target_loader)

        for s_batch in source_loader:
            # Source forward (DS already applies compress_raw + add_terrain_channels)
            x_s = s_batch['x'].to(device)
            y_s = s_batch['y'].to(device)

            # Target forward
            try:
                t_batch = next(t_iter)
            except StopIteration:
                t_iter = iter(target_loader)
                t_batch = next(t_iter)
            x_t = t_batch['x'].to(device)

            # Supervised loss (source only)
            out = model(x_s)
            mask_logits = out[0]
            y_s = s_batch['y'].to(device)
            sup_loss = F.binary_cross_entropy_with_logits(mask_logits, y_s)

            # Domain adversarial loss
            feat_s = extract_features(model, x_s)
            feat_t = extract_features(model, x_t)

            features = torch.cat([feat_s, feat_t], dim=0)
            domain_labels = torch.cat([
                torch.zeros(x_s.size(0), device=device),
                torch.ones(x_t.size(0), device=device)
            ])

            features_adv = grl(features)
            domain_logits = disc(features_adv)
            dom_loss = domain_loss_fn(domain_logits, domain_labels)

            total_loss = sup_loss + domain_weight * dom_loss

            optimizer.zero_grad()
            disc_opt.zero_grad()
            total_loss.backward()
            optimizer.step()
            disc_opt.step()

            total_sup += sup_loss.item()
            total_dom += dom_loss.item()
            count += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{epochs}: sup={total_sup/max(count,1):.4f} dom={total_dom/max(count,1):.4f}')

    # ── Save ──
    out_dir = ROOT / cfg.get('run_dir', 'outputs/uda_default')
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'disc': disc.state_dict(),
        'cfg': cfg,
        'epoch': epochs,
    }, out_dir / 'checkpoint_final.pt')
    print(f'\nSaved: {out_dir / "checkpoint_final.pt"}')


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    train_uda(cfg)
