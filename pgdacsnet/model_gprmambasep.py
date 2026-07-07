"""
pgdacsnet/model_gprmambasep.py

GprMambaSep: Multi-decoder architecture for separable A/S/G clutter suppression.

Architecture overview (current implementation direction):

1. Shared encoder (4 stages, 3 MaxPool):
   ConvNeXtStage(stem→c) → MaxPool → ConvNeXtStage(c→2c) + Mamba2DBlock(2c)
   → MaxPool → ConvNeXtStage(2c→4c) + Mamba2DBlock(4c)
   → MaxPool → ConvNeXtStage(4c→8c) + Mamba2DBlock(8c)

2. Decomposition bottleneck:
   DilatedBottleneck(8c) → Mamba2DBlock(8c) → DilatedBottleneck(8c)
   → 1x1 split conv (8c→3×4c) → 3× ConvNeXtBlock(4c) refinement blocks

3. Three decoders (A / S / G):
   each returns both a 1-channel component reconstruction and the last c-channel
   feature map before projection. The G branch feature map feeds the standard PGDA
   task heads.

4. Task heads on G branch features:
   - Mask: 1×1 conv (c→1)
   - Center: CenterRefineHead(c)
   - Presence: per-trace logits (B, 1, W) via height pooling + 1×1 conv

Signal model (locked):
    A = air-coupled direct wave
    S = surface reflection
    G = subsurface geological signal (target)

    Y_full = A + S + G
    A_hat, S_hat, G_hat = GprMambaSep(x)

    mask_logits   ~ G_hat activation (binary segmentation of target signal)
    center_logits ~ G_hat curve center refinement
    presence_logits ~ per-trace target presence indicator
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from pgdacsnet.model_raw_unet import (
    ConvNeXtStage,
    ConvNeXtBlock,
    DilatedBottleneck,
    CenterRefineHead,
)
from pgdacsnet.model_mamba import Mamba2DBlock
from pgdacsnet.model_interfaces import GprMambaSepOutput


class _ComponentDecoder(nn.Module):
    """Lightweight decoder returning both component image and task features."""

    def __init__(self, base_ch: int, dropout: float = 0.0):
        super().__init__()
        c = int(base_ch)
        dp = float(dropout)
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.stage1 = ConvNeXtStage(c * 2, c * 2, depth=2, dropout=dp)
        self.up2 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.stage2 = ConvNeXtStage(c, c, depth=2, dropout=dp)
        self.proj = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.up1(x)
        x = self.stage1(x)
        x = self.up2(x)
        feat = self.stage2(x)
        out = self.proj(feat)
        return out, feat


class GprMambaSep(nn.Module):
    """Separable A/S/G clutter suppression network with shared encoder.

    Uses a shared ConvNeXt-Mamba2D encoder to extract multi-scale features,
    a decomposition bottleneck that splits the latent into three 4c-channel
    streams (A, S, G), and three decoders that reconstruct each component.

    Task heads on the G branch provide:
    - mask_logits: binary target segmentation (1-ch map)
    - center_logits: curve center refinement (1-ch map)
    - presence_logits: per-trace target presence (B, 1, W)
    """

    def __init__(
        self,
        base_ch: int = 16,
        mamba_state_dim: int = 32,
        mamba_d_conv: int = 4,
        decoder_dropout: float = 0.0,
        input_channels: int = 1,
    ):
        super().__init__()
        c = int(base_ch)
        d_state = int(mamba_state_dim)
        d_conv = int(mamba_d_conv)
        dp = float(decoder_dropout)
        self.input_channels = int(input_channels)
        stem_channels = self.input_channels + 1  # raw -> raw + |raw|, plus auxiliaries

        # ---- Shared encoder ----
        self.e1 = ConvNeXtStage(stem_channels, c, depth=2, dropout=dp)
        self.p = nn.MaxPool2d(2)

        self.e2 = ConvNeXtStage(c, c * 2, depth=2, dropout=dp)
        self.mamba_2c = Mamba2DBlock(c * 2, d_state=d_state, d_conv=d_conv) if d_state > 0 else nn.Identity()

        self.e3 = ConvNeXtStage(c * 2, c * 4, depth=2, dropout=dp)
        self.mamba_4c = Mamba2DBlock(c * 4, d_state=d_state, d_conv=d_conv) if d_state > 0 else nn.Identity()

        self.e4 = ConvNeXtStage(c * 4, c * 8, depth=2, dropout=dp)
        self.mamba_8c = Mamba2DBlock(c * 8, d_state=d_state, d_conv=d_conv) if d_state > 0 else nn.Identity()

        # ---- Decomposition bottleneck ----
        self.bottleneck = nn.Sequential(
            DilatedBottleneck(c * 8),
            Mamba2DBlock(c * 8, d_state=d_state, d_conv=d_conv) if d_state > 0 else nn.Identity(),
            DilatedBottleneck(c * 8),
        )

        self.split_conv = nn.Conv2d(c * 8, c * 4 * 3, 1)
        self.a_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])
        self.s_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])
        self.g_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])

        # ---- Component decoders ----
        self.decoder_a = _ComponentDecoder(c, dropout=dp)
        self.decoder_s = _ComponentDecoder(c, dropout=dp)
        self.decoder_g = _ComponentDecoder(c, dropout=dp)

        # ---- Task heads on G-decoder features ----
        self.mask_head = nn.Conv2d(c, 1, 1)
        self.center_head = CenterRefineHead(c, dropout=dp)
        self.pres_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(c, 1, 1),
        )

    def _build_stem_input(self, x: torch.Tensor) -> torch.Tensor:
        """Build raw-domain stem input safely when terrain channels are appended.

        The first channel is always treated as raw radar amplitude. Any remaining
        channels are appended unchanged as auxiliary terrain/metadata channels.
        """
        raw = x[:, :1]
        aux = x[:, 1:] if x.shape[1] > 1 else None
        parts = [raw, torch.abs(raw)]
        if aux is not None and aux.shape[1] > 0:
            parts.append(aux)
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor) -> GprMambaSepOutput:
        """Forward pass.

        Args:
            x: Input B-scan of shape (B, C, H, W). Channel 0 is raw radar;
               optional remaining channels are terrain/metadata features.

        Returns:
            GprMambaSepOutput with standard PGDA heads and A/S/G components.
        """
        _, _, h, w = x.shape

        # ---- Encoder (shared) ----
        e1_in = self._build_stem_input(x)
        e1 = self.e1(e1_in)
        p1 = self.p(e1)

        e2_feat = self.mamba_2c(self.e2(p1))
        p2 = self.p(e2_feat)

        e3_feat = self.mamba_4c(self.e3(p2))
        p3 = self.p(e3_feat)

        e4_feat = self.mamba_8c(self.e4(p3))

        # ---- Bottleneck ----
        b = self.bottleneck(e4_feat)

        # ---- Split and refine ----
        split = self.split_conv(b)
        a_feat, s_feat, g_feat = split.chunk(3, dim=1)

        a_feat = self.a_refine(a_feat)
        s_feat = self.s_refine(s_feat)
        g_feat = self.g_refine(g_feat)

        # ---- Component decoders ----
        a_hat_raw, _ = self.decoder_a(a_feat)
        s_hat_raw, _ = self.decoder_s(s_feat)
        g_hat_raw, g_task_feat = self.decoder_g(g_feat)

        a_hat = F.interpolate(a_hat_raw, size=(h, w), mode='bilinear', align_corners=False)
        s_hat = F.interpolate(s_hat_raw, size=(h, w), mode='bilinear', align_corners=False)
        g_hat = F.interpolate(g_hat_raw, size=(h, w), mode='bilinear', align_corners=False)

        # ---- Standard PGDA heads on G task features ----
        mask_raw = self.mask_head(g_task_feat)
        mask_logits = F.interpolate(mask_raw, size=(h, w), mode='bilinear', align_corners=False)

        center_raw = self.center_head(g_task_feat)
        center_logits = F.interpolate(center_raw, size=(h, w), mode='bilinear', align_corners=False)

        presence_logits = self.pres_head(g_task_feat).squeeze(2)
        presence_logits = F.interpolate(presence_logits, size=w, mode='linear', align_corners=False)

        return GprMambaSepOutput(
            mask_logits=mask_logits,
            presence_logits=presence_logits,
            center_logits=center_logits,
            A_hat=a_hat,
            S_hat=s_hat,
            G_hat=g_hat,
        )


def build_gprmambasep(cfg: dict) -> GprMambaSep:
    """Construct a GprMambaSep from a config dict."""
    return GprMambaSep(
        base_ch=int(cfg.get('base_ch', 16)),
        mamba_state_dim=int(cfg.get('mamba_state_dim', 32)),
        mamba_d_conv=int(cfg.get('mamba_d_conv', 4)),
        decoder_dropout=float(cfg.get('decoder_dropout', 0.0)),
        input_channels=int(cfg.get('input_channels', 1)),
    )
