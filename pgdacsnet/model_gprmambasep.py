"""
pgdacsnet/model_gprmambasep.py

GprMambaSep: Multi-decoder architecture for separable A/S/G clutter suppression.

Architecture overview (from research plan TASK 4):

1. Shared encoder (4 stages, 3 MaxPool):
   ConvNeXtStage(2→16ch) → MaxPool → ConvNeXtStage(16→32ch) + Mamba2DBlock(32ch)
   → MaxPool → ConvNeXtStage(32→64ch) + Mamba2DBlock(64ch)
   → MaxPool → ConvNeXtStage(64→128ch) + Mamba2DBlock(128ch)

   The Mamba2DBlock at each stage provides a skip-connection feature that can
   be injected into the decoders (currently unused; skips are a future extension).

2. Decomposition bottleneck:
   DilatedBottleneck(128ch) → Mamba2DBlock(128ch) → DilatedBottleneck(128ch)
   → 1x1 split conv (128→3×64) → 3× ConvNeXtBlock(64ch) refinement blocks

3. Three decoders (A / S / G), same topology, unique weights:
   TransposedConv(64→32ch) → ConvNeXtStage → TransposedConv(32→16ch)
   → ConvNeXtStage → 1×1(16→1ch) → bilinear upsample to input resolution

4. Task heads on G_decoder 16ch features:
   - Mask: 1×1 conv (16→1ch)
   - Center: CenterRefineHead(16ch)
   - Presence: global pool → FC(16→1)

Signal model (locked):
    A = air-coupled direct wave
    S = surface reflection
    G = subsurface geological signal (target)

    Y_full = A + S + G
    A_hat, S_hat, G_hat = GprMambaSep(x)

    mask_logits  ~ G_hat activation (binary segmentation of target signal)
    center_logits ~ G_hat curve center refinement
    presence_logits ~ per-sample target presence indicator
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

from pgdacsnet.model_raw_unet import (
    ConvNeXtStage,
    ConvNeXtBlock,
    DilatedBottleneck,
    CenterRefineHead,
    SEBlock,
    SpectralGatingModule,
)
from pgdacsnet.model_mamba import Mamba2DBlock
from pgdacsnet.model_interfaces import GprMambaSepOutput


class GprMambaSep(nn.Module):
    """Separable A/S/G clutter suppression network with shared encoder.

    Uses a shared ConvNeXt-Mamba2D encoder to extract multi-scale features,
    a decomposition bottleneck that splits the latent into three 64-channel
    streams (A, S, G), and three independent decoders that reconstruct each
    component at full resolution.

    Task heads on the G-decoder provide:
    - mask_logits:  binary target segmentation (1-ch map)
    - center_logits: curve center refinement (1-ch map)
    - presence_logits: per-sample target presence (scalar)
    """

    def __init__(
        self,
        base_ch: int = 16,
        mamba_state_dim: int = 32,
        mamba_d_conv: int = 4,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()
        c = int(base_ch)
        d_state = int(mamba_state_dim)
        d_conv = int(mamba_d_conv)
        dp = float(decoder_dropout)

        # ---- Shared encoder ----
        # Stage 1: 2 → c  (input = raw + |raw|, no Mamba2D here)
        self.e1 = ConvNeXtStage(2, c, depth=2, dropout=dp)
        self.p = nn.MaxPool2d(2)

        # Stage 2: c → 2c, with Mamba2DBlock for skip
        self.e2 = ConvNeXtStage(c, c * 2, depth=2, dropout=dp)
        self.mamba_2c = Mamba2DBlock(c * 2, d_state=d_state, d_conv=d_conv)

        # Stage 3: 2c → 4c, with Mamba2DBlock for skip
        self.e3 = ConvNeXtStage(c * 2, c * 4, depth=2, dropout=dp)
        self.mamba_4c = Mamba2DBlock(c * 4, d_state=d_state, d_conv=d_conv)

        # Stage 4: 4c → 8c, with Mamba2DBlock for skip
        self.e4 = ConvNeXtStage(c * 4, c * 8, depth=2, dropout=dp)
        self.mamba_8c = Mamba2DBlock(c * 8, d_state=d_state, d_conv=d_conv)

        # ---- Decomposition bottleneck ----
        self.bottleneck = nn.Sequential(
            DilatedBottleneck(c * 8),
            Mamba2DBlock(c * 8, d_state=d_state, d_conv=d_conv),
            DilatedBottleneck(c * 8),
        )

        # Split conv: 8c → 3 × 4c (192 = 3 * 64 when c=16)
        self.split_conv = nn.Conv2d(c * 8, c * 4 * 3, 1)

        # Refinement per branch: 3 × ConvNeXtBlock(4c)
        self.a_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])
        self.s_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])
        self.g_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])

        # ---- Three decoders (same topology, unique weights) ----
        def _make_decoder():
            """Build one A/S/G decoder with 2× upsampling + final scale_factor=2."""
            return nn.Sequential(
                # Step 1: 4c → 2c, 2× up
                nn.ConvTranspose2d(c * 4, c * 2, 2, 2),
                ConvNeXtStage(c * 2, c * 2, depth=2, dropout=dp),
                # Step 2: 2c → c, 2× up
                nn.ConvTranspose2d(c * 2, c, 2, 2),
                ConvNeXtStage(c, c, depth=2, dropout=dp),
                # Step 3: project to 1 channel and upsample to input resolution
                nn.Conv2d(c, 1, 1),
            )

        self.decoder_a = _make_decoder()
        self.decoder_s = _make_decoder()
        self.decoder_g = _make_decoder()

        # ---- Task heads on G-decoder intermediate features ----
        # These operate on the 16ch features at H/2, W/2 resolution
        # (the ConvNeXtStage right before the final 1×1 conv in decoder_g)
        self.mask_head = nn.Conv2d(c, 1, 1)                  # 16→1
        self.center_head = CenterRefineHead(c, dropout=dp)   # CenterRefineHead(16)
        self.pres_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # (B, 16, 1, 1)
            nn.Flatten(),                                     # (B, 16)
            nn.Linear(c, 1),                                  # (B, 1)
        )

        # ---- Expose the G-decoder's inner 16ch ConvNeXtStage for task heads ----
        # We hack around nn.Sequential by storing references (see forward)
        self._g_decoder_up2_proj = self.decoder_g[3]  # ConvNeXtStage(c, c) at position 3

    def forward(self, x: torch.Tensor) -> GprMambaSepOutput:
        """Forward pass.

        Args:
            x: Input B-scan of shape (B, 1, H, W).

        Returns:
            GprMambaSepOutput with all six fields.
        """
        B, C_in, H, W = x.shape

        # ---- Encoder (shared) ----
        # Input: raw + |raw|
        e1_in = torch.cat([x, torch.abs(x)], dim=1)  # (B, 2, H, W)
        e1 = self.e1(e1_in)                          # (B, c, H, W)
        p1 = self.p(e1)                              # (B, c, H/2, W/2)

        e2_feat = self.e2(p1)                        # (B, 2c, H/2, W/2)
        e2_skip = self.mamba_2c(e2_feat)             # (B, 2c, H/2, W/2) — reserved for future skip
        p2 = self.p(e2_feat)                         # (B, 2c, H/4, W/4)

        e3_feat = self.e3(p2)                        # (B, 4c, H/4, W/4)
        e3_skip = self.mamba_4c(e3_feat)             # (B, 4c, H/4, W/4)
        p3 = self.p(e3_feat)                         # (B, 4c, H/8, W/8)

        e4_feat = self.e4(p3)                        # (B, 8c, H/8, W/8)
        e4_skip = self.mamba_8c(e4_feat)             # (B, 8c, H/8, W/8)

        # ---- Bottleneck ----
        b = self.bottleneck(e4_feat)                 # (B, 8c, H/8, W/8)

        # ---- Split and refine ----
        split = self.split_conv(b)                   # (B, 12c, H/8, W/8)
        a_feat, s_feat, g_feat = split.chunk(3, dim=1)  # each (B, 4c, H/8, W/8)

        a_feat = self.a_refine(a_feat)               # (B, 4c, H/8, W/8)
        s_feat = self.s_refine(s_feat)               # (B, 4c, H/8, W/8)
        g_feat = self.g_refine(g_feat)               # (B, 4c, H/8, W/8)

        # ---- Decoders (2× upsample per TConv = 4× total) ----
        # TConv 64→32 stride 2: (B, 4c, H/8, W/8) → (B, 2c, H/4, W/4)
        # ConvNeXtStage(2c→2c)
        # TConv 32→16 stride 2: (B, 2c, H/4, W/4) → (B, c, H/2, W/2)
        # ConvNeXtStage(c→c)
        # 1×1 conv c→1: (B, c, H/2, W/2) → (B, 1, H/2, W/2)
        # Final bilinear upsample: (B, 1, H/2, W/2) → (B, 1, H, W)
        A_hat_raw = self.decoder_a(a_feat)           # (B, 1, H/2, W/2)
        S_hat_raw = self.decoder_s(s_feat)           # (B, 1, H/2, W/2)
        G_decoder_out = self.decoder_g(g_feat)         # (B, 1, H/2, W/2)

        # Upsample to input resolution
        A_hat = F.interpolate(A_hat_raw, size=(H, W), mode='bilinear', align_corners=False)
        S_hat = F.interpolate(S_hat_raw, size=(H, W), mode='bilinear', align_corners=False)
        G_hat = F.interpolate(G_decoder_out, size=(H, W), mode='bilinear', align_corners=False)

        # ---- Task heads on G-decoder 16ch features ----
        # Extract the 16ch features from G_decoder's ConvNeXtStage
        # We need to replicate the decoder_g forward partially to get the
        # intermediate 16ch features at H/2, W/2.
        g_up1 = self.decoder_g[0](g_feat)      # TConv 64→32 (B, 2c, H/4, W/4)
        g_up1 = self.decoder_g[1](g_up1)       # ConvNeXtStage(2c, 2c)
        g_up2 = self.decoder_g[2](g_up1)       # TConv 32→16 (B, c, H/2, W/2)
        g_16ch = self.decoder_g[3](g_up2)      # ConvNeXtStage(c, c) — (B, c, H/2, W/2)

        # Mask head
        mask_raw = self.mask_head(g_16ch)      # (B, 1, H/2, W/2)
        mask_logits = F.interpolate(mask_raw, size=(H, W), mode='bilinear', align_corners=False)

        # Center head (already produces 1ch maps via its internal 1×1)
        center_raw = self.center_head(g_16ch)  # (B, 1, H/2, W/2)
        center_logits = F.interpolate(center_raw, size=(H, W), mode='bilinear', align_corners=False)

        # Presence head (per-sample scalar)
        presence_logits = self.pres_head(g_16ch)  # (B, 1)

        return GprMambaSepOutput(
            mask_logits=mask_logits,
            presence_logits=presence_logits,
            center_logits=center_logits,
            A_hat=A_hat,
            S_hat=S_hat,
            G_hat=G_hat,
        )


def build_gprmambasep(cfg: dict) -> GprMambaSep:
    """Construct a GprMambaSep from a config dict.

    Args:
        cfg: Dictionary with keys:
            - base_ch (int): Base channel count (default: 16).
            - mamba_state_dim (int): Mamba state dimension (default: 32).
            - mamba_d_conv (int): Mamba depthwise conv kernel (default: 4).
            - decoder_dropout (float): Dropout in decoder ConvNeXtStages (default: 0.0).

    Returns:
        Initialised GprMambaSep model.
    """
    return GprMambaSep(
        base_ch=int(cfg.get('base_ch', 16)),
        mamba_state_dim=int(cfg.get('mamba_state_dim', 32)),
        mamba_d_conv=int(cfg.get('mamba_d_conv', 4)),
        decoder_dropout=float(cfg.get('decoder_dropout', 0.0)),
    )
