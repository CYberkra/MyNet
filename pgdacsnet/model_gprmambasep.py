"""
pgdacsnet/model_gprmambasep.py

GprMambaSep: A/S/G explicit-decomposition network for UAV-GPR B-scans.

This v2 implementation keeps the first-round P0 fixes and adds second-round
architectural safeguards:

1. Shared ConvNeXt + axial sequence-mixer encoder.
2. Soft gated A/S/G latent allocation at the bottleneck, so the three component
   branches cannot be interpreted as three unrelated decoders over the same
   feature tensor.
3. Three branch-specific skip decoders with full 1/8 -> 1/4 -> 1/2 -> full
   resolution recovery.  This removes the previous two-upsample + final
   interpolation shortcut that was weak for trace-level interface picking.
4. Search-window-aware presence pooling retained through config fractions.

Terminology note:
    The default sequence mixer is AxialSSMLiteBlock, a project-local
    Mamba-like / SSM-lite axial mixer.  It is not the official Mamba-2 block.
    Set ssm_impl='official_mamba2' only in environments with mamba-ssm installed.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from pgdacsnet.model_raw_unet import (
    ConvNeXtStage,
    ConvNeXtBlock,
    DilatedBottleneck,
    CenterRefineHead,
)
from pgdacsnet.model_mamba import make_axial_sequence_block
from pgdacsnet.model_interfaces import GprMambaSepOutput


def _safe_logit(p: float) -> float:
    p = max(1e-4, min(1.0 - 1e-4, float(p)))
    return math.log(p / (1.0 - p))


class _SkipFuse(nn.Module):
    """Project and gate a skip connection before adding it to a decoder stream."""

    def __init__(self, skip_ch: int, out_ch: int, init_gate: float):
        super().__init__()
        self.proj = nn.Conv2d(int(skip_ch), int(out_ch), 1)
        self.gate_logit = nn.Parameter(torch.tensor(_safe_logit(init_gate), dtype=torch.float32))

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        if skip is None:
            return x
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + torch.sigmoid(self.gate_logit) * self.proj(skip)


class _ComponentDecoder(nn.Module):
    """Three-stage full-resolution branch decoder with branch-specific skips.

    Latent resolution is 1/8 of the input.  The decoder recovers 1/4, 1/2 and
    full resolution and returns both the reconstructed component and full-scale
    features for the G-task heads.
    """

    def __init__(
        self,
        base_ch: int,
        dropout: float = 0.0,
        skip_gate_priors: Sequence[float] = (0.75, 0.75, 0.75),
    ):
        super().__init__()
        c = int(base_ch)
        dp = float(dropout)
        g3, g2, g1 = [float(v) for v in skip_gate_priors]

        self.up1 = nn.ConvTranspose2d(c * 4, c * 4, 2, 2)
        self.fuse3 = _SkipFuse(c * 4, c * 4, g3)
        self.stage1 = ConvNeXtStage(c * 4, c * 4, depth=2, dropout=dp)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.fuse2 = _SkipFuse(c * 2, c * 2, g2)
        self.stage2 = ConvNeXtStage(c * 2, c * 2, depth=2, dropout=dp)

        self.up3 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.fuse1 = _SkipFuse(c, c, g1)
        self.stage3 = ConvNeXtStage(c, c, depth=2, dropout=dp)

        self.proj = nn.Conv2d(c, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        skip1: torch.Tensor | None,
        skip2: torch.Tensor | None,
        skip3: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.up1(x)
        x = self.stage1(self.fuse3(x, skip3))
        x = self.up2(x)
        x = self.stage2(self.fuse2(x, skip2))
        x = self.up3(x)
        feat = self.stage3(self.fuse1(x, skip1))
        out = self.proj(feat)
        return out, feat


class GprMambaSep(nn.Module):
    """Separable A/S/G clutter suppression network with gated decomposition."""

    def __init__(
        self,
        base_ch: int = 16,
        mamba_state_dim: int = 32,
        mamba_d_conv: int = 4,
        decoder_dropout: float = 0.0,
        input_channels: int = 1,
        presence_pool_lo_frac: float | None = None,
        presence_pool_hi_frac: float | None = None,
        ssm_impl: str = "ssm_lite",
        gated_decomposition: bool = True,
    ):
        super().__init__()
        c = int(base_ch)
        d_state = int(mamba_state_dim)
        d_conv = int(mamba_d_conv)
        dp = float(decoder_dropout)
        self.input_channels = int(input_channels)
        self.presence_pool_lo_frac = presence_pool_lo_frac
        self.presence_pool_hi_frac = presence_pool_hi_frac
        self.gated_decomposition = bool(gated_decomposition)
        self.ssm_impl = str(ssm_impl)
        stem_channels = self.input_channels + 1  # raw -> raw + |raw|, plus auxiliaries

        def seq_block(ch: int) -> nn.Module:
            if d_state <= 0:
                return nn.Identity()
            return make_axial_sequence_block(
                impl=self.ssm_impl,
                channels=ch,
                d_state=d_state,
                d_conv=d_conv,
            )

        # ---- Shared encoder ----
        self.e1 = ConvNeXtStage(stem_channels, c, depth=2, dropout=dp)
        self.p = nn.MaxPool2d(2)

        self.e2 = ConvNeXtStage(c, c * 2, depth=2, dropout=dp)
        self.mixer_2c = seq_block(c * 2)

        self.e3 = ConvNeXtStage(c * 2, c * 4, depth=2, dropout=dp)
        self.mixer_4c = seq_block(c * 4)

        self.e4 = ConvNeXtStage(c * 4, c * 8, depth=2, dropout=dp)
        self.mixer_8c = seq_block(c * 8)

        # ---- Decomposition bottleneck ----
        self.bottleneck = nn.Sequential(
            DilatedBottleneck(c * 8),
            seq_block(c * 8),
            DilatedBottleneck(c * 8),
        )

        self.split_conv = nn.Conv2d(c * 8, c * 4 * 3, 1)
        self.component_gate = nn.Conv2d(c * 8, 3, 1)
        nn.init.zeros_(self.component_gate.weight)
        nn.init.zeros_(self.component_gate.bias)

        self.a_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])
        self.s_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])
        self.g_refine = nn.Sequential(*[ConvNeXtBlock(c * 4, dropout=dp) for _ in range(3)])

        # Branch-specific skip priors: A emphasises early/high-resolution events,
        # S keeps balanced surface detail, G is conservative on the shallowest skip.
        self.decoder_a = _ComponentDecoder(c, dropout=dp, skip_gate_priors=(0.60, 0.85, 1.00))
        self.decoder_s = _ComponentDecoder(c, dropout=dp, skip_gate_priors=(0.75, 1.00, 0.80))
        self.decoder_g = _ComponentDecoder(c, dropout=dp, skip_gate_priors=(1.00, 0.80, 0.35))

        # ---- Task heads on full-resolution G-decoder features ----
        self.mask_head = nn.Conv2d(c, 1, 1)
        self.center_head = CenterRefineHead(c, dropout=dp)
        self.pres_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(c, 1, 1),
        )

    def _build_stem_input(self, x: torch.Tensor) -> torch.Tensor:
        """Build raw-domain stem input with raw, |raw| and optional auxiliaries."""
        raw = x[:, :1]
        aux = x[:, 1:] if x.shape[1] > 1 else None
        parts = [raw, torch.abs(raw)]
        if aux is not None and aux.shape[1] > 0:
            parts.append(aux)
        return torch.cat(parts, dim=1)

    def _presence_logits_from_feat(self, feat: torch.Tensor, out_w: int) -> torch.Tensor:
        """Compute per-trace presence logits with an optional height/time gate."""
        if self.presence_pool_lo_frac is not None and self.presence_pool_hi_frac is not None:
            h = feat.shape[2]
            lo = max(0, min(h - 1, int(round(float(self.presence_pool_lo_frac) * h))))
            hi = max(lo + 1, min(h, int(round(float(self.presence_pool_hi_frac) * h))))
            feat = feat[:, :, lo:hi, :]
        presence_logits = self.pres_head(feat).squeeze(2)
        return F.interpolate(presence_logits, size=out_w, mode="linear", align_corners=False)

    def _split_with_gates(self, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = b.shape
        split = self.split_conv(b).view(B, 3, -1, H, W)
        gate = torch.softmax(self.component_gate(b), dim=1)
        if self.gated_decomposition:
            # Equal softmax gates produce scale=1, so legacy magnitude is preserved at init.
            split = split * (3.0 * gate.unsqueeze(2))
        a_feat, s_feat, g_feat = split[:, 0], split[:, 1], split[:, 2]
        return a_feat, s_feat, g_feat, gate

    def forward(self, x: torch.Tensor) -> GprMambaSepOutput:
        """Forward pass.

        Args:
            x: Input B-scan of shape (B, C, H, W). Channel 0 is raw radar;
               optional remaining channels are terrain/metadata features.
        """
        _, _, h, w = x.shape

        e1_in = self._build_stem_input(x)
        e1 = self.e1(e1_in)             # full resolution, c
        e2 = self.mixer_2c(self.e2(self.p(e1)))   # 1/2, 2c
        e3 = self.mixer_4c(self.e3(self.p(e2)))   # 1/4, 4c
        e4 = self.mixer_8c(self.e4(self.p(e3)))   # 1/8, 8c

        b = self.bottleneck(e4)
        a_feat, s_feat, g_feat, gates = self._split_with_gates(b)

        a_feat = self.a_refine(a_feat)
        s_feat = self.s_refine(s_feat)
        g_feat = self.g_refine(g_feat)

        a_hat_raw, _ = self.decoder_a(a_feat, e1, e2, e3)
        s_hat_raw, _ = self.decoder_s(s_feat, e1, e2, e3)
        g_hat_raw, g_task_feat = self.decoder_g(g_feat, e1, e2, e3)

        # Keep exact output size for odd input dimensions.
        a_hat = F.interpolate(a_hat_raw, size=(h, w), mode="bilinear", align_corners=False)
        s_hat = F.interpolate(s_hat_raw, size=(h, w), mode="bilinear", align_corners=False)
        g_hat = F.interpolate(g_hat_raw, size=(h, w), mode="bilinear", align_corners=False)

        mask_logits = F.interpolate(self.mask_head(g_task_feat), size=(h, w), mode="bilinear", align_corners=False)
        center_logits = F.interpolate(self.center_head(g_task_feat), size=(h, w), mode="bilinear", align_corners=False)
        presence_logits = self._presence_logits_from_feat(g_task_feat, w)
        full_gates = F.interpolate(gates, size=(h, w), mode="bilinear", align_corners=False)

        return GprMambaSepOutput(
            mask_logits=mask_logits,
            presence_logits=presence_logits,
            center_logits=center_logits,
            A_hat=a_hat,
            S_hat=s_hat,
            G_hat=g_hat,
            component_gates=full_gates,
        )


def build_gprmambasep(cfg: dict) -> GprMambaSep:
    """Construct a GprMambaSep from a config dict."""
    return GprMambaSep(
        base_ch=int(cfg.get("base_ch", 16)),
        mamba_state_dim=int(cfg.get("mamba_state_dim", 32)),
        mamba_d_conv=int(cfg.get("mamba_d_conv", cfg.get("ssm_kernel", 4))),
        decoder_dropout=float(cfg.get("decoder_dropout", cfg.get("model_dropout", 0.0))),
        input_channels=int(cfg.get("input_channels", 1)),
        presence_pool_lo_frac=cfg.get("presence_pool_lo_frac", None),
        presence_pool_hi_frac=cfg.get("presence_pool_hi_frac", None),
        ssm_impl=str(cfg.get("ssm_impl", cfg.get("mamba_impl", "ssm_lite"))),
        gated_decomposition=bool(cfg.get("gated_decomposition", True)),
    )
