"""AeroPath-SSD: acquisition-conditioned structured UAV-GPR path picking.

The model deliberately treats a basal interface as one continuous path instead
of an unrelated set of segmentation pixels.  It uses local phase-aware convolu-
tions, non-causal axial state-space mixing, and a differentiable soft path
inference layer.  ``official_mamba2`` is opt-in; tests may use ``ssm_lite`` but
that fallback must never be presented as a Mamba-2 experiment.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from pgdacsnet.model_interfaces import AeroPathOutput
from pgdacsnet.model_mamba import make_axial_sequence_block


class _PhaseStem(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        self.time = nn.Conv2d(channels, width, (9, 3), padding=(4, 1))
        self.trace = nn.Conv2d(channels, width, (3, 9), padding=(1, 4))
        self.mix = nn.Sequential(
            nn.GroupNorm(4, width * 2), nn.GELU(), nn.Conv2d(width * 2, width, 1),
            nn.GroupNorm(4, width), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([self.time(x), self.trace(x)], dim=1))


class _FiLM(nn.Module):
    def __init__(self, metadata_channels: int, channels: int):
        super().__init__()
        self.metadata_channels = int(metadata_channels)
        self.net = nn.Sequential(
            nn.Linear(max(self.metadata_channels, 1), max(16, channels // 2)), nn.GELU(),
            nn.Linear(max(16, channels // 2), channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None) -> torch.Tensor:
        if self.metadata_channels <= 0 or metadata is None:
            return x
        pooled = metadata.mean(dim=(2, 3))
        gamma, beta = self.net(pooled).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class _AeroBlock(nn.Module):
    def __init__(self, channels: int, metadata_channels: int, *, ssm_impl: str, d_state: int, d_conv: int):
        super().__init__()
        self.film = _FiLM(metadata_channels, channels)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.GroupNorm(4, channels), nn.GELU(), nn.Conv2d(channels, channels, 1),
        )
        self.ssm = make_axial_sequence_block(ssm_impl, channels, d_state=d_state, d_conv=d_conv)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None) -> torch.Tensor:
        z = self.film(x, metadata)
        mixed = self.local(z) + (self.ssm(z) - z)
        return x + self.gate(z) * mixed


def _air_reduce(raw: torch.Tensor, altitude_m: torch.Tensor | None, time_window_ns: float) -> torch.Tensor:
    """Create a differentiable reduced-time view without discarding raw time.

    A missing altitude leaves the view unchanged.  Heights are explicitly passed
    from the dataset, never inferred from a normalised auxiliary image channel.
    """
    if altitude_m is None:
        return raw
    if altitude_m.dim() == 1:
        altitude_m = altitude_m[None]
    if altitude_m.dim() == 3:
        altitude_m = altitude_m.squeeze(1)
    if altitude_m.dim() != 2 or altitude_m.shape[0] != raw.shape[0]:
        raise ValueError(f"altitude must have shape (B,W), got {tuple(altitude_m.shape)}")
    b, _, h, w = raw.shape
    altitude = F.interpolate(altitude_m[:, None], size=w, mode="linear", align_corners=False).squeeze(1)
    air_ns = 2.0 * altitude.clamp_min(0.0) / 299_792_458.0 * 1e9
    y = torch.linspace(-1.0, 1.0, h, device=raw.device, dtype=raw.dtype)[None, :, None]
    x = torch.linspace(-1.0, 1.0, w, device=raw.device, dtype=raw.dtype)[None, None, :]
    shift = 2.0 * air_ns[:, None, :] / max(float(time_window_ns), 1e-6)
    grid = torch.stack((x.expand(b, h, w), (y + shift).expand(b, h, w)), dim=-1)
    return F.grid_sample(raw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class SoftPathInference(nn.Module):
    """Differentiable first-order Viterbi relaxation with forward/backward marginals."""
    def __init__(self, max_step: int = 6, initial_slope_penalty: float = 1.0, temperature: float = 0.35):
        super().__init__()
        self.max_step = int(max_step)
        self.log_slope_penalty = nn.Parameter(torch.tensor(math.log(math.expm1(initial_slope_penalty))))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(temperature))))

    def _transition(self, score: torch.Tensor) -> torch.Tensor:
        # score: (B,H); outputs logsumexp over admissible predecessor heights.
        penalty = F.softplus(self.log_slope_penalty)
        parts = []
        for delta in range(-self.max_step, self.max_step + 1):
            shifted = F.pad(score, (max(delta, 0), max(-delta, 0)), value=-float("inf"))
            shifted = shifted[:, max(-delta, 0): shifted.shape[1] - max(delta, 0)]
            parts.append(shifted - penalty * abs(delta))
        return torch.logsumexp(torch.stack(parts, dim=-1), dim=-1)

    def forward(self, unary_logits: torch.Tensor) -> torch.Tensor:
        if unary_logits.dim() != 4 or unary_logits.shape[1] != 1:
            raise ValueError(f"unary logits must be (B,1,H,W), got {tuple(unary_logits.shape)}")
        unary = unary_logits[:, 0]
        b, h, w = unary.shape
        temperature = F.softplus(self.log_temperature).clamp_min(0.05)
        score = unary / temperature
        forward = []
        alpha = score[:, :, 0]
        forward.append(alpha)
        for col in range(1, w):
            alpha = score[:, :, col] + self._transition(alpha)
            forward.append(alpha)
        backward = [None] * w
        beta = torch.zeros((b, h), device=unary.device, dtype=unary.dtype)
        backward[-1] = beta
        for col in range(w - 2, -1, -1):
            beta = self._transition(score[:, :, col + 1] + beta)
            backward[col] = beta
        alpha_all = torch.stack(forward, dim=-1)
        beta_all = torch.stack(backward, dim=-1)
        return torch.softmax(alpha_all + beta_all - score, dim=1)[:, None]


class AeroPathSSD(nn.Module):
    """Compact non-causal state-space path network for 501x256 UAV-GPR windows."""
    accepts_altitude = True

    def __init__(self, *, base_ch: int = 24, input_channels: int = 1, metadata_channels: int = 0,
                 ssm_impl: str = "official_mamba2", mamba_state_dim: int = 64,
                 mamba_d_conv: int = 4, time_window_ns: float = 700.0, max_path_step: int = 6):
        super().__init__()
        self.input_channels = int(input_channels)
        self.metadata_channels = int(metadata_channels)
        self.time_window_ns = float(time_window_ns)
        c = int(base_ch)
        self.stem = _PhaseStem(4, c)
        self.down1 = nn.Sequential(nn.Conv2d(c, c * 2, 3, stride=(2, 1), padding=1), nn.GroupNorm(4, c * 2), nn.GELU())
        self.block1 = _AeroBlock(c * 2, metadata_channels, ssm_impl=ssm_impl, d_state=mamba_state_dim, d_conv=mamba_d_conv)
        self.down2 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.GroupNorm(4, c * 4), nn.GELU())
        self.block2 = _AeroBlock(c * 4, metadata_channels, ssm_impl=ssm_impl, d_state=mamba_state_dim, d_conv=mamba_d_conv)
        self.bottleneck = _AeroBlock(c * 4, metadata_channels, ssm_impl=ssm_impl, d_state=mamba_state_dim, d_conv=mamba_d_conv)
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(c * 4, c * 2, 3, padding=1), nn.GroupNorm(4, c * 2), nn.GELU())
        self.up2 = nn.ConvTranspose2d(c * 2, c, (2, 1), stride=(2, 1))
        self.dec2 = nn.Sequential(nn.Conv2d(c * 2, c, 3, padding=1), nn.GroupNorm(4, c), nn.GELU())
        self.energy_head = nn.Conv2d(c, 1, 1)
        self.band_head = nn.Conv2d(c, 1, 1)
        self.presence_head = nn.Conv2d(c, 1, 1)
        self.no_pick_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, 1, 1))
        self.path = SoftPathInference(max_step=max_path_step)

    def forward(self, x: torch.Tensor, altitude: torch.Tensor | None = None) -> AeroPathOutput:
        raw = x[:, :1]
        metadata = x[:, 1:1 + self.metadata_channels] if self.metadata_channels else None
        reduced = _air_reduce(raw, altitude, self.time_window_ns)
        stem_input = torch.cat([raw, raw.abs(), reduced, reduced.abs()], dim=1)
        s0 = self.stem(stem_input)
        s1 = self.block1(self.down1(s0), metadata)
        s2 = self.bottleneck(self.block2(self.down2(s1), metadata), metadata)
        d1 = self.dec1(torch.cat([F.interpolate(self.up1(s2), size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], dim=1))
        d0 = self.dec2(torch.cat([F.interpolate(self.up2(d1), size=s0.shape[-2:], mode="bilinear", align_corners=False), s0], dim=1))
        energy = self.energy_head(d0)
        marginals = self.path(energy)
        presence = self.presence_head(d0).amax(dim=2)
        no_pick = self.no_pick_head(d0).flatten(1)
        return AeroPathOutput(
            self.band_head(d0), presence, energy, curve_logits=energy,
            path_marginals=marginals, no_pick_logits=no_pick, air_reduced_input=reduced,
        )
