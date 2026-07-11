"""AeroPath-SSD: acquisition-conditioned structured UAV-GPR path picking.

The model deliberately treats a basal interface as one continuous path instead
of an unrelated set of segmentation pixels.  It uses local anisotropic temporal/
trace convolutions, bidirectional axial state-space mixing, and a differentiable
soft path inference layer.  ``official_mamba2`` is opt-in; tests may use
``ssm_lite`` but that fallback must never be presented as a Mamba-2 experiment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from pgdacsnet.model_interfaces import AeroPathOutput
from pgdacsnet.model_mamba import make_axial_sequence_block


class _AnisotropicStem(nn.Module):
    """Local time/trace mixer; this is not an analytic phase or IQ encoder."""
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


class _TraceFiLM(nn.Module):
    def __init__(self, metadata_channels: int, channels: int):
        super().__init__()
        self.metadata_channels = int(metadata_channels)
        hidden = max(16, channels // 2)
        self.net = nn.Sequential(
            nn.Conv1d(max(self.metadata_channels, 1), hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, channels * 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None) -> torch.Tensor:
        if self.metadata_channels <= 0 or metadata is None:
            return x
        # Keep conditioning trace-resolved.  A global pooled vector would erase
        # the altitude, terrain, and acquisition variation this model claims to use.
        trace_metadata = metadata.mean(dim=2)
        trace_metadata = F.interpolate(trace_metadata, size=x.shape[-1], mode="linear", align_corners=False)
        gamma, beta = self.net(trace_metadata).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, :]) + beta[:, :, None, :]


class _AeroBlock(nn.Module):
    def __init__(self, channels: int, metadata_channels: int, *, ssm_impl: str, d_state: int, d_conv: int,
                 mamba_expand: int, mamba_headdim: int, bidirectional: bool):
        super().__init__()
        self.film = _TraceFiLM(metadata_channels, channels)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.GroupNorm(4, channels), nn.GELU(), nn.Conv2d(channels, channels, 1),
        )
        self.ssm = make_axial_sequence_block(
            ssm_impl, channels, d_state=d_state, d_conv=d_conv,
            expand=mamba_expand, headdim=mamba_headdim, bidirectional=bidirectional,
        )
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
    # Missing AGL metadata is represented by NaN in canonical datasets.  It
    # must leave only that trace unreduced, never turn the entire view into NaN.
    altitude = torch.where(torch.isfinite(altitude) & (altitude > 0), altitude, torch.zeros_like(altitude))
    air_ns = 2.0 * altitude / 299_792_458.0 * 1e9
    y = torch.linspace(-1.0, 1.0, h, device=raw.device, dtype=raw.dtype)[None, :, None]
    x = torch.linspace(-1.0, 1.0, w, device=raw.device, dtype=raw.dtype)[None, None, :]
    shift = 2.0 * air_ns[:, None, :] / max(float(time_window_ns), 1e-6)
    grid = torch.stack((x.expand(b, h, w), (y + shift).expand(b, h, w)), dim=-1)
    return F.grid_sample(raw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


@dataclass
class SoftPathResult:
    path_marginals: torch.Tensor
    null_marginals: torch.Tensor
    path_start_prob: torch.Tensor
    path_end_prob: torch.Tensor


class SoftPathInference(nn.Module):
    """First-order forward/backward inference with explicit NULL transitions.

    The physical states are time samples and the NULL state represents a trace
    with no interface pick.  The forward messages include the current emission;
    backward messages exclude it, so a physical marginal is simply
    ``softmax(alpha + beta)`` over physical and NULL states.
    """
    def __init__(self, max_step: int = 6, initial_slope_penalty: float = 1.0, temperature: float = 0.35,
                 initial_start_penalty: float = 1.0, initial_end_penalty: float = 1.0):
        super().__init__()
        self.max_step = int(max_step)
        self.log_slope_penalty = nn.Parameter(torch.tensor(math.log(math.expm1(initial_slope_penalty))))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(temperature))))
        self.log_start_penalty = nn.Parameter(torch.tensor(math.log(math.expm1(initial_start_penalty))))
        self.log_end_penalty = nn.Parameter(torch.tensor(math.log(math.expm1(initial_end_penalty))))

    def _transition(self, score: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        # score: (B,H); outputs logsumexp over admissible predecessor heights.
        penalty = F.softplus(self.log_slope_penalty)
        if scale is None:
            scale = torch.ones(score.shape[0], device=score.device, dtype=score.dtype)
        parts = []
        for delta in range(-self.max_step, self.max_step + 1):
            shifted = F.pad(score, (max(delta, 0), max(-delta, 0)), value=-float("inf"))
            shifted = shifted[:, max(-delta, 0): shifted.shape[1] - max(delta, 0)]
            parts.append(shifted - penalty * abs(delta) * scale[:, None])
        return torch.logsumexp(torch.stack(parts, dim=-1), dim=-1)

    @staticmethod
    def _transition_scale(chainage_m: torch.Tensor | None, width: int, reference: torch.Tensor) -> torch.Tensor:
        if chainage_m is None:
            return torch.ones((reference.shape[0], max(width - 1, 0)), device=reference.device, dtype=reference.dtype)
        if chainage_m.dim() == 3:
            chainage_m = chainage_m.squeeze(1)
        if chainage_m.dim() != 2 or chainage_m.shape[0] != reference.shape[0]:
            raise ValueError(f"chainage_m must be (B,W), got {tuple(chainage_m.shape)}")
        chainage_m = F.interpolate(chainage_m[:, None].to(reference.dtype), size=width, mode="linear", align_corners=False).squeeze(1)
        delta = (chainage_m[:, 1:] - chainage_m[:, :-1]).abs()
        finite = torch.isfinite(delta) & (delta > 1e-6)
        fallback = torch.ones_like(delta)
        delta = torch.where(finite, delta, fallback)
        median = delta.median(dim=1, keepdim=True).values.clamp_min(1e-6)
        return (median / delta).clamp(0.25, 4.0)

    def forward(self, unary_logits: torch.Tensor, *, null_logits: torch.Tensor | None = None,
                chainage_m: torch.Tensor | None = None, return_details: bool = False) -> torch.Tensor | SoftPathResult:
        if unary_logits.dim() != 4 or unary_logits.shape[1] != 1:
            raise ValueError(f"unary logits must be (B,1,H,W), got {tuple(unary_logits.shape)}")
        unary = unary_logits[:, 0]
        b, h, w = unary.shape
        temperature = F.softplus(self.log_temperature).clamp_min(0.05)
        score = unary / temperature
        if null_logits is None:
            null_score = torch.full((b, w), -float("inf"), device=unary.device, dtype=unary.dtype)
        else:
            if null_logits.dim() == 3:
                null_logits = null_logits[:, 0]
            if null_logits.shape[0] != b:
                raise ValueError(f"null logits batch size differs from unary logits: {tuple(null_logits.shape)}")
            null_score = F.interpolate(null_logits[:, None].to(unary.dtype), size=w, mode="linear", align_corners=False).squeeze(1) / temperature
        start_penalty = F.softplus(self.log_start_penalty)
        end_penalty = F.softplus(self.log_end_penalty)
        transition_scale = self._transition_scale(chainage_m, w, unary)

        forward_physical: list[torch.Tensor] = []
        forward_null: list[torch.Tensor] = []
        alpha = score[:, :, 0] - start_penalty
        alpha_null = null_score[:, 0]
        forward_physical.append(alpha)
        forward_null.append(alpha_null)
        for col in range(1, w):
            previous_alpha = alpha
            previous_null = alpha_null
            alpha = score[:, :, col] + torch.logaddexp(
                self._transition(previous_alpha, transition_scale[:, col - 1]),
                previous_null[:, None] - start_penalty,
            )
            alpha_null = null_score[:, col] + torch.logaddexp(
                previous_null,
                torch.logsumexp(previous_alpha, dim=1) - end_penalty,
            )
            forward_physical.append(alpha)
            forward_null.append(alpha_null)

        backward_physical: list[torch.Tensor | None] = [None] * w
        backward_null: list[torch.Tensor | None] = [None] * w
        beta = torch.zeros((b, h), device=unary.device, dtype=unary.dtype)
        beta_null = torch.zeros((b,), device=unary.device, dtype=unary.dtype)
        backward_physical[-1] = beta
        backward_null[-1] = beta_null
        for col in range(w - 2, -1, -1):
            next_physical = score[:, :, col + 1] + beta
            next_null = null_score[:, col + 1] + beta_null
            beta = torch.logaddexp(
                self._transition(next_physical, transition_scale[:, col]),
                next_null[:, None] - end_penalty,
            )
            beta_null = torch.logaddexp(
                next_null,
                torch.logsumexp(next_physical, dim=1) - start_penalty,
            )
            backward_physical[col] = beta
            backward_null[col] = beta_null

        alpha_all = torch.stack(forward_physical, dim=-1)
        alpha_null_all = torch.stack(forward_null, dim=-1)
        beta_all = torch.stack([value for value in backward_physical if value is not None], dim=-1)
        beta_null_all = torch.stack([value for value in backward_null if value is not None], dim=-1)
        joint = torch.cat([alpha_all + beta_all, (alpha_null_all + beta_null_all)[:, None]], dim=1)
        posterior = torch.softmax(joint, dim=1)
        path_marginals = posterior[:, :h, :][:, None]
        null_marginals = posterior[:, h, :][:, None]
        log_partition = torch.logsumexp(joint, dim=1)

        starts = torch.zeros((b, w), device=unary.device, dtype=unary.dtype)
        ends = torch.zeros_like(starts)
        starts[:, 0] = path_marginals[:, 0, :, 0].sum(dim=1)
        ends[:, -1] = path_marginals[:, 0, :, -1].sum(dim=1)
        for col in range(1, w):
            start_log = (
                alpha_null_all[:, col - 1, None] - start_penalty + score[:, :, col]
                + beta_all[:, :, col] - log_partition[:, col, None]
            )
            starts[:, col] = torch.exp(start_log).sum(dim=1)
            end_log = (
                alpha_all[:, :, col - 1] - end_penalty + null_score[:, col, None]
                + beta_null_all[:, col, None] - log_partition[:, col, None]
            )
            ends[:, col - 1] = torch.exp(end_log).sum(dim=1)
        result = SoftPathResult(path_marginals, null_marginals, starts[:, None], ends[:, None])
        return result if return_details else result.path_marginals


class AeroPathSSD(nn.Module):
    """Compact non-causal state-space path network for 501x256 UAV-GPR windows."""
    accepts_altitude = True

    def __init__(self, *, base_ch: int = 24, input_channels: int = 1, metadata_channels: int = 0,
                 ssm_impl: str = "official_mamba2", mamba_state_dim: int = 64,
                 mamba_d_conv: int = 4, mamba_expand: int = 2, mamba_headdim: int = 16, bidirectional_axial: bool = True,
                 time_window_ns: float = 700.0, max_path_step: int = 6):
        super().__init__()
        self.input_channels = int(input_channels)
        self.metadata_channels = int(metadata_channels)
        self.time_window_ns = float(time_window_ns)
        c = int(base_ch)
        self.stem = _AnisotropicStem(4, c)
        self.down1 = nn.Sequential(nn.Conv2d(c, c * 2, 3, stride=(2, 1), padding=1), nn.GroupNorm(4, c * 2), nn.GELU())
        block_args = dict(ssm_impl=ssm_impl, d_state=mamba_state_dim, d_conv=mamba_d_conv, mamba_expand=mamba_expand,
                          mamba_headdim=mamba_headdim, bidirectional=bidirectional_axial)
        self.block1 = _AeroBlock(c * 2, metadata_channels, **block_args)
        self.down2 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.GroupNorm(4, c * 4), nn.GELU())
        self.block2 = _AeroBlock(c * 4, metadata_channels, **block_args)
        self.bottleneck = _AeroBlock(c * 4, metadata_channels, **block_args)
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(c * 4, c * 2, 3, padding=1), nn.GroupNorm(4, c * 2), nn.GELU())
        self.up2 = nn.ConvTranspose2d(c * 2, c, (2, 1), stride=(2, 1))
        self.dec2 = nn.Sequential(nn.Conv2d(c * 2, c, 3, padding=1), nn.GroupNorm(4, c), nn.GELU())
        self.energy_head = nn.Conv2d(c, 1, 1)
        self.band_head = nn.Conv2d(c, 1, 1)
        self.presence_head = nn.Conv2d(c, 1, 1)
        self.null_head = nn.Conv2d(c, 1, 1)
        self.no_pick_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, 1, 1))
        # This is a log-variance field, not a confidence mask.  Losses consume
        # it only along the structured path distribution.
        self.uncertainty_head = nn.Conv2d(c, 1, 1)
        self.path = SoftPathInference(max_step=max_path_step)

    def forward(self, x: torch.Tensor, altitude: torch.Tensor | None = None,
                chainage_m: torch.Tensor | None = None) -> AeroPathOutput:
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
        path_result = self.path(
            energy, null_logits=self.null_head(d0).mean(dim=2), chainage_m=chainage_m, return_details=True,
        )
        presence = self.presence_head(d0).amax(dim=2)
        no_pick = self.no_pick_head(d0).flatten(1)
        return AeroPathOutput(
            self.band_head(d0), presence, energy, curve_logits=energy,
            path_marginals=path_result.path_marginals, null_marginals=path_result.null_marginals,
            path_start_prob=path_result.path_start_prob, path_end_prob=path_result.path_end_prob,
            no_pick_logits=no_pick, air_reduced_input=reduced,
            uncertainty_logits=self.uncertainty_head(d0),
        )
