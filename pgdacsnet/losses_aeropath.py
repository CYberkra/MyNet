"""Losses for AeroPath-SSD's structured interface-path objective.

The model still emits a band mask for comparison with historical baselines,
but its scientific target is a single time-distribution per trace.  This
module therefore keeps mask losses separate from path, abstention, and
uncertainty losses instead of treating a path probability as a segmentation
mask.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from pgdacsnet.losses_pgda import combine_pgda_losses, compute_segmentation_losses


def _normalise_target(target: torch.Tensor, valid_pix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.clamp_min(0.0) * valid_pix
    mass = target.sum(dim=2, keepdim=True)
    valid = (mass.squeeze(2) > 1e-6).to(target.dtype)
    return target / mass.clamp_min(1e-6), valid


def _trace_weights(batch: dict[str, torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
    weight = batch["weight"].to(device=valid.device, dtype=valid.dtype)
    return (0.25 + weight[:, None, :]) * valid


def structured_path_losses(output: Any, batch: dict[str, torch.Tensor], cfg: dict) -> dict[str, torch.Tensor]:
    """Supervise soft-DP marginals and calibrated path uncertainty."""
    path = output.path_marginals.clamp_min(1e-8)
    target, valid = _normalise_target(batch["y"].to(path.dtype), batch["valid_pix"].to(path.dtype))
    ignore = batch.get("ignore_mask")
    if ignore is not None:
        valid = valid * (ignore.to(path.dtype).mean(dim=2) < 0.5).to(path.dtype)
    weights = _trace_weights(batch, valid)
    denom = weights.sum().clamp_min(1e-6)
    per_trace_nll = -(target * path.log()).sum(dim=2)
    path_nll = (per_trace_nll * weights).sum() / denom

    h = path.shape[2]
    ys = torch.linspace(0.0, 1.0, h, device=path.device, dtype=path.dtype)[None, None, :, None]
    pred_center = (path * ys).sum(dim=2)
    target_center = (target * ys).sum(dim=2)
    center_l1 = (pred_center.sub(target_center).abs() * weights).sum() / denom
    if path.shape[-1] > 1:
        pair_weight = weights[..., 1:] * (valid[..., :-1] > 0).to(path.dtype)
        smooth = (pred_center[..., 1:] - pred_center[..., :-1]).abs()
        path_smooth = (smooth * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)
    else:
        path_smooth = path_nll * 0.0

    uncertainty = getattr(output, "uncertainty_logits", None)
    if uncertainty is None:
        uncertainty_nll = path_nll * 0.0
    else:
        if uncertainty.shape[-2:] != path.shape[-2:]:
            uncertainty = F.interpolate(uncertainty, size=path.shape[-2:], mode="bilinear", align_corners=False)
        log_variance = (path.detach() * uncertainty.clamp(-6.0, 3.0)).sum(dim=2)
        squared_error = (pred_center - target_center).square()
        uncertainty_nll = ((squared_error * torch.exp(-log_variance) + log_variance) * weights).sum() / denom

    presence = batch["presence"].to(path.dtype)
    presence_valid = batch["presence_valid"].to(path.dtype)
    if presence.ndim == 3:
        presence = presence[:, 0]
    if presence_valid.ndim == 3:
        presence_valid = presence_valid[:, 0]
    confirmed = presence_valid > 0.5
    # A window with only weak/unknown traces has no valid global target.
    window_valid = confirmed.any(dim=-1)
    no_pick_logits = output.no_pick_logits.reshape(-1)
    if window_valid.any():
        has_target = ((presence > 0.05) & confirmed).any(dim=-1)
        no_pick_target = (~has_target).to(path.dtype)
        no_pick_bce = F.binary_cross_entropy_with_logits(
            no_pick_logits[window_valid], no_pick_target[window_valid]
        )
    else:
        no_pick_bce = path_nll * 0.0
    return {
        "path_nll": path_nll,
        "path_center_l1": center_l1,
        "path_smooth": path_smooth,
        "path_uncertainty_nll": uncertainty_nll,
        "no_pick_bce": no_pick_bce,
        "path_supervised_trace_count": valid.sum(),
        "no_pick_supervised_window_count": window_valid.to(path.dtype).sum(),
    }


def compute_aeropath_loss(output: Any, batch: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine baseline-compatible segmentation and structured AeroPath losses."""
    lp = cfg.get("loss", {})
    seg = compute_segmentation_losses(
        {
            "mask_logits": output.mask_logits,
            "presence_logits": output.presence_logits,
            "center_logits": output.center_logits,
        },
        batch,
        cfg,
    )
    total = combine_pgda_losses(seg, {"decomp_total": output.mask_logits.mean() * 0.0}, cfg)
    structured = structured_path_losses(output, batch, cfg)
    total = total + (
        float(lp.get("aeropath_path_nll_weight", 1.0)) * structured["path_nll"]
        + float(lp.get("aeropath_path_center_weight", 0.6)) * structured["path_center_l1"]
        + float(lp.get("aeropath_path_smooth_weight", 0.03)) * structured["path_smooth"]
        + float(lp.get("aeropath_uncertainty_weight", 0.05)) * structured["path_uncertainty_nll"]
        + float(lp.get("aeropath_no_pick_weight", 0.25)) * structured["no_pick_bce"]
    )
    parts = {f"seg_{key}": float(value.detach().cpu()) for key, value in seg.items()}
    parts.update({key: float(value.detach().cpu()) for key, value in structured.items()})
    parts["loss"] = float(total.detach().cpu())
    return total, parts
