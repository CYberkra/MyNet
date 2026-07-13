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


def _probability_bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute BCE on probabilities in FP32 under a mixed-precision caller.

    ``binary_cross_entropy`` is intentionally rejected by CUDA autocast because
    its backward pass can be numerically unsafe in half precision.  AeroPath's
    NULL and boundary outputs are posterior probabilities rather than logits,
    so BCE-with-logits is not applicable here.
    """
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return F.binary_cross_entropy(prediction.float(), target.float())


def _normalise_target(target: torch.Tensor, valid_pix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.clamp_min(0.0) * valid_pix
    mass = target.sum(dim=2, keepdim=True)
    valid = (mass.squeeze(2) > 1e-6).to(target.dtype)
    return target / mass.clamp_min(1e-6), valid


def _trace_weights(batch: dict[str, torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
    weight = batch["weight"].to(device=valid.device, dtype=valid.dtype)
    return (0.25 + weight[:, None, :]) * valid


def _trace_state(batch: dict[str, torch.Tensor], path: torch.Tensor) -> torch.Tensor:
    """Return 0=negative, 1=confirmed positive, 2=weak, 3=ignore.

    Older debug fixtures do not carry ``trace_state``; derive the same contract
    from presence validity there, but production datasets must provide it.
    """
    state = batch.get("trace_state")
    if state is not None:
        if state.ndim == 3:
            state = state[:, 0]
        return state.to(device=path.device, dtype=torch.long)
    presence = batch["presence"].to(path.device)
    presence_valid = batch["presence_valid"].to(path.device)
    if presence.ndim == 3:
        presence = presence[:, 0]
    if presence_valid.ndim == 3:
        presence_valid = presence_valid[:, 0]
    return torch.where(
        presence_valid > 0.5,
        torch.where(presence > 0.05, torch.ones_like(presence, dtype=torch.long), torch.zeros_like(presence, dtype=torch.long)),
        torch.full_like(presence, 2, dtype=torch.long),
    )


def _chainage_transition_scale(
    chainage_m: torch.Tensor | None,
    width: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Scale adjacent-trace penalties by their measured physical spacing."""
    if width < 2:
        return reference.new_ones((*reference.shape[:-1], 0))
    if chainage_m is None:
        return reference.new_ones((*reference.shape[:-1], width - 1))
    if chainage_m.ndim == 3:
        chainage_m = chainage_m[:, 0]
    if chainage_m.ndim != 2 or chainage_m.shape[0] != reference.shape[0]:
        raise ValueError(f"chainage_m must be (B,W), got {tuple(chainage_m.shape)}")
    chainage_m = chainage_m.to(device=reference.device, dtype=reference.dtype)
    if chainage_m.shape[-1] != width:
        chainage_m = F.interpolate(
            chainage_m[:, None], size=width, mode="linear", align_corners=False
        )[:, 0]
    delta = (chainage_m[:, 1:] - chainage_m[:, :-1]).abs()
    finite = torch.isfinite(delta) & (delta > 1e-6)
    delta = torch.where(finite, delta, torch.ones_like(delta))
    median = delta.median(dim=1, keepdim=True).values.clamp_min(1e-6)
    return (median / delta).clamp(0.25, 4.0)[:, None, :]


def structured_path_losses(output: Any, batch: dict[str, torch.Tensor], cfg: dict) -> dict[str, torch.Tensor]:
    """Supervise soft-DP marginals and calibrated path uncertainty."""
    path = output.path_marginals.clamp_min(1e-8)
    state = _trace_state(batch, path)
    trace_valid = batch.get("valid_trace_mask")
    if trace_valid is None:
        trace_valid = (state != 3).to(path.dtype)
    elif trace_valid.ndim == 3:
        trace_valid = trace_valid[:, 0]
    trace_valid = trace_valid.to(device=path.device, dtype=path.dtype)
    target, valid = _normalise_target(batch["y"].to(path.dtype), batch["valid_pix"].to(path.dtype))
    # A path label is meaningful only for confirmed or weak-positive traces.
    valid = valid * (((state == 1) | (state == 2)).to(path.dtype) * trace_valid)[:, None, :]
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
        smooth = smooth * _chainage_transition_scale(
            batch.get("chainage_m"), path.shape[-1], pred_center
        )
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

    null = getattr(output, "null_marginals", None)
    hard_known = (state == 0) | (state == 1)
    null_valid = hard_known & (trace_valid > 0.5)
    if null is not None and null_valid.any():
        null = null.squeeze(1).clamp(1e-8, 1.0 - 1e-8)
        null_target = (state == 0).to(path.dtype)
        null_nll = _probability_bce(null[null_valid], null_target[null_valid])
    else:
        null_nll = path_nll * 0.0

    # Do not turn weak positives into negatives, and do not claim a window-level
    # no-pick target when any trace is weak or ignored.
    known_and_valid = hard_known & (trace_valid > 0.5)
    window_valid = known_and_valid.all(dim=-1)
    no_pick_logits = output.no_pick_logits.reshape(-1)
    if window_valid.any():
        has_target = (state == 1).any(dim=-1)
        no_pick_target = (~has_target).to(path.dtype)
        no_pick_bce = F.binary_cross_entropy_with_logits(
            no_pick_logits[window_valid], no_pick_target[window_valid]
        )
    else:
        no_pick_bce = path_nll * 0.0

    boundary_nll = path_nll * 0.0
    starts = getattr(output, "path_start_prob", None)
    ends = getattr(output, "path_end_prob", None)
    if starts is not None and ends is not None:
        starts, ends = starts.squeeze(1).clamp(1e-8, 1.0 - 1e-8), ends.squeeze(1).clamp(1e-8, 1.0 - 1e-8)
        previous = F.pad(state[:, :-1], (1, 0), value=0)
        following = F.pad(state[:, 1:], (0, 1), value=0)
        previous_known = F.pad(known_and_valid[:, :-1], (1, 0), value=True)
        following_known = F.pad(known_and_valid[:, 1:], (0, 1), value=True)
        start_known = known_and_valid & previous_known
        end_known = known_and_valid & following_known
        start_target = ((state == 1) & (previous == 0)).to(path.dtype)
        end_target = ((state == 1) & (following == 0)).to(path.dtype)
        terms = []
        if start_known.any():
            terms.append(_probability_bce(starts[start_known], start_target[start_known]))
        if end_known.any():
            terms.append(_probability_bce(ends[end_known], end_target[end_known]))
        if terms:
            boundary_nll = torch.stack(terms).mean()
    return {
        "path_nll": path_nll,
        "path_center_l1": center_l1,
        "path_smooth": path_smooth,
        "path_uncertainty_nll": uncertainty_nll,
        "no_pick_bce": no_pick_bce,
        "path_null_nll": null_nll,
        "path_boundary_nll": boundary_nll,
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
        + float(lp.get("aeropath_null_weight", 0.25)) * structured["path_null_nll"]
        + float(lp.get("aeropath_boundary_weight", 0.1)) * structured["path_boundary_nll"]
    )
    parts = {f"seg_{key}": float(value.detach().cpu()) for key, value in seg.items()}
    parts.update({key: float(value.detach().cpu()) for key, value in structured.items()})
    parts["loss"] = float(total.detach().cpu())
    return total, parts
