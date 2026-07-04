"""PGDA-CSNet physics-guided loss functions.

Architecture-agnostic loss computation consuming dict-like model outputs.
Suitable for U-Net, FNO, or any architecture that produces mask/presence/center heads.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


ZERO_PART_KEYS = (
    'clean_recon',
    'clutter_recon',
    'clean_consistency',
    'clutter_consistency',
    'decomp_total',
)


def dice_loss_from_prob(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    inter = (pred * target * weight).sum((1, 2, 3))
    den = ((pred + target) * weight).sum((1, 2, 3)) + eps
    return (1 - 2 * inter / den).mean()


def centerline_aux_losses(center_logits, target, weight, cfg, ignore=None):
    if center_logits is None:
        zero = target.mean() * 0.0
        return zero, zero
    lp = cfg.get('loss', {})
    prob = torch.sigmoid(center_logits)
    h = prob.shape[2]
    ys = torch.linspace(0.0, 1.0, h, device=prob.device, dtype=prob.dtype)[None, None, :, None]
    target_mass = target.sum(dim=2).clamp_min(1e-6)
    pred_mass = prob.sum(dim=2).clamp_min(1e-6)
    target_valid = (target.sum(dim=2) > float(lp.get('center_valid_min_sum', 1e-3))).float()
    if ignore is not None:
        target_valid = target_valid * (ignore.mean(dim=2) < 0.5).float()
    target_center = (target * ys).sum(dim=2) / target_mass
    pred_center = (prob * ys).sum(dim=2) / pred_mass
    col_w = (0.25 + weight[:, None, :]) * target_valid
    center_l1 = (torch.abs(pred_center - target_center) * col_w).sum() / col_w.sum().clamp_min(1e-6)
    if pred_center.shape[-1] > 1:
        smooth_valid = target_valid[..., 1:] * target_valid[..., :-1]
        smooth = torch.abs(pred_center[..., 1:] - pred_center[..., :-1])
        continuity = (smooth * smooth_valid).sum() / smooth_valid.sum().clamp_min(1e-6)
    else:
        continuity = pred_center.mean() * 0.0
    return center_l1, continuity


def spectral_consistency_loss(pred_prob, target, cfg):
    """Penalise divergence in f-k spectral energy between predicted and target masks."""
    lp = cfg.get('loss', {})
    lam = float(lp.get('spectral_consistency_weight', 0.0))
    if lam <= 0:
        return pred_prob.mean() * 0.0
    mask = (target > 0.01).float()
    if mask.sum() < 100:
        return pred_prob.mean() * 0.0
    pred_masked = pred_prob * mask
    tgt_masked = target * mask
    pred_fft = torch.fft.rfft2(pred_masked, norm='ortho')
    tgt_fft = torch.fft.rfft2(tgt_masked, norm='ortho')
    pred_mag = pred_fft.abs()
    tgt_mag = tgt_fft.abs()
    log_pred = torch.log1p(pred_mag)
    log_tgt = torch.log1p(tgt_mag)
    spec_loss = F.mse_loss(log_pred, log_tgt)
    return lam * spec_loss


def compute_segmentation_losses(outputs: dict[str, torch.Tensor | None], batch: dict[str, torch.Tensor], cfg: dict) -> dict[str, torch.Tensor]:
    """Compute segmentation, presence, and centerline losses.

    Args:
        outputs: Dict with keys 'mask_logits', 'presence_logits', 'center_logits'.
        batch: Dataset batch with keys 'y', 'y_core', 'presence', 'presence_valid', 'weight', 'valid_pix', 'valid_denom'.
        cfg: Training config dict.

    Returns:
        Dict of loss tensors.
    """
    lp = cfg.get('loss', {})
    y = batch['y']
    y_core = batch['y_core']
    prob = torch.sigmoid(outputs['mask_logits'])
    valid_pix = batch['valid_pix']
    valid_denom = batch['valid_denom']
    lw = batch['weight']
    pix_w = (float(lp.get('base_pixel_weight', 0.10)) + lw[:, None, None, :]) * valid_pix
    pos_boost = float(lp.get('positive_pixel_boost', 4.0))
    bce_w = pix_w * (1.0 + pos_boost * y)
    band_bce = (F.binary_cross_entropy_with_logits(outputs['mask_logits'], y, reduction='none') * bce_w).sum() / valid_denom
    band_dice = dice_loss_from_prob(prob, y, pix_w)
    core_w = pix_w * (0.5 + y_core)
    core_bce = (F.binary_cross_entropy_with_logits(outputs['mask_logits'], y_core, reduction='none') * core_w).sum() / valid_denom
    outside = (y < float(lp.get('outside_margin', 0.05))).float()
    outside_penalty = (prob * outside * (0.15 + pix_w) * valid_pix).sum() / valid_denom
    bg = (y < float(lp.get('outside_margin', 0.05))) & (valid_pix > 0.5)
    bg_prob = prob[bg]
    if bg_prob.numel() > 0:
        frac = float(lp.get('hard_negative_topk_frac', 0.02))
        k = max(1, int(bg_prob.numel() * frac))
        hard_negative = torch.topk(bg_prob.flatten(), k).values.mean()
    else:
        hard_negative = prob.mean() * 0.0
    pres = batch['presence']
    pres_valid = batch['presence_valid']
    pres_bce = F.binary_cross_entropy_with_logits(outputs['presence_logits'], pres, reduction='none')
    neg_boost = float(lp.get('presence_negative_weight', 5.0))
    pres_class_w = torch.where(pres <= 0.05, torch.full_like(pres, neg_boost), torch.ones_like(pres))
    pres_w = (0.25 + lw[:, None, :]) * pres_class_w * pres_valid[:, None, :]
    pres_loss = (pres_bce * pres_w).sum() / pres_w.sum().clamp_min(1e-6)
    center_l1, continuity = centerline_aux_losses(outputs['center_logits'], y, lw, cfg, batch.get('ignore_mask'))
    spec_loss = spectral_consistency_loss(prob, y, cfg)
    return {
        'band_bce': band_bce,
        'band_dice': band_dice,
        'core_bce': core_bce,
        'outside_penalty': outside_penalty,
        'hard_negative': hard_negative,
        'presence_loss': pres_loss,
        'centerline_l1': center_l1,
        'continuity': continuity,
        'spec_loss': spec_loss,
    }


def compute_decomposition_losses(outputs: dict[str, torch.Tensor | None], batch: dict[str, torch.Tensor | None], cfg: dict) -> dict[str, torch.Tensor]:
    """Compute clean/clutter decomposition losses.

    Args:
        outputs: Dict with keys 'clean_logits', 'clutter_logits'.
        batch: Dataset batch with optional keys 'x_clean', 'x_clutter', 'x_raw'.
        cfg: Training config dict.

    Returns:
        Dict of loss tensors (zero if heads or targets missing).
    """
    lp = cfg.get('loss', {})
    weights = {
        'clean_recon_weight': float(lp.get('clean_recon_weight', 0.0)),
        'clutter_recon_weight': float(lp.get('clutter_recon_weight', 0.0)),
        'clean_consistency_weight': float(lp.get('clean_consistency_weight', 0.0)),
        'clutter_consistency_weight': float(lp.get('clutter_consistency_weight', 0.0)),
    }
    enabled = any(v > 0 for v in weights.values())
    ref = batch['x'][:, :1]
    zero = ref.mean() * 0.0
    parts = {key: zero for key in ZERO_PART_KEYS}
    target_pairs = {
        'clean_logits': batch.get('x_clean'),
        'clutter_logits': batch.get('x_clutter'),
    }
    if not enabled:
        return parts
    for name, target in target_pairs.items():
        if target is None:
            continue
        pred = outputs.get(name)
        if pred is None:
            continue
        pred = pred[:, :1] if pred.ndim == 4 and pred.shape[1] > 1 else pred
        target = target[:, :1] if target.ndim == 4 and target.shape[1] > 1 else target
        if name == 'clean_logits':
            parts['clean_recon'] = F.smooth_l1_loss(pred, target)
        else:
            parts['clutter_recon'] = F.smooth_l1_loss(pred, target)
    clean_pred = outputs.get('clean_logits')
    clutter_pred = outputs.get('clutter_logits')
    if clean_pred is not None and clutter_pred is not None:
        clean_pred = clean_pred[:, :1] if clean_pred.ndim == 4 and clean_pred.shape[1] > 1 else clean_pred
        clutter_pred = clutter_pred[:, :1] if clutter_pred.ndim == 4 and clutter_pred.shape[1] > 1 else clutter_pred
        raw_target = batch.get('raw_target')
        if raw_target is not None:
            parts['clean_consistency'] = F.smooth_l1_loss(clean_pred + clutter_pred, raw_target)
        if batch.get('x_clean') is not None and batch.get('x_clutter') is not None:
            parts['clutter_consistency'] = F.smooth_l1_loss(clean_pred + clutter_pred, batch['x_clean'] + batch['x_clutter'])
    parts['decomp_total'] = (
        weights['clean_recon_weight'] * parts['clean_recon']
        + weights['clutter_recon_weight'] * parts['clutter_recon']
        + weights['clean_consistency_weight'] * parts['clean_consistency']
        + weights['clutter_consistency_weight'] * parts['clutter_consistency']
    )
    return parts


def combine_pgda_losses(seg_losses: dict[str, torch.Tensor], decomp_losses: dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    """Weighted sum of segmentation and decomposition losses."""
    lp = cfg.get('loss', {})
    total = (
        seg_losses['band_bce']
        + float(lp.get('dice_weight', 0.5)) * seg_losses['band_dice']
        + float(lp.get('core_weight', 0.25)) * seg_losses['core_bce']
        + float(lp.get('outside_weight', 0.40)) * seg_losses['outside_penalty']
        + float(lp.get('hard_negative_weight', 0.35)) * seg_losses['hard_negative']
        + float(lp.get('presence_weight', 0.25)) * seg_losses['presence_loss']
        + float(lp.get('centerline_weight', 0.0)) * seg_losses['centerline_l1']
        + float(lp.get('continuity_weight', 0.0)) * seg_losses['continuity']
        + seg_losses['spec_loss']
        + decomp_losses['decomp_total']
    )
    return total


def detach_parts(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in parts.items()}
