"""Physics-guided losses for GprMambaSep.

The model decomposes a UAV-GPR B-scan into A (air-coupled direct wave),
S (surface reflection), and G (subsurface geological signal).  This module
keeps the standard PGDA segmentation losses and adds component-aware losses.

Important implementation rule:
    A_hat + S_hat + G_hat is only a physically linear reconstruction when
    the batch supplies ``Y_full_component`` in a linear/affine component space.
    If the batch does not supply that key, the reconstruction loss falls back
    to the model input tensor space and is logged as tensor-space consistency.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgdacsnet.losses_pgda import compute_segmentation_losses

COMPONENT_TARGET_KEYS = ("Y_air", "Y_target_without_G", "X_clean", "G_target")


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer for optional domain-adversarial experiments."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    """Gradient reversal layer module."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradReverse.apply(x, self.lambd)


class ComponentDiscriminator(nn.Module):
    """MLP classifier over hand-crafted component statistics.

    The default feature vector has 8 dimensions and includes signed/global,
    early/mid/late, gradient and coherence statistics.  The discriminator must
    be owned by the trainer and included in the optimizer/checkpoint; this
    module intentionally does not get instantiated inside the loss function.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _zero_like(ref: torch.Tensor) -> torch.Tensor:
    return ref.mean() * 0.0


def _component_stats(x: torch.Tensor) -> torch.Tensor:
    """Return compact differentiable stats for one component, shape (B, 8)."""
    B, _, H, W = x.shape
    abs_x = x.abs()
    thirds = torch.chunk(abs_x, 3, dim=2)
    early = thirds[0].mean(dim=(1, 2, 3))
    mid = thirds[1].mean(dim=(1, 2, 3)) if len(thirds) > 1 else early * 0
    late = thirds[2].mean(dim=(1, 2, 3)) if len(thirds) > 2 else early * 0
    signed_mean = x.mean(dim=(1, 2, 3))
    abs_mean = abs_x.mean(dim=(1, 2, 3))
    std = x.flatten(1).std(dim=1, unbiased=False)
    if H > 1:
        grad_t = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    else:
        grad_t = abs_mean * 0
    if W > 1:
        # Adjacent-trace similarity proxy; higher for horizontally coherent events.
        a = x[:, :, :, 1:]
        b = x[:, :, :, :-1]
        coh = (a * b).mean(dim=(1, 2, 3)) / (
            (a.square().mean(dim=(1, 2, 3)).sqrt() * b.square().mean(dim=(1, 2, 3)).sqrt()).clamp_min(1e-6)
        )
    else:
        coh = abs_mean * 0
    return torch.stack([signed_mean, abs_mean, early, mid, late, std, grad_t, coh], dim=1)


def self_consistency_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    G_hat: torch.Tensor,
    Y_full: torch.Tensor,
) -> torch.Tensor:
    """Reconstruction consistency in the supplied target space."""
    recon = A_hat + S_hat + G_hat
    return F.l1_loss(recon, Y_full) + F.mse_loss(recon, Y_full)


def _valid_vector(valid: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if valid is None:
        return None
    v = valid.to(device=ref.device, dtype=ref.dtype).reshape(ref.shape[0], -1).mean(dim=1)
    return (v > 0.5).to(ref.dtype)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor | None, valid: torch.Tensor | None) -> torch.Tensor:
    if target is None:
        return _zero_like(pred)
    if valid is None:
        return F.l1_loss(pred, target)
    v = _valid_vector(valid, pred)
    if v is None or float(v.sum().detach().cpu()) <= 0:
        return _zero_like(pred)
    per = torch.abs(pred - target).mean(dim=(1, 2, 3))
    return (per * v).sum() / v.sum().clamp_min(1.0)


def sim_supervised_component_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    G_hat: torch.Tensor,
    Y_air: torch.Tensor | None = None,
    Y_target_without_G: torch.Tensor | None = None,
    X_clean: torch.Tensor | None = None,
    G_target: torch.Tensor | None = None,
    Y_air_valid: torch.Tensor | None = None,
    Y_target_without_G_valid: torch.Tensor | None = None,
    X_clean_valid: torch.Tensor | None = None,
    G_target_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Component-aware supervision with per-sample validity flags.

    Datasets always return placeholder tensors for optional component targets so
    mixed real/sim batches collate safely.  The *_valid flags decide whether a
    sample contributes to each component loss.
    """
    a_l1 = _masked_l1(A_hat, Y_air, Y_air_valid)
    s_l1 = _masked_l1(S_hat, Y_target_without_G, Y_target_without_G_valid)
    sg_clean_l1 = _masked_l1(S_hat + G_hat, X_clean, X_clean_valid)
    g_pure_l1 = _masked_l1(G_hat, G_target, G_target_valid)
    return {
        "a_l1": a_l1,
        "s_l1": s_l1,
        "sg_clean_l1": sg_clean_l1,
        "g_pure_l1": g_pure_l1,
    }


def contrastive_separation_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    G_hat: torch.Tensor,
    discriminator: ComponentDiscriminator,
    grl_layer: GRL | None = None,
    use_gradient_reversal: bool = False,
) -> torch.Tensor:
    """Separate A and G statistics using a persistent discriminator.

    By default this is a discriminative separation objective: both the
    discriminator and component pathways receive gradients that make A/G more
    distinguishable.  Set use_gradient_reversal=True only for experimental
    domain-invariance/adversarial ablations.
    """
    feat_a = _component_stats(A_hat)
    feat_g = _component_stats(G_hat)
    feat = torch.cat([feat_a, feat_g], dim=0)
    if use_gradient_reversal:
        feat = (grl_layer or GRL(1.0))(feat)
    labels = torch.cat(
        [
            torch.zeros(A_hat.size(0), 1, device=A_hat.device, dtype=A_hat.dtype),
            torch.ones(G_hat.size(0), 1, device=G_hat.device, dtype=G_hat.dtype),
        ],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(discriminator(feat), labels)


def arrival_time_prior_loss(G_hat: torch.Tensor, batch: dict[str, Any], cfg: dict[str, Any]) -> torch.Tensor:
    """Penalise G energy before the plausible earliest subsurface arrival."""
    B, C, H, W = G_hat.shape
    device, dtype = G_hat.device, G_hat.dtype
    c_air = float(cfg.get("c_air_m_per_ns", 0.3))
    v_earth = float(cfg.get("v_earth_m_per_ns", 0.07))
    z_min = float(cfg.get("g_min_depth_m", 3.0))
    altitude = batch.get("altitude", None)
    if altitude is None:
        altitude_t = torch.full((B,), float(cfg.get("default_altitude_m", 2.4)), device=device, dtype=dtype)
    elif isinstance(altitude, (float, int)):
        altitude_t = torch.full((B,), float(altitude), device=device, dtype=dtype)
    else:
        altitude_t = altitude.to(device=device, dtype=dtype).reshape(-1)
        if altitude_t.numel() == 1:
            altitude_t = altitude_t.expand(B)
    t_min_ns = 2 * altitude_t / c_air + 2 * z_min / v_earth
    time_window_ns = float(cfg.get("time_window_ns", 700.0))
    dt = time_window_ns / max(int(cfg.get("height_resize", H)), 1)
    t_min_samples = (t_min_ns / dt).long().clamp(0, H)
    t_grid = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    t_limit = t_min_samples.float().view(B, 1, 1, 1)
    w_prior = (t_grid < t_limit).float().expand(B, C, H, W)
    return (G_hat.abs() * w_prior).sum() / w_prior.sum().clamp_min(1.0)


def amplitude_ratio_prior_loss(A_hat: torch.Tensor, S_hat: torch.Tensor, weight: float = 0.01) -> torch.Tensor:
    """Very weak A/S amplitude-ratio prior; keep low-weight and metadata-aware later."""
    eps = 1e-8
    ratio = A_hat.abs().mean(dim=(2, 3), keepdim=True) / S_hat.abs().mean(dim=(2, 3), keepdim=True).clamp_min(eps)
    target_ratio = (0.42 + 0.63) / 2.0
    return weight * F.mse_loss(ratio, torch.full_like(ratio, target_ratio))


def g_envelope_mask_consistency_loss(G_hat: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Force G_hat energy/envelope to be explainable by the target mask."""
    target = batch.get("y_core", batch.get("y"))
    if target is None:
        return _zero_like(G_hat)
    env = G_hat.abs()
    env = env / env.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    valid = batch.get("valid_pix", torch.ones_like(env)).to(device=env.device, dtype=env.dtype)
    ignore = 1.0 - valid
    mse = ((env - target.to(env.device, env.dtype)).square() * valid).sum() / valid.sum().clamp_min(1.0)
    outside = (target.to(env.device, env.dtype) < 0.05).float() * valid
    outside_penalty = (env * outside).sum() / outside.sum().clamp_min(1.0)
    return mse + 0.25 * outside_penalty + ignore.mean() * 0.0





def _normalise_curve_target(target: torch.Tensor, valid_pix: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a wide/narrow mask into trace-wise P(t|trace) and validity."""
    if target.dim() != 4 or target.shape[1] != 1:
        raise ValueError(f"target must have shape (B,1,H,W), got {tuple(target.shape)}")
    t = target.clamp_min(0.0)
    if valid_pix is not None:
        t = t * valid_pix.to(device=t.device, dtype=t.dtype)
    mass = t.sum(dim=2, keepdim=True)
    valid = (mass.squeeze(2) > 1e-6).to(t.dtype)  # (B,1,W)
    probs = t / mass.clamp_min(1e-6)
    return probs, valid


def curve_distribution_loss(
    curve_logits: torch.Tensor,
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Trace-wise curve distribution supervision for P(t|trace).

    The target is derived from the existing soft/wide mask, then normalised over
    the time axis.  This makes the main head optimise the actual picking target
    instead of a generic pixel mask.
    """
    lp = cfg.get("loss", {})
    y = batch["y"].to(device=curve_logits.device, dtype=curve_logits.dtype)
    valid_pix = batch.get("valid_pix")
    if valid_pix is not None:
        valid_pix = valid_pix.to(device=curve_logits.device, dtype=curve_logits.dtype)
    weight = batch.get("weight")
    if weight is None:
        weight = torch.ones((curve_logits.shape[0], curve_logits.shape[-1]), device=curve_logits.device, dtype=curve_logits.dtype)
    else:
        weight = weight.to(device=curve_logits.device, dtype=curve_logits.dtype)
    ignore = batch.get("ignore_mask")
    if ignore is not None:
        ignore = ignore.to(device=curve_logits.device, dtype=curve_logits.dtype)

    target_probs, target_valid = _normalise_curve_target(y, valid_pix)  # (B,1,H,W), (B,1,W)
    if ignore is not None:
        target_valid = target_valid * (ignore.mean(dim=2) < 0.5).to(target_valid.dtype)

    log_probs = F.log_softmax(curve_logits, dim=2)
    probs = log_probs.exp()
    per_trace_ce = -(target_probs * log_probs).sum(dim=2)  # (B,1,W)
    col_w = (0.25 + weight[:, None, :]) * target_valid
    ce = (per_trace_ce * col_w).sum() / col_w.sum().clamp_min(1e-6)

    h = curve_logits.shape[2]
    ys = torch.linspace(0.0, 1.0, h, device=curve_logits.device, dtype=curve_logits.dtype)[None, None, :, None]
    pred_center = (probs * ys).sum(dim=2)
    target_center = (target_probs * ys).sum(dim=2)
    center = (F.smooth_l1_loss(pred_center, target_center, reduction="none") * col_w).sum() / col_w.sum().clamp_min(1e-6)

    if pred_center.shape[-1] > 1:
        smooth_valid = target_valid[..., 1:] * target_valid[..., :-1]
        first = (pred_center[..., 1:] - pred_center[..., :-1]).abs()
        smooth = (first * smooth_valid).sum() / smooth_valid.sum().clamp_min(1e-6)
    else:
        smooth = curve_logits.mean() * 0.0

    if pred_center.shape[-1] > 2:
        smooth2_valid = target_valid[..., 2:] * target_valid[..., 1:-1] * target_valid[..., :-2]
        second = (pred_center[..., 2:] - 2.0 * pred_center[..., 1:-1] + pred_center[..., :-2]).abs()
        curvature = (second * smooth2_valid).sum() / smooth2_valid.sum().clamp_min(1e-6)
    else:
        curvature = curve_logits.mean() * 0.0

    shallow_max_ns = float(lp.get("shallow_suppression_max_ns", -1.0))
    shallow = curve_logits.mean() * 0.0
    if shallow_max_ns > 0:
        time_window_ns = float(cfg.get("time_window_ns", cfg.get("time_window", 700.0)))
        hi = int(round(shallow_max_ns / max(time_window_ns, 1e-6) * h))
        hi = max(1, min(h, hi))
        # Penalise probability mass in shallow forbidden area, mostly on traces
        # without target support there.  This is intentionally soft: it should
        # suppress distractors, not make early targets impossible in future data.
        shallow_mass = probs[:, :, :hi, :].sum(dim=2)
        shallow = (shallow_mass * (0.25 + weight[:, None, :])).mean()

    return {
        "curve_ce": ce,
        "curve_center": center,
        "curve_smooth": smooth,
        "curve_curvature": curvature,
        "curve_shallow_suppression": shallow,
    }


def global_no_target_loss(
    global_no_target_logits: torch.Tensor,
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Binary line-level no-target supervision.

    The head predicts P(no target in this window).  A line/window is treated as
    no-target when no trace-level presence target exceeds the configured floor.
    """
    pres = batch.get("presence")
    if pres is None:
        return global_no_target_logits.mean() * 0.0
    pres = pres.to(device=global_no_target_logits.device, dtype=global_no_target_logits.dtype)
    thr = float(cfg.get("loss", {}).get("global_no_target_presence_thr", 0.05))
    target_no = (pres.amax(dim=-1) <= thr).to(global_no_target_logits.dtype)
    logits = global_no_target_logits.reshape_as(target_no)
    return F.binary_cross_entropy_with_logits(logits, target_no)



def uncertainty_nll_loss(
    curve_logits: torch.Tensor,
    uncertainty_logits: torch.Tensor,
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Heteroscedastic trace-wise center loss for the optional uncertainty head.

    The head predicts a per-pixel log-variance proxy.  We collapse it to one
    trace-wise log-sigma value by taking the expectation under ``P(t|trace)``.
    This gives the uncertainty head a real training signal without requiring a
    separate uncertainty label.  It is intentionally low-weight: the head is for
    calibration/diagnostics, not for rescuing poor curve logits.
    """
    if uncertainty_logits is None:
        return curve_logits.mean() * 0.0
    if uncertainty_logits.shape[-2:] != curve_logits.shape[-2:]:
        uncertainty_logits = F.interpolate(
            uncertainty_logits,
            size=curve_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    y = batch["y"].to(device=curve_logits.device, dtype=curve_logits.dtype)
    valid_pix = batch.get("valid_pix")
    if valid_pix is not None:
        valid_pix = valid_pix.to(device=curve_logits.device, dtype=curve_logits.dtype)
    ignore = batch.get("ignore_mask")
    if ignore is not None:
        ignore = ignore.to(device=curve_logits.device, dtype=curve_logits.dtype)
    weight = batch.get("weight")
    if weight is None:
        weight = torch.ones((curve_logits.shape[0], curve_logits.shape[-1]), device=curve_logits.device, dtype=curve_logits.dtype)
    else:
        weight = weight.to(device=curve_logits.device, dtype=curve_logits.dtype)

    target_probs, target_valid = _normalise_curve_target(y, valid_pix)
    if ignore is not None:
        target_valid = target_valid * (ignore.mean(dim=2) < 0.5).to(target_valid.dtype)

    probs = F.softmax(curve_logits, dim=2)
    h = curve_logits.shape[2]
    ys = torch.linspace(0.0, 1.0, h, device=curve_logits.device, dtype=curve_logits.dtype)[None, None, :, None]
    pred_center = (probs * ys).sum(dim=2)
    target_center = (target_probs * ys).sum(dim=2)
    err2 = (pred_center - target_center).square().detach()

    # Convert pixel-wise log-sigma logits to a trace-wise log-sigma.  Clamp for
    # stable NLL and to prevent the trivial high-uncertainty escape.
    log_sigma_map = uncertainty_logits.clamp(-5.0, 2.0)
    log_sigma = (probs.detach() * log_sigma_map).sum(dim=2)
    nll = 0.5 * (err2 * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma)
    col_w = (0.25 + weight[:, None, :]) * target_valid
    return (nll * col_w).sum() / col_w.sum().clamp_min(1e-6)


def component_gate_regularization_loss(gates: torch.Tensor, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Regularise soft A/S/G allocation gates to reduce branch collapse.

    gates has shape (B, 3, H, W) and sums to one along dim=1.  The balance term
    keeps the dataset-level branch allocation near a configurable prior; the
    entropy term can be enabled to avoid overly hard early collapse.
    """
    if gates is None:
        raise ValueError("gates must be a tensor")
    prior_cfg = cfg.get("component_gate_prior", [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    prior = torch.as_tensor(prior_cfg, device=gates.device, dtype=gates.dtype).view(3)
    prior = prior / prior.sum().clamp_min(1e-6)
    mean_gate = gates.mean(dim=(0, 2, 3))
    balance = F.mse_loss(mean_gate, prior)
    entropy = -(gates.clamp_min(1e-8) * gates.clamp_min(1e-8).log()).sum(dim=1).mean()
    target_entropy = float(cfg.get("component_gate_entropy_target", 0.70)) * torch.log(torch.tensor(3.0, device=gates.device, dtype=gates.dtype))
    entropy_match = (entropy - target_entropy).square()
    return {"balance": balance, "entropy": entropy, "entropy_match": entropy_match}

def co_prediction_cycle_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    G_hat: torch.Tensor,
    Y_full: torch.Tensor,
    model_fn: Callable[[torch.Tensor], Any],
    original_input: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage-3 cycle loss that preserves auxiliary channels when present."""
    Y_target_hat = A_hat + S_hat
    if original_input is not None and original_input.dim() == 4 and original_input.shape[1] > 1:
        Y_target_hat = torch.cat([Y_target_hat, original_input[:, 1:]], dim=1)
    outputs2 = model_fn(Y_target_hat)
    A_hat2, S_hat2, G_hat2 = outputs2["A_hat"], outputs2["S_hat"], outputs2["G_hat"]
    recon_loss = F.l1_loss(A_hat2 + S_hat2 + G_hat2, Y_full)
    return recon_loss + F.l1_loss(A_hat2, A_hat) + F.l1_loss(S_hat2, S_hat)


def _get_output(outputs: Any, *names: str):
    for name in names:
        try:
            value = outputs.get(name)
        except AttributeError:
            value = getattr(outputs, name, None)
        if value is not None:
            return value
    return None


def _loss_weight(lp: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in lp:
            return float(lp.get(name, default))
    return float(default)


def compute_gprmambasep_loss(
    outputs: Any,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    model: Any = None,
    discriminator: ComponentDiscriminator | None = None,
    grl_layer: GRL | None = None,
    stage3: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute standard PGDA + GprMambaSep component losses."""
    lp = cfg.get("loss", {})
    ref = batch.get("x")
    if ref is None:
        ref = _get_output(outputs, "G_hat", "A_hat")
    zero = ref.mean() * 0.0
    parts: dict[str, torch.Tensor] = {}
    total = zero.clone()

    mask_logits = _get_output(outputs, "mask_logits", "G_mask_logits")
    presence_logits = _get_output(outputs, "presence_logits", "G_presence_logits")
    center_logits = _get_output(outputs, "center_logits", "G_center_logits")
    if mask_logits is not None:
        seg_outputs = {"mask_logits": mask_logits, "presence_logits": presence_logits, "center_logits": center_logits}
        seg_parts = compute_segmentation_losses(seg_outputs, batch, cfg)
        seg_weight = float(lp.get("g_segmentation_weight", 1.0))
        for k, v in seg_parts.items():
            parts[f"seg_{k}"] = v.detach()
            if k == "spec_loss":
                total = total + v
            elif k == "band_bce":
                total = total + seg_weight * v
            else:
                w_key = {
                    "band_dice": "dice_weight",
                    "core_bce": "core_weight",
                    "outside_penalty": "outside_weight",
                    "hard_negative": "hard_negative_weight",
                    "presence_loss": "presence_weight",
                    "centerline_l1": "centerline_weight",
                    "continuity": "continuity_weight",
                }.get(k)
                total = total + seg_weight * (float(lp.get(w_key, 1.0)) if w_key else 1.0) * v

    curve_logits = _get_output(outputs, "curve_logits")
    curve_weight = _loss_weight(lp, "curve_distribution_weight", "curve_weight", default=0.0)
    if curve_logits is not None and curve_weight > 0:
        curve_parts = curve_distribution_loss(curve_logits, batch, cfg)
        curve_total = (
            curve_parts["curve_ce"]
            + float(lp.get("curve_center_weight", 0.5)) * curve_parts["curve_center"]
            + float(lp.get("curve_smooth_weight", 0.02)) * curve_parts["curve_smooth"]
            + float(lp.get("curve_curvature_weight", 0.02)) * curve_parts["curve_curvature"]
            + float(lp.get("shallow_suppression_weight", 0.0)) * curve_parts["curve_shallow_suppression"]
        )
        for k, v in curve_parts.items():
            parts[k] = v.detach()
        total = total + curve_weight * curve_total
    else:
        for k in ("curve_ce", "curve_center", "curve_smooth", "curve_curvature", "curve_shallow_suppression"):
            parts[k] = zero.detach()

    global_no_target_logits = _get_output(outputs, "global_no_target_logits")
    global_no_target_weight = _loss_weight(lp, "global_no_target_weight", "no_target_weight", default=0.0)
    if global_no_target_logits is not None and global_no_target_weight > 0:
        gnt = global_no_target_loss(global_no_target_logits, batch, cfg)
        parts["global_no_target"] = gnt.detach()
        total = total + global_no_target_weight * gnt
    else:
        parts["global_no_target"] = zero.detach()

    uncertainty_logits = _get_output(outputs, "uncertainty_logits")
    uncertainty_weight = _loss_weight(lp, "uncertainty_weight", "uncertainty_nll_weight", default=0.0)
    if uncertainty_logits is not None and curve_logits is not None and uncertainty_weight > 0:
        unc = uncertainty_nll_loss(curve_logits, uncertainty_logits, batch, cfg)
        parts["uncertainty_nll"] = unc.detach()
        total = total + uncertainty_weight * unc
    else:
        parts["uncertainty_nll"] = zero.detach()

    A_hat = _get_output(outputs, "A_hat")
    S_hat = _get_output(outputs, "S_hat")
    G_hat = _get_output(outputs, "G_hat")
    Y_full = batch.get("Y_full_component", batch.get("x"))
    if isinstance(Y_full, torch.Tensor) and Y_full.dim() == 4 and Y_full.shape[1] > 1:
        Y_full = Y_full[:, :1]

    # Coverage/provenance diagnostics are always logged.  Optional component
    # tensors may be zero placeholders; validity flags are authoritative.
    for key in COMPONENT_TARGET_KEYS:
        flag = batch.get(f"{key}_valid")
        if isinstance(flag, torch.Tensor):
            has_key = flag.to(zero.device, zero.dtype).float().mean()
        else:
            has_key = torch.as_tensor(1.0 if isinstance(batch.get(key), torch.Tensor) else 0.0, device=zero.device, dtype=zero.dtype)
        parts[f"component_has_{key}"] = has_key
    if isinstance(batch.get("has_component_targets"), torch.Tensor):
        parts["component_has_any"] = batch["has_component_targets"].to(zero.device, zero.dtype).float().mean()
    else:
        parts["component_has_any"] = torch.stack([parts[f"component_has_{k}"] for k in COMPONENT_TARGET_KEYS]).amax()

    # L1 reconstruction consistency.
    l1_weight = float(lp.get("self_consistency_weight", 0.0))
    if l1_weight > 0 and all(t is not None for t in (A_hat, S_hat, G_hat, Y_full)):
        l1_loss = self_consistency_loss(A_hat, S_hat, G_hat, Y_full)
        parts["self_consistency"] = l1_loss.detach()
        parts["self_consistency_linear_space"] = torch.as_tensor(
            1.0 if "Y_full_component" in batch else 0.0, device=zero.device, dtype=zero.dtype
        )
        total = total + l1_weight * l1_loss
    else:
        parts["self_consistency"] = zero.detach()
        parts["self_consistency_linear_space"] = zero.detach()

    # L2 component supervision.
    l2_weight = _loss_weight(lp, "sim_supervised_component_weight", "sim_supervised_weight", default=0.0)
    if l2_weight > 0 and all(t is not None for t in (A_hat, S_hat, G_hat)):
        l2_parts = sim_supervised_component_loss(
            A_hat,
            S_hat,
            G_hat,
            batch.get("Y_air"),
            batch.get("Y_target_without_G"),
            batch.get("X_clean"),
            batch.get("G_target"),
            batch.get("Y_air_valid"),
            batch.get("Y_target_without_G_valid"),
            batch.get("X_clean_valid"),
            batch.get("G_target_valid"),
        )
        l2_total = sum(l2_parts.values())
        for k, v in l2_parts.items():
            parts[f"l2_{k}"] = v.detach()
        total = total + l2_weight * l2_total
    else:
        for k in ("a_l1", "s_l1", "sg_clean_l1", "g_pure_l1"):
            parts[f"l2_{k}"] = zero.detach()

    # L3 separation.  The discriminator must be supplied by the trainer.
    l3_weight = _loss_weight(lp, "component_separation_weight", "contrastive_separation_weight", "contrastive_weight", default=0.0)
    if l3_weight > 0 and all(t is not None for t in (A_hat, S_hat, G_hat)):
        if discriminator is None:
            raise RuntimeError(
                "component/contrastive separation loss is enabled, but no persistent "
                "ComponentDiscriminator was supplied by the trainer. Disable the weight "
                "or create/pass a discriminator so it enters the optimizer/checkpoint."
            )
        l3_A, l3_S, l3_G = A_hat, S_hat, G_hat
        if bool(lp.get("component_separation_requires_targets", True)):
            valid_any = batch.get("has_component_targets")
            if isinstance(valid_any, torch.Tensor):
                mask = _valid_vector(valid_any, A_hat) > 0.5
                if bool(mask.any()):
                    l3_A, l3_S, l3_G = A_hat[mask], S_hat[mask], G_hat[mask]
                else:
                    l3_A = l3_S = l3_G = None
        if l3_A is not None:
            l3_loss = contrastive_separation_loss(
                l3_A,
                l3_S,
                l3_G,
                discriminator,
                grl_layer,
                use_gradient_reversal=bool(lp.get("use_gradient_reversal", False)),
            )
            parts["component_separation"] = l3_loss.detach()
            total = total + l3_weight * l3_loss
        else:
            parts["component_separation"] = zero.detach()
    else:
        parts["component_separation"] = zero.detach()

    # L4/L5 priors.
    l4_weight = _loss_weight(lp, "arrival_time_prior_weight", "arrival_prior_weight", default=0.0)
    if l4_weight > 0 and G_hat is not None:
        l4_loss = arrival_time_prior_loss(G_hat, batch, cfg)
        parts["arrival_time_prior"] = l4_loss.detach()
        total = total + l4_weight * l4_loss
    else:
        parts["arrival_time_prior"] = zero.detach()

    l5_weight = _loss_weight(lp, "amplitude_ratio_prior_weight", "amplitude_ratio_weight", default=0.0)
    if l5_weight > 0 and all(t is not None for t in (A_hat, S_hat)):
        l5_loss = amplitude_ratio_prior_loss(A_hat, S_hat, weight=1.0)
        parts["amplitude_ratio_prior"] = l5_loss.detach()
        total = total + l5_weight * l5_loss
    else:
        parts["amplitude_ratio_prior"] = zero.detach()

    # G-envelope/mask consistency to prevent a correct mask with meaningless G_hat.
    env_weight = _loss_weight(lp, "g_envelope_mask_weight", "g_mask_consistency_weight", default=0.0)
    if env_weight > 0 and G_hat is not None:
        env_loss = g_envelope_mask_consistency_loss(G_hat, batch)
        parts["g_envelope_mask"] = env_loss.detach()
        total = total + env_weight * env_loss
    else:
        parts["g_envelope_mask"] = zero.detach()

    # Soft component gate regularisation.
    gates = _get_output(outputs, "component_gates")
    gate_balance_weight = _loss_weight(lp, "component_gate_balance_weight", default=0.0)
    gate_entropy_weight = _loss_weight(lp, "component_gate_entropy_weight", default=0.0)
    if gates is not None and (gate_balance_weight > 0 or gate_entropy_weight > 0):
        gate_parts = component_gate_regularization_loss(gates, lp)
        parts["component_gate_balance"] = gate_parts["balance"].detach()
        parts["component_gate_entropy"] = gate_parts["entropy"].detach()
        parts["component_gate_entropy_match"] = gate_parts["entropy_match"].detach()
        total = total + gate_balance_weight * gate_parts["balance"] + gate_entropy_weight * gate_parts["entropy_match"]
    else:
        parts["component_gate_balance"] = zero.detach()
        parts["component_gate_entropy"] = zero.detach()
        parts["component_gate_entropy_match"] = zero.detach()

    # Stage-3 cycle.
    l6_weight = float(lp.get("co_prediction_cycle_weight", 0.0))
    if stage3 and l6_weight > 0 and all(t is not None for t in (A_hat, S_hat, G_hat, Y_full)) and model is not None:
        l6_loss = co_prediction_cycle_loss(A_hat, S_hat, G_hat, Y_full, model.forward, batch.get("x"))
        parts["co_prediction_cycle"] = l6_loss.detach()
        total = total + l6_weight * l6_loss
    else:
        parts["co_prediction_cycle"] = zero.detach()

    parts_float: dict[str, float] = {"loss": float(total.detach().cpu())}
    for k, v in parts.items():
        parts_float[k] = float(v.detach().cpu())
    return total, parts_float
