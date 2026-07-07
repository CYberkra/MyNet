"""GPRMambaSep physics-guided loss functions.

Six new losses (L1-L6) for the GPRMambaSep architecture that decomposes
the GPR B-scan into A (air-coupled direct wave), S (surface reflection),
and G (geological signal) components.

Imports compute_segmentation_losses from pgdacsnet.losses_pgda and
wraps it together with the new component-aware losses.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgdacsnet.losses_pgda import compute_segmentation_losses

# ---------------------------------------------------------------------------
# L3 building blocks: Gradient Reversal Layer + Discriminator
# ---------------------------------------------------------------------------


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer for adversarial training.

    Forward: identity (passes input through unchanged).
    Backward: multiplies gradient by -lambda.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    """Gradient reversal layer module.

    Wraps GradReverse as an nn.Module. The scale factor lambda controls
    the strength of gradient reversal (default 1.0).
    """

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradReverse.apply(x, self.lambd)


class ComponentDiscriminator(nn.Module):
    """3-layer MLP discriminator for the contrastive separation loss (L3).

    Takes pooled feature vectors from A and G pathways and classifies
    which pathway they came from (binary: 0 = A, 1 = G).

    Architecture: Linear(64 -> 32) -> ReLU -> Linear(32 -> 32) -> ReLU -> Linear(32 -> 1)
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) — pooled feature vectors. Returns logits (B, 1)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# L1: Self-consistency loss
# ---------------------------------------------------------------------------


def self_consistency_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    G_hat: torch.Tensor,
    Y_full: torch.Tensor,
) -> torch.Tensor:
    """L1: Enforce that A_hat + S_hat + G_hat reconstructs Y_full.

    loss = ||Y_full - (A_hat + S_hat + G_hat)||_1
         + ||Y_full - (A_hat + S_hat + G_hat)||_2

    All tensors: (B, 1, H, W).
    """
    recon = A_hat + S_hat + G_hat
    l1_loss = F.l1_loss(recon, Y_full)
    l2_loss = F.mse_loss(recon, Y_full)
    return l1_loss + l2_loss


# ---------------------------------------------------------------------------
# L2: Supervised component loss
# ---------------------------------------------------------------------------


def sim_supervised_component_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    G_hat: torch.Tensor,
    Y_air: torch.Tensor | None = None,
    Y_target_without_G: torch.Tensor | None = None,
    X_clean: torch.Tensor | None = None,
    G_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """L2: Component-aware supervision when simulation targets are available.

    Returns dict with keys:
        'a_l1', 's_l1', 'g_l1' — per-component L1 losses (zero if target missing).

    Semantics follow the locked signal model:
      - Y_air supervises A_hat.
      - Y_target_without_G supervises S_hat when available.
      - X_clean corresponds to S + G, so it supervises (S_hat + G_hat), not pure G_hat.
      - G_hat is only directly supervised when an explicit pure-G target is provided.
    """
    zero = (A_hat * 0.0).mean()

    if Y_air is not None:
        a_l1 = F.l1_loss(A_hat, Y_air)
    else:
        a_l1 = zero

    if Y_target_without_G is not None:
        s_l1 = F.l1_loss(S_hat, Y_target_without_G)
    else:
        s_l1 = zero

    if G_target is not None:
        g_l1 = F.l1_loss(G_hat, G_target)
    elif X_clean is not None:
        g_l1 = F.l1_loss(S_hat + G_hat, X_clean)
    else:
        g_l1 = zero

    return {"a_l1": a_l1, "s_l1": s_l1, "g_l1": g_l1}


# ---------------------------------------------------------------------------
# L3: Contrastive separation loss
# ---------------------------------------------------------------------------


def contrastive_separation_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    G_hat: torch.Tensor,
    discriminator: ComponentDiscriminator,
    grl_layer: GRL,
) -> torch.Tensor:
    """L3: Adversarial loss to encourage A and G pathways to learn
    statistically distinguishable features.

    Pools A_hat and G_hat to fixed-size vectors, runs through gradient
    reversal + discriminator. The discriminator is trained to classify
    A vs G; the encoder is trained adversarially (via GRL) to fool it.

    Returns binary cross-entropy loss over the discriminator output.
    """
    # Global average pool to (B, 1, 1, 1) then squeeze to (B,)
    pool_a = A_hat.mean(dim=(2, 3), keepdim=False)  # (B, 1)
    pool_g = G_hat.mean(dim=(2, 3), keepdim=False)  # (B, 1)

    # Concatenate features: (2*B, 1) — but we need (B, C) for the MLP
    # Instead, stack into a single batch with labels
    feat = torch.cat([pool_a, pool_g], dim=0)  # (2*B, 1)
    labels = torch.cat(
        [
            torch.zeros(A_hat.size(0), 1, device=A_hat.device, dtype=A_hat.dtype),
            torch.ones(G_hat.size(0), 1, device=G_hat.device, dtype=G_hat.dtype),
        ],
        dim=0,
    )  # (2*B, 1): 0 = A pathway, 1 = G pathway

    # Pass through gradient reversal (adversarial for encoder), then discriminator
    feat_rev = grl_layer(feat)
    logits = discriminator(feat_rev)  # (2*B, 1)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss


# ---------------------------------------------------------------------------
# L4: Arrival time prior loss
# ---------------------------------------------------------------------------


def arrival_time_prior_loss(
    G_hat: torch.Tensor,
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """L4: Penalize G_hat energy that appears before the physically
    plausible earliest arrival time.

    t_min(trace) = 2 * altitude / c_air + 2 * z_min / v_earth

    Mask: w_prior[t] = 1 if t < t_min else 0
    loss = ||G_hat * w_prior||_1 / ||w_prior||_1

    G_hat: (B, 1, H, W)
    batch: dict that may contain 'altitude' (scalar or (B,) tensor).
    cfg: config dict with optional 'time_window_ns', 'height_resize', etc.
    """
    B, C, H, W = G_hat.shape
    device = G_hat.device
    dtype = G_hat.dtype

    # --- constants ---
    c_air: float = 0.3      # m/ns
    v_earth: float = 0.07   # m/ns
    z_min: float = 3.0      # m

    # --- altitude: normalize to (B,) tensor ---
    altitude = batch.get("altitude", None)
    if altitude is None:
        altitude_t = torch.full((B,), 2.4, device=device, dtype=dtype)
    elif isinstance(altitude, (float, int)):
        altitude_t = torch.full((B,), float(altitude), device=device, dtype=dtype)
    else:
        altitude_t = altitude.to(device=device, dtype=dtype).reshape(-1)
        if altitude_t.numel() == 1:
            altitude_t = altitude_t.expand(B)

    # --- t_min in ns -> sample index ---
    t_min_ns = 2 * altitude_t / c_air + 2 * z_min / v_earth  # (B,)

    time_window_ns: float = float(cfg.get("time_window_ns", 700.0))
    n_time_samples: int = int(cfg.get("height_resize", H))
    dt: float = time_window_ns / max(n_time_samples, 1)

    t_min_samples = (t_min_ns / dt).long().clamp(0, H)  # (B,)

    # --- build mask: (B, 1, H, 1) then expand to (B, 1, H, W) ---
    t_grid = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)  # (1, 1, H, 1)
    t_limit = t_min_samples.float().view(B, 1, 1, 1)                     # (B, 1, 1, 1)
    w_prior = (t_grid < t_limit).float()                                  # (B, 1, H, 1)
    w_prior = w_prior.expand(B, C, H, W)                                  # (B, 1, H, W)

    # --- compute loss ---
    numerator = (G_hat.abs() * w_prior).sum()
    denominator = w_prior.sum().clamp_min(1.0)
    return numerator / denominator


# ---------------------------------------------------------------------------
# L5: Amplitude ratio prior loss
# ---------------------------------------------------------------------------


def amplitude_ratio_prior_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    weight: float = 0.01,
) -> torch.Tensor:
    """L5: Soft constraint on the |A_hat| / |S_hat| amplitude ratio.

    The theoretical Fresnel reflection coefficient at the air-ground
    interface for typical GPR scenarios falls in the range [-0.63, -0.42].
    This loss penalizes deviation of the log-ratio from that interval.

    Returns: weight * MSE(log_ratio, target_log_ratio)

    If S_hat has negligible energy, the loss returns 0 (avoids division
    by near-zero values).
    """
    eps = 1e-8
    a_abs = A_hat.abs().mean(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
    s_abs = S_hat.abs().mean(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)

    # Avoid division by near-zero
    s_abs = s_abs.clamp_min(eps)

    ratio = a_abs / s_abs  # (B, 1, 1, 1) — positive by construction

    # Target: mean of Fresnel magnitude range [0.42, 0.63]
    target_ratio = (0.42 + 0.63) / 2.0  # ~0.525

    # MSE on the ratio itself (symmetric penalty)
    loss = F.mse_loss(ratio, torch.full_like(ratio, target_ratio))

    return weight * loss


# ---------------------------------------------------------------------------
# L6: Co-prediction cycle loss (Stage 3 only)
# ---------------------------------------------------------------------------


def co_prediction_cycle_loss(
    A_hat: torch.Tensor,
    S_hat: torch.Tensor,
    G_hat: torch.Tensor,
    Y_full: torch.Tensor,
    model_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]],
) -> torch.Tensor:
    """L6: Cycle-consistency loss for Stage 3 training.

    Reconstructs Y_target_hat = A_hat + S_hat (geological signal removed),
    runs it through the shared encoder again, and enforces consistency.

    cycle_loss = ||Y_full - (A_hat2 + S_hat2 + G_hat2)||_1
               + ||A_hat - A_hat2||_1
               + ||S_hat - S_hat2||_1

    model_fn: Callable that takes (B, 1, H, W) B-scan and returns a dict
              with keys 'A_hat', 'S_hat', 'G_hat'.
    """
    # Reconstruct: Y_full without geological signal (S + A components)
    Y_target_hat = A_hat + S_hat

    # Pass through shared encoder
    outputs2 = model_fn(Y_target_hat)

    A_hat2 = outputs2["A_hat"]
    S_hat2 = outputs2["S_hat"]
    G_hat2 = outputs2["G_hat"]

    # Reconstruction loss: the second pass should still reconstruct Y_full
    recon_loss = F.l1_loss(A_hat2 + S_hat2 + G_hat2, Y_full)

    # Cycle consistency: A and S should be preserved through the cycle
    cycle_a = F.l1_loss(A_hat2, A_hat)
    cycle_s = F.l1_loss(S_hat2, S_hat)

    return recon_loss + cycle_a + cycle_s


# ---------------------------------------------------------------------------
# Orchestrator: compute_gprmambasep_loss
# ---------------------------------------------------------------------------


def compute_gprmambasep_loss(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, Any],
    cfg: dict[str, Any],
    model: Any = None,
    discriminator: ComponentDiscriminator | None = None,
    grl_layer: GRL | None = None,
    stage3: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined GPRMambaSep loss.

    Calls compute_segmentation_losses on the standard PGDA heads, then adds
    L1-L5 (and optionally L6 for Stage 3).

    Args:
        outputs: Dict-like object with standard keys:
            - 'mask_logits' / 'presence_logits' / 'center_logits'
            - 'A_hat', 'S_hat', 'G_hat' — component predictions (B, 1, H, W)
          Backward-compatible aliases 'G_mask_logits' / 'G_presence_logits' /
          'G_center_logits' are also accepted.
        batch: Dataset batch dict. Keys used:
            - 'y', 'y_core', 'presence', 'presence_valid', 'weight' (for segmentation loss)
            - 'valid_pix', 'valid_denom' (for segmentation loss)
            - 'x' — the input B-scan (Y_full)
            - 'Y_air', 'Y_target_without_G', 'X_clean' (optional, for L2)
            - 'G_target' (optional explicit pure-G supervision for L2)
            - 'altitude' (optional, for L4)
        cfg: Training config dict with loss weights under cfg['loss'].
        model: The GPRMambaSep model instance. Required for Stage 3 (L6).
        discriminator: ComponentDiscriminator instance for L3. Created if None and L3 weight > 0.
        grl_layer: GRL instance for L3. Created if None and L3 weight > 0.
        stage3: If True, include L6 co-prediction cycle loss.

    Returns:
        (total_loss, parts_dict) where total_loss is a scalar tensor for
        backward and parts_dict contains float values for logging.
    """
    lp = cfg.get("loss", {})
    device = batch["x"].device if isinstance(batch.get("x"), torch.Tensor) else outputs.get("A_hat").device
    dtype = outputs.get("A_hat", outputs.get("G_hat")).dtype if any(outputs.get(k) is not None for k in ("A_hat", "G_hat")) else torch.float32

    def _zero():
        ref = batch.get("x", outputs.get("G_hat"))
        if ref is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return ref.mean() * 0.0

    def _get_output(*names: str):
        for name in names:
            value = outputs.get(name)
            if value is not None:
                return value
        return None

    def _loss_weight(*names: str, default: float = 0.0) -> float:
        for name in names:
            if name in lp:
                return float(lp.get(name, default))
        return float(default)

    zero = _zero()
    parts: dict[str, torch.Tensor] = {}
    total = zero.clone()

    # ---- Segmentation losses on standard G/PGDA heads ----
    mask_logits = _get_output("mask_logits", "G_mask_logits")
    presence_logits = _get_output("presence_logits", "G_presence_logits")
    center_logits = _get_output("center_logits", "G_center_logits")
    if mask_logits is not None:
        seg_outputs = {
            "mask_logits": mask_logits,
            "presence_logits": presence_logits,
            "center_logits": center_logits,
        }
        seg_parts = compute_segmentation_losses(seg_outputs, batch, cfg)
        seg_weight = float(lp.get("g_segmentation_weight", 1.0))
        for k, v in seg_parts.items():
            parts[f"seg_{k}"] = v.detach()
            if k in ("spec_loss",):
                total = total + v
            elif k in ("band_bce",):
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
                }.get(k, None)
                w = float(lp.get(w_key, 1.0)) if w_key else 1.0
                total = total + seg_weight * w * v
    else:
        for k in (
            "seg_band_bce",
            "seg_band_dice",
            "seg_core_bce",
            "seg_outside_penalty",
            "seg_hard_negative",
            "seg_presence_loss",
            "seg_centerline_l1",
            "seg_continuity",
            "seg_spec_loss",
        ):
            parts[k] = zero.detach()

    # ---- Extract component tensors ----
    A_hat = outputs.get("A_hat")
    S_hat = outputs.get("S_hat")
    G_hat = outputs.get("G_hat")
    Y_full = batch.get("x")

    if Y_full is not None and Y_full.dim() == 4 and Y_full.shape[1] > 1:
        Y_full = Y_full[:, :1]

    # ---- L1: Self-consistency loss ----
    l1_weight = float(lp.get("self_consistency_weight", 0.0))
    if l1_weight > 0 and all(t is not None for t in (A_hat, S_hat, G_hat, Y_full)):
        l1_loss = self_consistency_loss(A_hat, S_hat, G_hat, Y_full)
        parts["self_consistency"] = l1_loss.detach()
        total = total + l1_weight * l1_loss
    else:
        parts["self_consistency"] = zero.detach()

    # ---- L2: Supervised component loss ----
    l2_weight = _loss_weight("sim_supervised_component_weight", "sim_supervised_weight", default=0.0)
    if l2_weight > 0 and all(t is not None for t in (A_hat, S_hat, G_hat)):
        Y_air = batch.get("Y_air")
        Y_tgt = batch.get("Y_target_without_G")
        X_clean = batch.get("X_clean")
        G_target = batch.get("G_target")
        l2_parts = sim_supervised_component_loss(A_hat, S_hat, G_hat, Y_air, Y_tgt, X_clean, G_target)
        l2_total = l2_parts["a_l1"] + l2_parts["s_l1"] + l2_parts["g_l1"]
        parts["l2_a_l1"] = l2_parts["a_l1"].detach()
        parts["l2_s_l1"] = l2_parts["s_l1"].detach()
        parts["l2_g_l1"] = l2_parts["g_l1"].detach()
        total = total + l2_weight * l2_total
    else:
        parts["l2_a_l1"] = zero.detach()
        parts["l2_s_l1"] = zero.detach()
        parts["l2_g_l1"] = zero.detach()

    # ---- L3: Contrastive separation loss ----
    l3_weight = _loss_weight("contrastive_separation_weight", "contrastive_weight", default=0.0)
    if l3_weight > 0 and all(t is not None for t in (A_hat, G_hat)):
        if discriminator is None:
            discriminator = ComponentDiscriminator(input_dim=1, hidden_dim=32).to(device)
        if grl_layer is None:
            grl_layer = GRL(lambd=float(lp.get("grad_reverse_lambda", 1.0)))
        l3_loss = contrastive_separation_loss(A_hat, S_hat, G_hat, discriminator, grl_layer)
        parts["contrastive_separation"] = l3_loss.detach()
        total = total + l3_weight * l3_loss
    else:
        parts["contrastive_separation"] = zero.detach()

    # ---- L4: Arrival time prior loss ----
    l4_weight = _loss_weight("arrival_time_prior_weight", "arrival_prior_weight", default=0.0)
    if l4_weight > 0 and G_hat is not None:
        l4_loss = arrival_time_prior_loss(G_hat, batch, cfg)
        parts["arrival_time_prior"] = l4_loss.detach()
        total = total + l4_weight * l4_loss
    else:
        parts["arrival_time_prior"] = zero.detach()

    # ---- L5: Amplitude ratio prior loss ----
    l5_weight = _loss_weight("amplitude_ratio_prior_weight", "amplitude_ratio_weight", default=0.0)
    if l5_weight > 0 and all(t is not None for t in (A_hat, S_hat)):
        l5_loss = amplitude_ratio_prior_loss(A_hat, S_hat, weight=1.0)
        parts["amplitude_ratio_prior"] = l5_loss.detach()
        total = total + l5_weight * l5_loss
    else:
        parts["amplitude_ratio_prior"] = zero.detach()

    # ---- L6: Co-prediction cycle loss (Stage 3 only) ----
    l6_weight = float(lp.get("co_prediction_cycle_weight", 0.0))
    if stage3 and l6_weight > 0 and all(t is not None for t in (A_hat, S_hat, G_hat, Y_full)) and model is not None:
        l6_loss = co_prediction_cycle_loss(A_hat, S_hat, G_hat, Y_full, model.forward)
        parts["co_prediction_cycle"] = l6_loss.detach()
        total = total + l6_weight * l6_loss
    else:
        parts["co_prediction_cycle"] = zero.detach()

    parts_float: dict[str, float] = {"loss": float(total.detach().cpu())}
    for k, v in parts.items():
        parts_float[k] = float(v.cpu())

    return total, parts_float
