#!/usr/bin/env python3
"""
eval_gprmambasep_separation.py — Separation quality evaluator for GprMambaSep.

Loads a trained GprMambaSep checkpoint, runs full B-scan inference, and
quantifies the A/S/G separation quality:

  - Per-component SNR (time-gated regions for real data, full-reference for sim)
  - Leakage ratio: fraction of A/S energy leaking into the G time window
  - G_mask IoU: intersection-over-union vs ground-truth soft mask (when available)

Saves a 4-panel diagnostic plot: A_hat / S_hat / G_hat / Residual.

Usage:
  "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/eval_gprmambasep_separation.py \\
      --checkpoint outputs/run_gprmambasep_pretrain_v2/checkpoint_best.pt \\
      --line LINE9_STYLE_001 \\
      --data-root data/PGDA_SYNTH_DATASET_V1/05_accepted_dataset

Real-data evaluation (using manual labels for G_mask IoU):
  "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/eval_gprmambasep_separation.py \\
      --checkpoint outputs/run_gprmambasep_mixed_v2/checkpoint_best.pt \\
      --line Line9 \\
      --data-root data_corrected_v1_4_terrain_direction
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ----- Path setup -----
# Both the worktree root and the main repo root are needed because the
# worktree's pgdacsnet/ may lack model_gprmambasep.py / model_interfaces.py.
_WT_ROOT = Path(__file__).resolve().parents[1]
_MAIN_REPO = _WT_ROOT.resolve().parent  # D:\Claude\PGDA-CSNet
for p in [str(_WT_ROOT), str(_MAIN_REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pgdacsnet.model_gprmambasep import build_gprmambasep
from pgdacsnet.model_interfaces import (
    GprMambaSepOutput,
    unpack_pgda_output,
)
from pgdacsnet.model_raw_unet import compress_raw


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def resolve_data_root(data_root_arg: str | None, cfg: dict | None = None) -> Path:
    """Resolve path relative to the main repo root if not absolute."""
    value = data_root_arg or (cfg or {}).get("data_root", "data")
    p = Path(value)
    return p if p.is_absolute() else _MAIN_REPO / p


def _stitch_one(
    ckpt_path: Path,
    line_name: str,
    data_root_arg: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run model on a full line and return stitched component predictions.

    Returns
    -------
    A_hat, S_hat, G_hat, residual, cfg
        All arrays shape (H0, W0); residual = raw - (A_hat + S_hat + G_hat).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    data_root = resolve_data_root(data_root_arg, cfg)

    # Build model
    model = build_gprmambasep(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load line — try lines/ format first, then case format
    line_path = data_root / "lines" / f"{line_name}.npz"
    case_raw = data_root / line_name / "input" / "raw_bscan.npy"
    case_label = data_root / line_name / "label" / "y_soft_501x128.npy"

    if line_path.exists():
        loaded = np.load(line_path)
        raw = loaded["raw_full_normalized"].astype(np.float32)
        gt = loaded.get("soft_mask_train") if "soft_mask_train" in loaded else None
    elif case_raw.exists():
        # Single-case sim dataset format
        raw = np.load(case_raw).astype(np.float32)
        gt = np.load(case_label).astype(np.float32) if case_label.exists() else None
    else:
        raise FileNotFoundError(
            f"Could not find data for line {line_name} under {data_root}\n"
            f"  Tried: {line_path}\n"
            f"  Tried: {case_raw}"
        )

    H0, W0 = raw.shape
    H, W = int(cfg["height_resize"]), int(cfg["width_resize"])

    # Accumulators for each component
    A_sum = np.zeros((H0, W0), dtype=np.float32)
    S_sum = np.zeros((H0, W0), dtype=np.float32)
    G_sum = np.zeros((H0, W0), dtype=np.float32)
    weight_sum = np.zeros((H0, W0), dtype=np.float32)

    # Try window_index stitching; if absent, run full line in one pass
    idx_path = data_root / "window_index.csv"
    if idx_path.exists():
        rows = [
            r
            for r in csv.DictReader(open(idx_path, encoding="utf-8"))
            if r["line"] == line_name
        ]
    else:
        # Single window covering the whole line
        rows = [{"line": line_name, "start": "0", "end": str(W0 - 1)}]

    for r in rows:
        s = int(r["start"])
        e = int(r["end"]) + 1
        x = torch.from_numpy(raw[:, s:e][None, None]).float().to(device)
        xrs = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        xrs = compress_raw(xrs, float(cfg.get("input_log_scale", 1e-3)))

        with torch.no_grad():
            out = model(xrs)  # GprMambaSepOutput

        # Upsample each component to original resolution
        def _up(t):
            return (
                F.interpolate(t, (H0, e - s), mode="bilinear", align_corners=False)[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

        A_piece = _up(out.A_hat)
        S_piece = _up(out.S_hat)
        G_piece = _up(out.G_hat)

        # Hanning blend
        ww = np.hanning(e - s).astype(np.float32)
        if ww.max() > 0:
            ww = ww / ww.max()
        ww = 0.15 + 0.85 * ww
        w2 = np.broadcast_to(ww[None, :], A_piece.shape).astype(np.float32)

        A_sum[:, s:e] += A_piece * w2
        S_sum[:, s:e] += S_piece * w2
        G_sum[:, s:e] += G_piece * w2
        weight_sum[:, s:e] += w2

    eps = 1e-12
    A_hat = A_sum / np.maximum(weight_sum, eps)
    S_hat = S_sum / np.maximum(weight_sum, eps)
    G_hat = G_sum / np.maximum(weight_sum, eps)
    residual = raw - (A_hat + S_hat + G_hat)

    return A_hat, S_hat, G_hat, residual, raw, gt, cfg


# ──────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────
def _snr_db(signal: np.ndarray, error: np.ndarray, eps: float = 1e-12) -> float:
    """Signal-to-noise ratio in dB: 10*log10(var(signal) / var(error))."""
    vs = float(np.var(signal))
    ve = float(np.var(error))
    if ve < eps:
        return float("inf")
    return float(10.0 * np.log10(max(vs, eps) / ve))


def compute_separation_metrics(
    A_hat: np.ndarray,
    S_hat: np.ndarray,
    G_hat: np.ndarray,
    residual: np.ndarray,
    raw: np.ndarray,
    gt: np.ndarray | None,
) -> dict:
    """Compute per-component SNR, leakage ratio, and G_mask IoU.

    For real data without ground-truth components, SNR is computed in
    time-gated regions:
      - Early gate (0–120 samples):  dominated by A + S
      - Late gate (250–400 samples): dominated by G  (geological target zone)

    Leakage ratio = fraction of (A+S) power that falls inside the G gate.
    """
    metrics = {}
    H, W = A_hat.shape

    # ---- Time gates (sample indices) ----
    early_start, early_end = 0, 120     # A+S zone
    late_start, late_end = 250, 400     # G target zone

    # Clip to valid range
    early_end = min(early_end, H)
    late_end = min(late_end, H)

    # ---- Per-component SNR in their respective time gates ----
    # A+S SNR in early gate
    as_early = (A_hat + S_hat)[early_start:early_end, :]
    raw_early = raw[early_start:early_end, :]
    err_early = raw_early - as_early
    metrics["snr_as_early_gate_db"] = _snr_db(raw_early, err_early)

    # G SNR in late gate
    g_late = G_hat[late_start:late_end, :]
    raw_late = raw[late_start:late_end, :]
    err_late = raw_late - g_late
    metrics["snr_g_late_gate_db"] = _snr_db(raw_late, err_late)

    # Overall residual SNR (full image)
    metrics["snr_residual_full_db"] = _snr_db(raw, residual)

    # ---- Leakage ratio: A/S energy in G gate ----
    as_total = np.sum((A_hat + S_hat) ** 2) + 1e-12
    as_in_g_gate = np.sum((A_hat + S_hat)[late_start:late_end, :] ** 2)
    metrics["leakage_as_into_g_gate"] = float(as_in_g_gate / as_total)

    # ---- G_mask IoU (against ground truth soft mask, if available) ----
    if gt is not None:
        # Binarise G_hat and gt at threshold 0.15 (standard pick threshold)
        thr = 0.15
        g_bin = (G_hat > thr).astype(np.float32)
        gt_bin = (gt > thr).astype(np.float32)
        inter = float(np.sum(g_bin * gt_bin))
        union = float(np.sum(np.clip(g_bin + gt_bin, 0, 1)))
        metrics["g_mask_iou_thr_0p15"] = inter / max(union, 1e-12)

        # Soft Dice (weighted by label weight if available)
        dice_num = 2.0 * np.sum(g_bin * gt_bin)
        dice_den = np.sum(g_bin) + np.sum(gt_bin) + 1e-12
        metrics["g_mask_soft_dice"] = float(dice_num / dice_den)

    # ---- Component energy balance ----
    a_energy = float(np.mean(A_hat ** 2))
    s_energy = float(np.mean(S_hat ** 2))
    g_energy = float(np.mean(G_hat ** 2))
    total = a_energy + s_energy + g_energy + 1e-12
    metrics["a_energy_frac"] = a_energy / total
    metrics["s_energy_frac"] = s_energy / total
    metrics["g_energy_frac"] = g_energy / total
    metrics["residual_energy_frac"] = float(np.mean(residual ** 2)) / (total + 1e-12)

    return metrics


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────
def save_diagnostic_plot(
    out_dir: Path,
    eval_name: str,
    raw: np.ndarray,
    A_hat: np.ndarray,
    S_hat: np.ndarray,
    G_hat: np.ndarray,
    residual: np.ndarray,
    gt: np.ndarray | None,
    metrics: dict,
):
    """4-panel figure: A_hat, S_hat, G_hat, Residual."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ["A_hat (direct wave)", "S_hat (surface)", "G_hat (geological)", "Residual"]
    panels = [A_hat, S_hat, G_hat, residual]
    vmax_raw = float(np.nanpercentile(np.abs(raw), 98))

    for ax, title, panel in zip(axes, titles, panels):
        if title == "Residual":
            v = float(np.nanpercentile(np.abs(panel), 98))
            im = ax.imshow(
                panel, aspect="auto", origin="upper", cmap="RdBu_r",
                vmin=-v, vmax=v,
            )
        else:
            v = float(np.nanpercentile(np.abs(panel), 95))
            im = ax.imshow(
                panel, aspect="auto", origin="upper", cmap="viridis",
                vmin=0, vmax=max(v, 1e-6),
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Overlay ground-truth contour if available
    if gt is not None:
        # draw on G_hat panel (index 2)
        axes[2].contour(
            gt, levels=[0.15, 0.5], colors="w", linewidths=0.6, alpha=0.7,
        )

    fig.suptitle(
        f"GprMambaSep Separation — {eval_name}\n"
        f"Leakage={metrics.get('leakage_as_into_g_gate', float('nan')):.4f}  "
        f"G_IoU={metrics.get('g_mask_iou_thr_0p15', float('nan')):.4f}  "
        f"SNR_G={metrics.get('snr_g_late_gate_db', float('nan')):.1f} dB",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.10, top=0.82, wspace=0.35)
    path = out_dir / f"{eval_name}_separation_diagnostic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────
#  Main CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="GprMambaSep separation quality evaluator."
    )
    ap.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint (.pt) file.",
    )
    ap.add_argument(
        "--line", required=True,
        help="Line name to evaluate (e.g. Line9, LINE9_STYLE_001).",
    )
    ap.add_argument(
        "--data-root", default="",
        help="Override dataset root. Defaults to checkpoint cfg data_root.",
    )
    ap.add_argument(
        "--out-dir", default="outputs/eval_gprmambasep_sep",
        help="Output directory for metrics CSV and diagnostic plot.",
    )
    ap.add_argument(
        "--force-cpu", action="store_true",
        help="Run on CPU even when CUDA is available.",
    )
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = _WT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = _MAIN_REPO / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # ---- Stitch ----
    print(f"Stitching {args.line} ...", flush=True)
    result = _stitch_one(
        ckpt_path, args.line, args.data_root, device,
    )
    A_hat, S_hat, G_hat, residual, raw, gt, cfg = result
    print(
        f"  Shapes: A={A_hat.shape}, S={S_hat.shape}, "
        f"G={G_hat.shape}, residual={residual.shape}",
        flush=True,
    )

    # ---- Metrics ----
    metrics = compute_separation_metrics(A_hat, S_hat, G_hat, residual, raw, gt)
    print("── Separation Metrics ──")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ---- Save outputs ----
    out_dir = _WT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Arrays
    np.save(out_dir / f"{args.line}_A_hat.npy", A_hat)
    np.save(out_dir / f"{args.line}_S_hat.npy", S_hat)
    np.save(out_dir / f"{args.line}_G_hat.npy", G_hat)
    np.save(out_dir / f"{args.line}_residual.npy", residual)

    # Metrics CSV
    csv_path = out_dir / f"{args.line}_separation_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")
    print(f"  Metrics: {csv_path}")

    # Diagnostic plot
    plot_path = save_diagnostic_plot(
        out_dir, args.line, raw, A_hat, S_hat, G_hat, residual, gt, metrics,
    )
    print(f"  Plot: {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
