from pathlib import Path
import argparse
import csv
import json
import re
import shutil
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.model_raw_unet import RawOnlyUNet, compress_raw


def preprocess_ez(ez):
    x = ez.astype(np.float32, copy=True)
    x -= np.nanmedian(x, axis=1, keepdims=True).astype(np.float32)
    scale = float(np.nanpercentile(np.abs(x), 99.5))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return np.clip(x / scale, -2.0, 2.0).astype(np.float32), scale


def parse_rx_positions(model_path, width):
    txt = Path(model_path).read_text(encoding="utf-8", errors="ignore")
    rx = re.search(r"^#rx:\s*([0-9.+-Ee]+)\s+([0-9.+-Ee]+)\s+([0-9.+-Ee]+)", txt, re.M)
    step = re.search(r"^#rx_steps:\s*([0-9.+-Ee]+)\s+([0-9.+-Ee]+)\s+([0-9.+-Ee]+)", txt, re.M)
    x0 = float(rx.group(1)) if rx else 0.2
    dx = float(step.group(1)) if step else 0.4
    return x0 + np.arange(width, dtype=np.float32) * dx


def read_merged(path):
    with h5py.File(path, "r") as f:
        ez = f["rxs/rx1/Ez"][:].astype(np.float32)
        dt_ns = float(f.attrs["dt"]) * 1e9
    return ez, dt_ns


def make_label(case_id, gpr_root, width, height, dt_ns, is_positive, v_eff_m_per_ns, sigma_ns):
    mask = np.zeros((height, width), np.float32)
    presence = np.zeros((width,), np.float32)
    if not is_positive:
        return mask, presence
    curve_path = gpr_root / "labels" / f"{case_id}_basal_curve.csv"
    model_path = gpr_root / "models" / f"{case_id}.in"
    curve = pd.read_csv(curve_path)
    trace_x = parse_rx_positions(model_path, width)
    depth = np.interp(trace_x, curve["x_m"].to_numpy(np.float32), curve["basal_depth_m"].to_numpy(np.float32))
    target_ns = 2.0 * depth / float(v_eff_m_per_ns)
    y = np.arange(height, dtype=np.float32)[:, None] * float(dt_ns)
    sigma = max(float(sigma_ns), float(dt_ns) * 2.0)
    mask = np.exp(-0.5 * ((y - target_ns[None, :]) / sigma) ** 2).astype(np.float32)
    mask[mask < 0.03] = 0.0
    presence[:] = 1.0
    return mask, presence


def load_base_checkpoint(run_dir, checkpoint, device):
    ckpt_path = Path(run_dir) / f"checkpoint_{checkpoint}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = RawOnlyUNet(cfg["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt_path, ckpt


def build_samples(gpr_root, cfg, device, v_eff, sigma_ns):
    manifest = pd.read_csv(gpr_root / "data" / "v074_label_manifest.csv")
    samples = []
    for _, row in manifest.iterrows():
        case_id = row["case_id"]
        out_path = gpr_root / "outputs_gprmax" / f"{case_id}_merged.out"
        ez, dt_ns = read_merged(out_path)
        raw, raw_scale = preprocess_ez(ez)
        mask, presence = make_label(
            case_id,
            gpr_root,
            raw.shape[1],
            raw.shape[0],
            dt_ns,
            bool(row["is_positive_gt"]),
            v_eff,
            sigma_ns,
        )
        h, w = int(cfg["height_resize"]), int(cfg["width_resize"])
        x = torch.from_numpy(raw[None, None]).float()
        y = torch.from_numpy(mask[None, None]).float()
        p = torch.from_numpy(presence[None, None]).float()
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)[0]
        x = compress_raw(x, cfg.get("input_log_scale", 1e-3)).to(device)
        y = F.interpolate(y, (h, w), mode="bilinear", align_corners=False)[0].to(device)
        p = F.interpolate(p, size=w, mode="nearest")[0].to(device)
        samples.append(
            {
                "case_id": case_id,
                "role": row["role"],
                "is_positive": bool(row["is_positive_gt"]),
                "x": x,
                "y": y,
                "presence": p,
                "raw_preview": raw,
                "mask_preview": mask,
                "dt_ns": dt_ns,
                "raw_scale": raw_scale,
            }
        )
    return samples


def loss_one(
    model,
    sample,
    pos_boost=80.0,
    positive_bg_weight=0.02,
    negative_bg_weight=0.65,
    neg_topk_weight=0.25,
    presence_weight=1.0,
):
    x = sample["x"][None]
    y = sample["y"][None]
    pres = sample["presence"][None]
    logits, pres_logits = model(x)
    prob = torch.sigmoid(logits)
    pix_w = positive_bg_weight + pos_boost * y
    if not sample["is_positive"]:
        pix_w = torch.full_like(y, negative_bg_weight)
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * pix_w).mean()
    if sample["is_positive"]:
        inter = (prob * y).sum()
        den = (prob + y).sum().clamp_min(1e-6)
        dice = 1.0 - 2.0 * inter / den
    else:
        dice = prob.mean() * 0.0
    bg = y < 0.03
    bg_prob = prob[bg]
    if bg_prob.numel():
        k = max(1, int(bg_prob.numel() * 0.02))
        hard_neg = torch.topk(bg_prob.flatten(), k).values.mean()
    else:
        hard_neg = prob.mean() * 0.0
    pres_loss = F.binary_cross_entropy_with_logits(pres_logits, pres)
    return bce + 0.6 * dice + neg_topk_weight * hard_neg + presence_weight * pres_loss


def evaluate(model, samples):
    model.eval()
    rows = []
    with torch.no_grad():
        for s in samples:
            logits, pres_logits = model(s["x"][None])
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pres = torch.sigmoid(pres_logits)[0, 0].cpu().numpy()
            y = s["y"][0].detach().cpu().numpy()
            pred_bin = prob >= 0.5
            gt_bin = y >= 0.25
            inter = np.logical_and(pred_bin, gt_bin).sum()
            union = np.logical_or(pred_bin, gt_bin).sum()
            rows.append(
                {
                    "case_id": s["case_id"],
                    "role": s["role"],
                    "is_positive": s["is_positive"],
                    "prob_mean": float(prob.mean()),
                    "prob_p95": float(np.percentile(prob, 95)),
                    "prob_max": float(prob.max()),
                    "presence_mean": float(pres.mean()),
                    "presence_max": float(pres.max()),
                    "area_prob_ge_050": float(pred_bin.mean()),
                    "iou_ge_050": float(inter / max(union, 1)) if s["is_positive"] else float("nan"),
                }
            )
    return rows


def save_previews(model, samples, out_dir, suffix):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for s in samples:
            logits, _ = model(s["x"][None])
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask = s["y"][0].detach().cpu().numpy()
            raw = s["x"][0].detach().cpu().numpy()
            fig, ax = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
            ax[0].imshow(raw, aspect="auto", cmap="gray")
            ax[0].set_title("input")
            ax[1].imshow(mask, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax[1].set_title("sim label")
            ax[2].imshow(prob, aspect="auto", cmap="magma", vmin=0, vmax=max(0.6, float(prob.max())))
            ax[2].set_title("prediction")
            ax[3].imshow(raw, aspect="auto", cmap="gray")
            ax[3].imshow(prob, aspect="auto", cmap="magma", alpha=np.clip(prob * 0.9, 0, 0.65), vmin=0, vmax=max(0.6, float(prob.max())))
            ax[3].set_title("overlay")
            fig.suptitle(f"{s['case_id']} {suffix}")
            fig.savefig(out_dir / f"{s['case_id']}_{suffix}.png", dpi=140)
            plt.close(fig)


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gprmax-root", required=True)
    ap.add_argument("--base-run-dir", default=str(ROOT / "outputs" / "run_gpu_final_line9dev"))
    ap.add_argument("--checkpoint", choices=["best", "last", "final"], default="final")
    ap.add_argument("--out-run-dir", default=str(ROOT / "outputs" / "run_gprmax_v074_adapter_v1"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--v-eff", type=float, default=0.074)
    ap.add_argument("--sigma-ns", type=float, default=9.0)
    ap.add_argument("--pos-boost", type=float, default=80.0)
    ap.add_argument("--positive-bg-weight", type=float, default=0.02)
    ap.add_argument("--negative-bg-weight", type=float, default=0.65)
    ap.add_argument("--neg-topk-weight", type=float, default=0.25)
    ap.add_argument("--presence-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_run_dir)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    model, cfg, ckpt_path, base_ckpt = load_base_checkpoint(args.base_run_dir, args.checkpoint, device)
    cfg = dict(cfg)
    cfg.update(
        {
            "run_dir": str(out_dir.relative_to(ROOT)) if out_dir.is_relative_to(ROOT) else str(out_dir),
            "adapter_source": "gprMax_v0.7.4_RELEASE_QC_LOCKED",
            "adapter_base_checkpoint": str(ckpt_path),
            "adapter_velocity_eff_m_per_ns": args.v_eff,
            "adapter_sigma_ns": args.sigma_ns,
            "adapter_pos_boost": args.pos_boost,
            "adapter_positive_bg_weight": args.positive_bg_weight,
            "adapter_negative_bg_weight": args.negative_bg_weight,
            "adapter_neg_topk_weight": args.neg_topk_weight,
            "adapter_presence_weight": args.presence_weight,
            "adapter_note": "Experimental gprMax-domain adapter; geometry labels are converted to time masks with an effective-velocity approximation.",
        }
    )
    samples = build_samples(Path(args.gprmax_root), cfg, device, args.v_eff, args.sigma_ns)
    baseline_rows = evaluate(model, samples)
    write_rows(out_dir / "logs" / "sim_metrics_before.csv", baseline_rows)
    save_previews(model, samples, out_dir / "previews", "before")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_state = None
    best_score = -1e9
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        order = np.random.permutation(len(samples))
        for idx in order:
            loss = loss_one(
                model,
                samples[int(idx)],
                pos_boost=args.pos_boost,
                positive_bg_weight=args.positive_bg_weight,
                negative_bg_weight=args.negative_bg_weight,
                neg_topk_weight=args.neg_topk_weight,
                presence_weight=args.presence_weight,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rows = evaluate(model, samples)
        pos_iou = np.nanmean([r["iou_ge_050"] for r in rows if r["is_positive"]])
        neg_area = np.mean([r["area_prob_ge_050"] for r in rows if not r["is_positive"]])
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "positive_iou_ge_050": float(pos_iou), "negative_area_ge_050": float(neg_area)}
        history.append(rec)
        print(rec, flush=True)
        score = float(pos_iou) - 0.5 * float(neg_area)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    after_rows = evaluate(model, samples)
    write_rows(out_dir / "logs" / "sim_metrics_after.csv", after_rows)
    save_previews(model, samples, out_dir / "previews", "after")
    json.dump(history, open(out_dir / "history.json", "w", encoding="utf-8"), indent=2)
    json.dump(cfg, open(out_dir / "used_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    torch.save({"model": model.state_dict(), "cfg": cfg, "history": history, "base_checkpoint": str(ckpt_path), "best_epoch": best_epoch, "best_score": best_score}, out_dir / "checkpoint_final.pt")
    shutil.copy2(out_dir / "checkpoint_final.pt", out_dir / "checkpoint_last.pt")
    shutil.copy2(out_dir / "checkpoint_final.pt", out_dir / "checkpoint_best.pt")
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
