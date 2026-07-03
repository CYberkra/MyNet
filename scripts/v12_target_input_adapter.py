from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.model_raw_unet import build_model, compress_raw  # noqa: E402
from scripts.eval_full_line import (  # noqa: E402
    add_terrain_channels,
    normalize_raw_channel_4d,
    unpack_model_output,
    resolve_data_root,
    write_centerline_csv,
    write_metrics,
)


LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]


class InputAdapter(nn.Module):
    """Small target-line adapter applied to the compressed raw channel only."""

    def __init__(self, height: int):
        super().__init__()
        self.scalar_log_gain = nn.Parameter(torch.zeros(1))
        self.scalar_bias = nn.Parameter(torch.zeros(1))
        self.depth_log_gain = nn.Parameter(torch.zeros(1, 1, height, 1))
        self.depth_bias = nn.Parameter(torch.zeros(1, 1, height, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x[:, :1]
        gain = torch.exp(torch.clamp(self.scalar_log_gain, -0.15, 0.15) + 0.08 * torch.tanh(self.depth_log_gain))
        bias = torch.clamp(self.scalar_bias, -0.08, 0.08) + 0.05 * torch.tanh(self.depth_bias)
        out = x.clone()
        out[:, :1] = torch.clamp(raw * gain + bias, -1.5, 1.5)
        return out

    def regularization(self) -> torch.Tensor:
        return (
            self.scalar_log_gain.square().mean()
            + self.scalar_bias.square().mean()
            + self.depth_log_gain.square().mean()
            + self.depth_bias.square().mean()
        )


def load_checkpoint(run_dir: Path, checkpoint: str, device: torch.device):
    ckpt_path = run_dir / f"checkpoint_{checkpoint}.pt"
    if not ckpt_path.exists() and checkpoint == "final":
        ckpt_path = run_dir / "checkpoint_last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ckpt_path


def line_rows(data_root: Path, line: str) -> list[dict[str, str]]:
    return [r for r in csv.DictReader(open(data_root / "window_index.csv", encoding="utf-8")) if r["line"] == line]


def make_window_tensor(raw: np.ndarray, row: dict[str, str], cfg: dict, data_root: Path, device: torch.device) -> torch.Tensor:
    s = int(row["start"])
    e = int(row["end"]) + 1
    h = int(cfg["height_resize"])
    w = int(cfg["width_resize"])
    x = torch.from_numpy(raw[:, s:e][None, None]).float().to(device)
    x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
    x = compress_raw(x, cfg.get("input_log_scale", 1e-3))
    x = normalize_raw_channel_4d(x, cfg)
    x = add_terrain_channels(x, row["line"], s, e, cfg, data_root)
    return x


def augment_for_consistency(x: torch.Tensor, noise_std: float = 0.006) -> torch.Tensor:
    y = x.clone()
    scale = torch.empty((x.shape[0], 1, 1, 1), device=x.device).uniform_(0.94, 1.06)
    y[:, :1] = y[:, :1] * scale + torch.randn_like(y[:, :1]) * noise_std
    if y.shape[-1] > 8:
        drop = (torch.rand((y.shape[0], 1, 1, y.shape[-1]), device=y.device) > 0.015).float()
        y[:, :1] = y[:, :1] * drop
    return y


def model_probs(model: nn.Module, x: torch.Tensor):
    mask_logits, pres_logits, center_logits = unpack_model_output(model(x))
    mask = torch.sigmoid(mask_logits)
    pres = torch.sigmoid(pres_logits)
    center = torch.sigmoid(center_logits) if center_logits is not None else mask
    return mask, pres, center


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ((pred - target).square() * weight).sum() / weight.sum().clamp_min(1e-6)


def adapt_line(
    model: nn.Module,
    cfg: dict,
    data_root: Path,
    line: str,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    raw = np.load(data_root / "lines" / f"{line}.npz")["raw_full_normalized"].astype(np.float32)
    rows = line_rows(data_root, line)
    adapter = InputAdapter(int(cfg["height_resize"])).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    model.eval()
    for step in range(int(steps)):
        idx = rng.choice(len(rows), size=min(batch_size, len(rows)), replace=len(rows) < batch_size)
        xb = torch.cat([make_window_tensor(raw, rows[int(i)], cfg, data_root, device) for i in idx], dim=0)
        with torch.no_grad():
            teach_mask, teach_pres, teach_center = model_probs(model, xb)
        xa = augment_for_consistency(xb)
        xb_ad = adapter(xb)
        xa_ad = adapter(xa)
        mask, pres, center = model_probs(model, xb_ad)
        mask_aug, pres_aug, center_aug = model_probs(model, xa_ad)

        teacher_conf = torch.maximum(teach_mask, 1.0 - teach_mask).detach()
        teacher_w = 0.08 + 0.92 * (teacher_conf > 0.72).float()
        teacher_loss = weighted_mse(mask, teach_mask, teacher_w)
        center_loss = weighted_mse(center, teach_center, teacher_w)
        pres_loss = F.mse_loss(pres, teach_pres)
        cons_loss = F.mse_loss(mask, mask_aug) + 0.5 * F.mse_loss(center, center_aug) + 0.25 * F.mse_loss(pres, pres_aug)
        entropy = -(mask.clamp(1e-5, 1 - 1e-5) * torch.log(mask.clamp(1e-5, 1 - 1e-5)) + (1 - mask).clamp(1e-5, 1 - 1e-5) * torch.log((1 - mask).clamp(1e-5, 1 - 1e-5))).mean()
        loss = 0.55 * teacher_loss + 0.20 * center_loss + 0.15 * pres_loss + 0.75 * cons_loss + 0.015 * entropy + 0.35 * adapter.regularization()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            losses.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "teacher_loss": float(teacher_loss.detach().cpu()),
                    "cons_loss": float(cons_loss.detach().cpu()),
                    "adapter_reg": float(adapter.regularization().detach().cpu()),
                }
            )
    return adapter.eval(), losses


@torch.no_grad()
def stitch_with_adapter(model: nn.Module, adapter: InputAdapter, cfg: dict, data_root: Path, line: str, device: torch.device):
    z = np.load(data_root / "lines" / f"{line}.npz")
    raw = z["raw_full_normalized"].astype(np.float32)
    h0, w0 = raw.shape
    pred_sum = np.zeros((h0, w0), np.float32)
    weight_sum = np.zeros((h0, w0), np.float32)
    center_sum = np.zeros((h0, w0), np.float32)
    center_wsum = np.zeros((h0, w0), np.float32)
    pres_sum = np.zeros((w0,), np.float32)
    pres_wsum = np.zeros((w0,), np.float32)
    rows = line_rows(data_root, line)
    for row in rows:
        s = int(row["start"])
        e = int(row["end"]) + 1
        x = make_window_tensor(raw, row, cfg, data_root, device)
        x = adapter(x)
        logits, pres_logits, center_logits = unpack_model_output(model(x))
        p = torch.sigmoid(logits)
        pp = torch.sigmoid(pres_logits)
        cp = torch.sigmoid(center_logits) if center_logits is not None else None
        p0 = F.interpolate(p, (h0, e - s), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        pp0 = F.interpolate(pp, size=e - s, mode="linear", align_corners=False)[0, 0].detach().cpu().numpy()
        cp0 = F.interpolate(cp, (h0, e - s), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy() if cp is not None else None
        ww = np.hanning(e - s).astype(np.float32)
        if ww.max() > 0:
            ww = ww / ww.max()
        ww = 0.15 + 0.85 * ww
        w2 = np.broadcast_to(ww[None, :], p0.shape).astype(np.float32)
        pred_sum[:, s:e] += p0 * w2
        weight_sum[:, s:e] += w2
        if cp0 is not None:
            center_sum[:, s:e] += cp0 * w2
            center_wsum[:, s:e] += w2
        pres_sum[s:e] += pp0 * ww
        pres_wsum[s:e] += ww
    pred = pred_sum / np.maximum(weight_sum, 1e-6)
    center = center_sum / np.maximum(center_wsum, 1e-6) if center_wsum.max() > 0 else None
    pres = pres_sum / np.maximum(pres_wsum, 1e-6)
    return pred.astype(np.float32), pres.astype(np.float32), center.astype(np.float32) if center is not None else None


def evaluate_arrays(out_dir: Path, line: str, pred: np.ndarray, pres: np.ndarray, center: np.ndarray | None, data_root: Path, args):
    z = np.load(data_root / "lines" / f"{line}.npz")
    gt = z["soft_mask_train"].astype(np.float32)
    label_w = z["label_weight"].astype(np.float32)
    status = z["status_code"].astype(np.int16)
    dt_ns = float(z["dt_ns"])
    fusion_w = max(0.0, min(1.0, float(args.center_fusion_weight)))
    path = pred
    curve_source = "mask_dp"
    if center is not None and fusion_w > 0:
        path = ((1.0 - fusion_w) * pred + fusion_w * center).astype(np.float32)
        curve_source = f"mask_center_fusion_{fusion_w:.2f}_dp"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{line}_pred_softmask.npy", pred)
    np.save(out_dir / f"{line}_presence_prob.npy", pres)
    if center is not None:
        np.save(out_dir / f"{line}_center_softmask.npy", center)
    if path is not pred:
        np.save(out_dir / f"{line}_path_softmask.npy", path)
    cmean, vmean, cdp, vdp, cgt, vgt, path_prob = write_centerline_csv(
        out_dir,
        line,
        path,
        pres,
        gt,
        dt_ns,
        args.search_min_ns,
        args.search_max_ns,
        args.presence_thr,
        args.path_prob_thr,
        0,
        args.dp_max_jump,
        args.dp_smooth_weight,
        True,
        args.dp_min_segment,
    )
    write_metrics(
        out_dir,
        line,
        pred,
        pres,
        gt,
        status,
        label_w,
        dt_ns,
        cmean,
        vmean,
        cdp,
        vdp,
        cgt,
        vgt,
        path_prob,
        args.presence_thr,
        args.path_prob_thr,
        0,
        pred.shape[1] - 1,
        args.dp_max_jump,
        args.dp_smooth_weight,
        curve_source,
        True,
        args.dp_min_segment,
    )


def read_metric(path: Path, name: str) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] == name:
                return float(row["value"])
    return float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_one(line: str, args, device: torch.device):
    run_dir = ROOT / f"outputs/run_gpu_paper_v1_10d_{line}_sourceval_v19d_baseline"
    model, cfg, ckpt_path = load_checkpoint(run_dir, args.checkpoint, device)
    data_root = resolve_data_root(args.data_root, cfg)
    adapter, losses = adapt_line(model, cfg, data_root, line, device, args.steps, args.batch_size, args.lr, args.seed + LINES.index(line))
    out_dir = ROOT / args.out_dir / f"{line}_adapted_p{int(round(args.path_prob_thr * 100)):03d}"
    pred, pres, center = stitch_with_adapter(model, adapter, cfg, data_root, line, device)
    evaluate_arrays(out_dir, line, pred, pres, center, data_root, args)
    write_csv(out_dir / "adaptation_losses.csv", losses)
    torch.save(
        {
            "line": line,
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "adapter_state": adapter.state_dict(),
            "args": vars(args),
        },
        out_dir / "input_adapter.pt",
    )
    metrics_path = out_dir / f"{line}_full_metrics.csv"
    return {
        "line": line,
        "eval_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "mae_ns": read_metric(metrics_path, "dp_center_mae_ns"),
        "pick_rate": read_metric(metrics_path, "final_pick_rate"),
        "mean_center_mae_ns": read_metric(metrics_path, "mean_center_mae_ns"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="*", default=LINES)
    ap.add_argument("--checkpoint", choices=["best", "last", "final"], default="last")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out-dir", default="outputs/eval_paper_v1_12_target_input_adapter")
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.035)
    ap.add_argument("--seed", type=int, default=1212)
    ap.add_argument("--presence-thr", type=float, default=0.45)
    ap.add_argument("--path-prob-thr", type=float, default=0.20)
    ap.add_argument("--center-fusion-weight", type=float, default=0.5)
    ap.add_argument("--search-min-ns", type=float, default=320.0)
    ap.add_argument("--search-max-ns", type=float, default=560.0)
    ap.add_argument("--dp-max-jump", type=int, default=6)
    ap.add_argument("--dp-smooth-weight", type=float, default=0.16)
    ap.add_argument("--dp-min-segment", type=int, default=16)
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for line in args.lines:
        print(f"v1.12 adapting {line} on {device}", flush=True)
        rows.append(run_one(line, args, device))
    report_dir = ROOT / "reports" / "v1_12_target_input_adapter"
    write_csv(report_dir / "v1_12_target_input_adapter_summary.csv", rows)
    lines = [
        "# PGDA-CSNet v1.12 Target Input Adapter",
        "",
        "## Purpose",
        "",
        "Unsupervised target-line input adaptation with frozen v1.9D source-validation models. Target labels are used only for final evaluation.",
        "",
        "## Results",
        "",
        "| Line | MAE ns | Pick rate | Eval dir |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['line']} | {row['mae_ns']:.3f} | {row['pick_rate']:.3f} | `{row['eval_dir']}` |")
    mae = np.array([r["mae_ns"] for r in rows], dtype=np.float32)
    pick = np.array([r["pick_rate"] for r in rows], dtype=np.float32)
    lines.extend(
        [
            "",
            f"Average MAE: `{float(np.nanmean(mae)):.3f} ns`",
            f"Average pick rate: `{float(np.nanmean(pick)):.3f}`",
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{(report_dir / 'v1_12_target_input_adapter_summary.csv').relative_to(ROOT).as_posix()}`",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "V1_12_TARGET_INPUT_ADAPTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(report_dir / "V1_12_TARGET_INPUT_ADAPTER_REPORT.md")


if __name__ == "__main__":
    main()
