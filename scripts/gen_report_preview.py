"""Comprehensive report preview: network prediction vs ground truth comparison for Line9 holdout."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path("D:/Claude/PGDA-CSNet/workspace/transfer_20260627_142748/PGDA-CSNet_transfer_bundle_20260627_142748/PGDA_CSNet_v0_9_6_SEARCH_WINDOW_GUARD")
eval_dir = ROOT / "outputs" / "eval_v3_pilot_mixed_3seed_ensemble"

# ── Load data ──
line_npz = np.load(ROOT / "data_corrected_v1_4_terrain_direction/lines/Line9.npz")
raw = line_npz["raw_full_normalized"].astype(np.float32)
gt_mask = line_npz["soft_mask_train"].astype(np.float32)
status = line_npz["status_code"].astype(np.int16)
label_w = line_npz["label_weight"].astype(np.float32)
dt_ns = float(line_npz["dt_ns"])

start, end = 1664, 2377
sl = slice(start, end + 1)
raw_sl = raw[:, sl]
gt_sl = gt_mask[:, sl]
label_w_sl = label_w[sl]
status_sl = status[sl]
n_traces = raw_sl.shape[1]
n_samples = raw_sl.shape[0]

# Prediction
pred_mask = np.load(eval_dir / "Line9_holdout_tr1664_2377_pred_softmask.npy").astype(np.float32)
pres_pred = np.load(eval_dir / "Line9_holdout_tr1664_2377_presence_prob.npy").astype(np.float32)
center_pred = np.load(eval_dir / "Line9_holdout_tr1664_2377_center_softmask.npy").astype(np.float32)
path_pred = np.load(eval_dir / "Line9_holdout_tr1664_2377_path_softmask.npy").astype(np.float32)

# ── Centerline extraction helpers ──
def centerline(arr, min_sum=1e-4):
    H, W = arr.shape
    ys = np.arange(H, dtype=np.float32)[:, None]
    s = arr.sum(axis=0)
    c = (arr * ys).sum(axis=0) / np.maximum(s, 1e-6)
    valid = s > min_sum
    c[~valid] = np.nan
    return c, valid

def dp_ridge_centerline(prob, max_jump=8, smooth_weight=0.08, search_min=None, search_max=None):
    H, W = prob.shape
    p = np.clip(prob.astype(np.float32), 1e-6, 1.0)
    unary = -np.log(p)
    if search_min is not None or search_max is not None:
        lo = 0 if search_min is None else max(0, int(search_min))
        hi = H - 1 if search_max is None else min(H - 1, int(search_max))
        mask = np.ones(H, dtype=bool)
        mask[lo:hi + 1] = False
        unary[mask, :] += 20.0
    dp = np.empty((H, W), np.float32)
    back = np.zeros((H, W), np.int16)
    dp[:, 0] = unary[:, 0]
    offsets = np.arange(-max_jump, max_jump + 1, dtype=np.int16)
    big = np.float32(1e6)
    for x in range(1, W):
        prev = dp[:, x - 1]
        cand = np.full((len(offsets), H), big, dtype=np.float32)
        for oi, off in enumerate(offsets):
            penalty = np.float32(smooth_weight * (int(off) ** 2))
            if off < 0:
                cand[oi, -off:] = prev[:off] + penalty
            elif off > 0:
                cand[oi, :-off] = prev[off:] + penalty
            else:
                cand[oi, :] = prev + penalty
        arg = np.argmin(cand, axis=0).astype(np.int16)
        best = cand[arg, np.arange(H)]
        dp[:, x] = unary[:, x] + best
        predecessor = np.arange(H, dtype=np.int32) + offsets[arg].astype(np.int32)
        back[:, x] = np.clip(predecessor, 0, H - 1).astype(np.int16)
    path = np.zeros(W, np.float32)
    y = int(np.argmin(dp[:, W - 1]))
    path[W - 1] = y
    for x in range(W - 1, 0, -1):
        y = int(back[y, x])
        path[x - 1] = y
    valid = np.ones(W, dtype=bool)
    return path, valid

# ── GT centerline ──
gt_center, gt_center_valid = centerline(gt_sl, 1e-3)

# ── DP predicted centerline ──
presence_thr = 0.45
path_prob_thr = 0.20
min_segment = 16
max_jump = 8
smooth_weight = 0.08
search_min = int(round(320.0 / dt_ns))
search_max = int(round(560.0 / dt_ns))

# breakable DP
local_peak = np.nanmax(path_pred[search_min:search_max + 1, :], axis=0)
gate = (pres_pred >= presence_thr) & (local_peak >= path_prob_thr)
dp_path = np.full(n_traces, np.nan, np.float32)
dp_valid = np.zeros(n_traces, dtype=bool)
seg_start = None
for i, ok in enumerate(np.r_[gate, False]):
    if ok and seg_start is None:
        seg_start = i
    if (not ok) and seg_start is not None:
        seg_end = i
        if seg_end - seg_start >= int(min_segment):
            sub_path, sub_valid = dp_ridge_centerline(
                path_pred[:, seg_start:seg_end],
                max_jump=max_jump, smooth_weight=smooth_weight,
                search_min=search_min, search_max=search_max)
            dp_path[seg_start:seg_end] = sub_path
            dp_valid[seg_start:seg_end] = sub_valid
        seg_start = None

# Compute per-trace error
both_valid = gt_center_valid & dp_valid & np.isfinite(dp_path)
per_trace_error = np.full(n_traces, np.nan, np.float32)
per_trace_error[both_valid] = np.abs(dp_path[both_valid] - gt_center[both_valid])

# ── Stats ──
mae = float(np.nanmean(per_trace_error))
mae_ns = mae * dt_ns
pick_rate = float(dp_valid.mean())
reject_rate = 1.0 - pick_rate

iou_thrs = {}
for thr in [0.2, 0.3, 0.5]:
    pb = pred_mask >= thr
    gb = gt_sl >= 0.1  # standard GT binarization
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    iou_thrs[f"IoU>{thr}"] = inter / max(union, 1)

# ── Figure: large comprehensive report ──
fig = plt.figure(figsize=(22, 16))
gs = gridspec.GridSpec(5, 4, figure=fig,
                       height_ratios=[0.5, 0.8, 0.8, 0.8, 0.8],
                       hspace=0.30, wspace=0.28)

# ── Title / Header ──
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis("off")
ax_title.text(0.5, 0.5,
    f"PGDA-CSNet v1.9D MambaVision Hybrid — 3-Seed Ensemble Evaluation\n"
    f"Line9 holdout (trace {start}-{end})  |  "
    f"Pick Rate: {pick_rate*100:.1f}%  |  "
    f"DP MAE: {mae:.2f} sample ({mae_ns:.2f} ns)  |  "
    f"Ensemble: seed1901/1902/1903",
    ha="center", va="center", fontsize=14, fontweight="bold",
    transform=ax_title.transAxes)

# ── 1. Input B-scan ──
ax1 = fig.add_subplot(gs[1, 0])
v = np.nanpercentile(np.abs(raw_sl), 98)
extent = (start, end, n_samples, 0)
ax1.imshow(raw_sl, aspect="auto", origin="upper", extent=extent,
           cmap="gray", vmin=-v, vmax=v)
ax1.set_title("(a) Input B-scan (raw)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Trace #")
ax1.set_ylabel("Sample")

# ── 2. Ground Truth ──
ax2 = fig.add_subplot(gs[1, 1])
im2 = ax2.imshow(gt_sl, aspect="auto", origin="upper", extent=extent,
                 cmap="viridis", vmin=0, vmax=max(0.6, float(gt_sl.max())))
ax2.set_title("(b) Ground Truth (soft mask)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Trace #")
ax2.set_ylabel("Sample")
plt.colorbar(im2, ax=ax2, fraction=0.046)

# ── 3. Prediction ──
ax3 = fig.add_subplot(gs[1, 2])
im3 = ax3.imshow(pred_mask, aspect="auto", origin="upper", extent=extent,
                 cmap="viridis", vmin=0, vmax=max(0.6, float(pred_mask.max())))
ax3.set_title("(c) Predicted (3-seed ensemble)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Trace #")
ax3.set_ylabel("Sample")
plt.colorbar(im3, ax=ax3, fraction=0.046)

# ── 4. Error map ──
ax4 = fig.add_subplot(gs[1, 3])
error = pred_mask - gt_sl
err_lim = max(abs(error).max(), 0.05)
im4 = ax4.imshow(error, aspect="auto", origin="upper", extent=extent,
                 cmap="RdBu_r", vmin=-err_lim, vmax=err_lim)
ax4.set_title("(d) Error (pred - GT)", fontsize=11, fontweight="bold")
ax4.set_xlabel("Trace #")
ax4.set_ylabel("Sample")
plt.colorbar(im4, ax=ax4, fraction=0.046)

# ── 5. Centerline overlay: GT vs Predicted ──
ax5 = fig.add_subplot(gs[2, :2])
ax5.imshow(raw_sl, aspect="auto", origin="upper", extent=extent,
           cmap="gray", vmin=-v, vmax=v)
ax5.imshow(path_pred, aspect="auto", origin="upper", extent=extent,
           cmap="magma", alpha=np.clip(path_pred * 0.75, 0, 0.60))
xcoords = np.arange(start, end + 1)
ax5.plot(xcoords, gt_center, linewidth=1.5, color="cyan", label="GT centerline", alpha=0.85)
ax5.plot(xcoords, dp_path, linewidth=1.5, color="lime", label=f"DP pred (PR={pick_rate*100:.0f}%)", alpha=0.85)
ax5.set_title("(e) Centerline overlay: GT(cyan) vs Pred(lime) on B-scan + prediction", fontsize=11, fontweight="bold")
ax5.set_xlabel("Trace #")
ax5.set_ylabel("Sample")
ax5.legend(fontsize=9, loc="lower right")

# ── 6. Error per trace (centerline) ──
ax6 = fig.add_subplot(gs[2, 2])
colors = np.where(both_valid, "green", "red")
ax6.bar(xcoords, per_trace_error, width=1.0, color="green", alpha=0.6, label=f"MAE={mae:.2f} smp")
ax6.axhline(mae, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mae:.2f}")
ax6.set_title("(f) Per-trace centerline |error| (sample)", fontsize=11, fontweight="bold")
ax6.set_xlabel("Trace #")
ax6.set_ylabel("|error| (sample)")
ax6.set_ylim(0, max(np.nanmax(per_trace_error) * 1.15, 5))
ax6.legend(fontsize=9)

# ── 7. Presence probability ──
ax7 = fig.add_subplot(gs[2, 3])
status_str = np.where(status_sl == 1, "Present",
                       np.where(status_sl == 2, "Weak", "No signal"))
for st in [1, 2]:
    idx = status_sl == st
    ax7.scatter(xcoords[idx], pres_pred[idx], s=3, alpha=0.5,
                label=f"{status_str[idx][0] if idx.any() else ''} ({st})")
idx0 = status_sl == 0
if idx0.any():
    ax7.scatter(xcoords[idx0], pres_pred[idx0], s=3, alpha=0.4, c="gray", label="No signal")
ax7.axhline(presence_thr, color="red", linestyle="--", alpha=0.7, label=f"thr={presence_thr}")
ax7.set_ylim(-0.02, 1.02)
ax7.set_title("(g) Presence prediction by status", fontsize=11, fontweight="bold")
ax7.set_xlabel("Trace #")
ax7.set_ylabel("Presence prob.")
ax7.legend(fontsize=8, markerscale=3)

# ── 8. Metrics table ──
ax8 = fig.add_subplot(gs[3, :])
ax8.axis("off")
metrics_text = (
    f"{'='*60}\n"
    f"  Key Metrics Summary (vs P0-3 Baseline)\n"
    f"{'='*60}\n"
    f"  {'Metric':<30s} {'v1.9D Ensemble':<16s} {'P0-3':<10s} {'Change':<10s}\n"
    f"  {'-'*66}\n"
    f"  {'DP Center MAE (sample)':<30s} {mae:<10.2f}        {'3.27':<10s} {mae - 3.27:<+7.2f}\n"
    f"  {'DP Center MAE (ns)':<30s} {mae_ns:<10.2f}        {'4.58':<10s} {mae_ns - 4.58:<+7.2f}\n"
    f"  {'Pick Rate (%)':<30s} {pick_rate*100:<10.1f}        {'56.2':<10s} {pick_rate*100 - 56.2:<+7.1f}\n"
    f"  {'Reject Rate (%)':<30s} {reject_rate*100:<10.1f}        {'43.8':<10s} {reject_rate*100 - 43.8:<+7.1f}\n"
    f"\n"
    f"  {'Soft Dice (weighted)':<30s} {'-':<16s} {'0.304':<10s}\n"
    f"  {'IoU @ thr=0.2':<30s} {iou_thrs['IoU>0.2']:<16.3f} {'0.311':<10s}\n"
    f"  {'IoU @ thr=0.3':<30s} {iou_thrs['IoU>0.3']:<16.3f} {'0.276':<10s}\n"
    f"  {'IoU @ thr=0.5':<30s} {iou_thrs['IoU>0.5']:<16.3f} {'0.217':<10s}\n"
    f"\n"
    f"  Presence thr={presence_thr}, Path prob thr={path_prob_thr}, DP max_jump={max_jump},\n"
    f"  smooth_w={smooth_weight}, breakable segments min={min_segment}\n"
)
ax8.text(0.02, 0.95, metrics_text, ha="left", va="top", fontsize=9,
         fontfamily="monospace", transform=ax8.transAxes)

# ── 9. Zoom: first 200 traces detail ──
zoom_start, zoom_end = start, min(start + 200, end)
z_sl = slice(zoom_start - start, zoom_end - start + 1)
ax9 = fig.add_subplot(gs[3, 2:])
ax9.imshow(raw_sl[:, z_sl], aspect="auto", origin="upper",
           extent=(zoom_start, zoom_end, n_samples, 0),
           cmap="gray", vmin=-v, vmax=v)
ax9.imshow(path_pred[:, z_sl], aspect="auto", origin="upper",
           extent=(zoom_start, zoom_end, n_samples, 0),
           cmap="magma", alpha=np.clip(path_pred[:, z_sl] * 0.75, 0, 0.60))
x_zoom = np.arange(zoom_start, zoom_end + 1)
ax9.plot(x_zoom, gt_center[z_sl.start:z_sl.stop][:len(x_zoom)],
         linewidth=1.8, color="cyan", label="GT", alpha=0.9)
ax9.plot(x_zoom, dp_path[z_sl.start:z_sl.stop][:len(x_zoom)],
         linewidth=1.8, color="lime", label="Pred", alpha=0.9)
ax9.set_title(f"(h) Zoom: traces {zoom_start}-{zoom_end}", fontsize=11, fontweight="bold")
ax9.set_xlabel("Trace #")
ax9.set_ylabel("Sample")
ax9.legend(fontsize=9, loc="upper right")

# ── 10. Histogram of error ──
ax10 = fig.add_subplot(gs[4, 0])
valid_err = per_trace_error[both_valid]
if len(valid_err) > 0:
    ax10.hist(valid_err, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax10.axvline(mae, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mae:.2f}")
    ax10.axvline(np.median(valid_err), color="orange", linestyle=":",
                 linewidth=1.5, label=f"Median={np.median(valid_err):.2f}")
ax10.set_title("(i) Centerline |error| distribution", fontsize=11, fontweight="bold")
ax10.set_xlabel("|error| (sample)")
ax10.set_ylabel("Count")
ax10.legend(fontsize=9)

# ── 11. Presence ROC-like ──
ax11 = fig.add_subplot(gs[4, 1])
# Categorize status into present (1) vs not present (0/2)
present_mask = status_sl == 1
weak_mask = status_sl == 2
no_signal_mask = status_sl == 0
# For traces where DP picked successfully
picked_traces = dp_valid
missed_traces = ~dp_valid & (pres_pred >= presence_thr)
rejected_by_presence = pres_pred < presence_thr
# Counts
n_present = present_mask.sum()
n_picked = picked_traces.sum()
n_missed = missed_traces.sum()
n_rejected = rejected_by_presence.sum()
categories = ["Correct\nPicked", "Correct\nRejected", "False\nMissed", "False\nPicked"]
# Simplified: correct pick = dp_valid & gt_center_valid
correct_pick = both_valid.sum()
false_pick = dp_valid.sum() - correct_pick
correct_reject = reject_rate * n_traces - false_pick  # simplified
false_reject = n_traces - dp_valid.sum() - correct_reject
vals = [correct_pick, max(0, int(correct_reject)), max(0, int(false_reject)), max(0, false_pick)]
bar_colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
bars = ax11.bar(categories, vals, color=bar_colors, alpha=0.8, edgecolor="white")
for bar, val in zip(bars, vals):
    ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
              str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax11.set_title("(j) Pick quality breakdown", fontsize=11, fontweight="bold")
ax11.set_ylabel("Trace count")

# ── 12. IoU bar chart ──
ax12 = fig.add_subplot(gs[4, 2])
iou_baseline = [0.311, 0.276, 0.217]
iou_ours = [iou_thrs["IoU>0.2"], iou_thrs["IoU>0.3"], iou_thrs["IoU>0.5"]]
x = np.arange(3)
width = 0.3
bars1 = ax12.bar(x - width / 2, iou_baseline, width, label="P0-3", color="gray", alpha=0.6)
bars2 = ax12.bar(x + width / 2, iou_ours, width, label="v1.9D", color="steelblue", alpha=0.8)
ax12.set_xticks(x)
ax12.set_xticklabels(["IoU@0.2", "IoU@0.3", "IoU@0.5"])
ax12.set_title("(k) Mask IoU: P0-3 vs v1.9D", fontsize=11, fontweight="bold")
ax12.set_ylabel("IoU")
ax12.legend(fontsize=9)
for bar in bars2:
    h = bar.get_height()
    ax12.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
              f"{h:.3f}", ha="center", va="bottom", fontsize=8)

# ── 13. Spacer / Data info ──
ax13 = fig.add_subplot(gs[4, 3])
ax13.axis("off")
info_text = (
    f"Evaluation Info\n"
    f"{'='*30}\n"
    f"Line: Line9\n"
    f"Traces: {start} - {end} ({n_traces})\n"
    f"Samples/trace: {n_samples}\n"
    f"dt: {dt_ns:.2f} ns\n"
    f"\n"
    f"Model: v1.9D\n"
    f"MambaVision Hybrid\n"
    f"base_ch=20, ssm_k=31\n"
    f"heads=4, dropout=0.06\n"
    f"\n"
    f"Training:\n"
    f"v3_pilot_mixed\n"
    f"(130 real + 60 sim)\n"
    f"80 epochs, lr=5.5e-4\n"
    f"sim_batch_ratio=0.3\n"
    f"\n"
    f"Data: data_corrected\n"
    f"_v1_4_terrain_direction"
)
ax13.text(0.05, 0.95, info_text, ha="left", va="top", fontsize=8.5,
          fontfamily="monospace", transform=ax13.transAxes)

# ── Save ──
out_path = eval_dir / "Line9_holdout_tr1664_2377_report_preview.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
