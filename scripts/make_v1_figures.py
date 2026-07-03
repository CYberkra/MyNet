"""
PGDA-CSNet v1 结果综合可视化 — 生成论文级对比图
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"
OUT = ROOT / "outputs" / "v1_results_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ──

def bgr(x):
    return x - np.median(x, axis=1, keepdims=True)

def agc(x, win=121):
    win = max(3, int(win))
    win += 1 - win % 2
    half = win // 2
    xp = np.pad(x * x, ((half, half), (0, 0)), mode="edge")
    cs = np.concatenate([np.zeros((1, xp.shape[1]), dtype=xp.dtype), np.cumsum(xp, axis=0)], axis=0)
    e = (cs[win:win+x.shape[0]] - cs[:x.shape[0]]) / float(win)
    return np.clip(x / np.sqrt(e + 1e-12), -5, 5)

def load_line(line):
    data = np.load(DATA_ROOT / "lines" / f"{line}.npz")
    raw = data["raw_full_normalized"].astype(np.float32)
    mask = data["soft_mask_train"].astype(np.float32)
    dt_ns = float(data["dt_ns"])
    return raw, mask, dt_ns

def svd_rank1(raw):
    U, S, Vt = np.linalg.svd(raw, full_matrices=False)
    return raw - U[:, :1] @ np.diag(S[:1]) @ Vt[:1, :]

def rpca(raw, lam=None, max_iter=150):
    n, m = raw.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, m))
    L = np.zeros_like(raw); S = np.zeros_like(raw); Y = np.zeros_like(raw)
    mu = 1.0; rho = 1.5
    for _ in range(max_iter):
        X = raw - S + Y / mu
        U, sig, Vt = np.linalg.svd(X, full_matrices=False)
        L = U @ np.diag(np.maximum(sig - 1.0 / mu, 0)) @ Vt
        Z = raw - L + Y / mu
        S = np.sign(Z) * np.maximum(np.abs(Z) - lam / mu, 0)
        Y = Y + mu * (raw - L - S)
        mu = min(mu * rho, 1e10)
    return S

def dp_pick(image, dt_ns, t_start_ns=320, t_end_ns=560):
    nt, nx = image.shape
    t0 = max(0, int(t_start_ns / dt_ns))
    t1 = min(nt, int(t_end_ns / dt_ns))
    region = image[t0:t1, :]
    pmin, pmax = np.percentile(region, 5), np.percentile(region, 95)
    prob = np.clip((region - pmin) / (pmax - pmin + 1e-10), 0, 1)
    n_rows, n_cols = prob.shape
    dp = np.zeros_like(prob); bp = np.zeros_like(prob, dtype=np.int32)
    dp[:, 0] = prob[:, 0]
    for j in range(1, n_cols):
        for i in range(n_rows):
            best_v, best_k = -1, i
            for di in range(-6, 7):
                k = i + di
                if 0 <= k < n_rows and dp[k, j-1] > best_v:
                    best_v, best_k = dp[k, j-1], k
            dp[i, j] = prob[i, j] + best_v * 0.95
            bp[i, j] = best_k
    picks = np.full(n_cols, np.nan, dtype=np.float32)
    valid = np.zeros(n_cols, dtype=bool)
    j = n_cols - 1; i = int(np.argmax(dp[:, j]))
    while j >= 0:
        picks[j] = i + t0
        valid[j] = dp[i, j] > 0.15 * n_rows
        i = bp[i, j]; j -= 1
    return picks, valid

def gt_picks(mask, dt_ns):
    """Extract GT peak per trace from soft mask."""
    nt, nx = mask.shape
    picks = np.full(nx, np.nan, dtype=np.float32)
    valid = np.zeros(nx, dtype=bool)
    for j in range(nx):
        col = mask[:, j]
        if col.max() < 0.01:
            continue
        picks[j] = float(np.argmax(col))
        valid[j] = True
    return picks, valid


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: B-scan comparison — Raw vs SVD vs v1.9D (Line9)
# ═══════════════════════════════════════════════════════════════
print("=== Figure 1: B-scan comparison ===")
line = "Line9"
raw, gt_mask, dt_ns = load_line(line)
svd = svd_rank1(raw)
gt_p, gt_v = gt_picks(gt_mask, dt_ns)

fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=150)

# Holdout region only
tr_start, tr_end = 1664, 2377
raw_h = raw[:, tr_start:tr_end]
svd_h = svd[:, tr_start:tr_end]
gt_mask_h = gt_mask[:, tr_start:tr_end]
gt_p_h = gt_p[tr_start:tr_end] - 0  # already relative to full; we need relative
gt_v_h = gt_v[tr_start:tr_end]

# BG removed
raw_bg = bgr(raw_h)
svd_bg = bgr(svd_h)
gt_bg = bgr(gt_mask_h)

rv = np.percentile(np.abs(raw_bg), 99.5)
sv = np.percentile(np.abs(svd_bg), 99.5)

axs[0, 0].imshow(raw_bg, aspect="auto", cmap="seismic", vmin=-rv, vmax=rv)
axs[0, 0].set_title("Raw B-scan (BG removed)", fontsize=12)
axs[0, 0].set_ylabel("sample")

axs[0, 1].imshow(svd_bg, aspect="auto", cmap="seismic", vmin=-sv, vmax=sv)
axs[0, 1].set_title("SVD rank-1 subtracted (BG removed)", fontsize=12)

# GT overlay on raw
axs[1, 0].imshow(raw_bg, aspect="auto", cmap="seismic", vmin=-rv, vmax=rv, alpha=0.8)
gt_y = np.arange(raw_h.shape[1])
gt_x = gt_p_h.copy()
gt_x[~gt_v_h] = np.nan
axs[1, 0].plot(gt_y, gt_x, color="#2ecc71", linewidth=1.5, alpha=0.9, label="GT interface")
axs[1, 0].set_title("Raw + GT interface", fontsize=12)
axs[1, 0].set_ylabel("sample")
axs[1, 0].set_xlabel("trace")
axs[1, 0].legend(fontsize=10, loc="upper right")

# v1.9D model output
eval_dir = ROOT / "outputs" / "eval_line9_holdout_full_frozen_v19d"
pred_softmask = np.load(eval_dir / "Line9_holdout_tr1664_2377_pred_softmask.npy")
pm = np.percentile(np.abs(pred_softmask), 99.5) + 1e-12
axs[1, 1].imshow(pred_softmask, aspect="auto", cmap="magma", vmin=0, vmax=pm)
axs[1, 1].set_title("v1.9D model prediction", fontsize=12)
axs[1, 1].set_xlabel("trace")

for ax in axs.flat:
    ax.set_xticks(np.linspace(0, raw_h.shape[1]-1, 5))
    ax.set_xticklabels([f"{int(tr_start + x * (tr_end - tr_start) / (raw_h.shape[1]-1))}" for x in np.linspace(0, raw_h.shape[1]-1, 5)])

fig.suptitle(f"Line9 Holdout ({tr_start}–{tr_end}) — Raw vs SVD vs v1.9D", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig1_bscan_comparison_line9.png", bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT / 'fig1_bscan_comparison_line9.png'}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Confidence abstention effect (Line9)
# ═══════════════════════════════════════════════════════════════
print("=== Figure 2: Confidence abstention ===")

# Load v1.9D results
pred_cl = np.load(eval_dir / "Line9_holdout_tr1664_2377_pred_softmask.npy")
path_sm = np.load(eval_dir / "Line9_holdout_tr1664_2377_path_softmask.npy")
center_sm = np.load(eval_dir / "Line9_holdout_tr1664_2377_center_softmask.npy")

# Load centerline CSV
cl_rows = {}
with (eval_dir / "Line9_holdout_tr1664_2377_pred_centerline.csv").open("r") as f:
    for row in csv.DictReader(f):
        cl_rows[int(row["trace_idx"])] = row

n_traces = pred_cl.shape[1]
dp_valid = np.zeros(n_traces, dtype=bool)
dp_sample = np.full(n_traces, np.nan)
path_prob = np.full(n_traces, np.nan)
for idx, row in cl_rows.items():
    if 0 <= idx < n_traces:
        dp_valid[idx] = int(row["dp_valid"]) == 1
        dp_sample[idx] = float(row["dp_center_sample"]) if row["dp_center_sample"] else np.nan
        path_prob[idx] = float(row["dp_path_prob"]) if row["dp_path_prob"] else np.nan

# Score: simple composite
scores = path_prob.copy()
scores[~dp_valid] = 0

# Threshold for 50% coverage
thr_50 = np.nanpercentile(scores[dp_valid], 50) if dp_valid.any() else 0.5
accepted_50 = dp_valid & (scores >= thr_50)
thr_80 = np.nanpercentile(scores[dp_valid], 20) if dp_valid.any() else 0.3
accepted_80 = dp_valid & (scores >= thr_80)

fig, axs = plt.subplots(3, 1, figsize=(16, 10), dpi=150, sharex=True)

# Top: all picks
axs[0].imshow(pred_cl, aspect="auto", cmap="magma", vmin=0, vmax=np.percentile(pred_cl, 99))
x_all = np.arange(n_traces)
y_all = dp_sample.copy(); y_all[~dp_valid] = np.nan
axs[0].plot(x_all, y_all, color="#ecf0f1", linewidth=0.8, alpha=0.7, label="All DP picks")
axs[0].set_title("All DP picks (picking_rate=100%)", fontsize=12)
axs[0].set_ylabel("sample")
axs[0].legend(fontsize=10)

# Middle: 80% coverage
axs[1].imshow(pred_cl, aspect="auto", cmap="magma", vmin=0, vmax=np.percentile(pred_cl, 99))
y_80 = dp_sample.copy(); y_80[~accepted_80] = np.nan
y_rej80 = dp_sample.copy(); y_rej80[accepted_80 | ~dp_valid] = np.nan
axs[1].plot(x_all, y_80, color="#2ecc71", linewidth=1.2, alpha=0.9, label=f"Kept (80% cov)")
axs[1].plot(x_all, y_rej80, color="#e74c3c", linewidth=0.5, alpha=0.4, label="Rejected")
axs[1].set_title(f"Confidence abstention (coverage≈80%, threshold={thr_80:.3f})", fontsize=12)
axs[1].set_ylabel("sample")
axs[1].legend(fontsize=10)

# Bottom: 50% coverage
axs[2].imshow(pred_cl, aspect="auto", cmap="magma", vmin=0, vmax=np.percentile(pred_cl, 99))
y_50 = dp_sample.copy(); y_50[~accepted_50] = np.nan
y_rej50 = dp_sample.copy(); y_rej50[accepted_50 | ~dp_valid] = np.nan
axs[2].plot(x_all, y_50, color="#2ecc71", linewidth=1.2, alpha=0.9, label=f"Kept (50% cov)")
axs[2].plot(x_all, y_rej50, color="#e74c3c", linewidth=0.5, alpha=0.4, label="Rejected")
axs[2].set_title(f"Confidence abstention (coverage≈50%, threshold={thr_50:.3f})", fontsize=12)
axs[2].set_ylabel("sample")
axs[2].set_xlabel("trace")
axs[2].legend(fontsize=10)

for ax in axs:
    ax.set_xticks(np.linspace(0, n_traces-1, 5))
    ax.set_xticklabels([f"{int(1664 + x * (2377 - 1664) / (n_traces-1))}" for x in np.linspace(0, n_traces-1, 5)])

fig.suptitle("Line9 Holdout — Confidence Abstention Effect", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig2_confidence_abstention_line9.png", bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT / 'fig2_confidence_abstention_line9.png'}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: MAE-Coverage Pareto (from v1.11 results)
# ═══════════════════════════════════════════════════════════════
print("=== Figure 3: MAE-Coverage Pareto ===")
sweep_csv = ROOT / "outputs" / "v11_frozen_v19d_confidence" / "pareto_sweep_results.csv"
if sweep_csv.exists():
    rows = list(csv.DictReader(sweep_csv.open("r", encoding="utf-8")))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=140)
    colors = {"score_composite": "#e74c3c", "score_crossview": "#3498db", "score_path": "#2ecc71", "score_presence": "#f39c12"}
    labels = {"score_composite": "Composite", "score_crossview": "Cross-view", "score_path": "Path Prob", "score_presence": "Presence Prob"}
    for score_name in ["score_composite", "score_crossview", "score_path", "score_presence"]:
        pts = [(float(r["avg_abstained_cov"]), float(r["avg_abstained_mae"])) for r in rows if r["score"] == score_name]
        pts = [(c, m) for c, m in pts if np.isfinite(m) and np.isfinite(c)]
        if pts:
            xs, ys = zip(*sorted(pts, reverse=True))
            ax.plot(xs, ys, "o-", color=colors[score_name], label=labels[score_name], markersize=5, linewidth=2)
    ax.axhline(y=3.768, color="gray", linestyle="--", alpha=0.6, label="Baseline frozen v1.9D (3.768 ns)")
    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("LOO Average MAE (ns)", fontsize=12)
    ax.set_title("v1.11 — Leave-One-Line-Out MAE–Coverage Pareto", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_pareto_lolo.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT / 'fig3_pareto_lolo.png'}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Method comparison bar chart
# ═══════════════════════════════════════════════════════════════
print("=== Figure 4: Method comparison ===")
trad_csv = ROOT / "outputs" / "traditional_baselines" / "traditional_baseline_comparison.csv"
if trad_csv.exists():
    trad_rows = list(csv.DictReader(trad_csv.open("r", encoding="utf-8")))
    # Line9 specific
    line9_trad = [r for r in trad_rows if r["line"] == "Line9"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)

    methods = ["raw_no_processing", "svd_rank1_subtract", "svd_rank2_subtract", "fk_filter", "rpca_sparse"]
    method_labels = ["Raw", "SVD-1", "SVD-2", "f-k filter", "RPCA"]
    maes = []
    for m in methods:
        hit = [r for r in line9_trad if r["method"] == m]
        maes.append(float(hit[0]["mae_ns"]) if hit and hit[0]["mae_ns"] != "nan" else 0)

    # Add v1.9D and v1.9D+abstain
    method_labels.extend(["v1.9D\n(all)", "v1.9D\n(50% cov)"])
    maes.extend([3.768, 0.809])

    colors_bar = ["#95a5a6"] * 5 + ["#e74c3c", "#2ecc71"]
    bars = ax1.bar(range(len(maes)), maes, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(method_labels)))
    ax1.set_xticklabels(method_labels, fontsize=9)
    ax1.set_ylabel("MAE (ns)", fontsize=11)
    ax1.set_title("Line9 Holdout — MAE Comparison", fontsize=12)
    ax1.set_yscale("log")
    for bar, val in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # Per-line comparison: SVD-1 vs v1.9D
    lines = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
    svd1_mae = []
    v19d_mae = [0.574, 0.512, 0.976, 3.765, 0.802]  # from report
    for ln in lines:
        hit = [r for r in trad_rows if r["line"] == ln and r["method"] == "svd_rank1_subtract"]
        svd1_mae.append(float(hit[0]["mae_ns"]) if hit and hit[0]["mae_ns"] != "nan" else 0)

    x = np.arange(len(lines))
    w = 0.35
    ax2.bar(x - w/2, svd1_mae, w, label="SVD rank-1", color="#3498db")
    ax2.bar(x + w/2, v19d_mae, w, label="v1.9D", color="#e74c3c")
    ax2.set_xticks(x)
    ax2.set_xticklabels(lines)
    ax2.set_ylabel("MAE (ns)", fontsize=11)
    ax2.set_title("Per-Line — SVD vs v1.9D", fontsize=12)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_method_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT / 'fig4_method_comparison.png'}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Multi-line B-scan overview
# ═══════════════════════════════════════════════════════════════
print("=== Figure 5: Multi-line overview ===")
all_lines = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
fig, axs = plt.subplots(1, 5, figsize=(22, 5), dpi=140)
for i, ln in enumerate(all_lines):
    raw_i, mask_i, dt_i = load_line(ln)
    bg = bgr(raw_i)
    v = np.percentile(np.abs(bg), 99.5)
    axs[i].imshow(bg, aspect="auto", cmap="seismic", vmin=-v, vmax=v)
    axs[i].set_title(ln, fontsize=12)
    axs[i].set_xlabel("trace")
    if i == 0:
        axs[i].set_ylabel("sample")
fig.suptitle("Corrected v1.4 — All Measured Lines (BG removed)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig5_multiline_overview.png", bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT / 'fig5_multiline_overview.png'}")

print("\nALL_FIGURES_DONE")
