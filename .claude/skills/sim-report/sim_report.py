#!/usr/bin/env python3
"""三件套生成器：任何仿真跑完后，强制生成 几何模型 + 原始B-scan + 处理后图。
兼容 run_dir/raw/ 和扁平目录两种结构。缺数据时 exit(1) 而非空文件。"""
import sys, os
from pathlib import Path

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
assets_dir = Path(r"D:\Claude\PGDA-CSNet\笔记\assets")

import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Auto-detect structure ──
# Formal PGDA: run_dir/raw/raw.in + run_dir/raw/bscan.npy
# Flat: out_dir/raw.in + out_dir/bscan.npy
if (out_dir / "raw" / "raw.in").exists():
    raw_dir = out_dir / "raw"
    stem = "raw"
elif (out_dir / "raw.in").exists():
    raw_dir = out_dir
    stem = "raw"
elif list(out_dir.rglob("raw.in")):
    raw_dir = next(out_dir.rglob("raw.in")).parent
    stem = "raw"
else:
    # Also try bscan.npy as anchor
    for cand in [out_dir / "raw" / "bscan.npy", out_dir / "bscan.npy"]:
        if cand.exists():
            raw_dir = cand.parent
            stem = cand.stem.replace("_bscan", "").replace("_native", "")
            break
    else:
        print(f"ERROR: No raw.in or bscan.npy found under {out_dir}")
        sys.exit(1)

in_file = raw_dir / f"{stem}.in"
bscan_path = raw_dir / "bscan.npy"
out_files = sorted(raw_dir.glob(f"{stem}*.out"))

had_geometry = False
had_bscan = False

# ── 1. Geometry ──
fig_geo, ax_geo = plt.subplots(figsize=(18, 6))
if in_file.exists():
    had_geometry = True
    src_lines = in_file.read_text().splitlines()
    mats = {}
    for l in src_lines:
        s = l.strip()
        if s.startswith("#material:"):
            p = s.split()
            mats[p[-1]] = (float(p[1]), float(p[2]))
    mc = dict(zip(sorted(mats.keys()), [plt.cm.tab20(i / len(mats)) for i in range(len(mats))]))
    for l in src_lines:
        s = l.strip()
        if s.startswith("#box:"):
            p = s.split()
            ax_geo.add_patch(mpatches.Rectangle((float(p[1]), float(p[2])), float(p[4]) - float(p[1]),
                float(p[5]) - float(p[2]), facecolor=mc.get(p[7], "gray"), edgecolor="black", lw=0.05, alpha=0.85))
        if s.startswith("#triangle:"):
            p = s.split()
            xs = [float(p[1]), float(p[4]), float(p[7])]
            ys = [float(p[2]), float(p[5]), float(p[8])]
            ax_geo.fill(xs, ys, facecolor=mc.get(p[-2], "gray"), edgecolor="black", lw=0.1, alpha=0.85)

    # TX/RX
    for l in src_lines:
        s = l.strip()
        if s.startswith("#hertzian_dipole:"): p = s.split(); ax_geo.plot(float(p[2]), float(p[3]), "rv", ms=10, label="TX")
        if s.startswith("#rx:"): p = s.split(); ax_geo.plot(float(p[1]), float(p[2]), "b^", ms=10, label="RX")

    ax_geo.set_ylim(40, 0)
    ax_geo.invert_yaxis()
    ax_geo.set_xlabel("x (m)")
    ax_geo.set_ylabel("y (m)")
    ax_geo.set_title(f"Geometry: {in_file.name}", fontweight="bold")
    lp = [mpatches.Patch(color=color, label=mat) for mat, color in mc.items()]
    ax_geo.legend(handles=lp, fontsize=6, loc="lower right", ncol=2)
else:
    ax_geo.text(0.5, 0.5, f"No .in file: {in_file}", ha="center", va="center", transform=ax_geo.transAxes)

# ── 2. B-scan (assemble from .out if needed) ──
if not bscan_path.exists() and out_files:
    files = sorted([f for f in out_files if f.stem.replace(stem, "").isdigit()],
                   key=lambda x: int(x.stem.replace(stem, "")))
    tr = []
    for f in files:
        with h5py.File(f, "r") as h:
            tr.append(np.asarray(h["rxs"]["rx1"]["Ez"]))
    arr = np.stack(tr, axis=1)
    np.save(bscan_path, arr)

fig_bscan, axes = plt.subplots(1, 3, figsize=(18, 5))
if bscan_path.exists():
    had_bscan = True
    arr = np.load(bscan_path)
    old = np.linspace(0, 700, arr.shape[0])
    new = np.linspace(0, 700, 501)
    bi = np.empty((501, arr.shape[1]))
    for i in range(arr.shape[1]):
        bi[:, i] = np.interp(new, old, arr[:, i])
    t = new
    diffs = [np.max(np.abs(arr[:, i] - arr[:, 0])) for i in range(arr.shape[1])]
    clip = np.percentile(np.abs(bi), 99.9)
    gain_t4 = (t / t[-1]) ** 4
    t4b = bi * (gain_t4[:, np.newaxis] + 0.01)

    def safe_agc(x, w=31, f=0.02):
        p = w // 2
        y = np.pad(x * x, ((p, p), (0, 0)), "reflect")
        r = np.empty_like(x)
        for i in range(x.shape[0]):
            r[i] = np.sqrt(np.mean(y[i:i + w], axis=0) + 1e-12)
        return x / np.maximum(r, np.percentile(r, 10) * f)

    vmax = np.percentile(np.abs(bi), 99.5)
    axes[0].imshow(bi, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax, extent=[0, bi.shape[1] - 1, 700, 0])
    axes[0].set_title(f"Raw B-scan (trace_var={max(diffs):.2f})", fontweight="bold")
    axes[0].set_xlabel("Trace")
    axes[1].imshow(t4b, cmap="gray", aspect="auto", vmin=-np.percentile(np.abs(t4b), 99.5),
                  vmax=np.percentile(np.abs(t4b), 99.5), extent=[0, bi.shape[1] - 1, 700, 0])
    axes[1].set_title("t^4 gain", fontweight="bold")
    axes[1].set_xlabel("Trace")
    axes[2].imshow(safe_agc(bi), cmap="gray", aspect="auto", vmin=-3, vmax=3, extent=[0, bi.shape[1] - 1, 700, 0])
    axes[2].set_title("Safe AGC", fontweight="bold")
    axes[2].set_xlabel("Trace")
else:
    for ax in axes:
        ax.text(0.5, 0.5, "No B-scan data", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()

# Save
fig_path = out_dir / "three_piece.png"
fig_bscan.savefig(fig_path, dpi=200, bbox_inches="tight")
fig_geo.savefig(out_dir / "geometry.png", dpi=200, bbox_inches="tight")
fig_path2 = Path(r"C:\Users\17844\Desktop") / f"{out_dir.name}_three_piece.png"
fig_bscan.savefig(fig_path2, dpi=200, bbox_inches="tight")

# Clean .vti
for v in raw_dir.glob("*.vti"):
    try:
        v.unlink()
    except:
        pass

n_files = len(out_files)
print(f"\n{'='*50}")
if had_geometry and had_bscan:
    print(f"三件套生成完成!")
else:
    print(f"三件套生成完成（部分缺失）!")
    if not had_geometry:
        print(f"  ⚠ 几何图：无 .in 文件 ({in_file})")
    if not had_bscan:
        print(f"  ⚠ B-scan：无数据 ({bscan_path})")
print(f"  仿真结果: {out_dir}")
print(f"  原始目录: {raw_dir}")
print(f"  文件数: {n_files}")
print(f"  几何图: {out_dir/'geometry.png'}")
print(f"  B-scan: {fig_path}")
print(f"  桌面: {fig_path2}")
print(f"  .vti清理: 完成")
print(f"{'='*50}")

# 铁律：缺几何或B-scan时退出码非零
if not had_geometry or not had_bscan:
    sys.exit(1)
