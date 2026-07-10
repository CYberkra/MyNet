#!/usr/bin/env python3
"""Generate, run, and report one shallow low-loss simulation case.

Runs a single raw.in with the V4 low-loss material parameters to verify
the bedrock interface is visible.  Uses h5py-based gprMax .out reader.

Usage:
    "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/v4_quick_test.py
"""
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from uavgpr_simlab.core.postprocess import merge_available_bscan_for_input

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs/v4_quick_test"
IN_FILE = str(OUT / "raw.in")
LABELS = OUT / "labels"

C_AIR = 0.3  # m/ns

# ── Geometry (y=0 = bottom of domain, positive y = upward) ──
DX = 0.08
DOMAIN_X = 52.0
DOMAIN_Y = 48.0
N_TRACES = 128
TRACE_STEP = 0.2
SCAN_X0 = 2.0
TX_RX_OFFSET = 1.4
TW_NS = 700.0
N_SAMPLES = 501
DT = TW_NS / N_SAMPLES

# ── Materials (V4 low-loss) ──
MATERIALS = {
    "air":                (1.0,  0.0,      1.0, 0.0),
    "silty_clay":         (13.0, 0.0015,   1.0, 0.0),
    "sandstone_bedrock":  (5.0,  0.0005,   1.0, 0.0),
}

GROUND_Y = 24.0
BEDROCK_DEPTH = 7.0
BEDROCK_THICK = 10.0
UAV_HEIGHT = 8.0
PLATFORM_Y = GROUND_Y + UAV_HEIGHT


def make_in() -> str:
    lines = ["#title: v4_quick_test bedrock=7.0m",
             f"#domain: {DOMAIN_X:.3f} {DOMAIN_Y:.3f} {DX:.5f}",
             f"#dx_dy_dz: {DX:.5f} {DX:.5f} {DX:.5f}",
             f"#time_window: {TW_NS*1e-9:.2e}",
             "#pml_cells: 60 60 0 60 60 0"]
    for name, (eps, sig, mu, ml) in MATERIALS.items():
        lines.append(f"#material: {eps:.6g} {sig:.6g} {mu:.6g} {ml:.6g} {name}")
    lines.append("#waveform: ricker 1 1e+08 uavgpr_wavelet")
    tx_x, tx_y = SCAN_X0, PLATFORM_Y
    rx_x, rx_y = SCAN_X0 + TX_RX_OFFSET, PLATFORM_Y
    lines.append(f"#hertzian_dipole: z {tx_x:.3f} {tx_y:.3f} {DX/2:.4f} uavgpr_wavelet")
    lines.append(f"#rx: {rx_x:.3f} {rx_y:.3f} {DX/2:.4f}")
    lines.append(f"#src_steps: {TRACE_STEP:.4f} 0 0")
    lines.append(f"#rx_steps: {TRACE_STEP:.4f} 0 0")
    # Bedrock base
    bedrock_top = GROUND_Y - BEDROCK_DEPTH
    lines.append(f"#box: 0 0 0 {DOMAIN_X:.3f} {bedrock_top:.3f} {DX:.3f} sandstone_bedrock")
    # Silty_clay overburden
    lines.append(f"#box: 0 {bedrock_top:.3f} 0 {DOMAIN_X:.3f} {GROUND_Y:.3f} {DX:.3f} silty_clay")
    return "\n".join(lines)


def compute_labels():
    OUT.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    time_ns = np.linspace(0, TW_NS, N_SAMPLES, dtype=np.float32)
    np.save(str(OUT / "time_501_ns.npy"), time_ns)
    tx_x = np.linspace(SCAN_X0, SCAN_X0 + TRACE_STEP * (N_TRACES - 1), N_TRACES)
    np.save(str(OUT / "trace_x_m.npy"), tx_x.astype(np.float32))
    air_twt = 2 * UAV_HEIGHT / C_AIR
    cover_v = C_AIR / math.sqrt(MATERIALS["silty_clay"][0])
    soil_twt = 2 * BEDROCK_DEPTH / cover_v
    total_twt = air_twt + soil_twt
    twt = np.full(N_TRACES, total_twt, dtype=np.float32)
    imask = np.zeros((N_SAMPLES, N_TRACES), dtype=np.float32)
    for i in range(N_TRACES):
        c = int(twt[i] / DT)
        lo, hi = max(0, c-3), min(N_SAMPLES, c+4)
        imask[lo:hi, i] = 1.0
    np.save(str(LABELS / "interface_mask_bscan.npy"), imask)
    y_soft = np.zeros((N_SAMPLES, N_TRACES), dtype=np.float32)
    for i in range(N_TRACES):
        center = twt[i]
        g = np.exp(-((time_ns - center) ** 2) / (2 * (8 * DT) ** 2))
        g /= g.max() + 1e-10
        y_soft[:, i] = g.astype(np.float32)
    np.save(str(LABELS / "y_soft_501x128.npy"), y_soft)
    print(f"  Labels saved, target TWT: {total_twt:.1f} ns")
    return twt


def run_gprmax():
    venv_py = Path("E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe")
    gprmax_root = Path("E:/gprMax/gprMax-v.3.1.7")
    env = dict(os.environ)
    extra_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64",
    ]
    for p in extra_paths:
        if os.path.isdir(p):
            env["PATH"] = p + ";" + env.get("PATH", "")
    msvc_inc = r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\include"
    sdk_inc = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt"
    msvc_lib = r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\lib\x64"
    sdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"
    ucrt_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64"
    inc_parts = [p for p in [msvc_inc, sdk_inc] if os.path.isdir(p)]
    if inc_parts:
        env["INCLUDE"] = ";".join(inc_parts) + ";" + env.get("INCLUDE", "")
    lib_parts = [p for p in [msvc_lib, sdk_lib, ucrt_lib] if os.path.isdir(p)]
    if lib_parts:
        env["LIB"] = ";".join(lib_parts) + ";" + env.get("LIB", "")
    cmd = [str(venv_py), "-m", "gprMax", IN_FILE, "-gpu", "--geometry-fixed", "-n", str(N_TRACES)]
    print("  Running gprMax (128 traces)...")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(gprmax_root), env=env, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    print(f"  gprMax finished in {elapsed/60:.1f} min (rc={proc.returncode})")
    if proc.returncode != 0:
        err = proc.stderr[-2000:]
        out = proc.stdout[-2000:]
        print(f"  stderr: {err}\n  stdout: {out}")
        raise RuntimeError(f"gprMax failed with code {proc.returncode}")
    # .out files are HDF5 — they stay in place; merge_bscan_from_outputs reads them.
    out_files = sorted(Path(IN_FILE).parent.glob("raw*.out"))
    if not out_files:
        raise FileNotFoundError(f"No raw*.out files found in {Path(IN_FILE).parent}")
    print(f"  {len(out_files)} .out files produced (HDF5 format)")
    return elapsed


def postprocess():
    """Merge .out HDF5 files via merge_available_bscan_for_input."""
    result = merge_available_bscan_for_input(IN_FILE)
    if result is None:
        raise RuntimeError("merge_available_bscan_for_input returned None")
    bscan, meta = result
    print(f"  Merged B-scan: {bscan.shape}, range=[{bscan.min():.2f}, {bscan.max():.2f}]")
    # Resample to N_SAMPLES if needed
    if bscan.shape[0] != N_SAMPLES:
        old = np.linspace(0, 1, bscan.shape[0])
        new = np.linspace(0, 1, N_SAMPLES)
        resampled = np.zeros((N_SAMPLES, bscan.shape[1]), dtype=np.float32)
        for i in range(bscan.shape[1]):
            resampled[:, i] = np.interp(new, old, bscan[:, i]).astype(np.float32)
        bscan = resampled
    np.save(str(OUT / "raw_bscan.npy"), bscan)
    print(f"  Resampled: {bscan.shape}, range=[{bscan.min():.2f}, {bscan.max():.2f}]")
    return bscan


def report(bscan):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    time_ns = np.load(str(OUT / "time_501_ns.npy"))
    tx_x = np.load(str(OUT / "trace_x_m.npy"))
    mask = np.load(str(LABELS / "interface_mask_bscan.npy")).astype(bool)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    # ── 1. Geometry preview ──
    ax = axes[0]; ax.set_title("Geometry: V4 Low-Loss, bedrock=7m", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.axhline(GROUND_Y, color="brown", linewidth=2, label="Ground surface")
    y_bt = GROUND_Y - BEDROCK_DEPTH; y_bb = y_bt - BEDROCK_THICK
    ax.fill_between([0, DOMAIN_X], y_bb, y_bt, color="gray", alpha=0.5, edgecolor="black", label="Bedrock")
    ax.fill_between([0, DOMAIN_X], y_bt, GROUND_Y, color="tan", alpha=0.3, label="Silty clay")
    ax.scatter(SCAN_X0, PLATFORM_Y, marker="v", color="red", s=80, label="TX")
    ax.scatter(SCAN_X0+TX_RX_OFFSET, PLATFORM_Y, marker="^", color="blue", s=80, label="RX")
    ax.set_xlim(0, DOMAIN_X); ax.set_ylim(0, DOMAIN_Y)
    ax.legend(fontsize=8, loc="upper right"); ax.set_facecolor("lightcyan")

    # ── 2. Raw B-scan ──
    ext = [tx_x[0], tx_x[-1], time_ns[-1], time_ns[0]]
    ax = axes[1]; vmax = max(abs(bscan.min()), abs(bscan.max()))
    ax.imshow(bscan, aspect="auto", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Raw B-scan  range=[{bscan.min():.2f}, {bscan.max():.2f}]")
    ax.set_xlabel("Trace x (m)"); ax.set_ylabel("Time (ns)")
    if mask.any():
        ax.imshow(np.ma.masked_where(~mask, np.ones_like(mask)),
                  aspect="auto", extent=ext, cmap="Greens", alpha=0.3, vmin=0, vmax=1)

    # ── 3. Processed ──
    b = bscan.copy()
    b = b - uniform_filter1d(b, size=51, axis=0)
    b = b - np.mean(b, axis=1, keepdims=True)
    gain = np.linspace(1, 80, b.shape[0]) ** 2.5
    b_gain = b * gain[:, np.newaxis]
    ax = axes[2]; vmax = max(abs(b_gain.min()), abs(b_gain.max()))
    ax.imshow(b_gain, aspect="auto", extent=ext, cmap="RdBu_r", vmin=-vmax/3, vmax=vmax/3)
    ax.set_title(f"Dewow+MeanSub+TimeGain  range=[{b_gain.min():.2f}, {b_gain.max():.2f}]")
    ax.set_xlabel("Trace x (m)"); ax.set_ylabel("Time (ns)")
    if mask.any():
        ax.imshow(np.ma.masked_where(~mask, np.ones_like(mask)),
                  aspect="auto", extent=ext, cmap="Greens", alpha=0.3, vmin=0, vmax=1)

    # ── Statistics ──
    int_vals = bscan[mask]; early = bscan[:45, :]
    if len(int_vals) > 0:
        int_m = float(np.mean(np.abs(int_vals)))
        early_m = float(np.mean(np.abs(early)))
        ratio_db = 20 * np.log10(int_m / (early_m + 1e-30))
        noise = bscan[400:, :]
        noise_m = float(np.mean(np.abs(noise)))
        snr_db = 20 * np.log10(int_m / (noise_m + 1e-30))
        txt = (f"Interface abs mean:  {int_m:.3e}\n"
               f"Early abs mean:      {early_m:.3e}\n"
               f"Interface/Early:     {ratio_db:.1f} dB\n"
               f"Interface/Noise:     {snr_db:.1f} dB\n"
               f"Interface range:     [{bscan[mask].min():.3e}, {bscan[mask].max():.3e}]")
        fig.text(0.02, 0.01, txt, fontsize=9, family='monospace',
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.savefig(str(OUT / "v4_quick_report.png"), dpi=150)
    plt.close(fig)
    print(f"  Report saved: {OUT / 'v4_quick_report.png'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print("V4 Quick Test — Shallow Low-Loss Simulation")
    print("=" * 50)
    # 1. .in file
    Path(IN_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(IN_FILE).write_text(make_in())
    print(f"\n[1/4] .in file: {IN_FILE}")
    print("  Materials: silty_clay eps=13 sig=0.0015, sandstone eps=5 sig=0.0005")
    # 2. Labels
    print("\n[2/4] Computing labels...")
    twt = compute_labels()
    # 3. Run gprMax
    print("\n[3/4] Running gprMax...")
    elapsed = run_gprmax()
    # 4. Postprocess
    print("\n[4/4] Postprocessing...")
    bscan = postprocess()
    # 5. Report
    report(bscan)
    full = "=" * 50
    print(f"\n{full}")
    print(f"Done!  Bedrock: {BEDROCK_DEPTH}m  TWT: {twt[0]:.1f} ns  Run: {elapsed/60:.1f} min")
    print(f"Report: {OUT / 'v4_quick_report.png'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
