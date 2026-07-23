"""
Synthetic B-scan preview generator for FORMAL10A.
Uses a simplified 1-D normal-incidence convolutional model to produce
a radar-like section from the epsilon profile.  This is NOT a gprMax
replacement — it is a fast pre-solver texture preview.
"""

import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_materials(path: Path) -> dict[int, float]:
    """Map material index -> epsilon_r."""
    mapping = {}
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == '#material:':
                epsilon = float(parts[1])
                mapping[idx] = epsilon
    return mapping


def ricker_wavelet(fc: float, dt: float, n_samples: int) -> np.ndarray:
    """Ricker wavelet (second derivative of Gaussian)."""
    t = np.arange(n_samples) * dt
    tau = t - 2.5 / fc  # shift so peak is well inside window
    sigma = 1.0 / (math.pi * fc)
    envelope = (1.0 - 2.0 * (math.pi * fc * tau) ** 2) * np.exp(-(math.pi * fc * tau) ** 2)
    return envelope / np.max(np.abs(envelope))


def generate_synthetic_bscan(
    case_dir: Path,
    fc_hz: float = 100e6,
    c0: float = 299_792_458.0,
    output_name: str = "preview_synthetic_bscan.png",
) -> Path:
    # ------------------------------------------------------------------
    # 1. Load geometry
    # ------------------------------------------------------------------
    h5_path = case_dir / "geology_indices.h5"
    with h5py.File(h5_path, 'r') as hf:
        indices = hf["data"][:, :, 0]  # (nx, ny)
        dl = float(hf.attrs["dx_dy_dz"][0])

    materials_path = case_dir / "materials_full.txt"
    epsilon_map = parse_materials(materials_path)

    # Build epsilon array
    epsilon = np.vectorize(epsilon_map.get)(indices)
    epsilon[indices < 0] = 1.0  # air

    nx, ny = epsilon.shape

    # ------------------------------------------------------------------
    # 2. Acquisition parameters (from labels or defaults)
    # ------------------------------------------------------------------
    labels_dir = case_dir / "labels"
    if (labels_dir / "source_x_m.npy").exists():
        source_x = np.load(labels_dir / "source_x_m.npy")
        trace_count = source_x.size
    else:
        trace_count = 256
        source_x = np.linspace(78.3, 78.3 + 255 * 0.09, trace_count)

    dt_s = dl / c0 * 0.5  # Courant-like sampling for display
    n_time = 1200
    wavelet = ricker_wavelet(fc_hz, dt_s, 401)

    # ------------------------------------------------------------------
    # 3. Synthetic trace generation (normal incidence, 1-D convolution)
    # ------------------------------------------------------------------
    bscan = np.zeros((n_time, trace_count), dtype=np.float32)
    y = (np.arange(ny) + 0.5) * dl

    for tr in range(trace_count):
        ix = int(np.clip(round(source_x[tr] / dl - 0.5), 0, nx - 1))
        eps_col = epsilon[ix, :]

        # Reflection coefficient at each interface
        rc = np.zeros(ny, dtype=np.float64)
        rc[1:] = (np.sqrt(eps_col[:-1]) - np.sqrt(eps_col[1:])) / (np.sqrt(eps_col[:-1]) + np.sqrt(eps_col[1:]))

        # Two-way travel time for each depth sample
        v = c0 / np.sqrt(eps_col)
        t_tw = 2.0 * np.cumsum(np.full(ny, dl) / v)

        # Map reflection coefficients to regular time grid
        trace = np.zeros(n_time, dtype=np.float64)
        for iy in range(1, ny):
            it = int(np.clip(round(t_tw[iy] / dt_s), 0, n_time - 1))
            trace[it] += rc[iy]

        # Convolve with wavelet
        trace = np.convolve(trace, wavelet, mode='same')
        bscan[:, tr] = trace.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Time-power gain
    # ------------------------------------------------------------------
    t = np.arange(n_time) * dt_s * 1e9  # ns
    t_ref = 250.0  # ns reference
    gain = np.clip((t / t_ref) ** 1.5, 0.0, 50.0)
    bscan_gain = bscan * gain[:, None]

    # ------------------------------------------------------------------
    # 5. Plot (raw vs tpower, matching audit gallery style)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw
    ax = axes[0]
    vmax_raw = np.percentile(np.abs(bscan), 99.5)
    ax.imshow(
        bscan,
        aspect='auto',
        cmap='gray',
        extent=[0, trace_count * 0.09, t[-1], 0],
        vmin=-vmax_raw,
        vmax=vmax_raw,
    )
    ax.set_title(f'FORMAL10A - Synthetic Raw\nP99.5 = {vmax_raw:.3e}', fontsize=11)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (ns)')

    # tpower gain
    ax = axes[1]
    vmax_gain = np.percentile(np.abs(bscan_gain), 99.5)
    ax.imshow(
        bscan_gain,
        aspect='auto',
        cmap='gray',
        extent=[0, trace_count * 0.09, t[-1], 0],
        vmin=-vmax_gain,
        vmax=vmax_gain,
    )
    ax.set_title(
        f'FORMAL10A - Synthetic Background-suppressed + t^1.5 gain\n'
        f'P99.5 = {vmax_gain:.3e}',
        fontsize=11,
    )
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (ns)')

    plt.suptitle(
        'FORMAL10A_MULTISCALE_HETEROGENEOUS_COVER - PRE-SOLVER SYNTHETIC PREVIEW\n'
        'Five-scale texture with lateral breaks | epsilon span 11-14 | 100 MHz Ricker',
        fontsize=12,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = case_dir / output_name
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved synthetic B-scan preview: {out_path}")
    return out_path


if __name__ == "__main__":
    CASE_DIR = Path(
        r"F:/codex/PSGN-CSNet/MyNet_master_cleanup/"
        r"data/simulations/v2/00_controls/"
        r"FORMAL10A_MULTISCALE_HETEROGENEOUS_COVER_DEVELOPMENT"
    )
    generate_synthetic_bscan(CASE_DIR)
