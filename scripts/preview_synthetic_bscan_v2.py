"""
Synthetic B-scan preview generator for FORMAL10A (numpy + PIL only).
"""

import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def parse_materials(path: Path) -> dict[int, float]:
    mapping = {}
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == '#material:':
                mapping[idx] = float(parts[1])
    return mapping


def ricker_wavelet(fc: float, dt: float, n_samples: int) -> np.ndarray:
    t = np.arange(n_samples) * dt
    tau = t - 2.5 / fc
    sigma = 1.0 / (math.pi * fc)
    envelope = (1.0 - 2.0 * (math.pi * fc * tau) ** 2) * np.exp(-(math.pi * fc * tau) ** 2)
    return envelope / np.max(np.abs(envelope))


def generate_synthetic_bscan(
    case_dir: Path,
    fc_hz: float = 100e6,
    c0: float = 299_792_458.0,
    output_name: str = "preview_synthetic_bscan.png",
) -> Path:
    h5_path = case_dir / "geology_indices.h5"
    with h5py.File(h5_path, 'r') as hf:
        indices = hf["data"][:, :, 0]
        dl = float(hf.attrs["dx_dy_dz"][0])

    epsilon_map = parse_materials(case_dir / "materials_full.txt")
    epsilon = np.vectorize(epsilon_map.get)(indices)
    epsilon[indices < 0] = 1.0
    nx, ny = epsilon.shape

    labels_dir = case_dir / "labels"
    if (labels_dir / "source_x_m.npy").exists():
        source_x = np.load(labels_dir / "source_x_m.npy")
        trace_count = source_x.size
    else:
        trace_count = 256
        source_x = np.linspace(78.3, 78.3 + 255 * 0.09, trace_count)

    dt_s = dl / c0 * 0.5
    n_time = 1200
    wavelet = ricker_wavelet(fc_hz, dt_s, 401)
    bscan = np.zeros((n_time, trace_count), dtype=np.float32)

    for tr in range(trace_count):
        ix = int(np.clip(round(source_x[tr] / dl - 0.5), 0, nx - 1))
        eps_col = epsilon[ix, :]
        rc = np.zeros(ny, dtype=np.float64)
        rc[1:] = (np.sqrt(eps_col[:-1]) - np.sqrt(eps_col[1:])) / (np.sqrt(eps_col[:-1]) + np.sqrt(eps_col[1:]))
        v = c0 / np.sqrt(eps_col)
        t_tw = 2.0 * np.cumsum(np.full(ny, dl) / v)
        trace = np.zeros(n_time, dtype=np.float64)
        for iy in range(1, ny):
            it = int(np.clip(round(t_tw[iy] / dt_s), 0, n_time - 1))
            trace[it] += rc[iy]
        trace = np.convolve(trace, wavelet, mode='same')
        bscan[:, tr] = trace.astype(np.float32)

    # Time-power gain
    t = np.arange(n_time) * dt_s * 1e9
    t_ref = 250.0
    gain = np.clip((t / t_ref) ** 1.5, 0.0, 50.0)
    bscan_gain = bscan * gain[:, None]

    # Render with PIL
    def to_gray(arr: np.ndarray) -> Image.Image:
        vmax = np.percentile(np.abs(arr), 99.5)
        normalized = np.clip((arr / vmax) * 0.5 + 0.5, 0.0, 1.0)
        img = (normalized * 255).astype(np.uint8)
        return Image.fromarray(img, mode='L')

    raw_img = to_gray(bscan)
    gain_img = to_gray(bscan_gain)

    # Combine side by side with labels
    w, h = raw_img.size
    canvas = Image.new('RGB', (w * 2 + 20, h + 80), (255, 255, 255))
    canvas.paste(Image.merge('RGB', (raw_img, raw_img, raw_img)), (0, 40))
    canvas.paste(Image.merge('RGB', (gain_img, gain_img, gain_img)), (w + 20, 40))

    out_path = case_dir / output_name
    canvas.save(out_path)
    print(f"Saved synthetic B-scan preview: {out_path}")
    return out_path


if __name__ == "__main__":
    CASE_DIR = Path(
        r"F:/codex/PSGN-CSNet/MyNet_master_cleanup/"
        r"data/simulations/v2/00_controls/"
        r"FORMAL10A_MULTISCALE_HETEROGENEOUS_COVER_DEVELOPMENT"
    )
    generate_synthetic_bscan(CASE_DIR)
