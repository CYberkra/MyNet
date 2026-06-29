from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    import h5py
except Exception:
    h5py = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _jsonable(v):
    try:
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return v.item() if hasattr(v, "item") else v
    except Exception:
        return str(v)


def read_gprmax_ascan(out_file: str | Path, rx: str = "rx1", component: str = "Ez") -> Tuple[np.ndarray, Dict]:
    if h5py is None:
        raise RuntimeError("h5py is not installed. Run: pip install h5py")
    out_file = Path(out_file)
    with h5py.File(out_file, "r") as f:
        attrs = {k: _jsonable(v) for k, v in f.attrs.items()}
        rx_group = f["rxs"][rx]
        if component not in rx_group:
            for comp in ["Ez", "Ey", "Ex", "Hz", "Hy", "Hx"]:
                if comp in rx_group:
                    component = comp
                    break
        data = np.asarray(rx_group[component])
        attrs["component"] = component
        attrs["rx_position"] = _jsonable(rx_group.attrs.get("Position", ""))
    return data, attrs


def merge_bscan_from_outputs(out_files: Sequence[str | Path], rx: str = "rx1", component: str = "Ez") -> Tuple[np.ndarray, Dict]:
    traces = []
    meta = {"files": []}
    for p in out_files:
        data, attrs = read_gprmax_ascan(p, rx=rx, component=component)
        traces.append(data.ravel())
        meta["files"].append({"file": str(p), "attrs": attrs})
    n = min(len(t) for t in traces)
    arr = np.stack([t[:n] for t in traces], axis=1)
    return arr, meta


def dewow(data: np.ndarray, window: int = 51) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if window <= 3:
        return x - np.mean(x, axis=0, keepdims=True)
    kernel = np.ones(window, dtype=float) / window
    trend = np.apply_along_axis(lambda a: np.convolve(a, kernel, mode="same"), 0, x)
    return x - trend


def zero_mean_background(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    return x - np.mean(x, axis=1, keepdims=True)


def time_gain(data: np.ndarray, power: float = 1.2, exponential: float = 0.0) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    t = np.linspace(0, 1, x.shape[0])[:, None]
    gain = (1.0 + t) ** power
    if exponential > 0:
        gain *= np.exp(exponential * t)
    return x * gain


def moving_average_trace(data: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda a: np.convolve(a, kernel, mode="same"), 1, x)


def svd_clutter_suppress(data: np.ndarray, remove_rank: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=float)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    rank = max(0, min(remove_rank, len(s)))
    clutter = (u[:, :rank] * s[:rank]) @ vt[:rank, :] if rank else np.zeros_like(x)
    return x - clutter, clutter


def fk_wedge_suppress(data: np.ndarray, vertical_width: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=float)
    F = np.fft.fftshift(np.fft.fft2(x))
    nx = x.shape[1]
    kx = np.linspace(-1, 1, nx)[None, :]
    mask = np.abs(kx) < vertical_width
    clean_F = F * (~mask)
    clutter_F = F * mask
    clean = np.real(np.fft.ifft2(np.fft.ifftshift(clean_F)))
    clutter = np.real(np.fft.ifft2(np.fft.ifftshift(clutter_F)))
    return clean, clutter


def snr_db(data: np.ndarray, signal_window: Tuple[int, int], noise_window: Tuple[int, int]) -> float:
    x = np.asarray(data, dtype=float)
    s0, s1 = signal_window
    n0, n1 = noise_window
    ps = float(np.mean(x[s0:s1, :] ** 2)) + 1e-12
    pn = float(np.mean(x[n0:n1, :] ** 2)) + 1e-12
    return 10.0 * math.log10(ps / pn)


def export_bscan_products(data: np.ndarray, out_dir: str | Path, stem: str, time_window_ns: float = 700.0) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    products: Dict[str, str] = {}
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, bscan=data, time_window_ns=float(time_window_ns))
    products["npz"] = str(npz_path)
    csv_path = out_dir / f"{stem}.csv"
    np.savetxt(csv_path, data, delimiter=",", fmt="%.8e")
    products["csv"] = str(csv_path)
    if plt is not None:
        fig_path = out_dir / f"{stem}.png"
        fig = plt.figure(figsize=(8, 4), dpi=160)
        vmax = np.percentile(np.abs(data), 99) if data.size else 1.0
        plt.imshow(data, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=[0, data.shape[1], time_window_ns, 0])
        plt.xlabel("Trace")
        plt.ylabel("Time (ns)")
        plt.title(stem)
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        products["png"] = str(fig_path)
    return products


def run_traditional_baselines(data: np.ndarray, out_dir: str | Path, stem: str = "bscan", time_window_ns: float = 700.0) -> Dict[str, Dict[str, str]]:
    out_dir = Path(out_dir)
    results: Dict[str, Dict[str, str]] = {}
    steps = {"raw": np.asarray(data, dtype=float)}
    steps["dewow"] = dewow(steps["raw"], window=51)
    steps["mean_subtract"] = zero_mean_background(steps["dewow"])
    steps["gain"] = time_gain(steps["mean_subtract"], power=1.2)
    clean_svd, clutter_svd = svd_clutter_suppress(steps["gain"], remove_rank=1)
    steps["svd_clean"] = clean_svd
    steps["svd_clutter"] = clutter_svd
    clean_fk, clutter_fk = fk_wedge_suppress(steps["gain"], vertical_width=0.08)
    steps["fk_clean"] = clean_fk
    steps["fk_clutter"] = clutter_fk
    for name, arr in steps.items():
        results[name] = export_bscan_products(arr, out_dir, f"{stem}_{name}", time_window_ns=time_window_ns)
    return results


def candidate_out_files_for_input(input_file: str | Path) -> list[Path]:
    """Return likely gprMax .out files for an input file.

    gprMax writes HDF5 .out files with names related to the input stem. Depending
    on whether -n is used, versions differ slightly in suffix style, so we scan a
    conservative set and return sorted unique files.
    """
    p = Path(input_file)
    parent = p.parent
    stem = p.stem
    patterns = [
        f"{stem}.out",
        f"{stem}[0-9]*.out",
        f"{stem}_[0-9]*.out",
        f"{stem}*.out",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(parent.glob(pat))
    # Natural-ish sort: stem.out first, then numeric suffixes if any.
    def key(x: Path):
        import re
        nums = re.findall(r"(\d+)", x.stem.replace(stem, ""))
        return (0 if x.name == f"{stem}.out" else 1, int(nums[-1]) if nums else -1, x.name)
    return sorted(set(files), key=key)


def merge_available_bscan_for_input(input_file: str | Path, rx: str = "rx1", component: str = "Ez") -> tuple[np.ndarray, Dict] | None:
    files = candidate_out_files_for_input(input_file)
    if not files:
        return None
    try:
        return merge_bscan_from_outputs(files, rx=rx, component=component)
    except Exception:
        # Some files can be half-written while a solver is still running. Try the
        # stable prefix only, preserving real-time preview instead of failing.
        stable: list[Path] = []
        for f in files:
            try:
                read_gprmax_ascan(f, rx=rx, component=component)
                stable.append(f)
            except Exception:
                continue
        if not stable:
            return None
        return merge_bscan_from_outputs(stable, rx=rx, component=component)


def export_gprmax_bscan_for_input(input_file: str | Path, out_dir: str | Path, stem: str | None = None, time_window_ns: float = 700.0) -> Dict[str, object]:
    merged = merge_available_bscan_for_input(input_file)
    if merged is None:
        raise FileNotFoundError(f"No readable gprMax .out files found for {input_file}")
    bscan, meta = merged
    products = run_traditional_baselines(bscan, out_dir, stem=stem or Path(input_file).stem, time_window_ns=time_window_ns)
    return {"bscan_shape": list(bscan.shape), "meta": meta, "products": products}
