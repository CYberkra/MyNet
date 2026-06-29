from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class UavGprCsvMeta:
    samples: int
    time_window_ns: float
    traces: int
    trace_interval_m: float
    columns: Tuple[str, str, str, str, str] = ("longitude", "latitude", "elevation_m", "amplitude", "flight_height_or_extra_m")


@dataclass
class RealCsvSummary:
    csv_path: str
    shape: Tuple[int, int]
    meta: UavGprCsvMeta
    amplitude_min: float
    amplitude_max: float
    amplitude_mean: float
    amplitude_std: float
    distance_m: float
    note: str = "shape is samples x traces"


def _parse_meta_line(line: str) -> tuple[str, float] | None:
    if "=" not in line:
        return None
    k, v = line.split("=", 1)
    k = k.strip().lower().replace(" ", "_").replace("(ns)", "ns").replace("(m)", "m")
    v = v.split(",", 1)[0].strip()
    try:
        return k, float(v)
    except Exception:
        return None


def read_uavgpr_csv(path: str | Path, max_traces: Optional[int] = None) -> tuple[UavGprCsvMeta, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(path)
    meta_vals: Dict[str, float] = {}
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for _ in range(10):
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            parsed = _parse_meta_line(line)
            if parsed is not None:
                meta_vals[parsed[0]] = parsed[1]
            else:
                f.seek(pos)
                break
        data_start = f.tell()
    samples = int(meta_vals.get("number_of_samples", 501))
    time_window_ns = float(meta_vals.get("time_windows_ns", meta_vals.get("time_window_ns", 700.0)))
    traces_total = int(meta_vals.get("number_of_traces", 0))
    trace_interval = float(meta_vals.get("trace_interval_m", 0.0))
    rows_needed = int(max_traces) * samples if max_traces and max_traces > 0 else None
    vals = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        f.seek(data_start)
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if rows_needed is not None and i >= rows_needed:
                break
            if len(row) < 4:
                continue
            try:
                vals.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]) if len(row) > 4 and row[4] != "" else np.nan])
            except Exception:
                continue
    if not vals:
        raise ValueError(f"No numeric rows found: {p}")
    arr = np.asarray(vals, dtype=np.float64)
    traces_read = arr.shape[0] // samples
    arr = arr[:traces_read * samples]
    data = arr.reshape(traces_read, samples, 5)
    bscan = data[:, :, 3].T
    lon = data[:, 0, 0]
    lat = data[:, 0, 1]
    elev = data[:, 0, 2]
    extra = data[:, 0, 4]
    if traces_total <= 0:
        traces_total = traces_read
    return UavGprCsvMeta(samples, time_window_ns, traces_total, trace_interval), bscan, lon, lat, elev, extra


def robust_normalize(x: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    scale = float(np.percentile(np.abs(x), clip_percentile))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(np.abs(x)) or 1.0)
    return np.clip(x / scale, -1.0, 1.0)


def subtract_mean_background(bscan: np.ndarray) -> np.ndarray:
    return bscan - np.mean(bscan, axis=1, keepdims=True)


def exponential_gain(bscan: np.ndarray, time_window_ns: float, strength: float = 1.4) -> np.ndarray:
    t = np.linspace(0.0, 1.0, bscan.shape[0]).reshape(-1, 1)
    return bscan * np.exp(strength * t)


def snr_db(bscan: np.ndarray, signal_window: tuple[int, int], noise_window: tuple[int, int]) -> float:
    s0, s1 = max(0, signal_window[0]), min(bscan.shape[0], signal_window[1])
    n0, n1 = max(0, noise_window[0]), min(bscan.shape[0], noise_window[1])
    ps = float(np.mean(np.square(bscan[s0:s1, :]))) if s1 > s0 else np.nan
    pn = float(np.mean(np.square(bscan[n0:n1, :]))) if n1 > n0 else np.nan
    if not np.isfinite(ps) or not np.isfinite(pn) or pn <= 0:
        return float("nan")
    return 10.0 * math.log10(ps / pn)


def summarize_real_csv(path: str | Path, max_traces: Optional[int] = 300) -> RealCsvSummary:
    meta, bscan, *_ = read_uavgpr_csv(path, max_traces=max_traces)
    distance = bscan.shape[1] * meta.trace_interval_m if meta.trace_interval_m else float("nan")
    return RealCsvSummary(str(Path(path)), tuple(int(x) for x in bscan.shape), meta, float(np.nanmin(bscan)), float(np.nanmax(bscan)), float(np.nanmean(bscan)), float(np.nanstd(bscan)), float(distance))


def _save_bscan_png(bscan: np.ndarray, path: Path, title: str, time_window_ns: float) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    v = float(np.percentile(np.abs(bscan), 99.0))
    if not np.isfinite(v) or v <= 0:
        v = float(np.max(np.abs(bscan)) or 1.0)
    fig = plt.figure(figsize=(10, 4.8), dpi=160)
    ax = fig.add_subplot(111)
    im = ax.imshow(bscan, aspect="auto", cmap="gray", vmin=-v, vmax=v, extent=[0, bscan.shape[1], time_window_ns, 0])
    ax.set_title(title); ax.set_xlabel("Trace index"); ax.set_ylabel("Time (ns)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def convert_real_csv(path: str | Path, out_dir: str | Path, max_traces: Optional[int] = None, make_baselines: bool = True) -> Dict[str, object]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta, bscan, lon, lat, elev, extra = read_uavgpr_csv(path, max_traces=max_traces)
    bg = subtract_mean_background(bscan)
    gained = exponential_gain(bg, meta.time_window_ns)
    norm = robust_normalize(gained)
    np.savez_compressed(out / "real_uavgpr_bscan_preview.npz", bscan=bscan, background_removed=bg, gained=gained, normalized=norm, lon=lon, lat=lat, elevation=elev, extra=extra, meta=asdict(meta))
    if make_baselines:
        _save_bscan_png(bscan, out / "bscan_raw.png", "Raw UavGPR B-scan", meta.time_window_ns)
        _save_bscan_png(bg, out / "bscan_mean_background_removed.png", "Mean-background removed", meta.time_window_ns)
        _save_bscan_png(norm, out / "bscan_gain_normalized.png", "Gain + robust normalized", meta.time_window_ns)
    n = bscan.shape[0]
    signal = (int(n * 350 / meta.time_window_ns), int(n * 500 / meta.time_window_ns)) if meta.time_window_ns else (int(n * 0.5), int(n * 0.7))
    noise = (int(n * 80 / meta.time_window_ns), int(n * 220 / meta.time_window_ns)) if meta.time_window_ns else (int(n * 0.1), int(n * 0.3))
    rep = {
        "out_dir": str(out),
        "bscan_shape": list(bscan.shape),
        "meta": asdict(meta),
        "snr_raw_db_default_windows": snr_db(bscan, signal, noise),
        "snr_background_removed_db_default_windows": snr_db(bg, signal, noise),
        "files": ["real_uavgpr_bscan_preview.npz", "bscan_raw.png", "bscan_mean_background_removed.png", "bscan_gain_normalized.png"],
    }
    (out / "qc_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep
