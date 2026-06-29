from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting backend
    plt = None


@dataclass
class BoreholePick:
    """One weak-supervision pick for a borehole or interpreted interface point.

    The generator accepts either ``trace_index`` directly, or ``distance_m`` / ``x_m``
    that can be converted to a trace index using ``trace_interval_m``.
    ``depth_m`` is converted to two-way travel time by ``velocity_m_per_ns``.
    """

    depth_m: float
    trace_index: Optional[float] = None
    distance_m: Optional[float] = None
    x_m: Optional[float] = None
    uncertainty_m: float = 1.0
    line_id: str = ""
    borehole_id: str = ""
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FIELD_ALIASES = {
    "line": "line_id",
    "line_name": "line_id",
    "lineid": "line_id",
    "borehole": "borehole_id",
    "borehole_name": "borehole_id",
    "zk": "borehole_id",
    "hole_id": "borehole_id",
    "trace": "trace_index",
    "trace_idx": "trace_index",
    "trace_id": "trace_index",
    "idx": "trace_index",
    "distance": "distance_m",
    "distance_m": "distance_m",
    "station_m": "distance_m",
    "x": "x_m",
    "x_m": "x_m",
    "depth": "depth_m",
    "depth_m": "depth_m",
    "interface_depth_m": "depth_m",
    "bedrock_depth_m": "depth_m",
    "uncertainty": "uncertainty_m",
    "uncertainty_m": "uncertainty_m",
    "sigma_m": "uncertainty_m",
    "half_width_m": "uncertainty_m",
    "weight": "weight",
}


def _norm_key(k: str) -> str:
    kk = k.strip().lower().replace(" ", "_").replace("-", "_")
    return _FIELD_ALIASES.get(kk, kk)


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, str) and not v.strip():
        return default
    try:
        x = float(v)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


def load_borehole_picks(path: str | Path, line_id: str | None = None) -> list[BoreholePick]:
    """Load borehole/interface weak labels from CSV or JSON.

    Supported CSV columns include: ``line_id``, ``borehole_id``, ``trace_index``,
    ``distance_m``, ``x_m``, ``depth_m``, ``uncertainty_m`` and ``weight``. Common
    aliases such as ``trace``, ``depth`` and ``ZK`` are normalized automatically.
    """

    p = Path(path)
    rows: list[dict[str, Any]] = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("picks", data.get("boreholes", data.get("items", [])))
        if not isinstance(data, list):
            raise ValueError(f"JSON weak-label file must contain a list of picks: {p}")
        rows = [dict(x) for x in data]
    else:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV weak-label file has no header: {p}")
            for row in reader:
                rows.append(dict(row))

    picks: list[BoreholePick] = []
    for row in rows:
        norm = {_norm_key(str(k)): v for k, v in row.items()}
        row_line = str(norm.get("line_id", "") or "")
        if line_id and row_line and row_line != str(line_id):
            continue
        depth = _to_float(norm.get("depth_m"), None)
        if depth is None:
            continue
        picks.append(
            BoreholePick(
                depth_m=float(depth),
                trace_index=_to_float(norm.get("trace_index"), None),
                distance_m=_to_float(norm.get("distance_m"), None),
                x_m=_to_float(norm.get("x_m"), None),
                uncertainty_m=float(_to_float(norm.get("uncertainty_m"), 1.0) or 1.0),
                line_id=row_line,
                borehole_id=str(norm.get("borehole_id", "") or ""),
                weight=float(_to_float(norm.get("weight"), 1.0) or 1.0),
            )
        )
    if not picks:
        raise ValueError(f"No usable borehole/interface picks found in {p}")
    return picks


def _extract_meta_from_npz(npz_path: str | Path) -> tuple[tuple[int, int], float, Optional[float], Optional[np.ndarray]]:
    """Return (shape, time_window_ns, trace_interval_m, preview_bscan)."""

    with np.load(npz_path, allow_pickle=True) as z:
        if "normalized" in z:
            bscan = np.asarray(z["normalized"])
        elif "bscan" in z:
            bscan = np.asarray(z["bscan"])
        else:
            first = next(iter(z.files))
            bscan = np.asarray(z[first])
        if bscan.ndim != 2:
            raise ValueError(f"Expected a 2-D bscan array in {npz_path}, got {bscan.shape}")
        time_window_ns = 700.0
        trace_interval_m: Optional[float] = None
        if "time_window_ns" in z:
            time_window_ns = float(np.asarray(z["time_window_ns"]).reshape(-1)[0])
        if "meta" in z:
            meta_obj = z["meta"]
            try:
                meta = meta_obj.item() if hasattr(meta_obj, "item") else meta_obj.tolist()
            except Exception:
                meta = {}
            if isinstance(meta, dict):
                time_window_ns = float(meta.get("time_window_ns", time_window_ns))
                ti = meta.get("trace_interval_m", None)
                if ti not in (None, ""):
                    try:
                        trace_interval_m = float(ti)
                    except Exception:
                        pass
        return (int(bscan.shape[0]), int(bscan.shape[1])), time_window_ns, trace_interval_m, bscan


def depth_to_time_ns(depth_m: float, velocity_m_per_ns: float, two_way: bool = True) -> float:
    if velocity_m_per_ns <= 0:
        raise ValueError("velocity_m_per_ns must be positive")
    factor = 2.0 if two_way else 1.0
    return factor * float(depth_m) / float(velocity_m_per_ns)


def _trace_position(pick: BoreholePick, trace_interval_m: Optional[float], n_traces: int) -> Optional[float]:
    if pick.trace_index is not None:
        return float(pick.trace_index)
    distance = pick.distance_m if pick.distance_m is not None else pick.x_m
    if distance is not None and trace_interval_m and trace_interval_m > 0:
        return float(distance) / float(trace_interval_m)
    if distance is not None and n_traces > 1:
        # Last-resort interpretation: x/distance is already normalized to the trace axis.
        return float(np.clip(distance, 0, n_traces - 1))
    return None


def build_borehole_soft_mask(
    shape: tuple[int, int],
    picks: Sequence[BoreholePick],
    time_window_ns: float = 700.0,
    velocity_m_per_ns: float = 0.10,
    trace_interval_m: Optional[float] = None,
    trace_sigma: float = 4.0,
    time_sigma_ns: Optional[float] = None,
    two_way_time: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create a Gaussian soft mask for borehole weak-supervision regions.

    The output mask has the same ``samples x traces`` shape as a B-scan. Values are
    in [0, 1]. Each pick contributes a localized Gaussian around its converted
    trace/time location, with the time uncertainty derived from ``uncertainty_m``
    unless ``time_sigma_ns`` is explicitly provided.
    """

    nt, nx = int(shape[0]), int(shape[1])
    yy = np.arange(nt, dtype=float)[:, None]
    xx = np.arange(nx, dtype=float)[None, :]
    mask = np.zeros((nt, nx), dtype=np.float32)
    used: list[dict[str, Any]] = []
    sample_per_ns = (nt - 1) / max(float(time_window_ns), 1e-9)
    for p in picks:
        trace = _trace_position(p, trace_interval_m, nx)
        if trace is None:
            continue
        time_ns = depth_to_time_ns(p.depth_m, velocity_m_per_ns, two_way=two_way_time)
        sample = time_ns * sample_per_ns
        if sample < -nt or sample > 2 * nt:
            # Still record the skipped pick so the report makes the mismatch obvious.
            used.append({**p.to_dict(), "trace_used": trace, "time_ns": time_ns, "sample_index": sample, "used": False, "reason": "outside_time_window"})
            continue
        sigma_t_ns = float(time_sigma_ns) if time_sigma_ns and time_sigma_ns > 0 else max(depth_to_time_ns(max(p.uncertainty_m, 0.05), velocity_m_per_ns, two_way=two_way_time), 1.0)
        sigma_y = max(sigma_t_ns * sample_per_ns, 1.0)
        sigma_x = max(float(trace_sigma), 0.5)
        contribution = np.exp(-0.5 * (((yy - sample) / sigma_y) ** 2 + ((xx - trace) / sigma_x) ** 2)) * max(float(p.weight), 0.0)
        mask = np.maximum(mask, contribution.astype(np.float32))
        used.append({**p.to_dict(), "trace_used": trace, "time_ns": time_ns, "sample_index": sample, "sigma_time_ns": sigma_t_ns, "sigma_trace": sigma_x, "used": True})
    if mask.size:
        maxv = float(mask.max())
        if maxv > 0:
            mask = np.clip(mask / maxv, 0.0, 1.0).astype(np.float32)
    return mask, used


def _save_mask_png(mask: np.ndarray, out_path: Path, title: str = "Borehole soft mask", bscan: Optional[np.ndarray] = None, time_window_ns: float = 700.0) -> None:
    if plt is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.0, 4.2), dpi=160)
    ax = fig.add_subplot(111)
    if bscan is not None and bscan.shape == mask.shape:
        vmax = float(np.percentile(np.abs(bscan), 99.0)) or 1.0
        ax.imshow(bscan, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=[0, mask.shape[1], time_window_ns, 0])
        ax.imshow(mask, aspect="auto", cmap="viridis", alpha=0.45, vmin=0, vmax=1, extent=[0, mask.shape[1], time_window_ns, 0])
    else:
        im = ax.imshow(mask, aspect="auto", vmin=0, vmax=1, extent=[0, mask.shape[1], time_window_ns, 0])
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time (ns)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_borehole_soft_mask(
    bscan_npz: str | Path,
    boreholes: str | Path,
    out_dir: str | Path,
    velocity_m_per_ns: float = 0.10,
    trace_interval_m: Optional[float] = None,
    line_id: str | None = None,
    trace_sigma: float = 4.0,
    time_sigma_ns: Optional[float] = None,
    two_way_time: bool = True,
) -> dict[str, Any]:
    """Generate and save weak-supervision soft masks from borehole picks."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shape, tw, ti_from_npz, bscan = _extract_meta_from_npz(bscan_npz)
    ti = trace_interval_m if trace_interval_m and trace_interval_m > 0 else ti_from_npz
    picks = load_borehole_picks(boreholes, line_id=line_id)
    mask, used = build_borehole_soft_mask(
        shape,
        picks,
        time_window_ns=tw,
        velocity_m_per_ns=velocity_m_per_ns,
        trace_interval_m=ti,
        trace_sigma=trace_sigma,
        time_sigma_ns=time_sigma_ns,
        two_way_time=two_way_time,
    )
    mask_path = out / "borehole_soft_mask.npy"
    np.save(mask_path, mask)
    npz_path = out / "borehole_soft_mask.npz"
    np.savez_compressed(npz_path, mask=mask, time_window_ns=float(tw), velocity_m_per_ns=float(velocity_m_per_ns), trace_interval_m=np.nan if ti is None else float(ti), picks=np.asarray(used, dtype=object))
    png_path = out / "borehole_soft_mask_overlay.png"
    _save_mask_png(mask, png_path, title="Borehole weak-supervision soft mask", bscan=bscan, time_window_ns=tw)
    used_csv = out / "borehole_picks_used.csv"
    fields = sorted({k for row in used for k in row.keys()}) or ["used"]
    with used_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in used:
            wr.writerow(row)
    rep = {
        "bscan_npz": str(Path(bscan_npz).resolve()),
        "boreholes": str(Path(boreholes).resolve()),
        "out_dir": str(out.resolve()),
        "mask_npy": str(mask_path.resolve()),
        "mask_npz": str(npz_path.resolve()),
        "mask_png": str(png_path.resolve()) if png_path.exists() else "",
        "picks_used_csv": str(used_csv.resolve()),
        "shape": list(mask.shape),
        "time_window_ns": float(tw),
        "trace_interval_m": None if ti is None else float(ti),
        "velocity_m_per_ns": float(velocity_m_per_ns),
        "line_id_filter": line_id or "",
        "total_picks_loaded": len(picks),
        "total_picks_used": int(sum(1 for x in used if x.get("used"))),
        "mask_nonzero_fraction": float(np.mean(mask > 1e-4)) if mask.size else 0.0,
        "mask_max": float(mask.max()) if mask.size else 0.0,
        "notes": [
            "Mask values are Gaussian soft weights in [0, 1].",
            "Depth-to-time conversion uses two-way travel time unless two_way_time=False.",
            "Use this mask as a weak preservation/consistency band; do not treat it as pixel-perfect clean-label ground truth.",
        ],
    }
    report_path = out / "soft_mask_report.json"
    report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep
