from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
except Exception:  # pragma: no cover - optional in headless minimal installs
    plt = None

from .postprocess import merge_available_bscan_for_input

_SCENEWORLD_BSCAN_ALIAS = {
    "raw": "raw_bscan.npy",
    "target_only": "target_only_bscan.npy",
    "background_only": "background_only_bscan.npy",
    "clutter_only": "clutter_only_bscan.npy",
    "air_only": "air_only_bscan.npy",
    "clutter_gt": "clutter_gt_bscan.npy",
}

_SCENEWORLD_LEGACY_BSCAN_ALIAS = {
    "target_only": "target_bscan.npy",
    "background_only": "background_bscan.npy",
    "clutter_only": "clutter_bscan.npy",
    "air_only": "air_bscan.npy",
}


@dataclass
class HistoryPreview:
    job_id: str
    case_id: str
    variant: str
    status: str
    input_file: str
    label_json: str = ""
    model_preview_png: str = ""
    bscan_preview_png: str = ""
    bscan_npz: str = ""
    bscan_shape: str = ""
    available_traces: int = 0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def find_label_json_for_input(input_file: str | Path, workspace: str | Path | None = None, case_id: str = "") -> Optional[Path]:
    """Resolve the labels.json created by scenario.generate_cases for a .in file.

    The generated layout is normally:
        <workspace>/models/<case_id>/<variant>.in
        <workspace>/datasets/<case_id>_labels.json
    We keep a few fallbacks so copied or older workspaces still work.
    """
    inp = Path(input_file)
    candidates: list[Path] = []
    cid = case_id or inp.parent.name
    if workspace:
        candidates.append(Path(workspace) / "datasets" / f"{cid}_labels.json")
    # models/case/raw.in -> workspace is parents[1]
    try:
        candidates.append(inp.parent.parent.parent / "datasets" / f"{cid}_labels.json")
    except Exception:
        pass
    try:
        candidates.append(inp.parent.parent / "datasets" / f"{cid}_labels.json")
    except Exception:
        pass
    candidates.append(inp.with_name(f"{cid}_labels.json"))
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def _read_label(label_json: str | Path) -> Dict[str, Any]:
    return json.loads(Path(label_json).read_text(encoding="utf-8"))


def render_model_preview(label_json: str | Path, out_png: str | Path, title: str | None = None, width: int = 760, height: int = 420) -> Optional[Path]:
    """Render a readable 2.5D model thumbnail from labels.json.

    This deliberately avoids heavy VTK/PyVista so it works in the same light GUI
    environment as the rest of the application. The main 3D page can still show a
    Matplotlib 3D surface; history thumbnails use this faster cross-section view.
    """
    if plt is None:
        return None
    p = Path(label_json)
    if not p.exists():
        return None
    data = _read_label(p)
    x = np.asarray(data.get("x_m", []), dtype=float)
    ground = np.asarray(data.get("ground_y_m", []), dtype=float)
    iface = np.asarray(data.get("interface_y_m", []), dtype=float)
    if x.size < 2 or ground.size != x.size or iface.size != x.size:
        return None
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    dpi = 120
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ymin = max(0.0, float(np.nanmin(iface)) - 1.0)
    ymax = float(np.nanmax(ground)) + float((data.get("radar") or {}).get("nominal_flight_height_m") or 8.0) + 1.0
    ax.fill_between(x, ymin, iface, alpha=0.28, label="基岩/下伏层")
    ax.fill_between(x, iface, ground, alpha=0.36, label="覆盖层")
    ax.plot(x, ground, linewidth=1.8, label="地表")
    ax.plot(x, iface, linewidth=2.0, label="基覆界面")
    fh = float((data.get("radar") or {}).get("nominal_flight_height_m") or 8.0)
    traj = data.get("trajectory") or {}
    if traj.get("mode") == "constant_level" and traj.get("source_y_m") is not None:
        sy = float(traj.get("source_y_m"))
        ax.plot([float(np.min(x)), float(np.max(x))], [sy, sy], linewidth=1.5, linestyle="--", label="constant-level flight path")
    else:
        ax.plot(x, ground + fh, linewidth=1.5, linestyle="--", label="UAV preview height")
    ax.set_xlabel("测线距离 x (m)")
    ax.set_ylabel("模型高度 y (m)")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_title(title or f"模型预览 {data.get('case_id', p.stem)}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out.resolve()


def _npz_from_postprocess(postprocess: Any) -> Optional[Path]:
    if not isinstance(postprocess, dict):
        return None
    products = postprocess.get("products")
    if isinstance(products, dict):
        raw = products.get("raw")
        if isinstance(raw, dict) and raw.get("npz"):
            p = Path(str(raw["npz"]))
            if p.exists():
                return p.resolve()
        # fallback: first NPZ product
        for val in products.values():
            if isinstance(val, dict) and val.get("npz"):
                p = Path(str(val["npz"]))
                if p.exists():
                    return p.resolve()
    return None


def load_bscan_for_history(input_file: str | Path, marker_data: Optional[Dict[str, Any]] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Load completed or live partial B-scan for a history record.

    Preference order:
    1) postprocess raw NPZ created after completed runs;
    2) currently readable gprMax .out files next to the .in file.
    The second path is what enables running-task previews while traces are being
    produced one by one.
    """
    meta: Dict[str, Any] = {}
    if marker_data:
        bscan_npy = marker_data.get("bscan_npy") or marker_data.get("output_bscan_npy")
        if bscan_npy:
            p = Path(str(bscan_npy))
            if p.exists():
                try:
                    arr = np.asarray(np.load(p), dtype=float)
                    meta.update({"source": "sceneworld_npy", "npy": str(p), "available_traces": int(arr.shape[1] if arr.ndim == 2 else 0)})
                    return arr, meta
                except Exception as exc:
                    meta["npy_error"] = repr(exc)
        npz = _npz_from_postprocess(marker_data.get("postprocess"))
        if npz:
            try:
                with np.load(npz) as z:
                    if "bscan" in z:
                        arr = np.asarray(z["bscan"], dtype=float)
                        meta.update({"source": "postprocess_npz", "npz": str(npz), "available_traces": int(arr.shape[1] if arr.ndim == 2 else 0)})
                        return arr, meta
            except Exception as exc:
                meta["npz_error"] = repr(exc)
    # SceneWorld fallback: models/<case>/<variant>.in -> models/<case>/outputs/<variant>_bscan.npy
    try:
        inp = Path(input_file)
        variant = inp.stem
        npy = inp.parent / "outputs" / _SCENEWORLD_BSCAN_ALIAS.get(variant, f"{variant}_bscan.npy")
        if not npy.exists() and variant in _SCENEWORLD_LEGACY_BSCAN_ALIAS:
            legacy = inp.parent / "outputs" / _SCENEWORLD_LEGACY_BSCAN_ALIAS[variant]
            if legacy.exists():
                npy = legacy
        if npy.exists():
            arr = np.asarray(np.load(npy), dtype=float)
            meta.update({"source": "sceneworld_inferred_npy", "npy": str(npy), "available_traces": int(arr.shape[1] if arr.ndim == 2 else 0)})
            return arr, meta
    except Exception as exc:
        meta["sceneworld_inferred_error"] = repr(exc)
    merged = merge_available_bscan_for_input(input_file)
    if merged is None:
        return None, meta
    arr, m = merged
    meta.update({"source": "live_out", "available_traces": int(arr.shape[1] if arr.ndim == 2 else 0), "merge_meta": m})
    return arr, meta


def render_bscan_preview(bscan: np.ndarray, out_png: str | Path, title: str = "B-scan", time_window_ns: float = 700.0, width: int = 760, height: int = 420) -> Optional[Path]:
    if plt is None:
        return None
    arr = np.asarray(bscan, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.nanpercentile(np.abs(arr), 99.0)) if arr.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(arr)) or 1.0)
    dpi = 120
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(arr, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=[0, arr.shape[1], float(time_window_ns), 0])
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time (ns)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out.resolve()


def build_history_preview(record: Any, workspace: str | Path, marker_data: Optional[Dict[str, Any]] = None, time_window_ns: float = 700.0, make_png: bool = True) -> HistoryPreview:
    """Build preview metadata and optional thumbnails for a HistoryRecord-like object."""
    root = Path(workspace)
    previews_dir = root / "previews" / "history"
    job_id = str(getattr(record, "job_id", ""))
    cid = str(getattr(record, "case_id", ""))
    variant = str(getattr(record, "variant", ""))
    status = str(getattr(record, "status", ""))
    input_file = str(getattr(record, "input_file", ""))
    label = find_label_json_for_input(input_file, root, cid)
    model_png = ""
    bscan_png = ""
    note_parts: list[str] = []
    if make_png and label:
        p = render_model_preview(label, previews_dir / f"{job_id}_model.png", title=f"{cid} | {variant} | 模型")
        if p:
            model_png = str(p)
    arr, meta = load_bscan_for_history(input_file, marker_data=marker_data)
    bscan_npz = str(meta.get("npz") or "")
    bshape = ""
    ntr = 0
    if arr is not None:
        bshape = "x".join(str(x) for x in arr.shape)
        ntr = int(arr.shape[1] if arr.ndim == 2 else 0)
        if make_png:
            p = render_bscan_preview(arr, previews_dir / f"{job_id}_bscan.png", title=f"{cid} | {variant} | B-scan {ntr} traces", time_window_ns=time_window_ns)
            if p:
                bscan_png = str(p)
    else:
        note_parts.append("no readable B-scan yet")
    if label is None:
        note_parts.append("label_json not found")
    return HistoryPreview(
        job_id=job_id,
        case_id=cid,
        variant=variant,
        status=status,
        input_file=input_file,
        label_json=str(label or ""),
        model_preview_png=model_png,
        bscan_preview_png=bscan_png,
        bscan_npz=bscan_npz,
        bscan_shape=bshape,
        available_traces=ntr,
        note="; ".join(note_parts),
    )
