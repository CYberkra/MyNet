from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uavgpr_simlab.core.real_data import (
    convert_real_csv,
    read_uavgpr_csv,
    robust_normalize,
    subtract_mean_background,
)


@dataclass(frozen=True)
class RealCsvPreview:
    """Prepared real-UavGPR CSV preview for advanced GUI rendering."""

    normalized_bscan: np.ndarray
    raw_bscan: np.ndarray
    info: dict[str, Any]
    time_window_ns: float


def load_real_csv_preview(path: str | Path, max_traces: int | None = 300) -> RealCsvPreview:
    """Read a real UAV-GPR CSV and prepare the normalized B-scan preview.

    This function centralizes the CSV preview pipeline used by the advanced GUI:
    read -> mean-background subtraction -> robust normalization -> metadata summary.
    It deliberately keeps the same processing semantics that previously lived in
    ``gui/main_window.py``.
    """

    meta, bscan, *_ = read_uavgpr_csv(path, max_traces=max_traces)
    background_removed = subtract_mean_background(bscan)
    normalized = robust_normalize(background_removed)
    info = {
        "shape_samples_x_traces": [int(x) for x in bscan.shape],
        "samples": int(meta.samples),
        "time_window_ns": float(meta.time_window_ns),
        "traces_total_in_file": int(meta.traces),
        "trace_interval_m": float(meta.trace_interval_m),
        "distance_preview_m": float(bscan.shape[1] * meta.trace_interval_m),
        "amplitude_min": float(np.nanmin(bscan)),
        "amplitude_max": float(np.nanmax(bscan)),
        "amplitude_std": float(np.nanstd(bscan)),
    }
    return RealCsvPreview(
        normalized_bscan=normalized,
        raw_bscan=bscan,
        info=info,
        time_window_ns=float(meta.time_window_ns),
    )


def export_real_csv_qc(
    path: str | Path,
    workspace: str | Path,
    max_traces: int | None = 300,
    *,
    make_baselines: bool = True,
) -> dict[str, object]:
    """Convert a real UAV-GPR CSV into the standard QC output directory."""

    csv_path = Path(path)
    out_dir = Path(workspace) / "real_csv_qc" / csv_path.stem
    return convert_real_csv(csv_path, out_dir, max_traces=max_traces, make_baselines=make_baselines)
