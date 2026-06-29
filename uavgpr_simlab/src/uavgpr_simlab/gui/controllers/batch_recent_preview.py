from __future__ import annotations

from typing import Any

import numpy as np


def reset_recent_bscan_preview(win: Any) -> None:
    """Reset the operator-facing recent-B-scan comparison strip."""

    win._recent_bscan_variants = []
    canvas = getattr(win, "batch_recent_canvas", None)
    if canvas is not None:
        canvas.show_bscan_grid([], "最近完成的 variant 对比", 700.0)


def append_recent_bscan_preview(win: Any, case_id: str, variant: str, bscan: np.ndarray, time_window_ns: float, label: str) -> None:
    """Append one completed B-scan to the recent preview strip and redraw it."""

    recent = list(getattr(win, "_recent_bscan_variants", []))
    arr = np.asarray(bscan, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return
    recent.append((f"{case_id}\n{label}", arr))
    recent = recent[-5:]
    win._recent_bscan_variants = recent
    canvas = getattr(win, "batch_recent_canvas", None)
    if canvas is not None:
        canvas.show_bscan_grid(recent, "最近完成的 B-scan（最多 5 个）", float(time_window_ns or 700.0))
