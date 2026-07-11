"""Canonical trace-level supervision semantics for measured and simulated data.

The legacy ``status_code`` alone is ambiguous because weak-positive and ignored
traces may both use status 2 in historical archives.  Runtime supervision must
therefore be derived jointly from status, label weight, and the ignore mask.
"""
from __future__ import annotations

from typing import Final

import numpy as np

CONFIRMED_NEGATIVE: Final[int] = 0
STRONG_POSITIVE: Final[int] = 1
WEAK_POSITIVE: Final[int] = 2
IGNORE: Final[int] = 3

SUPERVISION_STATE_NAMES: Final[dict[int, str]] = {
    CONFIRMED_NEGATIVE: "confirmed_negative",
    STRONG_POSITIVE: "strong_positive",
    WEAK_POSITIVE: "weak_positive",
    IGNORE: "ignore",
}


def _trace_ignore_mask(ignore_mask: np.ndarray | None, trace_count: int) -> np.ndarray:
    """Collapse a pixel/trace ignore mask to one boolean value per trace."""
    if ignore_mask is None:
        return np.zeros(trace_count, dtype=bool)
    array = np.asarray(ignore_mask, dtype=np.float32)
    if array.ndim == 0:
        return np.full(trace_count, bool(array > 0.5), dtype=bool)
    if array.shape[-1] != trace_count:
        raise ValueError(
            f"ignore_mask last dimension {array.shape[-1]} does not match trace count {trace_count}"
        )
    if array.ndim == 1:
        return array > 0.5
    reduce_axes = tuple(range(array.ndim - 1))
    return np.mean(array, axis=reduce_axes) > 0.5


def derive_supervision_state(
    status_code: np.ndarray,
    label_weight: np.ndarray,
    ignore_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Derive an unambiguous trace-level supervision state.

    State meanings:
      0 confirmed negative: non-ignored status 0 trace
      1 strong positive: non-ignored, positive-weight status 1 trace
      2 weak positive: non-ignored, positive-weight status 2 trace
      3 ignore: explicitly ignored, zero-weight positive, or unknown status

    ``label_weight`` governs curve/mask supervision. A confirmed-negative trace
    normally has zero curve weight but remains valid negative presence
    supervision unless explicitly ignored.

    This function deliberately does not infer negatives from missing, weak, or
    ambiguous labels.
    """
    status = np.asarray(status_code, dtype=np.int64)
    weight = np.asarray(label_weight, dtype=np.float32)
    if status.ndim != 1 or weight.ndim != 1:
        raise ValueError(f"status_code and label_weight must be 1-D, got {status.shape} and {weight.shape}")
    if status.shape != weight.shape:
        raise ValueError(f"status_code shape {status.shape} does not match label_weight shape {weight.shape}")
    if not np.isfinite(weight).all():
        raise ValueError("label_weight contains NaN/Inf")

    ignored = _trace_ignore_mask(ignore_mask, status.size)
    state = np.full(status.shape, IGNORE, dtype=np.int8)
    state[(~ignored) & (status == 0)] = CONFIRMED_NEGATIVE
    active_positive = (~ignored) & (weight > 0.0)
    state[active_positive & (status == 1)] = STRONG_POSITIVE
    state[active_positive & (status == 2)] = WEAK_POSITIVE
    return state
