#!/usr/bin/env python3
"""Audit continuity of a signed gprMax full-minus-control interface response."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


def collect(case_dir: Path, stem: str, expected_trace_count: int) -> list[Path]:
    stem_path = Path(stem)
    output_dir = case_dir / stem_path.parent
    basename = stem_path.name
    pattern = re.compile(rf"^{re.escape(basename)}(\d+)\.out$")
    indexed = []
    for path in output_dir.glob(f"{basename}*.out"):
        match = pattern.match(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    expected = list(range(1, expected_trace_count + 1))
    if [index for index, _ in indexed] != expected:
        raise RuntimeError(f"Expected {basename} outputs {expected[0]}..{expected[-1]} in {output_dir}")
    return [path for _, path in indexed]


def stack(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    columns: list[np.ndarray] = []
    time_ns: np.ndarray | None = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle["rxs"]["rx1"]["Ez"], dtype=np.float64)
            local_time = np.arange(values.size, dtype=np.float64) * float(handle.attrs["dt"]) * 1e9
        if time_ns is None:
            time_ns = local_time
        elif values.shape != columns[0].shape or not np.allclose(time_ns, local_time):
            raise RuntimeError(f"Inconsistent time contract at {path}")
        columns.append(values)
    if time_ns is None:
        raise RuntimeError("No receiver outputs")
    return np.column_stack(columns), time_ns


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > np.finfo(np.float64).eps else 0.0


def smooth_signed_difference_path(
    difference: np.ndarray,
    time_ns: np.ndarray,
    reference_ns: np.ndarray,
    *,
    search_half_width_ns: float = 55.0,
    max_jump_ns: float = 2.5,
    smooth_weight: float = 0.012,
) -> np.ndarray:
    """Pick one continuous high-energy lobe inside each reference search window.

    This is intentionally an audit-time path constraint, not a training-label
    generator. The physical model defines a continuous basal interface, so a
    valid visible phase must be explainable by one continuous wavelet lobe.
    """
    dt_ns = float(time_ns[1] - time_ns[0])
    lower = max(0, int(np.floor((float(np.min(reference_ns)) - search_half_width_ns) / dt_ns)))
    upper = min(difference.shape[0] - 1, int(np.ceil((float(np.max(reference_ns)) + search_half_width_ns) / dt_ns)))
    amplitude = np.abs(difference[lower : upper + 1])
    column_scale = np.maximum(np.max(amplitude, axis=0, keepdims=True), np.finfo(np.float64).eps)
    probability = amplitude / column_scale
    local_time = time_ns[lower : upper + 1]
    allowed = np.abs(local_time[:, None] - reference_ns[None, :]) <= search_half_width_ns
    unary = -np.log(np.clip(probability, 1e-8, 1.0))
    unary[~allowed] += 40.0
    height, width = unary.shape
    max_jump = max(1, int(round(max_jump_ns / dt_ns)))
    offsets = np.arange(-max_jump, max_jump + 1, dtype=np.int32)
    dp = np.empty((height, width), dtype=np.float64)
    back = np.zeros((height, width), dtype=np.int32)
    dp[:, 0] = unary[:, 0]
    large = 1e12
    for trace in range(1, width):
        previous = dp[:, trace - 1]
        candidates = np.full((offsets.size, height), large, dtype=np.float64)
        # ``offset`` is predecessor minus current sample in the recurrence.
        expected_offset = float((reference_ns[trace - 1] - reference_ns[trace]) / dt_ns)
        for offset_index, offset in enumerate(offsets):
            # Penalise departure from the locally expected geometric move,
            # rather than treating a real sloping interface as an error.
            penalty = smooth_weight * float((offset - expected_offset) ** 2)
            if offset < 0:
                candidates[offset_index, -offset:] = previous[:offset] + penalty
            elif offset > 0:
                candidates[offset_index, :-offset] = previous[offset:] + penalty
            else:
                candidates[offset_index] = previous
        choice = np.argmin(candidates, axis=0)
        dp[:, trace] = unary[:, trace] + candidates[choice, np.arange(height)]
        back[:, trace] = np.clip(np.arange(height) + offsets[choice], 0, height - 1)
    path = np.zeros(width, dtype=np.int32)
    path[-1] = int(np.argmin(dp[:, -1]))
    for trace in range(width - 1, 0, -1):
        path[trace - 1] = back[path[trace], trace]
    return local_time[path]


def path_wavelet_metrics(
    difference: np.ndarray, time_ns: np.ndarray, path_ns: np.ndarray
) -> dict[str, float]:
    half_window_samples = max(2, int(round(18.0 / (time_ns[1] - time_ns[0]))))
    snippets = []
    for trace, centre in enumerate(path_ns):
        peak_index = int(np.argmin(np.abs(time_ns - centre)))
        start = max(0, peak_index - half_window_samples)
        end = min(difference.shape[0], peak_index + half_window_samples + 1)
        snippet = difference[start:end, trace]
        snippets.append(np.pad(snippet, (0, 2 * half_window_samples + 1 - snippet.size)))
    correlations = np.asarray([safe_correlation(left, right) for left, right in zip(snippets[:-1], snippets[1:])])
    steps = np.abs(np.diff(path_ns))
    return {
        "path_adjacent_time_step_abs_ns_median": float(np.median(steps)),
        "path_adjacent_time_step_abs_ns_p95": float(np.quantile(steps, 0.95)),
        "path_adjacent_wavelet_correlation_median": float(np.median(correlations)),
        "path_adjacent_wavelet_correlation_p10": float(np.quantile(correlations, 0.10)),
        "path_weak_adjacent_correlation_fraction_below_0_5": float(np.mean(correlations < 0.5)),
    }


def response_metrics(difference: np.ndarray, time_ns: np.ndarray, reference_ns: np.ndarray) -> dict:
    """Extract a local signed-pair event without promoting it to a label."""
    peak_indices = []
    peak_amplitudes = []
    target_rms = []
    background_rms = []
    half_window_samples = max(2, int(round(18.0 / (time_ns[1] - time_ns[0]))))
    snippets = []
    for trace, centre in enumerate(reference_ns):
        target = np.flatnonzero(np.abs(time_ns - centre) <= 55.0)
        if not target.size:
            raise RuntimeError(f"Reference {centre} ns lies outside time window")
        peak_index = int(target[np.argmax(np.abs(difference[target, trace]))])
        peak_indices.append(peak_index)
        peak_amplitudes.append(float(abs(difference[peak_index, trace])))
        target_rms.append(float(np.sqrt(np.mean(np.square(difference[target, trace])))))
        adjacent = ((time_ns >= centre - 105.0) & (time_ns <= centre - 65.0)) | (
            (time_ns >= centre + 65.0) & (time_ns <= centre + 105.0)
        )
        background_rms.append(float(np.sqrt(np.mean(np.square(difference[adjacent, trace])))))
        start = max(0, peak_index - half_window_samples)
        end = min(difference.shape[0], peak_index + half_window_samples + 1)
        snippet = difference[start:end, trace]
        snippets.append(np.pad(snippet, (0, 2 * half_window_samples + 1 - snippet.size)))
    peaks_ns = time_ns[np.asarray(peak_indices)]
    peak_amplitudes = np.asarray(peak_amplitudes)
    target_rms = np.asarray(target_rms)
    background_rms = np.asarray(background_rms)
    correlations = np.asarray([safe_correlation(left, right) for left, right in zip(snippets[:-1], snippets[1:])])
    median_amplitude = max(float(np.median(peak_amplitudes)), np.finfo(np.float64).eps)
    constrained_path = smooth_signed_difference_path(difference, time_ns, reference_ns)
    return {
        "visible_phase_peak_ns": peaks_ns.tolist(),
        "visible_phase_offset_from_geometric_ns": (peaks_ns - reference_ns).tolist(),
        "target_to_adjacent_difference_rms": float(np.mean(target_rms) / max(float(np.mean(background_rms)), np.finfo(np.float64).eps)),
        "peak_amplitude_median": float(np.median(peak_amplitudes)),
        "dropout_fraction_below_25pct_median": float(np.mean(peak_amplitudes < 0.25 * median_amplitude)),
        "adjacent_peak_time_step_abs_ns_median": float(np.median(np.abs(np.diff(peaks_ns)))),
        "adjacent_peak_time_step_abs_ns_p95": float(np.quantile(np.abs(np.diff(peaks_ns)), 0.95)),
        "adjacent_wavelet_correlation_median": float(np.median(correlations)),
        "adjacent_wavelet_correlation_p10": float(np.quantile(correlations, 0.10)),
        "weak_adjacent_correlation_fraction_below_0_5": float(np.mean(correlations < 0.5)),
        "geometric_to_visible_offset_ns_median": float(np.median(peaks_ns - reference_ns)),
        "geometric_to_visible_offset_ns_p95_abs": float(np.quantile(np.abs(peaks_ns - reference_ns), 0.95)),
        "path_constrained_signed_difference_peak_ns": constrained_path.tolist(),
        "path_constrained_offset_from_geometric_ns": (constrained_path - reference_ns).tolist(),
        **path_wavelet_metrics(difference, time_ns, constrained_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--full-stem", default="full_scene")
    parser.add_argument("--control-stem", default="no_basal_contrast_control")
    parser.add_argument("--arrival-label", type=Path, default=Path("labels/geometric_reference_arrival_time_ns.npy"))
    parser.add_argument("--expected-trace-count", type=int, required=True)
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    full, time_ns = stack(collect(case_dir, args.full_stem, args.expected_trace_count))
    control, control_time_ns = stack(collect(case_dir, args.control_stem, args.expected_trace_count))
    if full.shape != control.shape or not np.allclose(time_ns, control_time_ns):
        raise RuntimeError("Full/control time contract mismatch")
    reference = np.asarray(np.load(case_dir / args.arrival_label), dtype=np.float64)[: args.expected_trace_count]
    if reference.shape != (args.expected_trace_count,):
        raise RuntimeError("Reference-arrival label count does not match expected trace count")
    metrics = response_metrics(full - control, time_ns, reference)
    report = {
        "case_id": case_dir.name,
        "development_only": True,
        "formal_training_allowed": False,
        "trace_count": args.expected_trace_count,
        "dt_ns": float(time_ns[1] - time_ns[0]),
        "metrics": metrics,
        "status": "descriptive only; visible-phase candidates require human review before any label use",
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        case_dir.name,
        f"corr_median={metrics['adjacent_wavelet_correlation_median']:.3f}",
        f"dropout={metrics['dropout_fraction_below_25pct_median']:.3%}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
