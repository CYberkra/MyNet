#!/usr/bin/env python3
"""Audit a complete full-scene B-scan without claiming causal attribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import extract_visible_phase  # noqa: E402
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _rms(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 0 else 0.0


def _aligned_traces(
    values: np.ndarray,
    time_ns: np.ndarray,
    path_ns: np.ndarray,
    *,
    half_width_ns: float,
    sample_step_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    relative = np.arange(-half_width_ns, half_width_ns + sample_step_ns * 0.5, sample_step_ns)
    aligned = np.empty((relative.size, values.shape[1]), dtype=np.float64)
    for trace, center in enumerate(path_ns):
        aligned[:, trace] = np.interp(center + relative, time_ns, values[:, trace])
    return relative, aligned


def _window_values(
    values: np.ndarray,
    time_ns: np.ndarray,
    centers_ns: np.ndarray,
    low_offset_ns: float,
    high_offset_ns: float,
) -> np.ndarray:
    chunks = []
    for trace, center in enumerate(centers_ns):
        keep = (time_ns >= center + low_offset_ns) & (time_ns <= center + high_offset_ns)
        chunks.append(values[keep, trace])
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def audit(
    run_dir: Path,
    output_dir: Path,
    *,
    component: str,
    line9_contract_path: Path | None,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = json.loads((run_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    trace_contract_path = run_dir / "run_logs" / "full_scene_trace_contract.json"
    trace_contract = json.loads(trace_contract_path.read_text(encoding="utf-8"))
    control_trace_count = len(list(run_dir.glob("no_basal_contrast_control[0-9]*.out")))
    control_merged = (run_dir / "no_basal_contrast_control_merged.out").is_file()
    dt_s, raw, _ = read_merged_bscan(run_dir / "full_scene_merged.out", component=component)
    raw = np.asarray(raw, dtype=np.float64)
    time_ns = np.arange(raw.shape[0], dtype=np.float64) * dt_s * 1e9
    protected_end_ns = float(scene["grid"]["protected_window_end_ns"])
    protected = time_ns <= protected_end_ns
    protected_time = time_ns[protected]
    protected_raw = raw[protected]
    declared_traces = int(scene["grid"]["trace_count"])
    selected = np.asarray(run.get("selected_trace_indices_zero_based", []), dtype=np.int64)
    expected_traces = int(run.get("requested_trace_count", selected.size or declared_traces))
    trace_stride = int(run.get("trace_stride", 1))
    finite = bool(np.isfinite(raw).all())

    reference_path = run_dir / "labels" / "source_referenced_arrival_time_ns.npy"
    reference_semantics = "geometric_interface_plus_explicit_source_reference_delay"
    if not reference_path.is_file():
        reference_path = run_dir / "labels" / "geometric_reference_arrival_time_ns.npy"
        reference_semantics = "geometric_interface_only"
    reference = np.load(reference_path).astype(np.float64)
    if reference.shape == (declared_traces,) and raw.shape[1] != declared_traces:
        if selected.shape != (raw.shape[1],):
            raise ValueError("run manifest selection does not match solved subset")
        reference = reference[selected]
    if reference.shape != (raw.shape[1],):
        raise ValueError(f"reference shape {reference.shape} does not match {raw.shape[1]} traces")
    common_mode = np.median(protected_raw, axis=1, keepdims=True)
    background_removed = protected_raw - common_mode
    morphology_path, support, _ = extract_visible_phase(
        background_removed,
        np.zeros_like(background_removed),
        protected_time,
        reference,
        search_half_width_ns=35.0,
        phase_half_width_ns=8.0,
        enforce_continuity=True,
        max_trace_step_ns=5.6 * trace_stride,
        geometric_anchor_weight=2.0,
    )
    if not np.isfinite(morphology_path).all():
        raise ValueError("full-only morphology path contains non-finite values")

    relative_ns, aligned = _aligned_traces(
        background_removed,
        protected_time,
        morphology_path,
        half_width_ns=56.0,
        sample_step_ns=1.4,
    )
    template = np.median(aligned, axis=1)
    correlations = np.asarray([_correlation(aligned[:, i], template) for i in range(aligned.shape[1])])
    aligned_envelope = np.abs(hilbert(aligned, axis=0))
    center_band = np.abs(relative_ns) <= 14.0
    target_amplitude = np.max(aligned_envelope[center_band], axis=0)
    median_amplitude = float(np.median(target_amplitude))
    dropout = float(np.mean(target_amplitude < 0.25 * median_amplitude))

    target_values = _window_values(background_removed, protected_time, morphology_path, -14.0, 14.0)
    background_values = np.concatenate(
        (
            _window_values(background_removed, protected_time, morphology_path, -98.0, -42.0),
            _window_values(background_removed, protected_time, morphology_path, 42.0, 98.0),
        )
    )
    target_to_background = _rms(target_values) / max(_rms(background_values), 1e-30)

    template_abs = np.abs(template)
    prominence = max(float(np.max(template_abs)) * 0.08, np.finfo(np.float64).tiny)
    lobe_indices, _ = find_peaks(template_abs, prominence=prominence, distance=2)
    lobe_signs = [int(np.sign(template[index])) for index in lobe_indices]
    tapered = (template - np.mean(template)) * np.hanning(template.size)
    spectrum = np.abs(np.fft.rfft(tapered)) ** 2
    frequency_hz = np.fft.rfftfreq(tapered.size, d=1.4e-9)
    positive = frequency_hz > 0
    peak_frequency_mhz = float(frequency_hz[positive][np.argmax(spectrum[positive])] / 1e6)
    spectral_centroid_mhz = float(
        np.sum(frequency_hz[positive] * spectrum[positive])
        / max(np.sum(spectrum[positive]), np.finfo(np.float64).tiny)
        / 1e6
    )

    steps = np.abs(np.diff(morphology_path))
    residual_steps = np.abs(np.diff(morphology_path - reference))
    reference_range = float(np.ptp(reference))
    path_range = float(np.ptp(morphology_path))
    result: dict[str, object] = {
        "audit_type": "FULL_SCENE_MORPHOLOGY_ONLY_V1",
        "case_id": scene["case_id"],
        "run_id": run.get("run_id") or run_dir.name,
        "component": component,
        "status": "development_only_full_scene_morphology",
        "claim_limits": {
            "causal_attribution_allowed": False,
            "formal_training_promotion_allowed": False,
            "visible_phase_training_label_allowed": False,
            "reason": (
                "This is a full-scene morphology audit only; a matched no-basal control is "
                f"not complete ({control_trace_count} unmerged control traces found)."
            ),
        },
        "execution_state": {
            "full_scene_merged": (run_dir / "full_scene_merged.out").is_file(),
            "full_scene_trace_contract_complete": bool(trace_contract.get("complete")),
            "full_scene_captured_trace_count": int(trace_contract.get("captured_trace_count", 0)),
            "control_completed_trace_files": control_trace_count,
            "control_merged": control_merged,
            "early_stop_reason": "User-approved stop after full-scene morphology became sufficient for the next design decision.",
        },
        "solver_contract": {
            "expected_trace_count": expected_traces,
            "declared_trace_count": declared_traces,
            "actual_trace_count": int(raw.shape[1]),
            "trace_stride": trace_stride,
            "selected_trace_indices_zero_based": selected.tolist(),
            "native_sample_count": int(raw.shape[0]),
            "dt_s": float(dt_s),
            "actual_end_ns": float(time_ns[-1]),
            "protected_end_ns": protected_end_ns,
            "finite": finite,
            "trace_contract_file": _portable_path(trace_contract_path),
            "reference_file": _portable_path(reference_path),
            "reference_semantics": reference_semantics,
        },
        "full_only_morphology": {
            "semantics": "continuous signed phase in background-removed full scene; not a causal full-minus-control phase",
            "path_range_ns": [float(np.min(morphology_path)), float(np.max(morphology_path))],
            "geometric_range_ns": [float(np.min(reference)), float(np.max(reference))],
            "path_to_geometric_correlation": _correlation(reference, morphology_path),
            "dynamic_range_retention": path_range / max(reference_range, 1e-30),
            "median_path_minus_geometric_ns": float(np.median(morphology_path - reference)),
            "path_step_p95_ns": float(np.percentile(steps, 95)),
            "path_step_max_ns": float(np.max(steps)),
            "path_residual_step_p95_ns": float(np.percentile(residual_steps, 95)),
            "path_residual_step_max_ns": float(np.max(residual_steps)),
            "path_residual_step_limit_ns": 5.6 * trace_stride,
            "target_to_adjacent_background_rms": target_to_background,
            "target_envelope_cv": float(np.std(target_amplitude) / max(np.mean(target_amplitude), 1e-30)),
            "target_dropout_fraction_below_25pct_median": dropout,
            "aligned_template_correlation_median": float(np.median(correlations)),
            "aligned_template_correlation_p10": float(np.percentile(correlations, 10)),
            "significant_signed_lobe_count": int(lobe_indices.size),
            "significant_lobe_signs": lobe_signs,
            "aligned_peak_frequency_mhz": peak_frequency_mhz,
            "aligned_spectral_centroid_mhz": spectral_centroid_mhz,
            "support_score_median": float(np.median(support)),
        },
    }
    morphology_gate = bool(
        finite
        and raw.shape[1] == expected_traces
        and result["full_only_morphology"]["path_to_geometric_correlation"] >= 0.80
        and result["full_only_morphology"]["dynamic_range_retention"] >= 0.60
        and result["full_only_morphology"]["path_residual_step_max_ns"] <= 5.6 * trace_stride
        and dropout <= 0.05
    )
    result["gates"] = {
        "full_output_contract_pass": bool(
            finite and raw.shape[1] == expected_traces and time_ns[-1] >= protected_end_ns
        ),
        "full_only_geometry_tracking_pass": morphology_gate,
        "full_resolution_causal_pair_pass": False,
        "formal_promotion_pass": False,
    }
    if line9_contract_path is not None:
        line9 = json.loads(line9_contract_path.resolve().read_text(encoding="utf-8"))
        result["development_only_line9_comparison"] = {
            "restriction": "diagnostic comparison only; must not condition a formal strict-holdout generator",
            "contract": _portable_path(line9_contract_path),
            "line9_signal": line9["signal"],
            "interpretation": "Numerical proximity is descriptive only because measured and FDTD processing domains differ.",
        }

    np.save(output_dir / "full_only_morphology_path_not_for_training.npy", morphology_path.astype(np.float32))
    np.save(output_dir / "full_only_aligned_template_not_for_training.npy", template.astype(np.float32))
    (output_dir / "full_only_morphology_audit.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component", default="Ez")
    parser.add_argument("--line9-contract", type=Path)
    args = parser.parse_args()
    result = audit(
        args.run_dir,
        args.output_dir,
        component=args.component,
        line9_contract_path=args.line9_contract,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
