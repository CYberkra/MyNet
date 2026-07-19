#!/usr/bin/env python3
"""Round 02: audit smooth per-trace gain and reject post-hoc delay drift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_realism_research import (  # noqa: E402
    FIT_LINES,
    HELDOUT_LINES,
    VALIDATION_LINES,
    aligned_packets,
    apply_trace_gain_delay,
    apply_zero_phase_response,
    bounded_system_response_db,
    measured_line_spectrum,
    packet_metrics,
    packet_spectrum,
    pool_spectra,
    smooth_unit_noise,
    valid_measured_traces,
)
from scripts.research_round01_system_response import (  # noqa: E402
    _blind_panel,
    _font,
    _load_simulation,
    _multiline_panel,
)


C0 = 299_792_458.0
GAIN_SCALES = (0.0, 0.03, 0.05, 0.08)
DELAY_PROBE_SCALE = 0.001


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _early_wave_lags_ns(values: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
    rows = (time_ns >= 0.0) & (time_ns <= 70.0)
    traces = np.asarray(values[rows], dtype=np.float64)
    traces -= np.mean(traces, axis=0, keepdims=True)
    traces /= np.maximum(
        np.sqrt(np.sum(np.square(traces), axis=0, keepdims=True)), 1e-30
    )
    template = np.median(traces, axis=1)
    template -= np.mean(template)
    template /= max(float(np.linalg.norm(template)), 1e-30)
    lags = []
    for trace in range(traces.shape[1]):
        scores = []
        for lag in range(-4, 5):
            if lag < 0:
                x, y = traces[-lag:, trace], template[: template.size + lag]
            elif lag > 0:
                x, y = traces[:-lag, trace], template[lag:]
            else:
                x, y = traces[:, trace], template
            scores.append(float(np.dot(x, y)))
        index = int(np.argmax(scores))
        lag = float(index - 4)
        if 0 < index < len(scores) - 1:
            left, center, right = scores[index - 1 : index + 2]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                lag += 0.5 * (left - right) / denominator
        lags.append(lag * 1.4)
    return np.asarray(lags)


def _local_window_detrended_std(values: np.ndarray, width: int = 64) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < width:
        return float(np.std(values))
    coordinate = np.linspace(-1.0, 1.0, width)
    denominator = float(np.sum(np.square(coordinate)))
    deviations = []
    for start in range(values.size - width + 1):
        window = values[start : start + width]
        centered = window - np.mean(window)
        trend = coordinate * float(np.sum(coordinate * centered)) / denominator
        deviations.append(float(np.std(centered - trend)))
    return float(np.median(deviations))


def _line_statistics(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as data:
        valid = valid_measured_traces(data)
        raw = np.asarray(data["raw_amplitude"], dtype=np.float64)[:, valid]
        time_ns = np.asarray(data["time_ns"], dtype=np.float64)
        path_ns = np.asarray(data["v15_final_center_time_ns"], dtype=np.float64)[valid]
        height = np.asarray(data["flight_height_agl_m"], dtype=np.float64)[valid]
        distance = np.asarray(data["gnss_cumulative_distance_m"], dtype=np.float64)[valid]
    dx = float(np.median(np.diff(distance)))
    common_removed = raw - np.median(raw, axis=1, keepdims=True)
    relative, packets = aligned_packets(common_removed, time_ns, path_ns)
    envelope = np.abs(hilbert(packets, axis=0))
    amplitude = np.max(envelope[np.abs(relative) <= 14.0], axis=0)
    log_amplitude = np.log(np.maximum(amplitude, 1e-30))
    smooth_log = gaussian_filter1d(
        log_amplitude, max(1.0, 2.0 / dx), mode="nearest"
    )
    residual_log = log_amplitude - smooth_log
    lags = _early_wave_lags_ns(raw, time_ns)
    measured_metrics = packet_metrics(raw, time_ns, path_ns)
    return {
        "trace_count": int(valid.sum()),
        "trace_spacing_m": dx,
        "target_to_background": measured_metrics["target_to_background"],
        "target_envelope_cv": float(np.std(amplitude) / np.mean(amplitude)),
        "aligned_template_correlation_median": measured_metrics[
            "aligned_template_correlation_median"
        ],
        "significant_lobe_count": measured_metrics["significant_lobe_count"],
        "local_log_gain_sigma": _robust_sigma(residual_log),
        "local_height_sigma_m": _local_window_detrended_std(height),
        "early_wave_jitter_sigma_ns": _robust_sigma(lags),
    }


def _pool_parameter(
    stats: dict[str, dict[str, float]], name: str
) -> float:
    return float(np.median([line_stats[name] for line_stats in stats.values()]))


def _trajectory_plot(
    output: Path,
    fields: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    width, height = 1500, 760
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 20), "Round 02 sampled acquisition trajectories", fill="black", font=_font(28, True))
    boxes = [(70, 100, 1360, 260), (70, 430, 1360, 260)]
    colours = [(30, 100, 210), (215, 80, 30), (40, 145, 70), (150, 65, 175)]
    for box_index, (left, top, box_width, box_height) in enumerate(boxes):
        draw.rectangle((left, top, left + box_width, top + box_height), outline="black", width=2)
        draw.text((left + 8, top + 8), "log gain" if box_index == 0 else "delay (ns)", fill="black", font=_font(17, True))
        all_values = np.concatenate(
            [entry[0 if box_index == 0 else 1] for entry in fields.values()]
        )
        maximum = max(float(np.max(np.abs(all_values))), 1e-6)
        for index, (name, entry) in enumerate(fields.items()):
            values = entry[0 if box_index == 0 else 1]
            x = left + np.rint(np.linspace(0, box_width, values.size)).astype(int)
            y = top + box_height // 2 - np.rint(
                values / maximum * (0.43 * box_height)
            ).astype(int)
            colour = colours[index % len(colours)]
            draw.line(list(zip(x.tolist(), y.tolist())), fill=colour, width=3)
            legend_y = top + 15 + index * 25
            draw.line((left + 1030, legend_y + 8, left + 1070, legend_y + 8), fill=colour, width=3)
            draw.text((left + 1080, legend_y), name, fill="black", font=_font(13))
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-root", type=Path, required=True)
    parser.add_argument("--round01-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071602)
    parser.add_argument("--heldout-diagnostic", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    measured_root = args.measured_root.resolve()
    round01 = json.loads(
        (args.round01_dir.resolve() / "round01_selection.json").read_text(
            encoding="utf-8"
        )
    )
    if round01["selected_candidate"] != "strength_1.00":
        raise RuntimeError("Round 02 expects the frozen Round 01 strength_1.00")

    time_ns, raw, base_path_ns, _ = _load_simulation(
        args.run_dir.resolve(), args.source_case_dir.resolve()
    )
    fit_spectra = [
        measured_line_spectrum(measured_root / f"{line}.npz") for line in FIT_LINES
    ]
    fit_spectrum = pool_spectra(fit_spectra)
    _, packets = aligned_packets(
        raw - np.median(raw, axis=1, keepdims=True), time_ns, base_path_ns
    )
    sim_spectrum = packet_spectrum(packets)
    response_db = bounded_system_response_db(sim_spectrum, fit_spectrum)
    round01_base = apply_zero_phase_response(
        raw,
        1.4,
        sim_spectrum.frequency_hz,
        response_db,
        strength=1.0,
    )

    fit_stats = {
        line: _line_statistics(measured_root / f"{line}.npz")
        for line in FIT_LINES
    }
    validation_stats = {
        line: _line_statistics(measured_root / f"{line}.npz")
        for line in VALIDATION_LINES
    }
    gain_sigma = _pool_parameter(fit_stats, "local_log_gain_sigma")
    height_sigma_m = _pool_parameter(fit_stats, "local_height_sigma_m")
    jitter_sigma_ns = _pool_parameter(fit_stats, "early_wave_jitter_sigma_ns")
    target_cv_fit = _pool_parameter(fit_stats, "target_envelope_cv")
    target_cv_validation = _pool_parameter(validation_stats, "target_envelope_cv")
    target_ratio_fit = _pool_parameter(fit_stats, "target_to_background")
    target_ratio_validation = _pool_parameter(
        validation_stats, "target_to_background"
    )

    rng = np.random.default_rng(args.seed)
    trace_count = raw.shape[1]
    trace_spacing_m = 0.09
    gain_basis = (
        0.72
        * smooth_unit_noise(trace_count, 0.55 / trace_spacing_m, rng)
        + 0.28
        * smooth_unit_noise(trace_count, 2.2 / trace_spacing_m, rng)
    )
    gain_basis /= np.std(gain_basis)
    height_basis = smooth_unit_noise(
        trace_count, 1.8 / trace_spacing_m, rng
    )
    jitter_basis = smooth_unit_noise(
        trace_count, 0.25 / trace_spacing_m, rng
    )

    candidates: dict[str, np.ndarray] = {}
    paths: dict[str, np.ndarray] = {}
    trajectories: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for gain_scale in GAIN_SCALES:
        name = f"gain_{gain_scale:.2f}"
        log_gain = gain_scale * gain_sigma * gain_basis
        relative_height = np.zeros(trace_count, dtype=np.float64)
        delay_ns = np.zeros(trace_count, dtype=np.float64)
        candidate = apply_trace_gain_delay(
            round01_base, time_ns, log_gain, delay_ns
        )
        path_ns = base_path_ns + delay_ns
        candidate_metrics = packet_metrics(
            candidate, time_ns, path_ns, spectrum_reference=fit_spectrum
        )
        cv = candidate_metrics["target_envelope_cv"]
        ratio = candidate_metrics["target_to_background"]
        score = (
            abs(np.log(ratio / target_ratio_validation))
            + 0.45 * abs(np.log(ratio / target_ratio_fit))
            + 0.45 * abs(cv - target_cv_validation)
            + 0.20 * abs(cv - target_cv_fit)
            + 20.0 * max(0.0, candidate_metrics["target_dropout_fraction"] - 0.03)
        )
        candidate_metrics.update(
            {
                "selection_score": float(score),
                "log_gain_sigma": float(np.std(log_gain)),
                "gain_min": float(np.min(np.exp(log_gain))),
                "gain_max": float(np.max(np.exp(log_gain))),
                "relative_height_sigma_m": float(np.std(relative_height)),
                "delay_p95_ns": float(np.percentile(np.abs(delay_ns), 95)),
            }
        )
        candidates[name] = candidate
        paths[name] = path_ns
        trajectories[name] = (log_gain, delay_ns, relative_height)
        metrics[name] = candidate_metrics

    delay_reference = (
        2.0 * height_sigma_m * height_basis / C0 * 1e9
        + jitter_sigma_ns * jitter_basis
    )
    delay_probe_ns = DELAY_PROBE_SCALE * delay_reference
    delay_probe = apply_trace_gain_delay(
        round01_base,
        time_ns,
        np.zeros(trace_count, dtype=np.float64),
        delay_probe_ns,
    )
    delay_probe_path = base_path_ns + delay_probe_ns
    delay_probe_metrics = packet_metrics(
        delay_probe,
        time_ns,
        delay_probe_path,
        spectrum_reference=fit_spectrum,
    )
    delay_probe_metrics.update(
        {
            "probe_scale": DELAY_PROBE_SCALE,
            "delay_p95_ns": float(np.percentile(np.abs(delay_probe_ns), 95)),
            "decision": "rejected_posthoc_approximation",
        }
    )
    trajectories["delay_probe_0.001"] = (
        np.zeros(trace_count, dtype=np.float64),
        delay_probe_ns,
        DELAY_PROBE_SCALE * height_sigma_m * height_basis,
    )

    selected_name = min(metrics, key=lambda name: metrics[name]["selection_score"])
    selection_path = output / "round02_selection.json"
    if args.heldout_diagnostic:
        if not selection_path.is_file():
            raise FileNotFoundError("heldout diagnostic requires a frozen selection")
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        if frozen["selected_candidate"] != selected_name:
            raise RuntimeError("recomputed selection differs from the frozen candidate")
        _multiline_panel(
            output / "round02_frozen_plus_heldout.png",
            measured_root,
            candidates[selected_name],
            time_ns,
            FIT_LINES + VALIDATION_LINES + HELDOUT_LINES,
            candidate_label="Round02 frozen candidate",
            heading="Round 02 frozen gain/delay candidate plus heldout diagnosis",
        )
        heldout_stats = {
            line: _line_statistics(measured_root / f"{line}.npz")
            for line in HELDOUT_LINES
        }
        frozen["status"] = "heldout_diagnostic_complete"
        frozen["heldout_lines_opened"] = list(HELDOUT_LINES)
        frozen["heldout_diagnostic"] = {
            "opened_after_freeze": True,
            "target_to_background": {
                line: stats["target_to_background"]
                for line, stats in heldout_stats.items()
            },
            "target_envelope_cv": {
                line: stats["target_envelope_cv"]
                for line, stats in heldout_stats.items()
            },
            "selected_candidate_target_envelope_cv": metrics[selected_name][
                "target_envelope_cv"
            ],
            "visual": "round02_frozen_plus_heldout.png",
        }
        selection_path.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(frozen, ensure_ascii=False, indent=2))
        return 0

    mapping = _blind_panel(
        output / "round02_blind_candidates.png",
        candidates,
        time_ns,
        args.seed,
        heading="Round 02 blind smooth gain-only candidates; shared scales; no labels",
    )
    _multiline_panel(
        output / "round02_fold_only_multiline.png",
        measured_root,
        candidates[selected_name],
        time_ns,
        FIT_LINES + VALIDATION_LINES,
        candidate_label="Round02 fold-selected gain-only candidate",
        heading="Round 02 equal-trace gain audit; independent P99.5 scales",
    )
    _trajectory_plot(output / "round02_trajectory_audit.png", trajectories)
    selection = {
        "round": 2,
        "factor": "smooth trace gain; posthoc delay audited as a rejected probe",
        "status": "fold_selected_heldout_closed",
        "formal_training_allowed": False,
        "fit_lines": list(FIT_LINES),
        "validation_lines": list(VALIDATION_LINES),
        "heldout_lines_opened": [],
        "round01_parent": round01["selected_candidate"],
        "fitted_parameters": {
            "local_log_gain_sigma": gain_sigma,
            "local_height_sigma_m": height_sigma_m,
            "early_wave_jitter_sigma_ns": jitter_sigma_ns,
            "fit_target_envelope_cv": target_cv_fit,
            "validation_target_envelope_cv": target_cv_validation,
            "fit_target_to_background": target_ratio_fit,
            "validation_target_to_background": target_ratio_validation,
        },
        "fit_line_statistics": fit_stats,
        "validation_line_statistics": validation_stats,
        "candidate_gain_scales": list(GAIN_SCALES),
        "metrics": metrics,
        "posthoc_delay_probe": delay_probe_metrics,
        "selected_candidate": selected_name,
        "blind_mapping": mapping,
        "visuals": [
            "round02_blind_candidates.png",
            "round02_fold_only_multiline.png",
            "round02_trajectory_audit.png",
        ],
        "next_gate": "human blind visual audit before heldout diagnostic",
        "interpretation_guard": (
            "posthoc trace shifting is not a flight-height model; physical height "
            "variation is deferred to the Round 05 FDTD experiment"
        ),
    }
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
