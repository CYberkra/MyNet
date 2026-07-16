#!/usr/bin/env python3
"""Generate FORMAL09B-2 lateral covariance and nonstationarity candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_formal09a_multiline_acquisition_realism import (  # noqa: E402
    add_panel,
    apply_realism,
    morphology_metrics,
    process,
    read_reference_segments,
    representative_crop,
    robust_limit,
    sha256,
)
from scripts.generate_formal09b1_empirical_spectrum import (  # noqa: E402
    DEVELOPMENT_LINES,
    PAPER_FIT_LINES,
    TARGET_VARIANT,
    SpectrumFit,
    _load_base,
    equal_line_log_pool,
    fit_spectrum,
    build_empirical_realism_basis,
)


COMMON_SPATIAL_FREQUENCY_CPM = np.linspace(0.0, 0.70, 141)


@dataclass(frozen=True)
class SpatialFit:
    name: str
    lines: tuple[str, ...]
    frequency_cpm: np.ndarray
    amplitude: np.ndarray
    line_amplitudes: dict[str, np.ndarray]
    frame_counts: dict[str, int]
    line_envelope_cv: dict[str, float]
    line_envelope_correlation_m: dict[str, float]
    pooled_envelope_cv: float
    pooled_envelope_correlation_m: float


def read_segments(
    path: Path, allowed_lines: tuple[str, ...]
) -> dict[str, list[tuple[int, int]]]:
    segments = {line: [] for line in allowed_lines}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["line"] in segments and row["purpose"] == "signal_style":
                segments[row["line"]].append(
                    (int(row["trace_start"]), int(row["trace_end"]))
                )
    missing = [line for line, spans in segments.items() if not spans]
    if missing:
        raise ValueError(f"missing signal-style segments for {missing}")
    return segments


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _spatial_frame_power(frame: np.ndarray, n_fft: int) -> np.ndarray | None:
    values = np.asarray(frame, dtype=np.float64)
    values = values - np.mean(values)
    scale = float(np.sqrt(np.mean(np.square(values))))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).tiny:
        return None
    tapered = (values / scale) * np.hanning(values.size)
    power = np.square(np.abs(np.fft.rfft(tapered, n=n_fft)))
    power /= max(float(np.sum(power[1:])), np.finfo(np.float64).tiny)
    return power


def _correlation_length(values: np.ndarray, spacing_m: float) -> float:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values)
    variance = float(np.sum(np.square(centered)))
    if variance <= np.finfo(np.float64).tiny:
        return spacing_m
    maximum_lag = min(256, centered.size // 2)
    for lag in range(1, maximum_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denominator = np.sqrt(
            np.sum(np.square(left)) * np.sum(np.square(right))
        )
        correlation = float(np.sum(left * right) / denominator) if denominator else 0.0
        if correlation <= math.exp(-1.0):
            return lag * spacing_m
    return maximum_lag * spacing_m


def fit_line_spatial_statistics(
    line_path: Path,
    spans: list[tuple[int, int]],
    *,
    n_fft: int = 256,
    frame_traces: int = 128,
    frame_stride: int = 64,
    fit_low_ns: float = 70.0,
    fit_high_ns: float = 260.0,
    target_guard_ns: float = 42.0,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    with np.load(line_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_amplitude"], dtype=np.float64)
        time_ns = np.asarray(data["time_ns"], dtype=np.float64)
        path_ns = np.asarray(data["v15_final_center_time_ns"], dtype=np.float64)
        distance_m = np.asarray(data["gnss_cumulative_distance_m"], dtype=np.float64)
        ignored = np.asarray(data["v15_final_ignore_trace"], dtype=bool)
        outside_height = np.asarray(
            data["flight_height_outside_planned_2_20_m"], dtype=bool
        )

    all_valid = np.zeros(raw.shape[1], dtype=bool)
    for start, end in spans:
        all_valid[start : end + 1] = True
    all_valid &= ~ignored & ~outside_height
    selected = np.flatnonzero(all_valid)
    common = np.median(raw[:, selected], axis=1, keepdims=True)
    residual = raw - common
    row_indices = np.flatnonzero((time_ns >= fit_low_ns) & (time_ns <= fit_high_ns))
    positive_steps = np.diff(distance_m)
    positive_steps = positive_steps[positive_steps > 1e-4]
    spacing_m = float(np.median(positive_steps))

    powers: list[np.ndarray] = []
    envelope_sequences: list[np.ndarray] = []
    for start, end in spans:
        local_valid = all_valid[start : end + 1]
        for run_start, run_end_exclusive in _contiguous_true_runs(local_valid):
            columns = np.arange(
                start + run_start, start + run_end_exclusive, dtype=np.int64
            )
            if columns.size < frame_traces:
                continue
            local = residual[np.ix_(row_indices, columns)]
            trace_rms = np.sqrt(np.mean(np.square(local), axis=0))
            smooth_sigma = max(1.0, 1.5 / spacing_m)
            smoothed = gaussian_filter1d(trace_rms, sigma=smooth_sigma, mode="reflect")
            smoothed /= max(float(np.median(smoothed)), np.finfo(np.float64).tiny)
            envelope_sequences.append(smoothed)

            for row_offset, row in enumerate(row_indices):
                target_clear = np.abs(time_ns[row] - path_ns[columns]) > target_guard_ns
                for frame_start in range(
                    0, columns.size - frame_traces + 1, frame_stride
                ):
                    frame_end = frame_start + frame_traces
                    if not np.all(target_clear[frame_start:frame_end]):
                        continue
                    power = _spatial_frame_power(
                        local[row_offset, frame_start:frame_end], n_fft
                    )
                    if power is not None:
                        powers.append(power)
    if not powers or not envelope_sequences:
        raise ValueError(f"no valid spatial frames in {line_path}")

    native_frequency = np.fft.rfftfreq(n_fft, d=spacing_m)
    log_power = np.log(np.maximum(np.stack(powers, axis=0), 1e-18))
    amplitude = np.sqrt(np.exp(np.median(log_power, axis=0)))
    amplitude = np.interp(
        COMMON_SPATIAL_FREQUENCY_CPM,
        native_frequency,
        amplitude,
        left=amplitude[0],
        right=0.0,
    )
    mask = COMMON_SPATIAL_FREQUENCY_CPM > 0.0
    amplitude /= max(
        float(np.sqrt(np.mean(np.square(amplitude[mask])))),
        np.finfo(np.float64).tiny,
    )
    envelope_cv = float(
        np.median(
            [
                np.std(sequence) / max(float(np.mean(sequence)), 1e-12)
                for sequence in envelope_sequences
            ]
        )
    )
    correlation_m = float(
        np.median(
            [_correlation_length(np.log(sequence), spacing_m) for sequence in envelope_sequences]
        )
    )
    return (
        COMMON_SPATIAL_FREQUENCY_CPM.copy(),
        amplitude,
        len(powers),
        envelope_cv,
        correlation_m,
    )


def fit_spatial_pool(
    name: str,
    lines: tuple[str, ...],
    lines_dir: Path,
    segment_csv: Path,
) -> SpatialFit:
    segments = read_segments(segment_csv, lines)
    line_amplitudes: dict[str, np.ndarray] = {}
    frame_counts: dict[str, int] = {}
    line_cv: dict[str, float] = {}
    line_corr: dict[str, float] = {}
    for line in lines:
        _, amplitude, count, cv, corr = fit_line_spatial_statistics(
            lines_dir / f"{line}.npz", segments[line]
        )
        line_amplitudes[line] = amplitude
        frame_counts[line] = count
        line_cv[line] = cv
        line_corr[line] = corr
    pooled = equal_line_log_pool([line_amplitudes[line] for line in lines])
    return SpatialFit(
        name=name,
        lines=lines,
        frequency_cpm=COMMON_SPATIAL_FREQUENCY_CPM.copy(),
        amplitude=pooled,
        line_amplitudes=line_amplitudes,
        frame_counts=frame_counts,
        line_envelope_cv=line_cv,
        line_envelope_correlation_m=line_corr,
        pooled_envelope_cv=float(np.median(list(line_cv.values()))),
        pooled_envelope_correlation_m=float(np.median(list(line_corr.values()))),
    )


def _temporal_response(
    rows: int, dt_s: float, temporal_fit: SpectrumFit
) -> np.ndarray:
    frequency = np.fft.rfftfreq(rows, d=dt_s)
    response = np.interp(
        frequency,
        temporal_fit.frequency_hz,
        temporal_fit.amplitude,
        left=0.0,
        right=0.0,
    )
    response[0] = 0.0
    response /= max(float(np.sqrt(np.mean(np.square(response[1:])))), 1e-12)
    return response


def build_lateral_covariance_basis(
    shape: tuple[int, int],
    dt_s: float,
    time_ns: np.ndarray,
    temporal_fit: SpectrumFit,
    spatial_fit: SpatialFit,
    trace_spacing_m: float,
    seed: int,
    *,
    nonstationary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, traces = shape
    temporal_response = _temporal_response(rows, dt_s, temporal_fit)
    spatial_frequency = np.abs(np.fft.fftfreq(traces, d=trace_spacing_m))
    spatial_response = np.interp(
        spatial_frequency,
        spatial_fit.frequency_cpm,
        spatial_fit.amplitude,
        left=spatial_fit.amplitude[0],
        right=0.0,
    )
    spatial_response /= max(float(np.sqrt(np.mean(np.square(spatial_response)))), 1e-12)

    white = rng.standard_normal(shape)
    diffuse = np.fft.irfft(
        np.fft.rfft(white, axis=0) * temporal_response[:, None],
        n=rows,
        axis=0,
    )
    diffuse = np.fft.ifft(
        np.fft.fft(diffuse, axis=1) * spatial_response[None, :], axis=1
    ).real

    low_rank = np.zeros(shape, dtype=np.float64)
    for _ in range(4):
        temporal = rng.standard_normal(rows)
        temporal = np.fft.irfft(
            np.fft.rfft(temporal) * temporal_response, n=rows
        )
        lateral = rng.standard_normal(traces)
        lateral = np.fft.ifft(
            np.fft.fft(lateral) * spatial_response
        ).real
        temporal = (temporal - np.mean(temporal)) / max(float(np.std(temporal)), 1e-12)
        lateral = (lateral - np.mean(lateral)) / max(float(np.std(lateral)), 1e-12)
        low_rank += temporal[:, None] * lateral[None, :]
    low_rank /= 4.0

    depth = np.clip((time_ns - 45.0) / 455.0, 0.0, 1.0)
    depth = 0.08 + 0.92 * np.power(depth, 0.85)
    combined = 0.72 * diffuse / max(float(np.std(diffuse)), 1e-12)
    combined += 0.28 * low_rank / max(float(np.std(low_rank)), 1e-12)

    gain = gaussian_filter1d(rng.standard_normal(traces), sigma=4.5, mode="reflect")
    gain = (gain - np.mean(gain)) / max(float(np.std(gain)), 1e-12)

    envelope = np.ones(traces, dtype=np.float64)
    if nonstationary:
        correlation_sigma = max(
            1.0, spatial_fit.pooled_envelope_correlation_m / trace_spacing_m
        )
        envelope_basis = gaussian_filter1d(
            rng.standard_normal(traces), sigma=correlation_sigma, mode="reflect"
        )
        envelope_basis = (envelope_basis - np.mean(envelope_basis)) / max(
            float(np.std(envelope_basis)), 1e-12
        )
        log_sigma = math.sqrt(
            math.log1p(min(spatial_fit.pooled_envelope_cv, 0.65) ** 2)
        )
        envelope = np.exp(log_sigma * envelope_basis - 0.5 * log_sigma**2)
        envelope = np.clip(envelope, 0.45, 2.2)
        combined *= envelope[None, :]

    basis = combined * depth[:, None]
    basis = (basis - np.mean(basis)) / max(float(np.std(basis)), 1e-12)
    return basis, gain, spatial_response, envelope


def background_lateral_metrics(
    values: np.ndarray,
    time_ns: np.ndarray,
    path_ns: np.ndarray,
    trace_spacing_m: float,
) -> dict[str, float]:
    suppressed = values - np.median(values, axis=1, keepdims=True)
    relative = np.concatenate(
        (np.arange(-98.0, -42.0 + 0.7, 1.4), np.arange(42.0, 98.0 + 0.7, 1.4))
    )
    aligned = np.column_stack(
        [
            np.interp(center + relative, time_ns, suppressed[:, trace])
            for trace, center in enumerate(path_ns)
        ]
    )
    envelope = np.abs(hilbert(aligned, axis=0))
    trace_energy = np.sqrt(np.mean(np.square(aligned), axis=0))
    correlations: list[float] = []
    coherence_m = 8.0 * trace_spacing_m
    for lag in range(1, min(8, aligned.shape[1] - 1) + 1):
        left = envelope[:, :-lag].ravel()
        right = envelope[:, lag:].ravel()
        left -= np.mean(left)
        right -= np.mean(right)
        denominator = np.sqrt(np.sum(np.square(left)) * np.sum(np.square(right)))
        correlation = float(np.sum(left * right) / denominator) if denominator else 0.0
        correlations.append(correlation)
        if correlation <= math.exp(-1.0) and coherence_m == 8.0 * trace_spacing_m:
            coherence_m = lag * trace_spacing_m
    return {
        "background_envelope_cv": float(
            np.std(trace_energy) / max(float(np.mean(trace_energy)), 1e-12)
        ),
        "background_adjacent_envelope_correlation": correlations[0],
        "background_coherence_length_m": float(coherence_m),
    }


def draw_spatial_spectrum_audit(
    development: SpatialFit, paper: SpatialFit, output_path: Path
) -> None:
    canvas = Image.new("RGB", (1280, 650), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 18),
        "FORMAL09B-2 target-excluded lateral spectra in physical cycles/m",
        fill="black",
    )
    colors = {
        "Line3": "#1f77b4",
        "Line6": "#ff7f0e",
        "Line7": "#2ca02c",
        "Line9": "#d62728",
        "LineL1": "#9467bd",
        "all-lines pooled": "#111111",
        "paper-fold pooled": "#00a6a6",
    }

    def panel(
        x: int,
        title: str,
        curves: list[tuple[str, np.ndarray]],
    ) -> None:
        y, width, height = 70, 585, 520
        draw.rectangle((x, y, x + width, y + height), outline="black")
        draw.text((x + 8, y + 8), title, fill="black")
        left, top, right, bottom = x + 55, y + 40, x + width - 18, y + height - 45
        draw.rectangle((left, top, right, bottom), outline="#888888")
        for frequency in np.linspace(0.0, 0.7, 8):
            px = left + (right - left) * frequency / 0.7
            draw.line((px, top, px, bottom), fill="#eeeeee")
            draw.text((int(px) - 10, bottom + 8), f"{frequency:.1f}", fill="black")
        for db in (-40, -30, -20, -10, 0):
            py = bottom - (bottom - top) * (db + 40.0) / 40.0
            draw.line((left, py, right, py), fill="#eeeeee")
            draw.text((left - 34, int(py) - 6), str(db), fill="black")
        draw.text((left + 165, bottom + 25), "Spatial frequency (cycles/m)", fill="black")
        legend_left = right - 160
        draw.rectangle(
            (legend_left, top + 7, right - 5, top + 20 + 18 * len(curves)),
            fill="white",
            outline="#bbbbbb",
        )
        for index, (name, amplitude) in enumerate(curves):
            db = 20.0 * np.log10(
                np.maximum(amplitude / max(float(np.max(amplitude)), 1e-12), 1e-4)
            )
            points = [
                (
                    left + (right - left) * float(frequency) / 0.7,
                    bottom - (bottom - top) * float(level + 40.0) / 40.0,
                )
                for frequency, level in zip(
                    development.frequency_cpm, np.clip(db, -40.0, 0.0)
                )
            ]
            draw.line(points, fill=colors[name], width=2 if "pooled" in name else 1)
            legend_y = top + 15 + 18 * index
            draw.line(
                (legend_left + 8, legend_y + 5, legend_left + 32, legend_y + 5),
                fill=colors[name],
                width=2,
            )
            draw.text((legend_left + 38, legend_y), name, fill="black")

    panel(
        55,
        "Per-line lateral spectra",
        [(line, development.line_amplitudes[line]) for line in development.lines],
    )
    panel(
        675,
        "Equal-line pooled lateral spectra",
        [
            ("all-lines pooled", development.amplitude),
            ("paper-fold pooled", paper.amplitude),
        ],
    )
    canvas.save(output_path)


def _spatial_payload(fit: SpatialFit) -> dict:
    return {
        "name": fit.name,
        "lines": list(fit.lines),
        "frequency_cpm": fit.frequency_cpm.tolist(),
        "pooled_amplitude": fit.amplitude.tolist(),
        "line_amplitudes": {
            line: values.tolist() for line, values in fit.line_amplitudes.items()
        },
        "frame_counts": fit.frame_counts,
        "line_envelope_cv": fit.line_envelope_cv,
        "line_envelope_correlation_m": fit.line_envelope_correlation_m,
        "pooled_envelope_cv": fit.pooled_envelope_cv,
        "pooled_envelope_correlation_m": fit.pooled_envelope_correlation_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-lines", type=Path, required=True)
    parser.add_argument("--reference-segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071611)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporal_fit = fit_spectrum(
        "paper_line9_holdout_fit",
        PAPER_FIT_LINES,
        args.measured_lines,
        args.reference_segments,
    )
    spatial_paper = fit_spatial_pool(
        "paper_line9_holdout_fit",
        PAPER_FIT_LINES,
        args.measured_lines,
        args.reference_segments,
    )
    spatial_all = fit_spatial_pool(
        "development_all_lines",
        DEVELOPMENT_LINES,
        args.measured_lines,
        args.reference_segments,
    )
    raw, time_ns, path_ns, dt_s, run, scene, merged = _load_base(
        args.release_dir, args.source_case_dir
    )
    selected = np.asarray(run["selected_trace_indices_zero_based"], dtype=np.int64)
    trace_spacing_m = float(scene["grid"]["trace_spacing_m"]) * float(
        np.median(np.diff(selected))
    )

    b1_basis, b1_gain, _ = build_empirical_realism_basis(
        raw.shape, dt_s, time_ns, temporal_fit, args.seed
    )
    b1_values, b1_report = apply_realism(
        raw, time_ns, path_ns, b1_basis, b1_gain, TARGET_VARIANT
    )
    realized = {"09B1-paper": b1_values}
    metrics = {
        "09B1-paper": {
            **b1_report,
            **morphology_metrics(b1_values[time_ns <= 500.0], time_ns[time_ns <= 500.0], path_ns),
            **background_lateral_metrics(b1_values, time_ns, path_ns, trace_spacing_m),
        }
    }
    responses: dict[str, list[float]] = {}
    envelopes: dict[str, list[float]] = {}
    candidates = (
        ("09B2-paper-covariance", spatial_paper, False),
        ("09B2-paper-nonstationary", spatial_paper, True),
        ("09B2-all-lines-nonstationary", spatial_all, True),
    )
    for name, spatial_fit, nonstationary in candidates:
        basis, gain, response, envelope = build_lateral_covariance_basis(
            raw.shape,
            dt_s,
            time_ns,
            temporal_fit,
            spatial_fit,
            trace_spacing_m,
            args.seed,
            nonstationary=nonstationary,
        )
        values, report = apply_realism(
            raw, time_ns, path_ns, basis, gain, TARGET_VARIANT
        )
        realized[name] = values
        responses[name] = response.tolist()
        envelopes[name] = envelope.tolist()
        protected = time_ns <= 500.0
        metrics[name] = {
            **report,
            **morphology_metrics(values[protected], time_ns[protected], path_ns),
            **background_lateral_metrics(values, time_ns, path_ns, trace_spacing_m),
        }

    crop_rows = (time_ns >= 250.0) & (time_ns <= 500.0)
    raw_crop = {name: values[crop_rows] for name, values in realized.items()}
    processed = {name: process(values, time_ns)[crop_rows] for name, values in realized.items()}
    raw_limit = robust_limit(np.concatenate(list(raw_crop.values()), axis=1))
    processed_limit = robust_limit(np.concatenate(list(processed.values()), axis=1))
    blind_order = list(realized)
    np.random.default_rng(args.seed + 1).shuffle(blind_order)
    blind_map = {chr(65 + index): name for index, name in enumerate(blind_order)}
    width, height, margin, header = 360, 430, 18, 48
    canvas = Image.new(
        "RGB", (margin * 5 + width * 4, header + margin * 3 + height * 2), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 14),
        "FORMAL09B-2 blind lateral covariance checkpoint; common scales; no labels",
        fill="black",
    )
    for column, blind_id in enumerate("ABCD"):
        name = blind_map[blind_id]
        x = margin + column * (width + margin)
        add_panel(canvas, raw_crop[name], f"{blind_id}: raw", (x, header, width, height), raw_limit)
        add_panel(
            canvas,
            processed[name],
            f"{blind_id}: common-mode suppressed + time^1.5",
            (x, header + height + margin, width, height),
            processed_limit,
        )
    blind_path = output / "FORMAL09B2_blind_lateral_covariance.png"
    canvas.save(blind_path)

    longest = read_reference_segments(args.reference_segments)
    span_m = float(selected[-1] - selected[0]) * float(scene["grid"]["trace_spacing_m"])
    measured = {
        line: representative_crop(
            args.measured_lines / f"{line}.npz",
            *longest[line],
            span_m,
            250.0,
            500.0,
        )[0]
        for line in DEVELOPMENT_LINES
    }
    comparison_items = [(line, measured[line]) for line in DEVELOPMENT_LINES]
    comparison_items.append(("FORMAL06C", process(raw, time_ns)[crop_rows]))
    comparison_items.extend((name, processed[name]) for name in realized)
    width, height, margin, header, columns = 330, 390, 16, 48, 5
    rows = math.ceil(len(comparison_items) / columns)
    canvas = Image.new(
        "RGB",
        (
            margin * (columns + 1) + width * columns,
            header + margin * (rows + 1) + height * rows,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 14),
        "Multi-line references vs FORMAL09B-2; independent P99.5 scales",
        fill="black",
    )
    for index, (title, matrix) in enumerate(comparison_items):
        row, column = divmod(index, columns)
        x = margin + column * (width + margin)
        y = header + row * (height + margin)
        add_panel(canvas, matrix, title, (x, y, width, height), robust_limit(matrix))
    comparison_path = output / "FORMAL09B2_multiline_visual_comparison.png"
    canvas.save(comparison_path)

    spatial_figure = output / "FORMAL09B2_spatial_spectrum_audit.png"
    draw_spatial_spectrum_audit(spatial_all, spatial_paper, spatial_figure)
    spatial_contract = output / "FORMAL09B2_spatial_statistics.json"
    spatial_contract.write_text(
        json.dumps(
            {
                "contract_id": "FORMAL09B2_TARGET_EXCLUDED_SPATIAL_STATISTICS_V1",
                "fit_time_ns": [70.0, 260.0],
                "target_guard_ns": 42.0,
                "spatial_frame_traces": 128,
                "spatial_frame_stride": 64,
                "spatial_n_fft": 256,
                "frequency_unit": "cycles_per_meter",
                "common_mode_removed": True,
                "fits": [_spatial_payload(spatial_all), _spatial_payload(spatial_paper)],
                "sampled_responses": responses,
                "sampled_envelopes": envelopes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "contract_id": "FORMAL09B2_LATERAL_COVARIANCE_DEVELOPMENT_V1",
        "status": "visual_candidates_only",
        "formal_training_allowed": False,
        "causal_pair_complete": False,
        "gprmax_solver_run_performed": False,
        "base_case": scene["case_id"],
        "base_output": merged.resolve().relative_to(ROOT).as_posix(),
        "base_output_sha256": sha256(merged),
        "seed": args.seed,
        "synthetic_trace_spacing_m": trace_spacing_m,
        "single_factor_group": "target-excluded lateral spectrum and optional nonstationary amplitude envelope",
        "locked_factors": [
            "FORMAL06C solved full scene",
            "FORMAL09B-1 paper-fold temporal spectrum",
            "FORMAL09A depth envelope",
            "FORMAL09A balanced gain budget",
            "target/background calibration target 4.5",
        ],
        "measured_trace_copying": False,
        "target_corridor_excluded": True,
        "paper_fit_lines": list(PAPER_FIT_LINES),
        "held_out_lines": ["Line9"],
        "validation_only_lines": ["Line6"],
        "metrics": metrics,
        "blind_mapping": blind_map,
        "visual_outputs": [blind_path.name, comparison_path.name, spatial_figure.name],
        "spatial_contract": spatial_contract.name,
        "next_gate": "blind human visual audit before metadata-conditioned FORMAL09B-3",
    }
    (output / "formal09b2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
