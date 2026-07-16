#!/usr/bin/env python3
"""Generate FORMAL09C sparse coherent-event acquisition candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, gaussian_filter, label
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
    seismic_rgb,
    sha256,
)
from scripts.generate_formal09b1_empirical_spectrum import (  # noqa: E402
    DEVELOPMENT_LINES,
    PAPER_FIT_LINES,
    TARGET_VARIANT,
    SpectrumFit,
    _load_base,
    build_empirical_realism_basis,
    fit_spectrum,
)
from scripts.generate_formal09b2_lateral_covariance import (  # noqa: E402
    background_lateral_metrics,
    read_segments,
)


@dataclass(frozen=True)
class DetectedEvent:
    length_m: float
    slope_ns_per_m: float
    curvature_ns_per_m2: float
    amplitude_p99_fraction: float
    center_time_ns: float


@dataclass(frozen=True)
class EventFit:
    lines: tuple[str, ...]
    line_event_counts: dict[str, int]
    line_lengths_m: dict[str, float]
    events_per_25m: dict[str, float]
    pooled_quantiles: dict[str, tuple[float, float, float]]
    pooled_events_per_25m: float


@dataclass(frozen=True)
class Variant:
    name: str
    coherent_fraction: float
    count_multiplier: float


VARIANTS = (
    Variant("light", 0.20, 0.65),
    Variant("balanced", 0.35, 1.00),
    Variant("rich", 0.50, 1.35),
)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    centered_x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    centered_y = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    return float(
        np.sum(centered_x * centered_y)
        / max(float(np.sum(np.square(centered_x))), np.finfo(np.float64).tiny)
    )


def detect_line_events(
    line_path: Path,
    spans: list[tuple[int, int]],
    *,
    low_ns: float = 250.0,
    high_ns: float = 500.0,
    target_guard_ns: float = 42.0,
) -> tuple[list[DetectedEvent], float]:
    with np.load(line_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_amplitude"], dtype=np.float64)
        time_ns = np.asarray(data["time_ns"], dtype=np.float64)
        distance_m = np.asarray(data["gnss_cumulative_distance_m"], dtype=np.float64)
        target_ns = np.asarray(data["v15_final_center_time_ns"], dtype=np.float64)
        ignored = np.asarray(data["v15_final_ignore_trace"], dtype=bool)
        outside_height = np.asarray(
            data["flight_height_outside_planned_2_20_m"], dtype=bool
        )

    valid_trace = np.zeros(raw.shape[1], dtype=bool)
    for start, end in spans:
        valid_trace[start : end + 1] = True
    valid_trace &= ~ignored & ~outside_height
    selected = np.flatnonzero(valid_trace)
    residual = raw - np.median(raw[:, selected], axis=1, keepdims=True)
    gain = np.power(np.clip(time_ns / 500.0, 0.02, 1.0), 1.5)
    image = residual * gain[:, None]
    rows = np.flatnonzero((time_ns >= low_ns) & (time_ns <= high_ns))
    local = image[rows]
    local_time = time_ns[rows]
    target_clear = (
        np.abs(local_time[:, None] - target_ns[None, :]) > target_guard_ns
    ) & valid_trace[None, :]
    local = np.where(target_clear, local, 0.0)

    steps = np.diff(distance_m)
    spacing_m = float(np.median(steps[steps > 1e-4]))
    dt_ns = float(np.median(np.diff(local_time)))
    grad_t = gaussian_filter(local, (1.4, 1.2), order=(1, 0)) / dt_ns
    grad_x = gaussian_filter(local, (1.4, 1.2), order=(0, 1)) / spacing_m
    jtt = gaussian_filter(np.square(grad_t), (2.0, 3.0))
    jxx = gaussian_filter(np.square(grad_x), (2.0, 3.0))
    jtx = gaussian_filter(grad_t * grad_x, (2.0, 3.0))
    trace = jtt + jxx
    coherence = np.sqrt(np.square(jtt - jxx) + 4.0 * np.square(jtx)) / np.maximum(
        trace, np.finfo(np.float64).tiny
    )
    amplitude = np.abs(local)
    threshold = float(np.quantile(amplitude[target_clear], 0.88))
    active = target_clear & (amplitude >= threshold) & (coherence >= 0.45)
    active = binary_closing(active, structure=np.ones((3, 5), dtype=bool))
    components, count = label(active, structure=np.ones((3, 3), dtype=np.int8))
    p99 = float(np.quantile(amplitude[target_clear], 0.99))

    events: list[DetectedEvent] = []
    for component_id in range(1, count + 1):
        event_rows, event_columns = np.where(components == component_id)
        unique_columns = np.unique(event_columns)
        if event_rows.size < 18 or unique_columns.size < 8:
            continue
        length_m = float(
            distance_m[event_columns].max() - distance_m[event_columns].min()
        )
        if length_m < 1.5:
            continue

        ridge_time: list[float] = []
        ridge_distance: list[float] = []
        for column in unique_columns:
            column_rows = event_rows[event_columns == column]
            weights = amplitude[column_rows, column]
            ridge_time.append(
                float(
                    np.sum(local_time[column_rows] * weights)
                    / max(float(np.sum(weights)), np.finfo(np.float64).tiny)
                )
            )
            ridge_distance.append(float(distance_m[column]))
        x = np.asarray(ridge_distance)
        y = np.asarray(ridge_time)
        if x.size < 12:
            continue
        slope = _linear_slope(x, y)
        middle = x.size // 2
        left_slope = _linear_slope(x[:middle], y[:middle])
        right_slope = _linear_slope(x[middle:], y[middle:])
        separation = float(np.mean(x[middle:]) - np.mean(x[:middle]))
        curvature = (right_slope - left_slope) / max(2.0 * separation, 1e-6)
        events.append(
            DetectedEvent(
                length_m=length_m,
                slope_ns_per_m=slope,
                curvature_ns_per_m2=curvature,
                amplitude_p99_fraction=float(
                    np.median(amplitude[event_rows, event_columns])
                    / max(p99, np.finfo(np.float64).tiny)
                ),
                center_time_ns=float(np.mean(y)),
            )
        )
    covered_length = float(
        sum(distance_m[end] - distance_m[start] for start, end in spans)
    )
    return events, covered_length


def _line_quantiles(events: list[DetectedEvent], field: str) -> np.ndarray:
    values = np.asarray([getattr(event, field) for event in events], dtype=np.float64)
    return np.quantile(values, (0.10, 0.50, 0.90))


def fit_event_contract(
    lines: tuple[str, ...], lines_dir: Path, segment_csv: Path
) -> tuple[EventFit, dict[str, list[DetectedEvent]]]:
    segments = read_segments(segment_csv, lines)
    by_line: dict[str, list[DetectedEvent]] = {}
    line_lengths: dict[str, float] = {}
    density: dict[str, float] = {}
    fields = (
        "length_m",
        "slope_ns_per_m",
        "curvature_ns_per_m2",
        "amplitude_p99_fraction",
        "center_time_ns",
    )
    line_quantiles: dict[str, dict[str, np.ndarray]] = {}
    for line in lines:
        events, length_m = detect_line_events(
            lines_dir / f"{line}.npz", segments[line]
        )
        if not events:
            raise ValueError(f"no coherent events detected for {line}")
        by_line[line] = events
        line_lengths[line] = length_m
        density[line] = len(events) / max(length_m, 1e-6) * 25.0
        line_quantiles[line] = {
            field: _line_quantiles(events, field) for field in fields
        }
    pooled = {
        field: tuple(
            float(value)
            for value in np.median(
                np.stack([line_quantiles[line][field] for line in lines], axis=0),
                axis=0,
            )
        )
        for field in fields
    }
    fit = EventFit(
        lines=lines,
        line_event_counts={line: len(by_line[line]) for line in lines},
        line_lengths_m=line_lengths,
        events_per_25m=density,
        pooled_quantiles=pooled,
        pooled_events_per_25m=float(np.median(list(density.values()))),
    )
    return fit, by_line


def _sample_quantiles(
    rng: np.random.Generator, quantiles: tuple[float, float, float]
) -> float:
    low, mode, high = quantiles
    if high <= low:
        return float(mode)
    mode = float(np.clip(mode, low, high))
    return float(rng.triangular(low, mode, high))


def _sample_quantile_span(
    rng: np.random.Generator, quantiles: tuple[float, float, float]
) -> float:
    """Sample across the observed central quantile span without a sharp mode."""
    low, median, high = quantiles
    probability = float(rng.uniform(0.10, 0.90))
    return float(np.interp(probability, (0.10, 0.50, 0.90), (low, median, high)))


def empirical_wavelet(temporal_fit: SpectrumFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_dt_ns = 1.4
    sample_count = 1024
    frequency = np.fft.rfftfreq(sample_count, d=sample_dt_ns * 1e-9)
    amplitude = np.interp(
        frequency,
        temporal_fit.frequency_hz,
        temporal_fit.amplitude,
        left=0.0,
        right=0.0,
    )
    amplitude[0] = 0.0
    even = np.fft.fftshift(np.fft.irfft(amplitude, n=sample_count))
    even /= max(float(np.max(np.abs(even))), 1e-12)
    quadrature = np.imag(hilbert(even))
    relative_ns = (np.arange(sample_count) - sample_count // 2) * sample_dt_ns
    keep = np.abs(relative_ns) <= 70.0
    return relative_ns[keep], even[keep], quadrature[keep]


def render_event_field(
    shape: tuple[int, int],
    time_ns: np.ndarray,
    target_path_ns: np.ndarray,
    trace_spacing_m: float,
    temporal_fit: SpectrumFit,
    event_fit: EventFit,
    variant: Variant,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    x = np.arange(shape[1], dtype=np.float64) * trace_spacing_m
    span_m = float(x[-1] - x[0])
    count = max(
        1,
        int(round(event_fit.pooled_events_per_25m * span_m / 25.0 * variant.count_multiplier)),
    )
    relative_ns, even_wavelet, quadrature_wavelet = empirical_wavelet(temporal_fit)
    field = np.zeros(shape, dtype=np.float64)
    records: list[dict[str, float]] = []
    attempts = 0
    while len(records) < count and attempts < 50 * count:
        attempts += 1
        length = float(
            np.clip(
                _sample_quantiles(rng, event_fit.pooled_quantiles["length_m"]),
                1.8,
                min(12.0, span_m * 0.75),
            )
        )
        center_x = float(rng.uniform(0.15 * span_m, 0.85 * span_m))
        slope = float(
            np.clip(
                _sample_quantiles(rng, event_fit.pooled_quantiles["slope_ns_per_m"]),
                -7.0,
                7.0,
            )
        )
        # Short detected components make second-derivative estimates unstable.
        # Keep curvature as a declared conservative geometry prior instead of
        # clipping noisy measured estimates into artificial hyperbolas.
        curvature = float(np.clip(rng.normal(0.0, 0.04), -0.12, 0.12))
        center_time = float(
            np.clip(
                _sample_quantile_span(
                    rng, event_fit.pooled_quantiles["center_time_ns"]
                ),
                265.0,
                475.0,
            )
        )
        support = np.abs(x - center_x) <= length / 2.0
        if np.count_nonzero(support) < 3:
            continue
        centered_x = x - center_x
        path = center_time + slope * centered_x + curvature * np.square(centered_x)
        target_overlap = np.mean(
            np.abs(path[support] - target_path_ns[support]) < 42.0
        )
        if target_overlap > 0.20 or np.min(path[support]) < 250.0 or np.max(path[support]) > 500.0:
            continue

        support_index = np.flatnonzero(support)
        taper = np.hanning(max(5, support_index.size + 2))[1:-1]
        if taper.size != support_index.size:
            taper = np.ones(support_index.size)
        phase = float(rng.uniform(-math.pi, math.pi))
        phase_wavelet = math.cos(phase) * even_wavelet + math.sin(phase) * quadrature_wavelet
        amplitude = float(
            np.clip(
                _sample_quantiles(
                    rng, event_fit.pooled_quantiles["amplitude_p99_fraction"]
                ),
                0.15,
                0.55,
            )
        )
        for local_index, trace in enumerate(support_index):
            shifted = np.interp(
                time_ns - path[trace],
                relative_ns,
                phase_wavelet,
                left=0.0,
                right=0.0,
            )
            field[:, trace] += amplitude * taper[local_index] * shifted
        records.append(
            {
                "center_x_m": center_x,
                "center_time_ns": center_time,
                "length_m": length,
                "slope_ns_per_m": slope,
                "curvature_ns_per_m2": curvature,
                "amplitude_weight": amplitude,
                "phase_rad": phase,
                "target_overlap_fraction": float(target_overlap),
            }
        )
    if not records:
        raise RuntimeError("failed to sample any target-excluded coherent event")
    field = (field - np.mean(field)) / max(float(np.std(field)), 1e-12)
    return field, records


def build_candidate_basis(
    diffuse_basis: np.ndarray,
    event_field: np.ndarray,
    coherent_fraction: float,
) -> np.ndarray:
    diffuse = diffuse_basis / max(
        float(np.quantile(np.abs(diffuse_basis), 0.995)), 1e-12
    )
    coherent = event_field / max(
        float(np.quantile(np.abs(event_field), 0.995)), 1e-12
    )
    mixed = (1.0 - coherent_fraction) * diffuse + coherent_fraction * coherent
    return (mixed - np.mean(mixed)) / max(float(np.std(mixed)), 1e-12)


def add_sparse_panel(
    canvas: Image.Image,
    matrix: np.ndarray,
    title: str,
    box: tuple[int, int, int, int],
    limit: float,
) -> None:
    """Resize sparse traces without inventing horizontal continuity."""
    x, y, width, height = box
    image = Image.fromarray(seismic_rgb(matrix, limit))
    image = image.resize((matrix.shape[1], height), Image.Resampling.BILINEAR)
    image = image.resize((width, height), Image.Resampling.NEAREST)
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + width - 1, y + height - 1), outline="black")
    draw.rectangle((x, y, x + width - 1, y + 25), fill="white")
    draw.text((x + 7, y + 6), title, fill="black")


def _fit_payload(fit: EventFit) -> dict:
    return {
        "lines": list(fit.lines),
        "line_event_counts": fit.line_event_counts,
        "line_lengths_m": fit.line_lengths_m,
        "events_per_25m": fit.events_per_25m,
        "pooled_quantiles_q10_q50_q90": {
            key: list(values) for key, values in fit.pooled_quantiles.items()
        },
        "pooled_events_per_25m": fit.pooled_events_per_25m,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-lines", type=Path, required=True)
    parser.add_argument("--reference-segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071613)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporal_fit = fit_spectrum(
        "paper_line9_holdout_fit",
        PAPER_FIT_LINES,
        args.measured_lines,
        args.reference_segments,
    )
    event_fit, detected = fit_event_contract(
        PAPER_FIT_LINES, args.measured_lines, args.reference_segments
    )
    raw, time_ns, path_ns, dt_s, run, scene, merged = _load_base(
        args.release_dir, args.source_case_dir
    )
    selected = np.asarray(run["selected_trace_indices_zero_based"], dtype=np.int64)
    trace_spacing_m = float(scene["grid"]["trace_spacing_m"]) * float(
        np.median(np.diff(selected))
    )
    diffuse_basis, gain_basis, _ = build_empirical_realism_basis(
        raw.shape, dt_s, time_ns, temporal_fit, args.seed
    )
    baseline, baseline_report = apply_realism(
        raw, time_ns, path_ns, diffuse_basis, gain_basis, TARGET_VARIANT
    )
    realized = {"09B1-paper": baseline}
    metrics = {"09B1-paper": baseline_report}
    generated_events: dict[str, list[dict[str, float]]] = {}
    event_fields: dict[str, np.ndarray] = {}
    for index, variant in enumerate(VARIANTS):
        event_field, records = render_event_field(
            raw.shape,
            time_ns,
            path_ns,
            trace_spacing_m,
            temporal_fit,
            event_fit,
            variant,
            args.seed + 100 + index,
        )
        basis = build_candidate_basis(
            diffuse_basis, event_field, variant.coherent_fraction
        )
        values, report = apply_realism(
            raw, time_ns, path_ns, basis, gain_basis, TARGET_VARIANT
        )
        name = f"09C-{variant.name}"
        realized[name] = values
        metrics[name] = report
        generated_events[name] = records
        event_fields[name] = event_field

    protected = time_ns <= 500.0
    for name, values in realized.items():
        metrics[name] = {
            **metrics[name],
            **morphology_metrics(values[protected], time_ns[protected], path_ns),
            **background_lateral_metrics(values, time_ns, path_ns, trace_spacing_m),
        }

    crop_rows = (time_ns >= 250.0) & (time_ns <= 500.0)
    raw_crop = {name: values[crop_rows] for name, values in realized.items()}
    processed = {name: process(values, time_ns)[crop_rows] for name, values in realized.items()}
    raw_limit = robust_limit(np.concatenate(list(raw_crop.values()), axis=1))
    processed_limit = robust_limit(np.concatenate(list(processed.values()), axis=1))
    blind_order = list(realized)
    np.random.default_rng(args.seed + 10).shuffle(blind_order)
    blind_map = {chr(65 + index): name for index, name in enumerate(blind_order)}
    width, height, margin, header = 360, 430, 18, 48
    canvas = Image.new(
        "RGB", (margin * 5 + width * 4, header + margin * 3 + height * 2), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 14),
        "FORMAL09C blind sparse coherent-event checkpoint; common scales; no labels",
        fill="black",
    )
    for column, blind_id in enumerate("ABCD"):
        name = blind_map[blind_id]
        x = margin + column * (width + margin)
        add_sparse_panel(
            canvas,
            raw_crop[name],
            f"{blind_id}: raw",
            (x, header, width, height),
            raw_limit,
        )
        add_sparse_panel(
            canvas,
            processed[name],
            f"{blind_id}: common-mode suppressed + time^1.5",
            (x, header + height + margin, width, height),
            processed_limit,
        )
    blind_path = output / "FORMAL09C_blind_sparse_events.png"
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
        "Multi-line references vs FORMAL09C sparse events; independent P99.5 scales",
        fill="black",
    )
    for index, (title, matrix) in enumerate(comparison_items):
        row, column = divmod(index, columns)
        x = margin + column * (width + margin)
        y = header + row * (height + margin)
        if title in DEVELOPMENT_LINES:
            add_panel(canvas, matrix, title, (x, y, width, height), robust_limit(matrix))
        else:
            add_sparse_panel(
                canvas, matrix, title, (x, y, width, height), robust_limit(matrix)
            )
    comparison_path = output / "FORMAL09C_multiline_visual_comparison.png"
    canvas.save(comparison_path)

    event_crop = {name: values[crop_rows] for name, values in event_fields.items()}
    event_limit = robust_limit(np.concatenate(list(event_crop.values()), axis=1))
    width, height, margin, header = 430, 420, 18, 58
    canvas = Image.new(
        "RGB", (margin * 4 + width * 3, header + margin + height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        "FORMAL09C generated event-only fields; sparse 0.72 m traces; horizontal nearest-neighbour display",
        fill="black",
    )
    for column, variant in enumerate(VARIANTS):
        name = f"09C-{variant.name}"
        x = margin + column * (width + margin)
        add_sparse_panel(
            canvas,
            event_crop[name],
            f"{name}: {len(generated_events[name])} finite events",
            (x, header, width, height),
            event_limit,
        )
    event_audit_path = output / "FORMAL09C_event_geometry_audit.png"
    canvas.save(event_audit_path)

    event_contract_path = output / "FORMAL09C_event_contract.json"
    event_contract_path.write_text(
        json.dumps(
            {
                "contract_id": "FORMAL09C_TARGET_EXCLUDED_EVENT_STATISTICS_V1",
                "fit_lines": list(PAPER_FIT_LINES),
                "validation_only_lines": ["Line6"],
                "held_out_lines": ["Line9"],
                "detection_time_ns": [250.0, 500.0],
                "target_guard_ns": 42.0,
                "event_fit": _fit_payload(event_fit),
                "curvature_contract": {
                    "source": "conservative_zero_mean_geometry_prior",
                    "standard_deviation_ns_per_m2": 0.04,
                    "absolute_limit_ns_per_m2": 0.12,
                    "reason": "short connected components make measured second derivatives unstable",
                },
                "detected_event_records_saved": False,
                "detected_event_records_reason": "quantiles retained; measured event coordinates are not copied into the synthetic contract",
                "generated_events": generated_events,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "contract_id": "FORMAL09C_SPARSE_COHERENT_EVENT_FIELD_DEVELOPMENT_V1",
        "status": "visual_candidates_only",
        "formal_training_allowed": False,
        "causal_pair_complete": False,
        "gprmax_solver_run_performed": False,
        "base_case": scene["case_id"],
        "base_output": merged.resolve().relative_to(ROOT).as_posix(),
        "base_output_sha256": sha256(merged),
        "seed": args.seed,
        "synthetic_trace_spacing_m": trace_spacing_m,
        "retained_component": "FORMAL09B-1 paper-fold diffuse temporal spectrum",
        "single_factor_group": "finite sparse coherent-event topology",
        "measured_trace_patch_waveform_or_coordinate_copying": False,
        "target_corridor_excluded_from_fit_and_generation": True,
        "variants": [asdict(variant) for variant in VARIANTS],
        "metrics": metrics,
        "blind_mapping": blind_map,
        "visual_outputs": [
            blind_path.name,
            comparison_path.name,
            event_audit_path.name,
        ],
        "sparse_display_contract": {
            "vertical_resize": "bilinear",
            "horizontal_resize": "nearest_neighbour",
            "horizontal_interpolation_forbidden": True,
        },
        "event_contract": event_contract_path.name,
        "next_gate": "blind human visual audit before any metadata conditioning or pair export",
    }
    (output / "formal09c_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
