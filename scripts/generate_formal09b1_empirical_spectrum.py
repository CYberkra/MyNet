#!/usr/bin/env python3
"""Generate FORMAL09B-1 empirical-spectrum acquisition candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import extract_visible_phase  # noqa: E402
from scripts.generate_formal09a_multiline_acquisition_realism import (  # noqa: E402
    Variant,
    add_panel,
    apply_realism,
    build_realism_basis,
    morphology_metrics,
    process,
    read_reference_segments,
    representative_crop,
    robust_limit,
    sha256,
)
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402


DEVELOPMENT_LINES = ("Line3", "Line6", "Line7", "Line9", "LineL1")
PAPER_FIT_LINES = ("Line3", "Line7", "LineL1")
TARGET_VARIANT = Variant("balanced", 4.5, 0.12)


@dataclass(frozen=True)
class SpectrumFit:
    name: str
    lines: tuple[str, ...]
    frequency_hz: np.ndarray
    amplitude: np.ndarray
    line_amplitudes: dict[str, np.ndarray]
    frame_counts: dict[str, int]


def _normalise_amplitude(amplitude: np.ndarray) -> np.ndarray:
    values = np.asarray(amplitude, dtype=np.float64)
    values = np.maximum(values, np.finfo(np.float64).tiny)
    values[0] = np.finfo(np.float64).tiny
    norm = np.sqrt(np.mean(np.square(values[1:])))
    return values / max(float(norm), np.finfo(np.float64).tiny)


def equal_line_log_pool(line_amplitudes: list[np.ndarray]) -> np.ndarray:
    """Pool spectra with equal weight per line, independent of trace count."""

    if not line_amplitudes:
        raise ValueError("at least one line spectrum is required")
    stacked = np.stack(
        [_normalise_amplitude(values) for values in line_amplitudes], axis=0
    )
    pooled = np.exp(np.mean(np.log(np.maximum(stacked, 1e-12)), axis=0))
    return _normalise_amplitude(pooled)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _frame_power(frame: np.ndarray, n_fft: int) -> np.ndarray | None:
    values = np.asarray(frame, dtype=np.float64)
    values = values - np.mean(values)
    scale = float(np.sqrt(np.mean(np.square(values))))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).tiny:
        return None
    tapered = (values / scale) * np.hanning(values.size)
    power = np.square(np.abs(np.fft.rfft(tapered, n=n_fft)))
    power /= max(float(np.sum(power[1:])), np.finfo(np.float64).tiny)
    return power


def read_signal_style_segments(
    path: Path, allowed_lines: tuple[str, ...]
) -> dict[str, list[tuple[int, int]]]:
    segments: dict[str, list[tuple[int, int]]] = {line: [] for line in allowed_lines}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            line = row["line"]
            if line in segments and row["purpose"] == "signal_style":
                segments[line].append(
                    (int(row["trace_start"]), int(row["trace_end"]))
                )
    missing = [line for line, spans in segments.items() if not spans]
    if missing:
        raise ValueError(f"missing signal-style segments for {missing}")
    return segments


def fit_line_residual_spectrum(
    line_path: Path,
    spans: list[tuple[int, int]],
    *,
    n_fft: int = 512,
    frame_samples: int = 96,
    frame_stride: int = 48,
    fit_low_ns: float = 70.0,
    fit_high_ns: float = 500.0,
    target_guard_ns: float = 42.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit spectral shape after removing common mode and the target corridor."""

    with np.load(line_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_amplitude"], dtype=np.float64)
        time_ns = np.asarray(data["time_ns"], dtype=np.float64)
        target_ns = np.asarray(data["v15_final_center_time_ns"], dtype=np.float64)
        ignored = np.asarray(data["v15_final_ignore_trace"], dtype=bool)
        outside_height = np.asarray(
            data["flight_height_outside_planned_2_20_m"], dtype=bool
        )

    selected = np.concatenate(
        [np.arange(start, end + 1, dtype=np.int64) for start, end in spans]
    )
    selected = np.unique(selected)
    selected = selected[~ignored[selected] & ~outside_height[selected]]
    if selected.size < 2:
        raise ValueError(f"not enough valid traces in {line_path}")

    residual = raw[:, selected] - np.median(raw[:, selected], axis=1, keepdims=True)
    powers: list[np.ndarray] = []
    base = (time_ns >= fit_low_ns) & (time_ns <= fit_high_ns)
    for local_trace, source_trace in enumerate(selected):
        allowed = base & (np.abs(time_ns - target_ns[source_trace]) > target_guard_ns)
        for start, end in _true_runs(allowed):
            count = end - start + 1
            if count < frame_samples:
                continue
            for frame_start in range(start, end - frame_samples + 2, frame_stride):
                power = _frame_power(
                    residual[frame_start : frame_start + frame_samples, local_trace],
                    n_fft,
                )
                if power is not None:
                    powers.append(power)
    if not powers:
        raise ValueError(f"no target-excluded spectral frames in {line_path}")

    log_power = np.log(np.maximum(np.stack(powers, axis=0), 1e-18))
    amplitude = np.sqrt(np.exp(np.median(log_power, axis=0)))
    amplitude = _normalise_amplitude(amplitude)
    dt_s = float(np.median(np.diff(time_ns))) * 1e-9
    return np.fft.rfftfreq(n_fft, d=dt_s), amplitude, len(powers)


def fit_spectrum(
    name: str,
    lines: tuple[str, ...],
    lines_dir: Path,
    segment_csv: Path,
) -> SpectrumFit:
    segments = read_signal_style_segments(segment_csv, lines)
    line_amplitudes: dict[str, np.ndarray] = {}
    frame_counts: dict[str, int] = {}
    frequency_hz: np.ndarray | None = None
    for line in lines:
        frequency, amplitude, count = fit_line_residual_spectrum(
            lines_dir / f"{line}.npz", segments[line]
        )
        if frequency_hz is None:
            frequency_hz = frequency
        elif not np.allclose(frequency_hz, frequency, rtol=0.0, atol=1e-6):
            raise ValueError("measured lines do not share one time-sampling contract")
        line_amplitudes[line] = amplitude
        frame_counts[line] = count
    assert frequency_hz is not None
    pooled = equal_line_log_pool([line_amplitudes[line] for line in lines])
    return SpectrumFit(
        name=name,
        lines=lines,
        frequency_hz=frequency_hz,
        amplitude=pooled,
        line_amplitudes=line_amplitudes,
        frame_counts=frame_counts,
    )


def build_empirical_realism_basis(
    shape: tuple[int, int],
    dt_s: float,
    time_ns: np.ndarray,
    fit: SpectrumFit,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the 09A operator fixed and replace only its temporal filter."""

    rng = np.random.default_rng(seed)
    rows, traces = shape
    target_frequency = np.fft.rfftfreq(rows, d=dt_s)
    response = np.interp(
        target_frequency,
        fit.frequency_hz,
        fit.amplitude,
        left=0.0,
        right=0.0,
    )
    response[0] = 0.0
    response /= max(float(np.sqrt(np.mean(np.square(response[1:])))), 1e-12)

    white = rng.standard_normal(shape)
    diffuse = np.fft.irfft(
        np.fft.rfft(white, axis=0) * response[:, None], n=rows, axis=0
    )
    diffuse = gaussian_filter1d(diffuse, sigma=0.75, axis=1, mode="reflect")

    low_rank = np.zeros(shape, dtype=np.float64)
    for _ in range(4):
        temporal = rng.standard_normal(rows)
        temporal = np.fft.irfft(np.fft.rfft(temporal) * response, n=rows)
        lateral = gaussian_filter1d(
            rng.standard_normal(traces), sigma=4.0, mode="reflect"
        )
        temporal /= max(float(np.std(temporal)), 1e-12)
        lateral = (lateral - np.mean(lateral)) / max(float(np.std(lateral)), 1e-12)
        low_rank += temporal[:, None] * lateral[None, :]
    low_rank /= 4.0

    depth = np.clip((time_ns - 45.0) / 455.0, 0.0, 1.0)
    depth = 0.08 + 0.92 * np.power(depth, 0.85)
    combined = 0.72 * diffuse / max(float(np.std(diffuse)), 1e-12)
    combined += 0.28 * low_rank / max(float(np.std(low_rank)), 1e-12)
    basis = combined * depth[:, None]
    basis = (basis - np.mean(basis)) / max(float(np.std(basis)), 1e-12)

    gain = gaussian_filter1d(rng.standard_normal(traces), sigma=4.5, mode="reflect")
    gain = (gain - np.mean(gain)) / max(float(np.std(gain)), 1e-12)
    return basis, gain, response


def _load_base(
    release_dir: Path, source_case_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict, dict, Path]:
    merged = release_dir / "full_scene_merged.out"
    run = json.loads((release_dir / "run_manifest.json").read_text(encoding="utf-8"))
    scene_path = release_dir.parents[1] / "source" / "scene_manifest.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    dt_s, raw, _ = read_merged_bscan(merged, component="Ez")
    raw = np.asarray(raw, dtype=np.float64)
    time_ns = np.arange(raw.shape[0], dtype=np.float64) * dt_s * 1e9
    selected = np.asarray(run["selected_trace_indices_zero_based"], dtype=np.int64)
    reference = np.load(
        source_case_dir / "labels" / "source_referenced_arrival_time_ns.npy"
    ).astype(np.float64)[selected]
    protected = time_ns <= 500.0
    suppressed = raw[protected] - np.median(raw[protected], axis=1, keepdims=True)
    path_ns, _, _ = extract_visible_phase(
        suppressed,
        np.zeros_like(suppressed),
        time_ns[protected],
        reference,
        search_half_width_ns=35.0,
        phase_half_width_ns=8.0,
        enforce_continuity=True,
        max_trace_step_ns=44.8,
        geometric_anchor_weight=2.0,
    )
    return raw, time_ns, path_ns, dt_s, run, scene, merged


def _spectrum_payload(fit: SpectrumFit) -> dict:
    return {
        "name": fit.name,
        "lines": list(fit.lines),
        "frequency_mhz": (fit.frequency_hz / 1e6).tolist(),
        "pooled_amplitude": fit.amplitude.tolist(),
        "line_amplitudes": {
            line: values.tolist() for line, values in fit.line_amplitudes.items()
        },
        "frame_counts": fit.frame_counts,
    }


def spectrum_summary(fit: SpectrumFit) -> dict[str, float]:
    frequency_mhz = fit.frequency_hz / 1e6
    mask = (frequency_mhz > 0.0) & (frequency_mhz <= 250.0)
    power = np.square(fit.amplitude[mask])
    return {
        "peak_frequency_mhz": float(
            frequency_mhz[mask][np.argmax(fit.amplitude[mask])]
        ),
        "spectral_centroid_mhz": float(
            np.sum(frequency_mhz[mask] * power)
            / max(float(np.sum(power)), np.finfo(np.float64).tiny)
        ),
    }


def draw_spectrum_audit(
    development_fit: SpectrumFit, paper_fit: SpectrumFit, output_path: Path
) -> None:
    width, height = 1280, 660
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 18),
        "FORMAL09B-1 target-excluded measured residual spectra (line-normalised)",
        fill="black",
    )

    panels = ((55, 70, 600, 540), (670, 70, 600, 540))
    colors = {
        "Line3": "#1f77b4",
        "Line6": "#ff7f0e",
        "Line7": "#2ca02c",
        "Line9": "#d62728",
        "LineL1": "#9467bd",
        "all-lines pooled": "#111111",
        "paper-fold pooled": "#00a6a6",
        "09A Gaussian band": "#cc00cc",
    }

    def draw_panel(
        box: tuple[int, int, int, int],
        title: str,
        curves: list[tuple[str, np.ndarray, np.ndarray]],
    ) -> None:
        x, y, w, h = box
        draw.rectangle((x, y, x + w, y + h), outline="black")
        draw.text((x + 8, y + 8), title, fill="black")
        plot = (x + 52, y + 35, x + w - 16, y + h - 42)
        draw.rectangle(plot, outline="#888888")
        left, top, right, bottom = plot
        for frequency in (0, 50, 100, 150, 200, 250):
            px = left + (right - left) * frequency / 250.0
            draw.line((px, top, px, bottom), fill="#eeeeee")
            draw.text((int(px) - 8, bottom + 7), str(frequency), fill="black")
        for db in (-40, -30, -20, -10, 0):
            py = bottom - (bottom - top) * (db + 40.0) / 40.0
            draw.line((left, py, right, py), fill="#eeeeee")
            draw.text((left - 34, int(py) - 6), str(db), fill="black")
        draw.text((left + 190, bottom + 23), "Frequency (MHz)", fill="black")
        draw.text((left - 45, top - 18), "dB", fill="black")
        legend_width = 175
        legend_height = 12 + 18 * len(curves)
        legend_left = right - legend_width
        draw.rectangle(
            (legend_left, top + 6, right - 5, top + 6 + legend_height),
            fill="white",
            outline="#bbbbbb",
        )
        for index, (name, frequency_mhz, amplitude) in enumerate(curves):
            mask = (frequency_mhz >= 0.0) & (frequency_mhz <= 250.0)
            values = amplitude[mask]
            db = 20.0 * np.log10(
                np.maximum(values / max(float(np.max(values)), 1e-12), 1e-4)
            )
            points = [
                (
                    left + (right - left) * float(frequency) / 250.0,
                    bottom - (bottom - top) * float(level + 40.0) / 40.0,
                )
                for frequency, level in zip(frequency_mhz[mask], np.clip(db, -40, 0))
            ]
            draw.line(points, fill=colors[name], width=2 if "pooled" in name else 1)
            legend_y = top + 14 + 18 * index
            draw.line(
                (legend_left + 8, legend_y + 5, legend_left + 34, legend_y + 5),
                fill=colors[name],
                width=2,
            )
            draw.text((legend_left + 41, legend_y), name, fill="black")

    line_curves = [
        (
            line,
            development_fit.frequency_hz / 1e6,
            development_fit.line_amplitudes[line],
        )
        for line in development_fit.lines
    ]
    frequency_mhz = development_fit.frequency_hz / 1e6
    gaussian = np.exp(-0.5 * np.square((development_fit.frequency_hz - 82e6) / 42e6))
    draw_panel(panels[0], "Per-line robust spectra", line_curves)
    draw_panel(
        panels[1],
        "Equal-line pooled spectra vs 09A hand-shaped band",
        [
            ("all-lines pooled", frequency_mhz, development_fit.amplitude),
            ("paper-fold pooled", frequency_mhz, paper_fit.amplitude),
            ("09A Gaussian band", frequency_mhz, gaussian),
        ],
    )
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-lines", type=Path, required=True)
    parser.add_argument("--reference-segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071610)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    development_fit = fit_spectrum(
        "development_all_lines",
        DEVELOPMENT_LINES,
        args.measured_lines,
        args.reference_segments,
    )
    paper_fit = fit_spectrum(
        "paper_line9_holdout_fit",
        PAPER_FIT_LINES,
        args.measured_lines,
        args.reference_segments,
    )

    raw, time_ns, path_ns, dt_s, run, scene, merged = _load_base(
        args.release_dir, args.source_case_dir
    )
    protected = time_ns <= 500.0
    protected_time = time_ns[protected]

    hand_basis, hand_gain = build_realism_basis(raw.shape, dt_s, time_ns, args.seed)
    hand, hand_report = apply_realism(
        raw, time_ns, path_ns, hand_basis, hand_gain, TARGET_VARIANT
    )
    realized: dict[str, np.ndarray] = {"baseline": raw, "09A-balanced": hand}
    metrics: dict[str, dict[str, float]] = {
        "baseline": morphology_metrics(raw[protected], protected_time, path_ns),
        "09A-balanced": {
            **hand_report,
            **morphology_metrics(hand[protected], protected_time, path_ns),
        },
    }
    responses: dict[str, np.ndarray] = {}
    for fit in (development_fit, paper_fit):
        basis, gain, response = build_empirical_realism_basis(
            raw.shape, dt_s, time_ns, fit, args.seed
        )
        values, report = apply_realism(
            raw, time_ns, path_ns, basis, gain, TARGET_VARIANT
        )
        realized[fit.name] = values
        responses[fit.name] = response
        metrics[fit.name] = {
            **report,
            **morphology_metrics(values[protected], protected_time, path_ns),
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
        "FORMAL09B-1 blind empirical-spectrum checkpoint; common scales; no labels",
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
    blind_path = output / "FORMAL09B1_blind_empirical_spectrum.png"
    canvas.save(blind_path)

    longest = read_reference_segments(args.reference_segments)
    selected = np.asarray(run["selected_trace_indices_zero_based"], dtype=np.int64)
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
    comparison_items.extend(
        [
            ("FORMAL06C", processed["baseline"]),
            ("09A-balanced", processed["09A-balanced"]),
            ("09B1-all-lines", processed["development_all_lines"]),
            ("09B1-paper-fold", processed["paper_line9_holdout_fit"]),
        ]
    )
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
        "Multi-line references vs FORMAL09B-1 empirical spectra; independent P99.5 scales",
        fill="black",
    )
    for index, (title, matrix) in enumerate(comparison_items):
        row, column = divmod(index, columns)
        x = margin + column * (width + margin)
        y = header + row * (height + margin)
        add_panel(canvas, matrix, title, (x, y, width, height), robust_limit(matrix))
    comparison_path = output / "FORMAL09B1_multiline_visual_comparison.png"
    canvas.save(comparison_path)

    spectrum_path = output / "FORMAL09B1_fitted_spectra.json"
    spectrum_path.write_text(
        json.dumps(
            {
                "contract_id": "FORMAL09B1_TARGET_EXCLUDED_SPECTRA_V1",
                "target_guard_ns": 42.0,
                "fit_time_ns": [70.0, 500.0],
                "frame_samples": 96,
                "frame_stride": 48,
                "n_fft": 512,
                "common_mode_removed": True,
                "line_pooling": "equal-weight geometric mean of line-normalised amplitudes",
                "fits": [_spectrum_payload(development_fit), _spectrum_payload(paper_fit)],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    spectrum_figure = output / "FORMAL09B1_spectrum_audit.png"
    draw_spectrum_audit(development_fit, paper_fit, spectrum_figure)

    shared_frequency_mhz = development_fit.frequency_hz / 1e6
    comparison_mask = (shared_frequency_mhz >= 20.0) & (shared_frequency_mhz <= 250.0)
    development_db = 20.0 * np.log10(
        development_fit.amplitude[comparison_mask]
        / np.max(development_fit.amplitude[comparison_mask])
    )
    paper_db = 20.0 * np.log10(
        paper_fit.amplitude[comparison_mask]
        / np.max(paper_fit.amplitude[comparison_mask])
    )

    manifest = {
        "contract_id": "FORMAL09B1_EMPIRICAL_SPECTRUM_DEVELOPMENT_V1",
        "status": "visual_candidates_only",
        "formal_training_allowed": False,
        "causal_pair_complete": False,
        "gprmax_solver_run_performed": False,
        "base_case": scene["case_id"],
        "base_output": merged.resolve().relative_to(ROOT).as_posix(),
        "base_output_sha256": sha256(merged),
        "seed": args.seed,
        "single_factor_change": "09A temporal Gaussian band replaced by target-excluded empirical residual spectrum",
        "locked_factors": [
            "FORMAL06C solved full scene",
            "09A diffuse/low-rank mixture",
            "09A lateral smoothing",
            "09A depth envelope",
            "09A balanced gain jitter",
            "target/background calibration target 4.5",
        ],
        "measured_trace_copying": False,
        "label_conditioned_enhancement": False,
        "target_corridor_excluded_from_spectrum_fit": True,
        "fits": [
            {
                "name": development_fit.name,
                "lines": list(development_fit.lines),
                "line9_conditioned": True,
                **spectrum_summary(development_fit),
            },
            {
                "name": paper_fit.name,
                "lines": list(paper_fit.lines),
                "line9_conditioned": False,
                **spectrum_summary(paper_fit),
            },
        ],
        "all_lines_vs_paper_fold_log_spectrum_rmse_db_20_250mhz": float(
            np.sqrt(np.mean(np.square(development_db - paper_db)))
        ),
        "variant": asdict(TARGET_VARIANT),
        "metrics": metrics,
        "blind_mapping": blind_map,
        "visual_outputs": [
            blind_path.name,
            comparison_path.name,
            spectrum_figure.name,
        ],
        "spectrum_contract": spectrum_path.name,
        "next_gate": "blind human visual audit before any 09B-2 covariance modelling",
    }
    (output / "formal09b1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
