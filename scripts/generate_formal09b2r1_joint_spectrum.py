#!/usr/bin/env python3
"""Generate FORMAL09B-2R1 candidates from empirical joint 2D spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d

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
    _load_base,
    build_empirical_realism_basis,
    fit_spectrum,
)
from scripts.generate_formal09b2_lateral_covariance import (  # noqa: E402
    SpatialFit,
    background_lateral_metrics,
    build_lateral_covariance_basis,
    fit_spatial_pool,
    read_segments,
)


COMMON_TEMPORAL_MHZ = np.linspace(-250.0, 250.0, 201)
COMMON_SPATIAL_CPM = np.linspace(-0.70, 0.70, 141)


@dataclass(frozen=True)
class JointSpectrumFit:
    name: str
    lines: tuple[str, ...]
    temporal_mhz: np.ndarray
    spatial_cpm: np.ndarray
    amplitude: np.ndarray
    line_amplitudes: dict[str, np.ndarray]
    patch_counts: dict[str, int]


def _normalise_joint(amplitude: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(amplitude, dtype=np.float64), 1e-12)
    norm = float(np.sqrt(np.mean(np.square(values))))
    return values / max(norm, 1e-12)


def equal_line_joint_pool(amplitudes: list[np.ndarray]) -> np.ndarray:
    if not amplitudes:
        raise ValueError("at least one joint spectrum is required")
    stacked = np.stack([_normalise_joint(values) for values in amplitudes], axis=0)
    pooled = np.exp(np.mean(np.log(np.maximum(stacked, 1e-12)), axis=0))
    power = np.square(pooled)
    power = 0.5 * (power + power[:, ::-1])
    return _normalise_joint(np.sqrt(power))


def _joint_patch_power(patch: np.ndarray, n_time: int, n_space: int) -> np.ndarray | None:
    values = np.asarray(patch, dtype=np.float64)
    values = values - np.mean(values, axis=1, keepdims=True)
    scale = float(np.sqrt(np.mean(np.square(values))))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).tiny:
        return None
    window = np.hanning(values.shape[0])[:, None] * np.hanning(values.shape[1])[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(values / scale * window, s=(n_time, n_space)))
    power = np.square(np.abs(spectrum))
    power /= max(float(np.sum(power)), np.finfo(np.float64).tiny)
    return power


def fit_line_joint_spectrum(
    line_path: Path,
    spans: list[tuple[int, int]],
    *,
    patch_time_samples: int = 96,
    patch_time_stride: int = 32,
    patch_traces: int = 128,
    patch_trace_stride: int = 64,
    n_time: int = 256,
    n_space: int = 256,
    fit_low_ns: float = 70.0,
    fit_high_ns: float = 260.0,
    target_guard_ns: float = 42.0,
) -> tuple[np.ndarray, int]:
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
    residual = raw - np.median(raw[:, selected], axis=1, keepdims=True)
    time_rows = np.flatnonzero((time_ns >= fit_low_ns) & (time_ns <= fit_high_ns))
    dt_s = float(np.median(np.diff(time_ns))) * 1e-9
    steps = np.diff(distance_m)
    spacing_m = float(np.median(steps[steps > 1e-4]))

    native_temporal = np.fft.fftshift(np.fft.fftfreq(n_time, d=dt_s)) / 1e6
    native_spatial = np.fft.fftshift(np.fft.fftfreq(n_space, d=spacing_m))
    query_t, query_x = np.meshgrid(
        COMMON_TEMPORAL_MHZ, COMMON_SPATIAL_CPM, indexing="ij"
    )
    query = np.column_stack((query_t.ravel(), query_x.ravel()))
    interpolated_powers: list[np.ndarray] = []

    for start, end in spans:
        local_valid = all_valid[start : end + 1]
        padded = np.concatenate(([False], local_valid, [False]))
        changes = np.diff(padded.astype(np.int8))
        run_starts = np.flatnonzero(changes == 1)
        run_ends = np.flatnonzero(changes == -1)
        for run_start, run_end in zip(run_starts, run_ends):
            columns = np.arange(start + run_start, start + run_end, dtype=np.int64)
            if columns.size < patch_traces:
                continue
            for time_offset in range(
                0, time_rows.size - patch_time_samples + 1, patch_time_stride
            ):
                rows = time_rows[time_offset : time_offset + patch_time_samples]
                for trace_offset in range(
                    0, columns.size - patch_traces + 1, patch_trace_stride
                ):
                    traces = columns[trace_offset : trace_offset + patch_traces]
                    if not np.all(
                        np.abs(time_ns[rows, None] - path_ns[traces][None, :])
                        > target_guard_ns
                    ):
                        continue
                    power = _joint_patch_power(
                        residual[np.ix_(rows, traces)], n_time, n_space
                    )
                    if power is None:
                        continue
                    interpolator = RegularGridInterpolator(
                        (native_temporal, native_spatial),
                        np.log(np.maximum(power, 1e-20)),
                        bounds_error=False,
                        fill_value=np.log(1e-20),
                    )
                    interpolated_powers.append(
                        np.exp(interpolator(query)).reshape(query_t.shape)
                    )
    if not interpolated_powers:
        raise ValueError(f"no target-excluded joint patches in {line_path}")
    log_power = np.log(np.maximum(np.stack(interpolated_powers, axis=0), 1e-20))
    amplitude = np.sqrt(np.exp(np.median(log_power, axis=0)))
    power = np.square(amplitude)
    power = 0.5 * (power + power[::-1, ::-1])
    return _normalise_joint(np.sqrt(power)), len(interpolated_powers)


def fit_joint_pool(
    name: str,
    lines: tuple[str, ...],
    lines_dir: Path,
    segment_csv: Path,
) -> JointSpectrumFit:
    segments = read_segments(segment_csv, lines)
    line_amplitudes: dict[str, np.ndarray] = {}
    patch_counts: dict[str, int] = {}
    for line in lines:
        amplitude, count = fit_line_joint_spectrum(
            lines_dir / f"{line}.npz", segments[line]
        )
        line_amplitudes[line] = amplitude
        patch_counts[line] = count
    pooled = equal_line_joint_pool([line_amplitudes[line] for line in lines])
    return JointSpectrumFit(
        name,
        lines,
        COMMON_TEMPORAL_MHZ.copy(),
        COMMON_SPATIAL_CPM.copy(),
        pooled,
        line_amplitudes,
        patch_counts,
    )


def _joint_response(
    shape: tuple[int, int],
    dt_s: float,
    trace_spacing_m: float,
    fit: JointSpectrumFit,
) -> np.ndarray:
    temporal = np.fft.fftfreq(shape[0], d=dt_s) / 1e6
    spatial = np.fft.fftfreq(shape[1], d=trace_spacing_m)
    query_t, query_x = np.meshgrid(temporal, spatial, indexing="ij")
    interpolator = RegularGridInterpolator(
        (fit.temporal_mhz, fit.spatial_cpm),
        fit.amplitude,
        bounds_error=False,
        fill_value=0.0,
    )
    response = interpolator(
        np.column_stack((query_t.ravel(), query_x.ravel()))
    ).reshape(shape)
    response[0, 0] = 0.0
    response /= max(float(np.sqrt(np.mean(np.square(response)))), 1e-12)
    return response


def build_joint_basis(
    shape: tuple[int, int],
    dt_s: float,
    time_ns: np.ndarray,
    joint_fit: JointSpectrumFit,
    spatial_fit: SpatialFit,
    trace_spacing_m: float,
    seed: int,
    *,
    nonstationary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    response = _joint_response(shape, dt_s, trace_spacing_m, joint_fit)
    rng_field = np.random.default_rng(seed)
    diffuse = np.fft.ifft2(np.fft.fft2(rng_field.standard_normal(shape)) * response).real

    temporal_marginal = np.sqrt(np.mean(np.square(response), axis=1))
    spatial_marginal = np.sqrt(np.mean(np.square(response), axis=0))
    rng_modes = np.random.default_rng(seed + 1)
    low_rank = np.zeros(shape, dtype=np.float64)
    for _ in range(4):
        temporal = np.fft.ifft(
            np.fft.fft(rng_modes.standard_normal(shape[0])) * temporal_marginal
        ).real
        lateral = np.fft.ifft(
            np.fft.fft(rng_modes.standard_normal(shape[1])) * spatial_marginal
        ).real
        temporal = (temporal - np.mean(temporal)) / max(float(np.std(temporal)), 1e-12)
        lateral = (lateral - np.mean(lateral)) / max(float(np.std(lateral)), 1e-12)
        low_rank += temporal[:, None] * lateral[None, :]
    low_rank /= 4.0

    depth = np.clip((time_ns - 45.0) / 455.0, 0.0, 1.0)
    depth = 0.08 + 0.92 * np.power(depth, 0.85)
    combined = 0.82 * diffuse / max(float(np.std(diffuse)), 1e-12)
    combined += 0.18 * low_rank / max(float(np.std(low_rank)), 1e-12)

    rng_gain = np.random.default_rng(seed + 2)
    gain = gaussian_filter1d(rng_gain.standard_normal(shape[1]), sigma=4.5, mode="reflect")
    gain = (gain - np.mean(gain)) / max(float(np.std(gain)), 1e-12)
    envelope = np.ones(shape[1], dtype=np.float64)
    if nonstationary:
        rng_envelope = np.random.default_rng(seed + 3)
        sigma = max(1.0, spatial_fit.pooled_envelope_correlation_m / trace_spacing_m)
        basis = gaussian_filter1d(
            rng_envelope.standard_normal(shape[1]), sigma=sigma, mode="reflect"
        )
        basis = (basis - np.mean(basis)) / max(float(np.std(basis)), 1e-12)
        log_sigma = math.sqrt(math.log1p(min(spatial_fit.pooled_envelope_cv, 0.65) ** 2))
        envelope = np.exp(log_sigma * basis - 0.5 * log_sigma**2)
        envelope = np.clip(envelope, 0.45, 2.2)
        combined *= envelope[None, :]
    basis = combined * depth[:, None]
    basis = (basis - np.mean(basis)) / max(float(np.std(basis)), 1e-12)
    return basis, gain, response, envelope


def _joint_rgb(amplitude: np.ndarray) -> np.ndarray:
    db = 20.0 * np.log10(
        np.maximum(amplitude / max(float(np.max(amplitude)), 1e-12), 1e-3)
    )
    unit = np.clip((db + 45.0) / 45.0, 0.0, 1.0)
    red = np.clip(2.2 * unit, 0.0, 1.0)
    green = np.clip(2.2 * unit - 0.65, 0.0, 1.0)
    blue = np.clip(2.4 * unit - 1.45, 0.0, 1.0)
    return np.rint(255.0 * np.stack((red, green, blue), axis=-1)).astype(np.uint8)


def draw_joint_audit(fit: JointSpectrumFit, output_path: Path) -> None:
    items = [(line, fit.line_amplitudes[line]) for line in fit.lines]
    items.append(("equal-line pooled", fit.amplitude))
    width, height, margin, header = 360, 390, 18, 48
    columns = 2
    rows = math.ceil(len(items) / columns)
    canvas = Image.new(
        "RGB",
        (margin * 3 + width * columns, header + margin * (rows + 1) + height * rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 14),
        "FORMAL09B-2R1 joint temporal-frequency/spatial-frequency spectra",
        fill="black",
    )
    for index, (title, amplitude) in enumerate(items):
        row, column = divmod(index, columns)
        x = margin + column * (width + margin)
        y = header + row * (height + margin)
        image = Image.fromarray(_joint_rgb(amplitude)).resize(
            (width, height), Image.Resampling.BILINEAR
        )
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline="black")
        draw.rectangle((x, y, x + width - 1, y + 25), fill="white")
        draw.text((x + 7, y + 6), title, fill="black")
        draw.text((x + 7, y + height - 20), "x: -0.70..0.70 cycles/m", fill="white")
        draw.text((x + 190, y + height - 20), "y: -250..250 MHz", fill="white")
    canvas.save(output_path)


def _joint_payload(fit: JointSpectrumFit) -> dict:
    return {
        "name": fit.name,
        "lines": list(fit.lines),
        "temporal_mhz": fit.temporal_mhz.tolist(),
        "spatial_cpm": fit.spatial_cpm.tolist(),
        "pooled_amplitude": fit.amplitude.tolist(),
        "line_amplitude_summaries": {
            line: {
                "sha256_float64_le": hashlib.sha256(
                    np.asarray(values, dtype="<f8").tobytes()
                ).hexdigest(),
                "rms": float(np.sqrt(np.mean(np.square(values)))),
                "maximum": float(np.max(values)),
            }
            for line, values in fit.line_amplitudes.items()
        },
        "patch_counts": fit.patch_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-lines", type=Path, required=True)
    parser.add_argument("--reference-segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071612)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporal_fit = fit_spectrum(
        "paper_line9_holdout_fit",
        PAPER_FIT_LINES,
        args.measured_lines,
        args.reference_segments,
    )
    spatial_fit = fit_spatial_pool(
        "paper_line9_holdout_fit",
        PAPER_FIT_LINES,
        args.measured_lines,
        args.reference_segments,
    )
    joint_fit = fit_joint_pool(
        "paper_line9_holdout_fit",
        PAPER_FIT_LINES,
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
    separable_basis, separable_gain, _, _ = build_lateral_covariance_basis(
        raw.shape,
        dt_s,
        time_ns,
        temporal_fit,
        spatial_fit,
        trace_spacing_m,
        args.seed,
        nonstationary=True,
    )
    separable_values, separable_report = apply_realism(
        raw, time_ns, path_ns, separable_basis, separable_gain, TARGET_VARIANT
    )
    realized = {
        "09B1-paper": b1_values,
        "09B2-separable-rejected": separable_values,
    }
    metrics = {
        "09B1-paper": b1_report,
        "09B2-separable-rejected": separable_report,
    }
    response_summaries: dict[str, dict[str, float | list[int]]] = {}
    envelopes: dict[str, list[float]] = {}
    for name, nonstationary in (
        ("09B2R1-joint-stationary", False),
        ("09B2R1-joint-nonstationary", True),
    ):
        basis, gain, response, envelope = build_joint_basis(
            raw.shape,
            dt_s,
            time_ns,
            joint_fit,
            spatial_fit,
            trace_spacing_m,
            args.seed,
            nonstationary=nonstationary,
        )
        values, report = apply_realism(
            raw, time_ns, path_ns, basis, gain, TARGET_VARIANT
        )
        realized[name] = values
        metrics[name] = report
        response_summaries[name] = {
            "shape": list(response.shape),
            "rms": float(np.sqrt(np.mean(np.square(response)))),
            "minimum": float(np.min(response)),
            "maximum": float(np.max(response)),
        }
        envelopes[name] = envelope.tolist()

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
        "FORMAL09B-2R1 blind joint-2D spectrum checkpoint; common scales; no labels",
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
    blind_path = output / "FORMAL09B2R1_blind_joint_spectrum.png"
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
        "Multi-line references vs FORMAL09B-2R1 joint 2D spectrum; independent P99.5 scales",
        fill="black",
    )
    for index, (title, matrix) in enumerate(comparison_items):
        row, column = divmod(index, columns)
        x = margin + column * (width + margin)
        y = header + row * (height + margin)
        add_panel(canvas, matrix, title, (x, y, width, height), robust_limit(matrix))
    comparison_path = output / "FORMAL09B2R1_multiline_visual_comparison.png"
    canvas.save(comparison_path)

    joint_figure = output / "FORMAL09B2R1_joint_spectrum_audit.png"
    draw_joint_audit(joint_fit, joint_figure)
    contract_path = output / "FORMAL09B2R1_joint_spectrum.json"
    contract_path.write_text(
        json.dumps(
            {
                "contract_id": "FORMAL09B2R1_TARGET_EXCLUDED_JOINT_SPECTRUM_V1",
                "fit_lines": list(PAPER_FIT_LINES),
                "held_out_lines": ["Line9"],
                "validation_only_lines": ["Line6"],
                "fit_time_ns": [70.0, 260.0],
                "target_guard_ns": 42.0,
                "patch_shape": [96, 128],
                "axes": ["temporal_frequency_mhz", "spatial_frequency_cycles_per_m"],
                "acquisition_direction_symmetrised": True,
                "measured_patch_copying": False,
                "fit": _joint_payload(joint_fit),
                "sampled_response_summaries": response_summaries,
                "sampled_envelopes": envelopes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "contract_id": "FORMAL09B2R1_JOINT_2D_SPECTRUM_DEVELOPMENT_V1",
        "status": "visual_candidates_only",
        "formal_training_allowed": False,
        "causal_pair_complete": False,
        "gprmax_solver_run_performed": False,
        "base_case": scene["case_id"],
        "base_output": merged.resolve().relative_to(ROOT).as_posix(),
        "base_output_sha256": sha256(merged),
        "seed": args.seed,
        "synthetic_trace_spacing_m": trace_spacing_m,
        "single_factor_group": "joint temporal-frequency/spatial-frequency nuisance spectrum",
        "measured_trace_or_patch_copying": False,
        "target_corridor_excluded": True,
        "metrics": metrics,
        "blind_mapping": blind_map,
        "visual_outputs": [blind_path.name, comparison_path.name, joint_figure.name],
        "joint_spectrum_contract": contract_path.name,
        "next_gate": "blind human visual audit before any metadata conditioning",
    }
    (output / "formal09b2r1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
