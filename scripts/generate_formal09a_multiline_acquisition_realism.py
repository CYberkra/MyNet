#!/usr/bin/env python3
"""Generate deterministic FORMAL06C acquisition-realism visual candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import extract_visible_phase  # noqa: E402
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    target_to_background: float
    gain_jitter_fraction: float


VARIANTS = (
    Variant("mild", 8.0, 0.06),
    Variant("balanced", 4.5, 0.12),
    Variant("strong", 2.5, 0.18),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0


def window_values(
    values: np.ndarray,
    time_ns: np.ndarray,
    centers_ns: np.ndarray,
    low_ns: float,
    high_ns: float,
) -> np.ndarray:
    return np.concatenate(
        [
            values[
                (time_ns >= center + low_ns) & (time_ns <= center + high_ns),
                trace,
            ]
            for trace, center in enumerate(centers_ns)
        ]
    )


def target_background_ratio(
    values: np.ndarray, time_ns: np.ndarray, path_ns: np.ndarray
) -> float:
    suppressed = values - np.median(values, axis=1, keepdims=True)
    target = window_values(suppressed, time_ns, path_ns, -14.0, 14.0)
    background = np.concatenate(
        (
            window_values(suppressed, time_ns, path_ns, -98.0, -42.0),
            window_values(suppressed, time_ns, path_ns, 42.0, 98.0),
        )
    )
    return rms(target) / max(rms(background), np.finfo(np.float64).tiny)


def morphology_metrics(
    values: np.ndarray, time_ns: np.ndarray, path_ns: np.ndarray
) -> dict[str, float]:
    suppressed = values - np.median(values, axis=1, keepdims=True)
    relative = np.arange(-56.0, 56.0 + 0.7, 1.4)
    aligned = np.column_stack(
        [
            np.interp(center + relative, time_ns, suppressed[:, trace])
            for trace, center in enumerate(path_ns)
        ]
    )
    template = np.median(aligned, axis=1)
    envelope = np.abs(hilbert(aligned, axis=0))
    amplitudes = np.max(envelope[np.abs(relative) <= 14.0], axis=0)
    correlations = []
    centered_template = template - np.mean(template)
    for trace in range(aligned.shape[1]):
        centered = aligned[:, trace] - np.mean(aligned[:, trace])
        denominator = np.sqrt(
            np.sum(np.square(centered)) * np.sum(np.square(centered_template))
        )
        correlations.append(
            float(np.sum(centered * centered_template) / denominator)
            if denominator > 0
            else 0.0
        )
    tapered = centered_template * np.hanning(centered_template.size)
    spectrum = np.abs(np.fft.rfft(tapered)) ** 2
    frequency = np.fft.rfftfreq(tapered.size, d=1.4e-9)
    positive = frequency > 0
    return {
        "target_to_background": target_background_ratio(values, time_ns, path_ns),
        "target_envelope_cv": float(
            np.std(amplitudes) / max(np.mean(amplitudes), np.finfo(np.float64).tiny)
        ),
        "target_dropout_fraction": float(
            np.mean(amplitudes < 0.25 * np.median(amplitudes))
        ),
        "aligned_template_correlation_median": float(np.median(correlations)),
        "aligned_peak_frequency_mhz": float(
            frequency[positive][np.argmax(spectrum[positive])] / 1e6
        ),
        "aligned_spectral_centroid_mhz": float(
            np.sum(frequency[positive] * spectrum[positive])
            / max(np.sum(spectrum[positive]), np.finfo(np.float64).tiny)
            / 1e6
        ),
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    return centered / max(float(np.std(centered)), np.finfo(np.float64).tiny)


def build_realism_basis(
    shape: tuple[int, int], dt_s: float, time_ns: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, traces = shape
    frequency = np.fft.rfftfreq(rows, d=dt_s)
    band = np.exp(-0.5 * np.square((frequency - 82e6) / 42e6))
    band[0] = 0.0

    white = rng.standard_normal(shape)
    diffuse = np.fft.irfft(np.fft.rfft(white, axis=0) * band[:, None], n=rows, axis=0)
    diffuse = gaussian_filter1d(diffuse, sigma=0.75, axis=1, mode="reflect")

    low_rank = np.zeros(shape, dtype=np.float64)
    for _ in range(4):
        temporal = rng.standard_normal(rows)
        temporal = np.fft.irfft(np.fft.rfft(temporal) * band, n=rows)
        lateral = gaussian_filter1d(
            rng.standard_normal(traces), sigma=4.0, mode="reflect"
        )
        low_rank += _standardize(temporal)[:, None] * _standardize(lateral)[None, :]
    low_rank /= 4.0

    depth = np.clip((time_ns - 45.0) / 455.0, 0.0, 1.0)
    depth = 0.08 + 0.92 * np.power(depth, 0.85)
    basis = _standardize((0.72 * _standardize(diffuse) + 0.28 * _standardize(low_rank)) * depth[:, None])

    gain = gaussian_filter1d(rng.standard_normal(traces), sigma=4.5, mode="reflect")
    gain = _standardize(gain)
    return basis, gain


def apply_realism(
    raw: np.ndarray,
    time_ns: np.ndarray,
    path_ns: np.ndarray,
    basis: np.ndarray,
    gain_basis: np.ndarray,
    variant: Variant,
    calibration_end_ns: float = 500.0,
) -> tuple[np.ndarray, dict[str, float]]:
    gain = np.clip(1.0 + variant.gain_jitter_fraction * gain_basis, 0.65, 1.35)
    gained = raw * gain[None, :]

    calibration = time_ns <= calibration_end_ns
    calibration_time = time_ns[calibration]
    low, high = 0.0, max(rms(gained), np.finfo(np.float64).tiny)
    while target_background_ratio(
        gained[calibration] + high * basis[calibration], calibration_time, path_ns
    ) > variant.target_to_background:
        high *= 2.0
        if high > 1e6 * max(rms(gained), 1e-30):
            raise RuntimeError("failed to bracket acquisition-realism noise scale")
    for _ in range(48):
        middle = 0.5 * (low + high)
        ratio = target_background_ratio(
            gained[calibration] + middle * basis[calibration], calibration_time, path_ns
        )
        if ratio > variant.target_to_background:
            low = middle
        else:
            high = middle
    scale = 0.5 * (low + high)
    realized = gained + scale * basis
    return realized, {
        "noise_scale": float(scale),
        "gain_min": float(np.min(gain)),
        "gain_median": float(np.median(gain)),
        "gain_max": float(np.max(gain)),
        "achieved_target_to_background": target_background_ratio(
            realized[calibration], calibration_time, path_ns
        ),
    }


def process(values: np.ndarray, time_ns: np.ndarray, power: float = 1.5) -> np.ndarray:
    suppressed = values - np.median(values, axis=1, keepdims=True)
    gain = np.power(np.clip(time_ns / 500.0, 0.02, 1.0), power)
    return suppressed * gain[:, None]


def robust_limit(values: np.ndarray, quantile: float = 0.995) -> float:
    return max(float(np.quantile(np.abs(values), quantile)), np.finfo(np.float64).eps)


def seismic_rgb(values: np.ndarray, limit: float) -> np.ndarray:
    unit = np.clip(values / limit, -1.0, 1.0)
    red = np.where(unit >= 0.0, 255, np.rint(255 * (1.0 + unit)))
    blue = np.where(unit <= 0.0, 255, np.rint(255 * (1.0 - unit)))
    green = np.rint(255 * (1.0 - np.abs(unit)))
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def add_panel(
    canvas: Image.Image,
    matrix: np.ndarray,
    title: str,
    box: tuple[int, int, int, int],
    limit: float,
) -> None:
    x, y, width, height = box
    image = Image.fromarray(seismic_rgb(matrix, limit)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + width - 1, y + height - 1), outline="black")
    draw.rectangle((x, y, x + width - 1, y + 25), fill="white")
    draw.text((x + 7, y + 6), title, fill="black")


def representative_crop(
    line_path: Path,
    start: int,
    end: int,
    span_m: float,
    low_ns: float,
    high_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(line_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_amplitude"], dtype=np.float64)
        time_ns = np.asarray(data["time_ns"], dtype=np.float64)
        distance = np.asarray(data["gnss_cumulative_distance_m"], dtype=np.float64)
    middle = (start + end) // 2
    target_low = distance[middle] - span_m / 2.0
    target_high = distance[middle] + span_m / 2.0
    columns = np.flatnonzero(
        (distance >= target_low)
        & (distance <= target_high)
        & (np.arange(distance.size) >= start)
        & (np.arange(distance.size) <= end)
    )
    rows = (time_ns >= low_ns) & (time_ns <= high_ns)
    return process(raw[:, columns], time_ns)[rows], time_ns[rows]


def read_reference_segments(path: Path) -> dict[str, tuple[int, int]]:
    longest: dict[str, tuple[int, int]] = {}
    lengths: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["purpose"] != "interface_morphology":
                continue
            count = int(row["trace_count"])
            if count > lengths.get(row["line"], -1):
                longest[row["line"]] = (int(row["trace_start"]), int(row["trace_end"]))
                lengths[row["line"]] = count
    return longest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-lines", type=Path, required=True)
    parser.add_argument("--reference-segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071609)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    merged = args.release_dir / "full_scene_merged.out"
    run_path = args.release_dir / "run_manifest.json"
    scene_path = args.release_dir.parents[1] / "source" / "scene_manifest.json"
    dt_s, raw, _ = read_merged_bscan(merged, component="Ez")
    raw = np.asarray(raw, dtype=np.float64)
    time_ns = np.arange(raw.shape[0], dtype=np.float64) * dt_s * 1e9
    run = json.loads(run_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    selected = np.asarray(run["selected_trace_indices_zero_based"], dtype=np.int64)
    reference = np.load(
        args.source_case_dir / "labels" / "source_referenced_arrival_time_ns.npy"
    ).astype(np.float64)[selected]
    protected = time_ns <= 500.0
    protected_raw = raw[protected]
    protected_time = time_ns[protected]
    baseline_suppressed = protected_raw - np.median(protected_raw, axis=1, keepdims=True)
    path_ns, _, _ = extract_visible_phase(
        baseline_suppressed,
        np.zeros_like(baseline_suppressed),
        protected_time,
        reference,
        search_half_width_ns=35.0,
        phase_half_width_ns=8.0,
        enforce_continuity=True,
        max_trace_step_ns=44.8,
        geometric_anchor_weight=2.0,
    )

    basis, gain_basis = build_realism_basis(raw.shape, dt_s, time_ns, args.seed)
    realized: dict[str, np.ndarray] = {"baseline": raw}
    metrics: dict[str, dict[str, float]] = {
        "baseline": morphology_metrics(raw[protected], protected_time, path_ns)
    }
    for variant in VARIANTS:
        values, report = apply_realism(raw, time_ns, path_ns, basis, gain_basis, variant)
        realized[variant.name] = values
        metrics[variant.name] = {
            **report,
            **morphology_metrics(values[protected], protected_time, path_ns),
        }

    crop_rows = (time_ns >= 250.0) & (time_ns <= 500.0)
    processed = {name: process(values, time_ns)[crop_rows] for name, values in realized.items()}
    raw_crop = {name: values[crop_rows] for name, values in realized.items()}
    raw_limit = robust_limit(np.concatenate(list(raw_crop.values()), axis=1))
    processed_limit = robust_limit(np.concatenate(list(processed.values()), axis=1))

    blind_order = list(realized)
    np.random.default_rng(args.seed + 1).shuffle(blind_order)
    blind_map = {chr(65 + index): name for index, name in enumerate(blind_order)}
    width, height, margin, header = 360, 430, 18, 48
    canvas = Image.new("RGB", (margin * 5 + width * 4, header + margin * 3 + height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), "FORMAL09A blind acquisition-realism checkpoint; common scales; no labels", fill="black")
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
    blind_path = output / "FORMAL09A_blind_variants.png"
    canvas.save(blind_path)

    longest = read_reference_segments(args.reference_segments)
    line_order = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
    span_m = float(selected[-1] - selected[0]) * float(scene["grid"]["trace_spacing_m"])
    measured: dict[str, np.ndarray] = {}
    measured_metrics: dict[str, dict[str, float]] = {}
    for line in line_order:
        start, end = longest[line]
        measured[line], _ = representative_crop(
            args.measured_lines / f"{line}.npz", start, end, span_m, 250.0, 500.0
        )
        with np.load(args.measured_lines / f"{line}.npz", allow_pickle=False) as data:
            measured_raw = np.asarray(data["raw_amplitude"], dtype=np.float64)
            measured_time = np.asarray(data["time_ns"], dtype=np.float64)
            measured_path = np.asarray(data["v15_final_center_time_ns"], dtype=np.float64)
            valid = (
                (np.asarray(data["status_code"]) == 1)
                & (np.asarray(data["label_weight"]) > 0)
                & ~np.asarray(data["v15_final_ignore_trace"], dtype=bool)
                & ~np.asarray(
                    data["flight_height_outside_planned_2_20_m"], dtype=bool
                )
            )
        measured_metrics[line] = morphology_metrics(
            measured_raw[:, valid], measured_time, measured_path[valid]
        )
    comparison_items = [(line, measured[line]) for line in line_order]
    comparison_items.append(("FORMAL06C", processed["baseline"]))
    for variant in VARIANTS:
        comparison_items.append((f"09A-{variant.name}", processed[variant.name]))
    width, height, margin, header = 330, 390, 16, 48
    columns = 4
    rows = math.ceil(len(comparison_items) / columns)
    canvas = Image.new(
        "RGB",
        (margin * (columns + 1) + width * columns, header + margin * (rows + 1) + height * rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), "Multi-line measured reference pool vs FORMAL06C/09A; independent P99.5 scales", fill="black")
    for index, (title, matrix) in enumerate(comparison_items):
        row, column = divmod(index, columns)
        x = margin + column * (width + margin)
        y = header + row * (height + margin)
        add_panel(canvas, matrix, title, (x, y, width, height), robust_limit(matrix))
    multi_path = output / "FORMAL09A_multiline_visual_comparison.png"
    canvas.save(multi_path)

    manifest = {
        "contract_id": "FORMAL09A_MULTILINE_ACQUISITION_REALISM_DEVELOPMENT_V1",
        "status": "full_scene_visual_candidates_only",
        "formal_training_allowed": False,
        "causal_pair_complete": False,
        "base_case": scene["case_id"],
        "base_output": merged.resolve().relative_to(ROOT).as_posix(),
        "base_output_sha256": sha256(merged),
        "seed": args.seed,
        "line9_conditioned": True,
        "synthesis_reads_measured_arrays": False,
        "measured_arrays_used_for_visual_comparison_only": True,
        "shared_pair_policy": "future full/control transformation must share the same basis and gain realization",
        "components": [
            "smooth trace gain jitter",
            "band-limited laterally correlated additive noise",
            "low-rank coherent acquisition residue",
        ],
        "forbidden_components": [
            "copied measured arrays",
            "label-conditioned enhancement",
            "point targets",
            "synthetic hyperbola injection",
            "hard dropout",
        ],
        "variants": [asdict(variant) for variant in VARIANTS],
        "metrics": metrics,
        "measured_reference_metrics": measured_metrics,
        "blind_mapping": blind_map,
        "visual_outputs": [blind_path.name, multi_path.name],
        "next_gate": "blind human visual audit before any matched-control or solver run",
    }
    (output / "formal09a_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
