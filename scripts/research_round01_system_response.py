#!/usr/bin/env python3
"""Round 01: fit a bounded fold-safe effective system response."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_realism_research import (  # noqa: E402
    FIT_LINES,
    HELDOUT_LINES,
    VALIDATION_LINES,
    aligned_packets,
    apply_zero_phase_response,
    bounded_system_response_db,
    display_process,
    measured_line_spectrum,
    packet_metrics,
    packet_spectrum,
    pool_spectra,
    valid_measured_traces,
)
from pgdacsnet.simulation_v2 import extract_visible_phase, resample_time_axis  # noqa: E402
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402


STRENGTHS = (0.0, 0.35, 0.65, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _seismic_rgb(values: np.ndarray, limit: float) -> np.ndarray:
    unit = np.clip(values / max(limit, np.finfo(np.float64).tiny), -1.0, 1.0)
    red = np.where(unit >= 0.0, 255, np.rint(255 * (1.0 + unit)))
    blue = np.where(unit <= 0.0, 255, np.rint(255 * (1.0 - unit)))
    green = np.rint(255 * (1.0 - np.abs(unit)))
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _panel(
    canvas: Image.Image,
    matrix: np.ndarray,
    title: str,
    box: tuple[int, int, int, int],
    limit: float,
) -> None:
    left, top, width, height = box
    image = Image.fromarray(_seismic_rgb(matrix, limit)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    canvas.paste(image, (left, top))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (left, top, left + width - 1, top + height - 1), outline="black", width=2
    )
    draw.rectangle((left, top, left + width - 1, top + 31), fill="white")
    draw.text((left + 8, top + 7), title, fill="black", font=_font(16, True))


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    if starts.size == 0:
        raise ValueError("no valid measured traces")
    lengths = ends - starts
    index = int(np.argmax(lengths))
    return int(starts[index]), int(ends[index])


def _measured_window(
    path: Path, trace_count: int, time_low_ns: float, time_high_ns: float
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        valid = valid_measured_traces(data)
        start, end = _longest_true_run(valid)
        if end - start < trace_count:
            raise ValueError(f"{path.stem} has fewer than {trace_count} contiguous traces")
        center = (start + end) // 2
        low = max(start, min(end - trace_count, center - trace_count // 2))
        high = low + trace_count
        time_ns = np.asarray(data["time_ns"], dtype=np.float64)
        raw = np.asarray(data["raw_amplitude"], dtype=np.float64)[:, low:high]
    rows = (time_ns >= time_low_ns) & (time_ns <= time_high_ns)
    return display_process(raw, time_ns)[rows]


def _load_simulation(
    run_dir: Path, source_case_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    dt_s, raw, _ = read_merged_bscan(run_dir / "full_scene_merged.out", component="Ez")
    sample_count = int(round(499.8 / 1.4)) + 1
    time_ns, raw = resample_time_axis(
        np.asarray(raw, dtype=np.float64),
        dt_s,
        time_window_ns=499.8,
        output_samples=sample_count,
    )
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    selected = np.asarray(run["selected_trace_indices_zero_based"], dtype=np.int64)
    reference = np.load(
        source_case_dir / "labels" / "source_referenced_arrival_time_ns.npy"
    ).astype(np.float64)[selected]
    suppressed = raw - np.median(raw, axis=1, keepdims=True)
    path_ns, _, _ = extract_visible_phase(
        suppressed,
        np.zeros_like(suppressed),
        time_ns,
        reference,
        search_half_width_ns=35.0,
        phase_half_width_ns=8.0,
        enforce_continuity=True,
        max_trace_step_ns=44.8,
        geometric_anchor_weight=2.0,
    )
    return time_ns, raw, path_ns, run


def _blind_panel(
    output: Path,
    candidates: dict[str, np.ndarray],
    time_ns: np.ndarray,
    seed: int,
    *,
    heading: str = "Round 01 blind system-response candidates; shared scales; no labels",
) -> dict[str, str]:
    rows = (time_ns >= 120.0) & (time_ns <= 500.0)
    raw = {name: values[rows] for name, values in candidates.items()}
    processed = {
        name: display_process(values, time_ns)[rows]
        for name, values in candidates.items()
    }
    raw_limit = float(np.quantile(np.abs(np.concatenate(list(raw.values()), axis=1)), 0.995))
    processed_limit = float(
        np.quantile(np.abs(np.concatenate(list(processed.values()), axis=1)), 0.995)
    )
    names = list(candidates)
    np.random.default_rng(seed).shuffle(names)
    mapping = {chr(65 + index): name for index, name in enumerate(names)}
    width, height, gap, top = 390, 455, 18, 72
    canvas = Image.new(
        "RGB", (gap * 5 + width * 4, top + gap * 3 + height * 2), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 17),
        heading,
        fill="black",
        font=_font(25, True),
    )
    for column, blind_id in enumerate("ABCD"):
        name = mapping[blind_id]
        left = gap + column * (width + gap)
        _panel(canvas, raw[name], f"{blind_id}: raw", (left, top, width, height), raw_limit)
        _panel(
            canvas,
            processed[name],
            f"{blind_id}: common suppressed + time^1.5",
            (left, top + height + gap, width, height),
            processed_limit,
        )
    canvas.save(output)
    return mapping


def _multiline_panel(
    output: Path,
    measured_root: Path,
    candidate: np.ndarray,
    time_ns: np.ndarray,
    lines: tuple[str, ...],
    *,
    candidate_label: str = "Round01 frozen candidate",
    heading: str = "Round 01 equal-trace morphology; independent P99.5 scales",
) -> None:
    rows = (time_ns >= 120.0) & (time_ns <= 500.0)
    items: list[tuple[str, np.ndarray]] = [
        (candidate_label, display_process(candidate, time_ns)[rows])
    ]
    for line in lines:
        items.append((line, _measured_window(measured_root / f"{line}.npz", candidate.shape[1], 120.0, 500.0)))
    columns = 3
    row_count = math.ceil(len(items) / columns)
    width, height, gap, top = 520, 430, 28, 95
    canvas = Image.new(
        "RGB",
        (gap * (columns + 1) + width * columns, top + gap * (row_count + 1) + height * row_count),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 18),
        heading,
        fill="black",
        font=_font(27, True),
    )
    draw.text(
        (gap, 56),
        "Fit: Line3/Line7/LineL1; validation: Line6; heldout appears only after freeze",
        fill="black",
        font=_font(17),
    )
    for index, (name, matrix) in enumerate(items):
        row, column = divmod(index, columns)
        left = gap + column * (width + gap)
        panel_top = top + row * (height + gap)
        limit = float(np.quantile(np.abs(matrix), 0.995))
        _panel(canvas, matrix, name, (left, panel_top, width, height), limit)
    canvas.save(output)


def _spectrum_plot(
    output: Path,
    frequency_hz: np.ndarray,
    response_db: np.ndarray,
    fit_log: np.ndarray,
    validation_log: np.ndarray,
    candidate_spectra: dict[str, np.ndarray],
) -> None:
    frequency_mhz = frequency_hz / 1e6
    canvas = Image.new("RGB", (1680, 650), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 18), "Round 01 effective system-response audit", fill="black", font=_font(29, True))

    def axes_box(left: int, title: str, y_label: str) -> tuple[int, int, int, int]:
        top, width, height = 100, 730, 470
        draw.rectangle((left, top, left + width, top + height), outline="black", width=2)
        draw.text((left, top - 34), title, fill="black", font=_font(20, True))
        draw.text((left + width // 2 - 60, top + height + 18), "Frequency (MHz)", fill="black", font=_font(16))
        draw.text((left + 8, top + 8), y_label, fill="black", font=_font(14))
        for value in (0, 50, 100, 150, 200):
            x = left + int(width * value / 220.0)
            draw.line((x, top, x, top + height), fill=(225, 225, 225), width=1)
            draw.text((x - 10, top + height + 3), str(value), fill="black", font=_font(13))
        return left, top, width, height

    def line_points(
        box: tuple[int, int, int, int],
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_low: float,
        y_high: float,
    ) -> list[tuple[int, int]]:
        left, top, width, height = box
        valid = (x_values >= 0.0) & (x_values <= 220.0) & np.isfinite(y_values)
        x = left + np.rint(width * x_values[valid] / 220.0).astype(int)
        y = top + height - np.rint(
            height * np.clip((y_values[valid] - y_low) / (y_high - y_low), 0.0, 1.0)
        ).astype(int)
        return list(zip(x.tolist(), y.tolist()))

    response_box = axes_box(70, "Bounded fold-fit response", "Gain (dB)")
    zero_y = response_box[1] + response_box[3] // 2
    draw.line((response_box[0], zero_y, response_box[0] + response_box[2], zero_y), fill=(170, 170, 170), width=1)
    draw.line(line_points(response_box, frequency_mhz, response_db, -6.5, 6.5), fill=(20, 20, 20), width=3)
    for value in (-6, -3, 0, 3, 6):
        y = response_box[1] + response_box[3] - int(response_box[3] * (value + 6.5) / 13.0)
        draw.text((response_box[0] - 34, y - 7), str(value), fill="black", font=_font(13))

    spectrum_box = axes_box(880, "Aligned target packet spectra", "Shape (dB)")
    colours = [(0, 90, 190), (210, 70, 20), (0, 135, 75), (145, 60, 170), (100, 100, 100), (210, 150, 0)]
    series = [("fit equal-line pool", fit_log), ("Line6 validation", validation_log), *candidate_spectra.items()]
    for index, (name, values) in enumerate(series):
        normalised = (values - np.max(values)) * (20.0 / np.log(10.0))
        colour = colours[index % len(colours)]
        draw.line(line_points(spectrum_box, frequency_mhz, normalised, -42.0, 2.0), fill=colour, width=3 if index < 2 else 2)
        legend_y = spectrum_box[1] + 15 + index * 25
        draw.line((spectrum_box[0] + 410, legend_y + 8, spectrum_box[0] + 450, legend_y + 8), fill=colour, width=3)
        draw.text((spectrum_box[0] + 460, legend_y), name, fill="black", font=_font(13))
    for value in (-40, -30, -20, -10, 0):
        y = spectrum_box[1] + spectrum_box[3] - int(spectrum_box[3] * (value + 42.0) / 44.0)
        draw.text((spectrum_box[0] - 40, y - 7), str(value), fill="black", font=_font(13))
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071601)
    parser.add_argument("--heldout-diagnostic", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    source_case_dir = args.source_case_dir.resolve()
    measured_root = args.measured_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    time_ns, raw, path_ns, run = _load_simulation(run_dir, source_case_dir)
    fit_by_line = {
        line: measured_line_spectrum(measured_root / f"{line}.npz")
        for line in FIT_LINES
    }
    validation_by_line = {
        line: measured_line_spectrum(measured_root / f"{line}.npz")
        for line in VALIDATION_LINES
    }
    fit = pool_spectra(list(fit_by_line.values()))
    validation = pool_spectra(list(validation_by_line.values()))
    suppressed = raw - np.median(raw, axis=1, keepdims=True)
    _, sim_packets = aligned_packets(suppressed, time_ns, path_ns)
    simulated_spectrum = packet_spectrum(sim_packets)
    response_db = bounded_system_response_db(simulated_spectrum, fit)

    candidates: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    spectra: dict[str, np.ndarray] = {}
    for strength in STRENGTHS:
        name = f"strength_{strength:.2f}"
        candidate = apply_zero_phase_response(
            raw,
            1.4,
            simulated_spectrum.frequency_hz,
            response_db,
            strength=strength,
        )
        candidates[name] = candidate
        fit_metrics = packet_metrics(
            candidate, time_ns, path_ns, spectrum_reference=fit
        )
        validation_metrics = packet_metrics(
            candidate, time_ns, path_ns, spectrum_reference=validation
        )
        score = (
            validation_metrics["shape_spectrum_rmse_db"]
            + 0.35 * fit_metrics["shape_spectrum_rmse_db"]
            + 25.0 * max(0.0, fit_metrics["target_dropout_fraction"] - 0.03)
        )
        metrics[name] = {
            **fit_metrics,
            "fit_shape_spectrum_rmse_db": fit_metrics["shape_spectrum_rmse_db"],
            "validation_shape_spectrum_rmse_db": validation_metrics[
                "shape_spectrum_rmse_db"
            ],
            "selection_score": float(score),
        }
        _, packets = aligned_packets(
            candidate - np.median(candidate, axis=1, keepdims=True), time_ns, path_ns
        )
        spectra[name] = packet_spectrum(packets).log_amplitude

    selected_name = min(metrics, key=lambda name: metrics[name]["selection_score"])
    selection_path = output / "round01_selection.json"
    if args.heldout_diagnostic:
        if not selection_path.is_file():
            raise FileNotFoundError(
                "heldout diagnostic requires an existing frozen round01_selection.json"
            )
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        if frozen["selected_candidate"] != selected_name:
            raise RuntimeError("recomputed candidate disagrees with frozen selection")
        _multiline_panel(
            output / "round01_frozen_plus_heldout.png",
            measured_root,
            candidates[selected_name],
            time_ns,
            FIT_LINES + VALIDATION_LINES + HELDOUT_LINES,
        )
        heldout = {
            line: measured_line_spectrum(measured_root / f"{line}.npz")
            for line in HELDOUT_LINES
        }
        frozen["status"] = "heldout_diagnostic_complete"
        frozen["heldout_lines_opened"] = list(HELDOUT_LINES)
        frozen["heldout_diagnostic"] = {
            "opened_after_freeze": True,
            "lines": list(HELDOUT_LINES),
            "candidate_shape_spectrum_rmse_db": {
                candidate_name: {
                    line: packet_metrics(
                        candidate,
                        time_ns,
                        path_ns,
                        spectrum_reference=estimate,
                    )["shape_spectrum_rmse_db"]
                    for line, estimate in heldout.items()
                }
                for candidate_name, candidate in candidates.items()
            },
            "visual": "round01_frozen_plus_heldout.png",
        }
        selection_path.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(frozen, ensure_ascii=False, indent=2))
        return 0

    blind_mapping = _blind_panel(
        output / "round01_blind_candidates.png", candidates, time_ns, args.seed
    )
    _multiline_panel(
        output / "round01_fold_only_multiline.png",
        measured_root,
        candidates[selected_name],
        time_ns,
        FIT_LINES + VALIDATION_LINES,
    )
    _spectrum_plot(
        output / "round01_spectrum_audit.png",
        simulated_spectrum.frequency_hz,
        response_db,
        fit.log_amplitude,
        validation.log_amplitude,
        spectra,
    )
    selection = {
        "round": 1,
        "factor": "bounded zero-phase effective system response",
        "status": "fold_selected_heldout_closed",
        "formal_training_allowed": False,
        "fit_lines": list(FIT_LINES),
        "validation_lines": list(VALIDATION_LINES),
        "heldout_lines_opened": [],
        "source_run": str(run_dir.relative_to(ROOT).as_posix()),
        "source_output_sha256": _sha256(run_dir / "full_scene_merged.out"),
        "selected_trace_indices_zero_based": run[
            "selected_trace_indices_zero_based"
        ],
        "response_contract": {
            "phase": "zero; measured phase is never imported",
            "maximum_absolute_gain_db": 6.0,
            "passband_mhz": [20.0, 180.0],
            "global_rms_preserved": True,
            "same_operator_required_for_future_full_control_pair": True,
        },
        "candidate_strengths": list(STRENGTHS),
        "metrics": metrics,
        "selected_candidate": selected_name,
        "blind_mapping": blind_mapping,
        "visuals": [
            "round01_blind_candidates.png",
            "round01_fold_only_multiline.png",
            "round01_spectrum_audit.png",
        ],
        "next_gate": "human blind visual audit before heldout diagnostic",
    }
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
