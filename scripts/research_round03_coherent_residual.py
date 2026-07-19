#!/usr/bin/env python3
"""Round 03: audit target-excluded low-rank coherent residual clutter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

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
    rms,
    smooth_unit_noise,
    valid_measured_traces,
)
from scripts.research_round01_system_response import (  # noqa: E402
    _blind_panel,
    _font,
    _load_simulation,
    _multiline_panel,
)


RANKS = (0, 1, 3, 6)
TARGET_EXCLUSION_HALF_WIDTH_NS = 35.0
SPATIAL_CORRELATION_M = (0.55, 1.4, 3.2, 0.9, 2.1, 4.8)


def _load_measured(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        valid = valid_measured_traces(data)
        raw = np.asarray(data["raw_amplitude"], dtype=np.float64)[:, valid]
        time_ns = np.asarray(data["time_ns"], dtype=np.float64)
        path_ns = np.asarray(
            data["v15_final_center_time_ns"], dtype=np.float64
        )[valid]
        distance = np.asarray(
            data["gnss_cumulative_distance_m"], dtype=np.float64
        )[valid]
    return time_ns, raw, path_ns, float(np.median(np.diff(distance)))


def _fill_target_corridor(
    residual: np.ndarray,
    time_ns: np.ndarray,
    path_ns: np.ndarray,
) -> np.ndarray:
    """Remove target support without copying its lateral path into the basis."""

    output = np.asarray(residual, dtype=np.float64).copy()
    target = (
        np.abs(time_ns[:, None] - path_ns[None, :])
        <= TARGET_EXCLUSION_HALF_WIDTH_NS
    )
    for row in range(output.shape[0]):
        available = ~target[row]
        fill = float(np.median(output[row, available])) if np.any(available) else 0.0
        output[row, ~available] = fill
    return output


def _line_modes(path: Path, output_time_ns: np.ndarray) -> list[dict[str, object]]:
    time_ns, raw, path_ns, spacing_m = _load_measured(path)
    scale = float(np.quantile(np.abs(raw), 0.995))
    residual = raw / max(scale, np.finfo(np.float64).tiny)
    residual -= np.median(residual, axis=1, keepdims=True)
    residual = _fill_target_corridor(residual, time_ns, path_ns)
    residual = gaussian_filter1d(
        residual,
        sigma=max(1.0, 0.75 / spacing_m),
        axis=1,
        mode="nearest",
    )
    rows = (time_ns >= 70.0) & (time_ns <= 500.0)
    working = residual[rows]
    working -= np.mean(working, axis=1, keepdims=True)
    tensor = torch.from_numpy(np.ascontiguousarray(working))
    u_tensor, singular_tensor, _ = torch.linalg.svd(tensor, full_matrices=False)
    u = u_tensor.cpu().numpy()
    singular = singular_tensor.cpu().numpy()
    total = max(float(np.sum(np.square(singular))), np.finfo(np.float64).tiny)
    modes: list[dict[str, object]] = []
    for index in range(min(4, singular.size)):
        atom = np.interp(
            output_time_ns,
            time_ns[rows],
            u[:, index],
            left=0.0,
            right=0.0,
        )
        if atom[np.argmax(np.abs(atom))] < 0.0:
            atom *= -1.0
        atom /= max(rms(atom), np.finfo(np.float64).tiny)
        modes.append(
            {
                "source_line": path.stem,
                "source_mode": index + 1,
                "relative_energy": float(np.square(singular[index]) / total),
                "atom": atom,
            }
        )
    return modes


def _coherent_energy_fractions(
    values: np.ndarray, time_ns: np.ndarray
) -> dict[str, float]:
    rows = (time_ns >= 120.0) & (time_ns <= 500.0)
    matrix = display_process(values, time_ns)[rows]
    matrix -= np.mean(matrix, axis=1, keepdims=True)
    singular = (
        torch.linalg.svdvals(torch.from_numpy(np.ascontiguousarray(matrix)))
        .cpu()
        .numpy()
    )
    energy = np.square(singular)
    cumulative = np.cumsum(energy) / max(
        float(np.sum(energy)), np.finfo(np.float64).tiny
    )
    result: dict[str, float] = {}
    for rank in (1, 3, 6, 12):
        index = min(rank, cumulative.size) - 1
        result[f"rank_{rank}_energy_fraction"] = float(cumulative[index])
    return result


def _measured_statistics(path: Path) -> dict[str, float]:
    time_ns, raw, path_ns, spacing_m = _load_measured(path)
    metrics = packet_metrics(raw, time_ns, path_ns)
    metrics.update(_coherent_energy_fractions(raw, time_ns))
    metrics["trace_count"] = int(raw.shape[1])
    metrics["trace_spacing_m"] = spacing_m
    return metrics


def _candidate_residual(
    modes: list[dict[str, object]],
    rank: int,
    trace_count: int,
    trace_spacing_m: float,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if rank == 0:
        return np.zeros((len(modes[0]["atom"]), trace_count)), []
    rng = np.random.default_rng(seed)
    components = []
    coefficients = []
    for index in range(rank):
        sigma = SPATIAL_CORRELATION_M[index] / trace_spacing_m
        coefficient = smooth_unit_noise(trace_count, sigma, rng)
        atom = np.asarray(modes[index]["atom"], dtype=np.float64)
        components.append(atom[:, None] * coefficient[None, :])
        coefficients.append(coefficient)
    residual = np.sum(components, axis=0) / math.sqrt(rank)
    residual /= max(rms(residual), np.finfo(np.float64).tiny)
    return residual, coefficients


def _calibrate_scale(
    parent: np.ndarray,
    residual: np.ndarray,
    time_ns: np.ndarray,
    path_ns: np.ndarray,
    target_ratio: float,
) -> tuple[float, np.ndarray]:
    if not np.any(residual):
        return 0.0, parent.copy()
    parent_scale = rms(parent)
    grid = parent_scale * np.geomspace(1e-5, 2.0, 160)
    ratios = np.asarray(
        [
            packet_metrics(parent + scale * residual, time_ns, path_ns)[
                "target_to_background"
            ]
            for scale in grid
        ]
    )
    index = int(np.argmin(np.abs(np.log(ratios / target_ratio))))
    return float(grid[index]), parent + grid[index] * residual


def _basis_plot(
    output: Path,
    modes: list[dict[str, object]],
    time_ns: np.ndarray,
    coefficients: list[np.ndarray],
) -> None:
    width, height = 1580, 850
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (35, 20),
        "Round 03 target-excluded temporal modes and new spatial coefficients",
        fill="black",
        font=_font(27, True),
    )
    colours = [(35, 100, 210), (215, 75, 35), (35, 145, 75), (145, 65, 180), (35, 155, 165), (190, 125, 25)]
    left, top, plot_width, plot_height = 80, 95, 1400, 290
    for panel in range(2):
        panel_top = top + panel * 390
        draw.rectangle(
            (left, panel_top, left + plot_width, panel_top + plot_height),
            outline="black",
            width=2,
        )
        draw.text(
            (left + 8, panel_top + 8),
            "temporal modes" if panel == 0 else "independent spatial coefficients",
            fill="black",
            font=_font(17, True),
        )
        for index in range(6):
            if panel == 0:
                values = np.asarray(modes[index]["atom"], dtype=np.float64)
                x = left + np.rint(
                    (time_ns - time_ns.min())
                    / max(float(np.ptp(time_ns)), 1e-30)
                    * plot_width
                ).astype(int)
                label = f"{modes[index]['source_line']} mode {modes[index]['source_mode']}"
            else:
                values = coefficients[index]
                x = left + np.rint(np.linspace(0, plot_width, values.size)).astype(int)
                label = f"coef {index + 1}, L={SPATIAL_CORRELATION_M[index]:.2f} m"
            maximum = max(float(np.max(np.abs(values))), 1e-30)
            y = panel_top + plot_height // 2 - np.rint(
                values / maximum * (0.40 * plot_height)
            ).astype(int)
            colour = colours[index]
            draw.line(list(zip(x.tolist(), y.tolist())), fill=colour, width=2)
            legend_y = panel_top + 12 + index * 31
            draw.line(
                (left + 1020, legend_y + 9, left + 1060, legend_y + 9),
                fill=colour,
                width=3,
            )
            draw.text(
                (left + 1070, legend_y), label, fill="black", font=_font(13)
            )
    canvas.save(output)


def _load_parent(
    run_dir: Path,
    source_case_dir: Path,
    measured_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    time_ns, raw, path_ns, _ = _load_simulation(run_dir, source_case_dir)
    fit_spectrum = pool_spectra(
        [measured_line_spectrum(measured_root / f"{line}.npz") for line in FIT_LINES]
    )
    _, packets = aligned_packets(
        raw - np.median(raw, axis=1, keepdims=True), time_ns, path_ns
    )
    simulated_spectrum = packet_spectrum(packets)
    response_db = bounded_system_response_db(simulated_spectrum, fit_spectrum)
    parent = apply_zero_phase_response(
        raw,
        1.4,
        simulated_spectrum.frequency_hz,
        response_db,
        strength=1.0,
    )
    return time_ns, parent, path_ns, fit_spectrum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-root", type=Path, required=True)
    parser.add_argument("--round01-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071603)
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
        raise RuntimeError("Round 03 expects the frozen Round 01 response")

    time_ns, parent, path_ns, fit_spectrum = _load_parent(
        args.run_dir.resolve(), args.source_case_dir.resolve(), measured_root
    )
    fit_stats = {
        line: _measured_statistics(measured_root / f"{line}.npz")
        for line in FIT_LINES
    }
    validation_stats = {
        line: _measured_statistics(measured_root / f"{line}.npz")
        for line in VALIDATION_LINES
    }
    fit_ratio = float(
        np.median([stats["target_to_background"] for stats in fit_stats.values()])
    )
    validation_reference = validation_stats[VALIDATION_LINES[0]]

    modes = []
    for line in FIT_LINES:
        modes.extend(_line_modes(measured_root / f"{line}.npz", time_ns))
    modes.sort(key=lambda item: float(item["relative_energy"]), reverse=True)
    if len(modes) < max(RANKS):
        raise RuntimeError("insufficient target-excluded coherent modes")

    candidates: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    scales: dict[str, float] = {}
    rank6_coefficients: list[np.ndarray] = []
    for rank in RANKS:
        name = f"rank_{rank}"
        residual, coefficients = _candidate_residual(
            modes,
            rank,
            parent.shape[1],
            0.09,
            args.seed,
        )
        scale, candidate = _calibrate_scale(
            parent, residual, time_ns, path_ns, fit_ratio
        )
        candidate_metrics = packet_metrics(
            candidate,
            time_ns,
            path_ns,
            spectrum_reference=fit_spectrum,
        )
        candidate_metrics.update(_coherent_energy_fractions(candidate, time_ns))
        low_rank_error = sum(
            abs(
                candidate_metrics[f"rank_{value}_energy_fraction"]
                - validation_reference[f"rank_{value}_energy_fraction"]
            )
            for value in (1, 3, 6, 12)
        )
        candidate_metrics["selection_score"] = float(
            abs(
                np.log(
                    candidate_metrics["target_to_background"]
                    / validation_reference["target_to_background"]
                )
            )
            + 2.0 * low_rank_error
            + 20.0
            * max(0.0, candidate_metrics["target_dropout_fraction"] - 0.03)
        )
        candidate_metrics["added_residual_scale"] = scale
        candidates[name] = candidate
        metrics[name] = candidate_metrics
        scales[name] = scale
        if rank == 6:
            rank6_coefficients = coefficients

    selected_name = min(metrics, key=lambda name: metrics[name]["selection_score"])
    selection_path = output / "round03_selection.json"
    if args.heldout_diagnostic:
        if not selection_path.is_file():
            raise FileNotFoundError("heldout diagnostic requires a frozen selection")
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        if frozen["selected_candidate"] != selected_name:
            raise RuntimeError("recomputed selection differs from frozen candidate")
        _multiline_panel(
            output / "round03_frozen_plus_heldout.png",
            measured_root,
            candidates[selected_name],
            time_ns,
            FIT_LINES + VALIDATION_LINES + HELDOUT_LINES,
            candidate_label="Round03 frozen coherent-residual candidate",
            heading="Round 03 frozen candidate plus heldout diagnosis",
        )
        heldout_stats = {
            line: _measured_statistics(measured_root / f"{line}.npz")
            for line in HELDOUT_LINES
        }
        frozen["status"] = "heldout_diagnostic_complete"
        frozen["heldout_lines_opened"] = list(HELDOUT_LINES)
        frozen["heldout_diagnostic"] = {
            "opened_after_freeze": True,
            "line_statistics": heldout_stats,
            "visual": "round03_frozen_plus_heldout.png",
        }
        selection_path.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(frozen, ensure_ascii=False, indent=2))
        return 0

    mapping = _blind_panel(
        output / "round03_blind_candidates.png",
        candidates,
        time_ns,
        args.seed,
        heading="Round 03 blind target-excluded coherent-residual ranks; no labels",
    )
    _multiline_panel(
        output / "round03_fold_only_multiline.png",
        measured_root,
        candidates[selected_name],
        time_ns,
        FIT_LINES + VALIDATION_LINES,
        candidate_label="Round03 fold-selected coherent residual",
        heading="Round 03 equal-trace coherent-residual audit",
    )
    _basis_plot(
        output / "round03_basis_audit.png",
        modes,
        time_ns,
        rank6_coefficients,
    )
    selection = {
        "round": 3,
        "factor": "target-excluded low-rank coherent residual clutter",
        "status": "fold_selected_heldout_closed",
        "formal_training_allowed": False,
        "fit_lines": list(FIT_LINES),
        "validation_lines": list(VALIDATION_LINES),
        "heldout_lines_opened": [],
        "round01_parent": round01["selected_candidate"],
        "target_exclusion_half_width_ns": TARGET_EXCLUSION_HALF_WIDTH_NS,
        "spatial_correlation_m": list(SPATIAL_CORRELATION_M),
        "fit_target_to_background": fit_ratio,
        "fit_line_statistics": fit_stats,
        "validation_line_statistics": validation_stats,
        "mode_provenance": [
            {
                "source_line": item["source_line"],
                "source_mode": item["source_mode"],
                "relative_energy": item["relative_energy"],
            }
            for item in modes[:6]
        ],
        "candidate_ranks": list(RANKS),
        "candidate_scales": scales,
        "metrics": metrics,
        "selected_candidate": selected_name,
        "blind_mapping": mapping,
        "visuals": [
            "round03_blind_candidates.png",
            "round03_fold_only_multiline.png",
            "round03_basis_audit.png",
        ],
        "leakage_guard": (
            "fit-line target corridors were removed before temporal SVD; only "
            "temporal atoms are reused and all spatial coefficients are newly sampled"
        ),
        "next_gate": "human blind visual audit before heldout diagnostic",
    }
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
