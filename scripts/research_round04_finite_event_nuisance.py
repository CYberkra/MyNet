#!/usr/bin/env python3
"""Round 04: test target-excluded non-Gaussian finite-event nuisance.

This is deliberately a measurement-domain prototype, not a gprMax scene and
not a training export.  It samples distributions measured only on the frozen
fit fold, never waveform patches, coordinates, or target-path values.  Its
purpose is to decide whether the next expensive physical FDTD factor should
model finite mid-cover events, rather than to manufacture a realistic image by
post-processing.
"""

from __future__ import annotations

import argparse
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
from scripts.generate_formal09c_sparse_event_field import (  # noqa: E402
    EventFit,
    fit_event_contract,
)
from scripts.research_round01_system_response import (  # noqa: E402
    _blind_panel,
    _font,
    _load_simulation,
    _multiline_panel,
)


TARGET_GUARD_NS = 42.0
DEPTH_WINDOW_NS = (250.0, 500.0)


@dataclass(frozen=True)
class Variant:
    """Declared finite-event density and amplitude budget."""

    name: str
    density_multiplier: float
    amplitude_multiplier: float


VARIANTS = (
    Variant("none", 0.0, 0.0),
    Variant("sparse", 0.35, 0.55),
    Variant("moderate", 0.65, 0.75),
    Variant("dense", 1.00, 0.95),
)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def _event_wavelet(dt_ns: float, *, center_mhz: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact, zero-mean multi-cycle packet without copied waveforms."""

    support_ns = 58.0
    relative_ns = np.arange(-support_ns, support_ns + dt_ns / 2.0, dt_ns)
    envelope = np.exp(-0.5 * np.square(relative_ns / 14.0))
    packet = envelope * np.cos(2.0 * math.pi * center_mhz * 1e-3 * relative_ns)
    packet -= float(np.mean(packet))
    packet /= max(_rms(packet), np.finfo(np.float64).tiny)
    return relative_ns, packet


def _sample_triangular(rng: np.random.Generator, q: tuple[float, float, float]) -> float:
    low, middle, high = (float(value) for value in q)
    if not np.isfinite([low, middle, high]).all() or high <= low:
        return middle
    return float(rng.triangular(low, min(max(middle, low), high), high))


def _event_field(
    shape: tuple[int, int],
    time_ns: np.ndarray,
    target_ns: np.ndarray,
    spacing_m: float,
    fit: EventFit,
    variant: Variant,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Sample finite, tapered, weakly curved nuisance events.

    A support must contain at least six native traces and stay outside the
    target corridor.  That avoids the sparse-trace interpolation artefact that
    invalidated the original FORMAL09C image-space prototype.
    """

    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=np.float64)
    x_m = np.arange(shape[1], dtype=np.float64) * spacing_m
    span_m = max(float(np.ptp(x_m)), spacing_m)
    expected = fit.pooled_events_per_25m * span_m / 25.0 * variant.density_multiplier
    wanted = int(round(expected))
    if wanted <= 0:
        return field, []
    relative_ns, base_wavelet = _event_wavelet(float(np.median(np.diff(time_ns))))
    records: list[dict[str, float]] = []
    attempts = 0
    while len(records) < wanted and attempts < 120 * wanted:
        attempts += 1
        length_m = float(np.clip(_sample_triangular(rng, fit.pooled_quantiles["length_m"]), 1.0, min(8.0, span_m * 0.8)))
        center_x = float(rng.uniform(0.12 * span_m, 0.88 * span_m))
        support = np.flatnonzero(np.abs(x_m - center_x) <= length_m / 2.0)
        if support.size < 6:
            continue
        slope = float(np.clip(_sample_triangular(rng, fit.pooled_quantiles["slope_ns_per_m"]), -4.5, 4.5))
        # Curvature is intentionally independently bounded: noisy short
        # components cannot support measured curvature estimates reliably.
        curvature = float(np.clip(rng.normal(0.0, 0.025), -0.065, 0.065))
        center_time = float(np.clip(_sample_triangular(rng, fit.pooled_quantiles["center_time_ns"]), *DEPTH_WINDOW_NS))
        local_x = x_m[support] - center_x
        path = center_time + slope * local_x + curvature * np.square(local_x)
        if np.any((path < DEPTH_WINDOW_NS[0]) | (path > DEPTH_WINDOW_NS[1])):
            continue
        overlap = float(np.mean(np.abs(path - target_ns[support]) < TARGET_GUARD_NS))
        if overlap > 0.0:
            continue
        taper = np.hanning(support.size + 2)[1:-1]
        phase = float(rng.uniform(-math.pi, math.pi))
        # Hilbert-like phase rotation, generated parametrically rather than by
        # pulling a measured trace patch.
        shifted_wavelet = np.cos(phase) * base_wavelet + np.sin(phase) * np.gradient(base_wavelet)
        amplitude = float(
            np.clip(
                _sample_triangular(rng, fit.pooled_quantiles["amplitude_p99_fraction"])
                * variant.amplitude_multiplier,
                0.02,
                0.48,
            )
        )
        for local_index, trace_index in enumerate(support):
            wavelet = np.interp(
                time_ns - path[local_index], relative_ns, shifted_wavelet, left=0.0, right=0.0
            )
            field[:, trace_index] += amplitude * taper[local_index] * wavelet
        records.append(
            {
                "center_x_m": center_x,
                "center_time_ns": center_time,
                "length_m": length_m,
                "slope_ns_per_m": slope,
                "curvature_ns_per_m2": curvature,
                "amplitude_weight": amplitude,
                "phase_rad": phase,
                "target_overlap_fraction": overlap,
                "support_trace_count": int(support.size),
            }
        )
    if len(records) != wanted:
        raise RuntimeError(f"only generated {len(records)} of {wanted} finite events")
    return field, records


def _calibrate(parent: np.ndarray, event: np.ndarray, time_ns: np.ndarray, path_ns: np.ndarray, target_ratio: float) -> tuple[float, np.ndarray]:
    if not np.any(event):
        return 0.0, parent.copy()
    base = _rms(parent)
    grid = base * np.geomspace(1e-5, 1.4, 160)
    ratios = np.asarray(
        [packet_metrics(parent + factor * event, time_ns, path_ns)["target_to_background"] for factor in grid]
    )
    choice = int(np.argmin(np.abs(np.log(np.maximum(ratios, 1e-12) / target_ratio))))
    return float(grid[choice]), parent + grid[choice] * event


def _native_event_panel(
    output: Path,
    fields: dict[str, np.ndarray],
    time_ns: np.ndarray,
    records: dict[str, list[dict[str, float]]],
) -> None:
    rows = (time_ns >= DEPTH_WINDOW_NS[0]) & (time_ns <= DEPTH_WINDOW_NS[1])
    width, height, margin, header = 430, 410, 22, 74
    canvas = Image.new("RGB", (margin * 5 + width * 4, header + margin + height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Round 04 event-only fields at native 0.09 m trace spacing", fill="black", font=_font(25, True))
    combined = np.concatenate([values[rows] for values in fields.values()], axis=1)
    limit = max(float(np.quantile(np.abs(combined), 0.995)), 1e-12)
    for index, variant in enumerate(VARIANTS):
        left = margin + index * (width + margin)
        values = fields[variant.name][rows]
        normalized = np.clip(values / limit, -1.0, 1.0)
        image = np.rint((normalized + 1.0) * 127.5).astype(np.uint8)
        panel = Image.fromarray(image, mode="L").resize((width, height), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(panel, (left, header))
        draw.rectangle((left, header, left + width, header + height), outline="black", width=2)
        draw.text((left + 7, header + 7), f"{variant.name}: {len(records[variant.name])} events", fill="white", font=_font(16, True))
    canvas.save(output)


def _append_ledger(root: Path, payload: dict[str, object]) -> None:
    ledger_path = root / "research_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    rounds = [entry for entry in ledger["rounds"] if int(entry.get("round", -1)) != 4]
    rounds.append(payload)
    rounds.sort(key=lambda entry: int(entry["round"]))
    ledger["rounds"] = rounds
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _measured_target_ratio(path: Path) -> float:
    """Read one measured line without turning weak/ignored traces into truth."""

    with np.load(path, allow_pickle=False) as data:
        valid = valid_measured_traces(data)
        return float(
            packet_metrics(
                np.asarray(data["raw_amplitude"], dtype=np.float64)[:, valid],
                np.asarray(data["time_ns"], dtype=np.float64),
                np.asarray(data["v15_final_center_time_ns"], dtype=np.float64)[valid],
            )["target_to_background"]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--measured-root", type=Path, required=True)
    parser.add_argument("--reference-segments", type=Path, required=True)
    parser.add_argument("--round01-dir", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026071604)
    parser.add_argument("--heldout-diagnostic", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    measured_root = args.measured_root.resolve()
    time_ns, raw, path_ns, run = _load_simulation(args.run_dir.resolve(), args.source_case_dir.resolve())
    response_selection = json.loads((args.round01_dir.resolve() / "round01_selection.json").read_text(encoding="utf-8"))
    if response_selection["selected_candidate"] != "strength_1.00":
        raise RuntimeError("Round 04 requires frozen Round 01 response strength_1.00")
    fit_spectrum = pool_spectra([measured_line_spectrum(measured_root / f"{line}.npz") for line in FIT_LINES])
    suppressed = raw - np.median(raw, axis=1, keepdims=True)
    _, packets = aligned_packets(suppressed, time_ns, path_ns)
    simulated_spectrum = packet_spectrum(packets)
    response = bounded_system_response_db(simulated_spectrum, fit_spectrum)
    parent = apply_zero_phase_response(
        raw, 1.4, simulated_spectrum.frequency_hz, response, strength=1.0
    )
    event_fit, _ = fit_event_contract(FIT_LINES, measured_root, args.reference_segments.resolve())
    validation_target = float(
        np.median([_measured_target_ratio(measured_root / f"{line}.npz") for line in VALIDATION_LINES])
    )
    # The parent has native trace spacing. Its full span is only 64 traces, so
    # the event density estimate is converted from metres, not copied by count.
    scene = json.loads((args.source_case_dir.resolve() / "scene_manifest.json").read_text(encoding="utf-8"))
    spacing_m = float(scene["grid"]["trace_spacing_m"])
    candidates: dict[str, np.ndarray] = {}
    fields: dict[str, np.ndarray] = {}
    records: dict[str, list[dict[str, float]]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for index, variant in enumerate(VARIANTS):
        field, events = _event_field(parent.shape, time_ns, path_ns, spacing_m, event_fit, variant, args.seed + index)
        scale, candidate = _calibrate(parent, field, time_ns, path_ns, validation_target)
        candidates[variant.name] = candidate
        fields[variant.name] = field
        records[variant.name] = events
        packet = packet_metrics(candidate, time_ns, path_ns, spectrum_reference=fit_spectrum)
        metrics[variant.name] = {
            **packet,
            "event_scale": scale,
            "event_count": float(len(events)),
            "mean_native_support_traces": float(np.mean([event["support_trace_count"] for event in events])) if events else 0.0,
            "selection_score": abs(math.log(max(packet["target_to_background"], 1e-12) / validation_target))
            + 8.0 * max(packet["target_dropout_fraction"] - 0.03, 0.0),
        }
    selected = min(metrics, key=lambda name: metrics[name]["selection_score"])
    selection_path = output / "round04_selection.json"
    if args.heldout_diagnostic:
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        if frozen["selected_candidate"] != selected:
            raise RuntimeError("frozen Round 04 selection changed on recomputation")
        _multiline_panel(
            output / "round04_frozen_plus_heldout.png",
            measured_root,
            candidates[selected],
            time_ns,
            FIT_LINES + VALIDATION_LINES + HELDOUT_LINES,
            candidate_label=f"Round04 frozen {selected}",
            heading="Round 04 finite-event nuisance: frozen candidate plus held-out diagnostic",
        )
        frozen["heldout_diagnostic"] = {
            "opened_after_freeze": True,
            "lines": list(HELDOUT_LINES),
            "metrics": {
                line: packet_metrics(candidates[selected], time_ns, path_ns, spectrum_reference=measured_line_spectrum(measured_root / f"{line}.npz"))
                for line in HELDOUT_LINES
            },
            "visual": "round04_frozen_plus_heldout.png",
        }
        frozen["status"] = "heldout_diagnostic_complete"
        selection_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(frozen, ensure_ascii=False, indent=2))
        return 0
    blind = _blind_panel(
        output / "round04_blind_candidates.png",
        {f"event_{name}": value for name, value in candidates.items()},
        time_ns,
        args.seed,
        heading="Round 04 target-excluded finite-event candidates; shared scales; no labels",
    )
    _multiline_panel(
        output / "round04_fold_only_multiline.png",
        measured_root,
        candidates[selected],
        time_ns,
        FIT_LINES + VALIDATION_LINES,
        candidate_label=f"Round04 selected {selected}",
        heading="Round 04 finite-event nuisance: fold-only comparison",
    )
    _native_event_panel(output / "round04_native_event_geometry.png", fields, time_ns, records)
    payload: dict[str, object] = {
        "round": 4,
        "factor": "target_excluded_non_gaussian_finite_event_nuisance",
        "hypothesis": "finite, tapered, non-Gaussian event topology can improve measured-like background texture without a low-rank band or copied measured patch",
        "fit_lines": list(FIT_LINES),
        "validation_lines": list(VALIDATION_LINES),
        "heldout_opened_after_freeze": list(HELDOUT_LINES),
        "selected_candidate": selected,
        "selection_metrics": metrics[selected],
        "variants": [asdict(item) for item in VARIANTS],
        "event_fit": {
            "events_per_25m": event_fit.events_per_25m,
            "pooled_events_per_25m": event_fit.pooled_events_per_25m,
            "pooled_quantiles_q10_q50_q90": {key: list(value) for key, value in event_fit.pooled_quantiles.items()},
        },
        "source_run": str(args.run_dir.resolve().relative_to(ROOT).as_posix()),
        "native_trace_spacing_m": spacing_m,
        "event_records": records,
        "blind_mapping": blind,
        "visuals": ["round04_blind_candidates.png", "round04_fold_only_multiline.png", "round04_native_event_geometry.png"],
        "status": "fold_selection_complete",
        "formal_training_allowed": False,
        "causal_pair_complete": False,
        "gprmax_solver_run_performed": False,
        "next_gate": "frozen held-out diagnostic and visual audit; reject if the local topology reads as synthetic or competes with the target",
    }
    selection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _append_ledger(args.research_root.resolve(), {
        "round": 4,
        "factor": payload["factor"],
        "hypothesis": payload["hypothesis"],
        "fit_lines": list(FIT_LINES),
        "validation_lines": list(VALIDATION_LINES),
        "heldout_opened_after_freeze": list(HELDOUT_LINES),
        "selected_variant": selected,
        "decision": "pending_frozen_heldout_and_visual_review",
        "standalone_realism_candidate": False,
        "key_result": metrics[selected],
        "report": "round04_finite_event_nuisance/ROUND_DECISION.md",
    })
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
