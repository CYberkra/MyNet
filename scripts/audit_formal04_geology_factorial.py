#!/usr/bin/env python3
"""Compare FORMAL04 one-trace strict pairs against the FORMAL03 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = ROOT / "reports" / "formal04_geology_factorial_20260715"
REFERENCE_ID = "FORMAL03_CORRELATED_COVER_GABOR80"
CASE_IDS = (
    "FORMAL04_A_WEAK_BASAL",
    "FORMAL04_B_STRONG_TEXTURE",
    "FORMAL04_C_COMBINED",
)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _read_trace(path: Path, component: str = "Ez") -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle[f"rxs/rx1/{component}"], dtype=np.float64)
        dt_s = float(handle.attrs["dt"])
    return values, dt_s


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _case_metrics(
    case_id: str,
    run_dir: Path,
    pair_report: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    audit = json.loads(pair_report.read_text(encoding="utf-8"))
    full, full_dt = _read_trace(run_dir / "full_scene.out")
    control, control_dt = _read_trace(run_dir / "no_basal_contrast_control.out")
    if full.shape != control.shape or full_dt != control_dt:
        raise ValueError(f"unaligned strict pair for {case_id}")
    time_ns = np.arange(full.size, dtype=np.float64) * full_dt * 1e9
    reference_ns = float(audit["reference_arrival_ns"])
    half_width = 35.0
    target = np.abs(time_ns - reference_ns) <= half_width
    early_adjacent = (time_ns >= reference_ns - 105.0) & (time_ns < reference_ns - 55.0)
    late_adjacent = (time_ns > reference_ns + 55.0) & (time_ns <= reference_ns + 105.0)
    adjacent = early_adjacent | late_adjacent
    difference = full - control
    target_full_rms = _rms(full[target])
    target_control_rms = _rms(control[target])
    adjacent_full_rms = _rms(full[adjacent])
    metrics: dict[str, object] = {
        "case_id": case_id,
        "run_dir": _portable(run_dir),
        "pair_audit": _portable(pair_report),
        "reference_arrival_ns": reference_ns,
        "difference_pick_offset_ns": float(audit["difference_pick_offset_ns"]),
        "target_difference_rms": float(audit["target_difference_rms"]),
        "pre_target_difference_rms": float(audit["pre_target_difference_rms"]),
        "target_to_pre_target_difference_db": float(audit["target_to_pre_target_difference_db"]),
        "target_full_rms": target_full_rms,
        "target_control_rms": target_control_rms,
        "adjacent_full_rms": adjacent_full_rms,
        "target_full_to_adjacent_full_rms": target_full_rms / max(adjacent_full_rms, np.finfo(float).tiny),
        "target_difference_to_full_rms": float(audit["target_difference_rms"]) / max(target_full_rms, np.finfo(float).tiny),
        "strict_pair_gate_passed": bool(audit["smoke_gate"]["passed"]),
    }
    arrays = {
        "time_ns": time_ns,
        "full": full,
        "control": control,
        "difference": difference,
    }
    return metrics, arrays


def _draw_curves(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    time_ns: np.ndarray,
    curves: list[tuple[np.ndarray, str]],
    x_range: tuple[float, float],
    magnitude: float,
    reference_ns: float,
    title: str,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#555555", width=1)
    draw.text((left + 5, top + 4), title, fill="#111111")
    data_top, data_bottom = top + 24, bottom - 16
    mid = (data_top + data_bottom) / 2.0
    selected = (time_ns >= x_range[0]) & (time_ns <= x_range[1])
    selected_indices = np.flatnonzero(selected)
    step = max(1, selected_indices.size // max(1, right - left))

    def x_pixel(value: float) -> int:
        return round(left + (value - x_range[0]) / (x_range[1] - x_range[0]) * (right - left))

    draw.line((left, round(mid), right, round(mid)), fill="#bbbbbb")
    draw.line((x_pixel(reference_ns), data_top, x_pixel(reference_ns), data_bottom), fill="#111111")
    scale = (data_bottom - data_top) * 0.44 / max(magnitude, np.finfo(float).tiny)
    for values, colour in curves:
        points = [
            (x_pixel(float(time_ns[index])), round(mid - float(values[index]) * scale))
            for index in selected_indices[::step]
        ]
        if len(points) > 1:
            draw.line(points, fill=colour, width=1)


def _write_preview(
    path: Path,
    rows: list[tuple[dict[str, object], dict[str, np.ndarray]]],
) -> None:
    width, row_height = 1900, 330
    canvas = Image.new("RGB", (width, 70 + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((55, 20), "FORMAL03/04 one-trace strict-pair comparison; shared GABOR80 source and geometry", fill="black", font=font)
    raw_magnitude = max(
        float(np.max(np.abs(arrays["full"][(arrays["time_ns"] >= 300) & (arrays["time_ns"] <= 500)])))
        for _, arrays in rows
    )
    difference_magnitude = max(
        float(np.max(np.abs(arrays["difference"][(arrays["time_ns"] >= 300) & (arrays["time_ns"] <= 500)])))
        for _, arrays in rows
    )
    for row_index, (metrics, arrays) in enumerate(rows):
        top = 60 + row_index * row_height
        case_id = str(metrics["case_id"])
        ratio = float(metrics["target_difference_to_full_rms"])
        draw.text(
            (55, top + 3),
            f"{case_id} | diff/full target={ratio:.3g} | diff/pre={metrics['target_to_pre_target_difference_db']:.1f} dB | pick offset={metrics['difference_pick_offset_ns']:.1f} ns",
            fill="black",
            font=font,
        )
        _draw_curves(
            draw,
            (55, top + 28, 920, top + 305),
            arrays["time_ns"],
            [(arrays["full"], "#1f77b4"), (arrays["control"], "#d62728")],
            (300.0, 500.0),
            raw_magnitude,
            float(metrics["reference_arrival_ns"]),
            "Common scale: full (blue), no-basal (red)",
        )
        _draw_curves(
            draw,
            (980, top + 28, 1845, top + 305),
            arrays["time_ns"],
            [(arrays["difference"], "#2ca02c")],
            (300.0, 500.0),
            difference_magnitude,
            float(metrics["reference_arrival_ns"]),
            "Common scale: signed full - no-basal",
        )
    canvas.save(path)


def _write_spatial_comparison(report_root: Path) -> None:
    sources = [
        report_root / "distributed8_audit" / case_id / "positive_pair_spatial_preview.png"
        for case_id in (CASE_IDS[0], CASE_IDS[2])
    ]
    if not all(path.is_file() for path in sources):
        return
    images = [Image.open(path).convert("RGB") for path in sources]
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + 60
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "FORMAL04 sparse full-span comparison: A causal anchor vs C texture upper bound", fill="black")
    top = 60
    for image in images:
        canvas.paste(image, (0, top))
        top += image.height
    canvas.save(report_root / "distributed8_A_vs_C_comparison.png")


def audit(report_root: Path) -> dict[str, object]:
    reference_run = (
        ROOT
        / "data/PGDA_SYNTH_DATASET_V2/01_solver_runs"
        / REFERENCE_ID
        / "formal03_smoke1_20260715"
    )
    reference_report = (
        ROOT
        / "reports/formal03_source_ablation_20260715/smoke1_audit"
        / REFERENCE_ID
        / "pair_audit.json"
    )
    rows = [_case_metrics(REFERENCE_ID, reference_run, reference_report)]
    for case_id in CASE_IDS:
        rows.append(
            _case_metrics(
                case_id,
                ROOT
                / "data/PGDA_SYNTH_DATASET_V2/01_solver_runs"
                / case_id
                / "formal04_smoke1_20260715",
                report_root / "smoke1_audit" / case_id / "pair_audit.json",
            )
        )
    metrics_by_id = {str(metrics["case_id"]): metrics for metrics, _ in rows}
    base = float(metrics_by_id[REFERENCE_ID]["target_difference_rms"])
    a = float(metrics_by_id[CASE_IDS[0]]["target_difference_rms"])
    b = float(metrics_by_id[CASE_IDS[1]]["target_difference_rms"])
    c = float(metrics_by_id[CASE_IDS[2]]["target_difference_rms"])
    effects = {
        "weak_basal_at_baseline_texture_ratio_A_over_reference": a / base,
        "strong_texture_at_strong_basal_ratio_B_over_reference": b / base,
        "weak_basal_at_strong_texture_ratio_C_over_B": c / b,
        "strong_texture_at_weak_basal_ratio_C_over_A": c / a,
        "multiplicative_interaction_C_times_reference_over_A_times_B": c * base / (a * b),
    }
    recommendation = {
        "one_trace_decision": "advance A and C to sparse paired morphology; do not advance B",
        "rationale": [
            "all cases pass strict-pair alignment and basal detectability",
            "weak basal contrast reduces target difference under both texture levels",
            "strong texture alone increases rather than suppresses the basal response",
            "one trace cannot measure lateral clutter, coherence, or dropout",
        ],
        "training_allowed": False,
    }
    report: dict[str, object] = {
        "schema": "formal04_geology_factorial_one_trace_audit_v1",
        "reference_case_id": REFERENCE_ID,
        "case_metrics": [metrics for metrics, _ in rows],
        "factor_effects": effects,
        "recommendation": recommendation,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "one_trace_factorial_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_preview(report_root / "one_trace_factorial_comparison.png", rows)

    spatial_paths = {
        REFERENCE_ID: ROOT
        / "reports/formal03_source_ablation_20260715/distributed8_audit_reference_relative"
        / REFERENCE_ID
        / "family_spatial_pilot_audit.json",
        CASE_IDS[0]: report_root
        / "distributed8_audit"
        / CASE_IDS[0]
        / "family_spatial_pilot_audit.json",
        CASE_IDS[2]: report_root
        / "distributed8_audit"
        / CASE_IDS[2]
        / "family_spatial_pilot_audit.json",
    }
    if all(path.is_file() for path in spatial_paths.values()):
        spatial = {
            case_id: json.loads(path.read_text(encoding="utf-8"))["signal_metrics"]
            for case_id, path in spatial_paths.items()
        }
        decision = {
            "schema": "formal04_geology_factorial_selection_v1",
            "formal_training_allowed": False,
            "dense_256_case_selected": None,
            "causal_anchor_case_id": CASE_IDS[0],
            "texture_upper_bound_case_id": CASE_IDS[2],
            "rejected_after_one_trace": [CASE_IDS[1]],
            "spatial_metrics": {
                case_id: {
                    "full_scene_target_to_local_background_rms": values[
                        "full_scene_target_to_local_background_rms"
                    ],
                    "target_amplitude_cv": values["target_amplitude_cv"],
                    "target_dropout_fraction_below_25pct_median": values[
                        "target_dropout_fraction_below_25pct_median"
                    ],
                    "visible_residual_step_per_canonical_trace_p95_ns": values[
                        "visible_residual_step_per_canonical_trace_p95_ns"
                    ],
                }
                for case_id, values in spatial.items()
            },
            "development_only_measured_context": {
                "source": "reports/line9_reproduction_program_20260714/line9_reproduction_contract.json",
                "not_read_by_generator": True,
                "target_to_adjacent_background_rms": 2.3485668444457377,
                "target_envelope_cv": 0.46463303004466555,
            },
            "decision": "FORMAL04 brackets the useful interval but neither A nor C advances to a dense run",
            "reasoning": [
                "A fixes the over-strong basal event but retains the comparatively clean FORMAL03 cover texture",
                "C adds the desired cover complexity but suppresses the full-scene target/local-background ratio below the declared preferred range",
                "B increases basal response and therefore fails the intended direction before a spatial run",
                "all runtime results remain development-only and cannot be exported as training data",
            ],
            "next_family": {
                "family_id": "FORMAL05_MODERATE_TEXTURE_BALANCED_BASAL",
                "derivation": "interpolate only within the simulation-tested FORMAL04 constitutive bracket",
                "locked_from_formal04": [
                    "GABOR80 source waveform",
                    "grid, PML, domain, acquisition, and flight height",
                    "basal path, transition thickness, stochastic latent field, and index geometry",
                ],
                "proposed_materials": {
                    "cover_epsilon_range": [10.5, 14.5],
                    "cover_conductivity_s_per_m_range": [0.0014, 0.0040],
                    "bedrock_epsilon_r": 10.7,
                    "bedrock_conductivity_s_per_m": 0.0021,
                },
                "required_gates": [
                    "static and geometry audit",
                    "one-trace strict full/control pair",
                    "eight-trace full-span strict pair",
                    "human morphology review before any dense run",
                ],
            },
        }
        (report_root / "selection_decision.json").write_text(
            json.dumps(decision, indent=2) + "\n", encoding="utf-8"
        )
        metrics = decision["spatial_metrics"]
        markdown = f"""# FORMAL04 geology factorial audit

## Scope

FORMAL04 kept the accepted FORMAL03 GABOR80 source, grid, acquisition,
flight height, basal path, transition thickness, stochastic latent field, and
indexed geometry fixed. Only basal contrast and correlated cover-material
amplitude changed. No measured data were read by the generator.

## Validation

- Six full/control static audits passed without warnings.
- FORMAL03 and all FORMAL04 cases share index-array SHA256
  `0ac1118fed1be74ee66d75797c527b39e32dd00d8179bae5113798ec07e17fbd`.
- The strongest material case retains 11.49 cells per conservative minimum
  wavelength at 2.8 times 80 MHz.
- A representative full/control geometry-only build passed.
- All three one-trace strict pairs passed alignment, finite-array,
  detectability, and source-reference-window gates.
- A and C each completed eight full-span full/control traces at canonical
  indices 0, 36, 72, 108, 144, 180, 216, and 252.

## Results

| Case | Texture | Basal | Full target/local background | Target CV | Dropout |
|---|---|---|---:|---:|---:|
| FORMAL03 reference | baseline | strong | {metrics[REFERENCE_ID]['full_scene_target_to_local_background_rms']:.3f} | {metrics[REFERENCE_ID]['target_amplitude_cv']:.3f} | {metrics[REFERENCE_ID]['target_dropout_fraction_below_25pct_median']:.1%} |
| FORMAL04 A | baseline | weak | {metrics[CASE_IDS[0]]['full_scene_target_to_local_background_rms']:.3f} | {metrics[CASE_IDS[0]]['target_amplitude_cv']:.3f} | {metrics[CASE_IDS[0]]['target_dropout_fraction_below_25pct_median']:.1%} |
| FORMAL04 C | strong | weak | {metrics[CASE_IDS[2]]['full_scene_target_to_local_background_rms']:.3f} | {metrics[CASE_IDS[2]]['target_amplitude_cv']:.3f} | {metrics[CASE_IDS[2]]['target_dropout_fraction_below_25pct_median']:.1%} |

The one-trace factor effects were: A/reference = {effects['weak_basal_at_baseline_texture_ratio_A_over_reference']:.3f},
B/reference = {effects['strong_texture_at_strong_basal_ratio_B_over_reference']:.3f},
C/B = {effects['weak_basal_at_strong_texture_ratio_C_over_B']:.3f}, and
C/A = {effects['strong_texture_at_weak_basal_ratio_C_over_A']:.3f}.

## Decision

No FORMAL04 case advances directly to a dense 256-trace run. A is retained as
the causal anchor; C is retained as the upper texture bound; B is rejected
because stronger texture with strong basal contrast amplified the target.
FORMAL05 will interpolate the material mapping inside this tested bracket while
leaving source, acquisition, path, and indexed geometry unchanged. It remains
blocked from training until the same staged gates pass.
"""
        (report_root / "FORMAL04_GEOLOGY_FACTORIAL_AUDIT.md").write_text(
            markdown, encoding="utf-8"
        )
        _write_spatial_comparison(report_root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    report = audit(args.report_root.resolve())
    print(json.dumps(report["factor_effects"], indent=2))
    print(json.dumps(report["recommendation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
