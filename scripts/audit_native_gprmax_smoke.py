#!/usr/bin/env python3
"""Audit one strict full/control/air gprMax smoke triplet before a B-scan run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def select_arrival_reference(source_case: Path) -> tuple[Path, str]:
    """Prefer a source-delay-aware audit reference when the case declares one."""

    reference_candidates = (
        ("source_referenced_arrival_time_ns.npy", "geometric_interface_plus_explicit_source_reference_delay"),
        ("reference_arrival_time_ns.npy", "legacy_case_reference"),
        ("geometric_reference_arrival_time_ns.npy", "geometric_interface_only"),
    )
    for filename, semantics in reference_candidates:
        candidate = source_case / "labels" / filename
        if candidate.is_file():
            return candidate, semantics
    raise FileNotFoundError(f"no supported arrival reference exists below {source_case / 'labels'}")


def _read_trace(path: Path, component: str) -> tuple[np.ndarray, dict[str, object]]:
    with h5py.File(path, "r") as handle:
        rx = handle["rxs/rx1"]
        values = np.asarray(rx[component], dtype=np.float64)
        metadata = {
            "iterations": int(handle.attrs["Iterations"]),
            "dt_s": float(handle.attrs["dt"]),
            "grid_m": np.asarray(handle.attrs["dx_dy_dz"], dtype=float).tolist(),
            "source_position_m": np.asarray(handle["srcs/src1"].attrs["Position"], dtype=float).tolist(),
            "receiver_position_m": np.asarray(rx.attrs["Position"], dtype=float).tolist(),
        }
    return values, metadata


def _envelope(values: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(values)
    weights = np.zeros(values.size)
    weights[0] = 1
    if values.size % 2 == 0:
        weights[values.size // 2] = 1
        weights[1 : values.size // 2] = 2
    else:
        weights[1 : (values.size + 1) // 2] = 2
    return np.abs(np.fft.ifft(spectrum * weights))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mask(time_ns: np.ndarray, start_ns: float, end_ns: float) -> np.ndarray:
    return (time_ns >= start_ns) & (time_ns <= end_ns)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    time_ns: np.ndarray,
    curves: list[tuple[np.ndarray, str]],
    *,
    x_range: tuple[float, float],
    title: str,
    target_window: tuple[float, float] | None = None,
    reference_ns: float | None = None,
    pick_ns: float | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#555555", width=1)
    draw.text((left + 4, top + 3), title, fill="#111111")
    x0, x1 = x_range
    selected = (time_ns >= x0) & (time_ns <= x1)
    shown_time = time_ns[selected]
    shown_curves = [values[selected] for values, _ in curves]
    magnitude = max(float(np.max(np.abs(values))) for values in shown_curves) or 1.0
    data_top, data_bottom = top + 22, bottom - 18
    mid = (data_top + data_bottom) / 2
    scale = (data_bottom - data_top) / (2 * magnitude)
    def x_coord(value: float) -> int:
        return round(left + (value - x0) / (x1 - x0) * (right - left))
    if target_window is not None:
        draw.rectangle((x_coord(target_window[0]), data_top, x_coord(target_window[1]), data_bottom), fill="#fff3c4")
    draw.line((left, round(mid), right, round(mid)), fill="#bbbbbb")
    for value, color in curves:
        clipped = value[selected]
        step = max(1, clipped.size // max(1, right - left))
        points = [(x_coord(float(shown_time[i])), round(mid - clipped[i] * scale)) for i in range(0, clipped.size, step)]
        if len(points) > 1:
            draw.line(points, fill=color, width=1)
    if reference_ns is not None:
        x = x_coord(reference_ns)
        draw.line((x, data_top, x, data_bottom), fill="#111111", width=1)
    if pick_ns is not None:
        x = x_coord(pick_ns)
        draw.line((x, data_top, x, data_bottom), fill="#9467bd", width=2)
    draw.text((left + 4, bottom - 15), f"{x0:.0f}-{x1:.0f} ns", fill="#333333")


def _write_preview(
    output_path: Path,
    time_ns: np.ndarray,
    traces: dict[str, np.ndarray],
    difference: np.ndarray,
    envelope: np.ndarray,
    reference_ns: float,
    pick_ns: float,
    half_window_ns: float,
    contrast_db: float,
) -> None:
    image = Image.new("RGB", (1500, 1050), "white")
    draw = ImageDraw.Draw(image)
    window = (reference_ns - half_window_ns, reference_ns + half_window_ns)
    raw_curves = [
        (traces["full_scene"], "#1f77b4"),
        (traces["no_basal_contrast_control"], "#d62728"),
    ]
    raw_title = "Raw: full (blue), no-basal (red)"
    if "air_reference" in traces:
        raw_curves.append((traces["air_reference"], "#777777"))
        raw_title += ", air (gray)"
    _draw_panel(draw, (70, 50, 1430, 350), time_ns, raw_curves, x_range=(0, 700), title=raw_title)
    _draw_panel(draw, (70, 390, 1430, 690), time_ns, [(difference, "#2ca02c")], x_range=(max(0.0, reference_ns - 100), min(700.0, reference_ns + 100)), title=f"Signed full - no-basal difference, target/pre-target = {contrast_db:.1f} dB", target_window=window, reference_ns=reference_ns, pick_ns=pick_ns)
    _draw_panel(draw, (70, 730, 1430, 1030), time_ns, [(envelope, "#9467bd")], x_range=(max(0.0, reference_ns - 100), min(700.0, reference_ns + 100)), title="Difference analytic-signal envelope", target_window=window, reference_ns=reference_ns, pick_ns=pick_ns)
    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="Completed one-trace solver run directory.")
    parser.add_argument("--component", default="Ez")
    parser.add_argument("--target-half-window-ns", type=float, default=35.0)
    parser.add_argument(
        "--skip-air-reference",
        action="store_true",
        help="Audit only the causal full/control pair and record that air was intentionally omitted.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_case = Path(json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))["source_case_dir"])
    source_x = np.load(source_case / "labels" / "source_x_m.npy")
    reference_path, reference_semantics = select_arrival_reference(source_case)
    reference_arrival = np.load(reference_path)

    traces: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, object]] = {}
    names = ["full_scene", "no_basal_contrast_control"]
    if not args.skip_air_reference:
        names.append("air_reference")
    for name in names:
        traces[name], metadata[name] = _read_trace(case_dir / f"{name}.out", args.component)

    full_meta = metadata["full_scene"]
    alignment_ok = all(
        metadata[name]["iterations"] == full_meta["iterations"]
        and metadata[name]["dt_s"] == full_meta["dt_s"]
        and np.allclose(metadata[name]["source_position_m"], full_meta["source_position_m"])
        and np.allclose(metadata[name]["receiver_position_m"], full_meta["receiver_position_m"])
        for name in metadata
    )
    finite_ok = all(np.isfinite(values).all() for values in traces.values())
    time_ns = np.arange(traces["full_scene"].size) * float(full_meta["dt_s"]) * 1e9
    trace_index = int(np.argmin(np.abs(source_x - float(full_meta["source_position_m"][0]))))
    reference_ns = float(reference_arrival[trace_index])
    difference = traces["full_scene"] - traces["no_basal_contrast_control"]
    target = _mask(time_ns, reference_ns - args.target_half_window_ns, reference_ns + args.target_half_window_ns)
    background = _mask(time_ns, 150.0, max(150.0, reference_ns - args.target_half_window_ns - 20.0))
    direct = _mask(time_ns, 0.0, 150.0)
    target_rms = _rms(difference[target])
    background_rms = _rms(difference[background])
    contrast_db = float(20 * np.log10(max(target_rms, np.finfo(float).tiny) / max(background_rms, np.finfo(float).tiny)))
    envelope = _envelope(difference)
    pick_index = int(np.argmax(envelope[target]))
    target_indices = np.flatnonzero(target)
    pick_ns = float(time_ns[target_indices[pick_index]])
    pick_offset_ns = pick_ns - reference_ns
    report = {
        "schema": "native_gprmax_smoke_pair_audit_v2",
        "case_dir": _portable_path(case_dir),
        "source_case_dir": _portable_path(source_case),
        "component": args.component,
        "air_reference_included": "air_reference" in traces,
        "trace_index": trace_index,
        "reference_path": _portable_path(reference_path),
        "reference_semantics": reference_semantics,
        "reference_arrival_ns": reference_ns,
        "difference_envelope_pick_ns": pick_ns,
        "difference_pick_offset_ns": pick_offset_ns,
        "target_window_ns": [reference_ns - args.target_half_window_ns, reference_ns + args.target_half_window_ns],
        "alignment_ok": alignment_ok,
        "finite_ok": finite_ok,
        "target_difference_rms": target_rms,
        "pre_target_difference_rms": background_rms,
        "target_to_pre_target_difference_db": contrast_db,
        "full_direct_rms": _rms(traces["full_scene"][direct]),
        "control_direct_rms": _rms(traces["no_basal_contrast_control"][direct]),
        "air_direct_rms": _rms(traces["air_reference"][direct]) if "air_reference" in traces else None,
        "metadata": metadata,
        "smoke_gate": {
            "passed": bool(alignment_ok and finite_ok and contrast_db >= 6.0 and abs(pick_offset_ns) <= args.target_half_window_ns),
            "criteria": {
                "strict_pair_alignment": True,
                "finite_arrays": True,
                "target_difference_over_pre_target_db_min": 6.0,
                "difference_pick_within_target_half_window_ns": args.target_half_window_ns,
            },
        },
    }
    (output_dir / "pair_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    _write_preview(output_dir / "pair_audit.png", time_ns, traces, difference, envelope, reference_ns, pick_ns, args.target_half_window_ns, contrast_db)
    print(json.dumps(report["smoke_gate"], indent=2))
    return 0 if report["smoke_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
