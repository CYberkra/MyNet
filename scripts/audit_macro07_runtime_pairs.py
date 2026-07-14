"""Audit strict gprMax full/control outputs without promoting development data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


def load_ez(path: Path) -> tuple[np.ndarray, float, int]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["rxs"]["rx1"]["Ez"], dtype=np.float64)
        return values, float(handle.attrs["dt"]), int(handle.attrs["Iterations"])


def collect(case_dir: Path, stem: str) -> dict[int, Path]:
    stem_path = Path(stem)
    output_dir = case_dir / stem_path.parent
    basename = stem_path.name
    pattern = re.compile(rf"^{re.escape(basename)}(\d+)\.out$")
    outputs: dict[int, Path] = {}
    for path in output_dir.glob(f"{basename}*.out"):
        if path.name == f"{basename}.out":
            outputs[1] = path
            continue
        match = pattern.match(path.name)
        if match:
            outputs[int(match.group(1))] = path
    return outputs


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def audit(
    case_dir: Path,
    full_stem: str,
    control_stem: str,
    arrival_label: Path,
    expected_trace_count: int,
) -> dict:
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    full = collect(case_dir, full_stem)
    control = collect(case_dir, control_stem)
    paired = sorted(set(full) & set(control))
    if not paired:
        raise RuntimeError(f"No paired full/control outputs found in {case_dir}")
    reference_path = case_dir / arrival_label
    if not reference_path.is_file() and arrival_label.parent == Path("."):
        candidate = case_dir / "labels" / arrival_label
        if candidate.is_file():
            reference_path = candidate
    if not reference_path.is_file():
        raise RuntimeError(f"Missing arrival-label array: {reference_path}")
    references = np.load(reference_path)
    rows = []
    expected_dt = None
    expected_iterations = None
    for trace_id in paired:
        full_ez, full_dt, full_iterations = load_ez(full[trace_id])
        control_ez, control_dt, control_iterations = load_ez(control[trace_id])
        if full_ez.shape != control_ez.shape or full_iterations != control_iterations:
            raise RuntimeError(f"{case_dir.name} trace {trace_id}: output shape mismatch")
        if not np.isclose(full_dt, control_dt):
            raise RuntimeError(f"{case_dir.name} trace {trace_id}: dt mismatch")
        if expected_dt is None:
            expected_dt, expected_iterations = full_dt, full_iterations
        elif not np.isclose(expected_dt, full_dt) or expected_iterations != full_iterations:
            raise RuntimeError(f"{case_dir.name}: time contract changes across traces")
        if not np.isfinite(full_ez).all() or not np.isfinite(control_ez).all():
            raise RuntimeError(f"{case_dir.name} trace {trace_id}: non-finite receiver values")
        time_ns = np.arange(full_ez.size) * full_dt * 1e9
        difference = full_ez - control_ez
        reference_ns = float(references[trace_id - 1])
        early = time_ns <= 120.0
        target = (time_ns >= reference_ns - 55.0) & (time_ns <= reference_ns + 55.0)
        pre_target = (time_ns >= 180.0) & (time_ns <= reference_ns - 80.0)
        peak = int(np.argmax(np.abs(difference)))
        rows.append({
            "trace_id": trace_id,
            "reference_arrival_ns": reference_ns,
            "early_difference_rms": rms(difference[early]),
            "pre_target_difference_rms": rms(difference[pre_target]),
            "target_difference_rms": rms(difference[target]),
            "difference_peak_ns": float(time_ns[peak]),
            "difference_peak_abs": float(abs(difference[peak])),
        })
    early_max = max(row["early_difference_rms"] for row in rows)
    target_min = min(row["target_difference_rms"] for row in rows)
    return {
        "case_id": manifest["case_id"],
        "development_only": True,
        "formal_training_allowed": False,
        "paired_trace_ids": paired,
        "paired_trace_count": len(paired),
        "full_stem": full_stem,
        "control_stem": control_stem,
        "arrival_label": reference_path.relative_to(case_dir).as_posix(),
        "expected_trace_count": expected_trace_count,
        "trace_count_complete": len(paired) == expected_trace_count,
        "dt_ns": expected_dt * 1e9,
        "iterations": expected_iterations,
        "early_difference_rms_max": early_max,
        "target_difference_rms_min": target_min,
        "pair_causality_pass": bool(early_max < 1e-3 and target_min > 1e-3),
        "per_trace": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--case-pattern",
        default="MACRO07_SIGNAL_CHARACTER_*",
        help="Glob selecting case directories below --controls-root.",
    )
    parser.add_argument("--full-stem", default="pilot24_full_scene")
    parser.add_argument("--control-stem", default="pilot24_no_basal_contrast_control")
    parser.add_argument(
        "--arrival-label",
        type=Path,
        default=Path("labels/pilot24_geometric_reference_arrival_time_ns.npy"),
    )
    parser.add_argument("--expected-trace-count", type=int, default=24)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    reports = []
    for case_dir in sorted(args.controls_root.resolve().glob(args.case_pattern)):
        if case_dir.is_dir():
            try:
                reports.append(
                    audit(
                        case_dir,
                        args.full_stem,
                        args.control_stem,
                        args.arrival_label,
                        args.expected_trace_count,
                    )
                )
            except RuntimeError as exc:
                if args.allow_partial and "No paired pilot24 outputs" in str(exc):
                    continue
                raise
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    for report in reports:
        print(
            report["case_id"],
            f"pairs={report['paired_trace_count']}",
            f"early={report['early_difference_rms_max']:.3e}",
            f"target={report['target_difference_rms_min']:.3e}",
            f"pass={report['pair_causality_pass']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
