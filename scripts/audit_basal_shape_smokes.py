#!/usr/bin/env python3
"""Audit one-trace full/no-basal smoke outputs for the shape-family pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "data" / "simulations" / "v2" / "01_solver_runs"
OUT = ROOT / "reports" / "basal_shape_family_pilot_20260720"

RUNS = [
    ("BS01_FLAT_REFERENCE", "SHAPE01_TRACE_01", 0),
    ("BS02_BROAD_RISE", "SHAPE01_CENTER_02", 128),
    ("BS03_DOUBLE_RELIEF", "SHAPE01_CENTER_03", 128),
    ("BS04_GENTLE_MULTISCALE", "SHAPE01_CENTER_04", 128),
]


def read_ez(path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["rxs"]["rx1"]["Ez"], dtype=np.float64), float(handle.attrs["dt"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for row_index, (family, run_id, trace_index) in enumerate(RUNS):
        run_dir = RUN_ROOT / f"{family}_POS" / run_id
        full, dt = read_ez(run_dir / "full_scene.out")
        control, dt_control = read_ez(run_dir / "no_basal_contrast_control.out")
        if full.shape != control.shape or abs(dt - dt_control) > 1e-15:
            raise RuntimeError(f"incompatible pair: {run_dir}")
        time_ns = np.arange(full.size) * dt * 1e9
        signed = full - control
        target = (time_ns >= 250.0) & (time_ns <= 650.0)
        peak_index = int(np.argmax(np.abs(signed[target])))
        target_indices = np.flatnonzero(target)
        peak_index = int(target_indices[peak_index])
        full_rms = float(np.sqrt(np.mean(full[target] ** 2)))
        control_rms = float(np.sqrt(np.mean(control[target] ** 2)))
        signed_rms = float(np.sqrt(np.mean(signed[target] ** 2)))
        row = {
            "family": family,
            "run_id": run_id,
            "trace_index": trace_index,
            "iterations": int(full.size),
            "dt_ns": dt * 1e9,
            "signed_target_peak_time_ns": float(time_ns[peak_index]),
            "signed_target_peak_amplitude": float(signed[peak_index]),
            "signed_target_peak_abs": float(abs(signed[peak_index])),
            "full_target_rms": full_rms,
            "control_target_rms": control_rms,
            "signed_target_rms": signed_rms,
            "signed_to_full_rms": signed_rms / max(full_rms, 1e-30),
            "pair_complete": True,
            "formal_training_allowed": False,
        }
        rows.append(row)
    with (OUT / "shape_family_one_trace_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "shape_family_one_trace_metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (OUT / "SHAPE01_PILOT_AUDIT.md").write_text(
        "# SHAPE01 basal geometry one-trace audit\n\n"
        "All four cases completed a causal full/no-basal pair. This is a smoke audit, not a 256-trace release.\n\n"
        "The signed difference is the only causal target evidence used here; separate full-scene energy is not treated as basal proof.\n\n"
        + "\n".join(
            f"- **{item['family']}**: signed peak {item['signed_target_peak_time_ns']:.2f} ns, "
            f"abs peak {item['signed_target_peak_abs']:.3e}, signed/full RMS {item['signed_to_full_rms']:.3f}."
            for item in rows
        )
        + "\n\nFormal promotion remains blocked until dense/canonical spacing, visible-phase extraction, and human semantic review pass.\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
