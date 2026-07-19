#!/usr/bin/env python3
"""Round 05: verify flight-height effects with FDTD full/control pairs.

This is intentionally a three-height, one-trace causal probe.  It is not a
training export and it cannot establish B-scan morphology.  Its purpose is to
replace the rejected post-solve timing warp from Round 02 with an auditable
physical result: source and receiver height are changed before gprMax solves
the same positive/control geology pair.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.runtime import load_runtime, require_gprmax  # noqa: E402
from scripts.run_native_256_release_pilot import (  # noqa: E402
    STATIC_AUDIT,
    _load_vcvars_environment,
    stage_case,
)


INPUTS = ("full_scene.in", "no_basal_contrast_control.in")
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def _read_position(line: str, keyword: str) -> tuple[float, float, float]:
    tokens = line.split()
    if keyword == "#hertzian_dipole":
        coordinates = tokens[2:5]
    elif keyword == "#rx":
        coordinates = tokens[1:4]
    else:
        raise ValueError(f"unsupported position keyword: {keyword}")
    if len(coordinates) != 3:
        raise RuntimeError(f"could not parse {keyword} line: {line}")
    return tuple(float(item) for item in coordinates)


def _set_height(input_path: Path, height_m: float, ground_y_m: float) -> None:
    """Move source and receiver together while retaining their 0.18 m offset."""

    lines = input_path.read_text(encoding="utf-8").splitlines()
    source_index = next(index for index, line in enumerate(lines) if line.startswith("#hertzian_dipole:"))
    receiver_index = next(index for index, line in enumerate(lines) if line.startswith("#rx:"))
    source = _read_position(lines[source_index], "#hertzian_dipole")
    receiver = _read_position(lines[receiver_index], "#rx")
    source_y = ground_y_m + height_m
    receiver_y = ground_y_m + height_m
    source_tokens = lines[source_index].split()
    receiver_tokens = lines[receiver_index].split()
    source_tokens[3] = f"{source_y:.6f}"
    receiver_tokens[1] = f"{receiver[0]:.6f}"
    receiver_tokens[2] = f"{receiver_y:.6f}"
    if not np.isclose(source[0], 0.0) and not np.isclose(receiver[0] - source[0], 0.18):
        raise RuntimeError("unexpected source/receiver lateral offset in source deck")
    lines[source_index] = " ".join(source_tokens)
    lines[receiver_index] = " ".join(receiver_tokens)
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_time_window(input_path: Path, time_window_ns: float) -> None:
    text = input_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^#time_window:\s+[^\r\n]+$",
        f"#time_window: {time_window_ns * 1e-9:.9g}",
        text,
    )
    if count != 1:
        raise RuntimeError(f"expected one #time_window declaration in {input_path}")
    input_path.write_text(updated, encoding="utf-8")


def _read_trace(path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["rxs/rx1/Ez"], dtype=np.float64)
        dt_s = float(handle.attrs["dt"])
    return values, dt_s


def _peak_time_ns(values: np.ndarray, dt_s: float, start_ns: float, end_ns: float) -> float:
    time_ns = np.arange(values.size, dtype=np.float64) * dt_s * 1e9
    selected = (time_ns >= start_ns) & (time_ns <= end_ns)
    if not np.any(selected):
        raise RuntimeError("peak window is outside solver output")
    return float(time_ns[selected][np.argmax(np.abs(hilbert(values[selected])))])


def _execute(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _write_visual(output: Path, records: list[dict[str, object]]) -> None:
    width, panel_height, margin = 1280, 245, 30
    canvas = Image.new("RGB", (width, panel_height * 3 + margin * 4), "white")
    draw = ImageDraw.Draw(canvas)
    colors = ((31, 119, 180), (214, 39, 40), (44, 160, 44))
    panels = (
        ("Round 05 physical height probe: full scene", "full"),
        ("Same geometry and height: no-basal control", "control"),
        ("Causal residual: full minus no-basal", "residual"),
    )
    for panel_index, (title, field) in enumerate(panels):
        top = margin + panel_index * (panel_height + margin)
        draw.text((margin, top), title, fill="black")
        plot_top, plot_bottom = top + 22, top + panel_height
        draw.rectangle((margin, plot_top, width - margin, plot_bottom), outline="black")
        values = [np.asarray(record[field], dtype=np.float64) for record in records]
        limit = max(float(np.quantile(np.abs(np.concatenate(values)), 0.998)), 1e-14)
        for color, record, series in zip(colors, records, values):
            time_ns = np.asarray(record["time_ns"], dtype=np.float64)
            valid = time_ns <= 550.0
            x = margin + (time_ns[valid] / 550.0) * (width - 2 * margin)
            y = (plot_top + plot_bottom) / 2.0 - np.clip(series[valid] / limit, -1.0, 1.0) * (plot_bottom - plot_top) * 0.44
            points = [(int(xi), int(yi)) for xi, yi in zip(x, y)]
            draw.line(points, fill=color, width=1)
            label = f"{float(record['height_m']):.2f} m"
            legend_y = plot_top + 4 + colors.index(color) * 16
            draw.text((width - 155, legend_y), label, fill=color)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heights-m", type=float, nargs="+", default=(7.5, 8.01, 8.49))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--time-window-ns",
        type=float,
        default=550.0,
        help="Short probe window; it retains the basal residual but reduces GPU work.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete full/control probe pairs after an interrupted long GPU run.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Read complete existing pairs and write metrics without launching gprMax.",
    )
    args = parser.parse_args()

    source_case = args.source_case.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source_case / "scene_manifest.json").read_text(encoding="utf-8"))
    fixed_height = float(manifest["geometry"]["fixed_flight_height_m"])
    source_line = next(line for line in (source_case / "full_scene.in").read_text(encoding="utf-8").splitlines() if line.startswith("#hertzian_dipole:"))
    _, base_source_y, _ = _read_position(source_line, "#hertzian_dipole")
    ground_y_m = base_source_y - fixed_height

    runtime = load_runtime()
    gprmax_python, gprmax_root = require_gprmax(runtime)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(gprmax_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["TEMP"] = str(runtime.scratch_root / "toolchain_tmp")
    env["TMP"] = env["TEMP"]
    pycuda_cache = runtime.scratch_root / "toolchain_tmp" / "pycuda_cache"
    pycuda_cache.mkdir(parents=True, exist_ok=True)
    env["PYCUDA_CACHE_DIR"] = str(pycuda_cache)
    if args.execute:
        env = _load_vcvars_environment(env, runtime.gprmax_vcvars)
        configured_cuda_bin = os.environ.get("PGDA_CUDA_BIN")
        if not configured_cuda_bin and runtime.profile_path:
            configured_cuda_bin = json.loads(runtime.profile_path.read_text(encoding="utf-8")).get("cuda_bin")
        candidates = [
            Path(configured_cuda_bin) / "nvcc.exe" if configured_cuda_bin else None,
            gprmax_python.parent / "Library" / "bin" / "nvcc.exe",
            Path(shutil.which("nvcc", path=env.get("PATH", "")) or ""),
        ]
        nvcc = next((candidate for candidate in candidates if candidate and candidate.is_file()), None)
        if nvcc is None or not nvcc.is_file():
            raise RuntimeError(
                "Round 05 requires nvcc.exe. Set PGDA_CUDA_BIN or configure cuda_bin "
                "in the ignored runtime profile."
            )
        env["CUDA_PATH"] = str(nvcc.parents[1])
        env["PATH"] = str(nvcc.parent) + os.pathsep + env["PATH"]

    records: list[dict[str, object]] = []
    for height_m in args.heights_m:
        identifier = f"simr05_height_{height_m:.2f}m".replace(".", "p")
        run_dir = (args.run_root.resolve() / "SIMR05_F01_HEIGHT_PHYSICAL_PROBE" / identifier)
        complete_pair = (run_dir / "full_scene.out").is_file() and (
            run_dir / "no_basal_contrast_control.out"
        ).is_file()
        if run_dir.exists() and not ((args.resume or args.summarize_existing) and complete_pair):
            raise FileExistsError(
                f"refusing to reuse incomplete or unapproved run directory: {run_dir}"
            )
        if not args.execute and not args.summarize_existing:
            continue
        if args.summarize_existing and not complete_pair:
            raise FileNotFoundError(f"missing complete full/control pair for summary: {run_dir}")
        if args.execute and not complete_pair:
            stage_case(
                source_case,
                run_dir,
                requested_trace_count=1,
                geometry_only=False,
                include_air_reference=False,
                full_scene_only=False,
            )
            for input_name in INPUTS:
                _set_height(run_dir / input_name, height_m, ground_y_m)
                _set_time_window(run_dir / input_name, args.time_window_ns)
                _execute(
                    [str(gprmax_python), str(STATIC_AUDIT), input_name, "--json", f"preflight_{Path(input_name).stem}.json"],
                    cwd=run_dir,
                    env=env,
                )
            for input_name in INPUTS:
                _execute(
                    [str(gprmax_python), "-m", "gprMax", input_name, "-n", "1", "--geometry-fixed", "-gpu", str(runtime.gpu_index)],
                    cwd=run_dir,
                    env=env,
                )
        full, dt_s = _read_trace(run_dir / "full_scene.out")
        control, control_dt_s = _read_trace(run_dir / "no_basal_contrast_control.out")
        if not np.isclose(dt_s, control_dt_s):
            raise RuntimeError("positive/control solver sample intervals differ")
        time_ns = np.arange(full.size, dtype=np.float64) * dt_s * 1e9
        residual = full - control
        realised_source_line = next(
            line
            for line in (run_dir / "full_scene.in").read_text(encoding="utf-8").splitlines()
            if line.startswith("#hertzian_dipole:")
        )
        _, realised_source_y, _ = _read_position(realised_source_line, "#hertzian_dipole")
        dl_m = float(manifest["grid"]["dl_m"])
        realised_source_y = np.floor((realised_source_y + 1e-12) / dl_m) * dl_m
        records.append(
            {
                "requested_height_m": float(height_m),
                "height_m": float(realised_source_y - ground_y_m),
                "run_dir": str(run_dir),
                "time_ns": time_ns,
                "full": full,
                "control": control,
                "residual": residual,
                "direct_peak_ns": _peak_time_ns(full, dt_s, 0.0, 180.0),
                "residual_peak_ns": _peak_time_ns(
                    residual, dt_s, 180.0, min(520.0, args.time_window_ns)
                ),
                "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
            }
        )

    if not args.execute and not args.summarize_existing:
        print("PLAN ONLY: no gprMax run executed. Add --execute to run the full/control probes.")
        return 0
    baseline = next(record for record in records if np.isclose(float(record["height_m"]), fixed_height))
    serializable: list[dict[str, object]] = []
    for record in records:
        expected_delta_ns = 2.0 * (float(record["height_m"]) - fixed_height) / SPEED_OF_LIGHT_M_PER_S * 1e9
        serializable.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"time_ns", "full", "control", "residual"}
            }
            | {
                "expected_air_round_trip_delta_ns": expected_delta_ns,
                "observed_direct_peak_delta_ns": float(record["direct_peak_ns"]) - float(baseline["direct_peak_ns"]),
                "observed_residual_peak_delta_ns": float(record["residual_peak_ns"]) - float(baseline["residual_peak_ns"]),
            }
        )
    result = {
        "round": 5,
        "factor": "physical_source_receiver_height",
        "source_case": str(source_case),
        "source_case_line9_conditioned": bool(manifest.get("line9_conditioned")),
        "strict_pair_complete": True,
        "air_reference_intentionally_omitted": True,
        "air_reference_reason": "height comparison needs full/no-basal attribution; the air-only solve cannot substitute for the control",
        "fixed_ground_y_m": ground_y_m,
        "base_height_m": fixed_height,
        "solver_time_window_ns": args.time_window_ns,
        "records": serializable,
        "status": "physical_probe_complete",
        "formal_training_allowed": False,
        "next_gate": "use measured per-trace height only after validating its physical semantics; do not emulate height with post-solve delays",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "round05_height_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_visual(output / "round05_height_probe.png", records)
    ledger_path = output.parent / "research_ledger.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        round_record = {
            "round": 5,
            "factor": result["factor"],
            "decision": "physical_height_accepted_postsolve_delay_rejected",
            "standalone_realism_candidate": False,
            "formal_training_allowed": False,
            "key_result": {
                "residual_peak_shift_ns": [
                    float(record["observed_residual_peak_delta_ns"])
                    for record in serializable
                ],
                "requested_vs_grid_realised_height_m": [
                    [float(record["requested_height_m"]), float(record["height_m"])]
                    for record in serializable
                ],
            },
            "report": "round05_height_probe/ROUND_DECISION.md",
        }
        ledger["rounds"] = [
            item for item in ledger["rounds"] if int(item.get("round", -1)) != 5
        ] + [round_record]
        ledger["rounds"].sort(key=lambda item: int(item["round"]))
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
