#!/usr/bin/env python3
"""Run one native-256 pilot case with pre-merge per-trace provenance capture."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.runtime import RuntimeConfigError, load_runtime, require_gprmax  # noqa: E402


def _command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _run(command: list[str], *, cwd: Path, env: dict[str, str], execute: bool) -> None:
    print(_command_text(command))
    if execute:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def _remove_geometry_views(case_dir: Path) -> None:
    for view in case_dir.glob("*.vti"):
        view.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--gprmax-python", help="Override gprmax_python from the local runtime profile.")
    parser.add_argument("--gprmax-root", type=Path, help="Override gprmax_source from the local runtime profile.")
    parser.add_argument("--gpu", type=int, help="Override gpu_index from the local runtime profile.")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Check source-deck geometry only, then remove the disposable VTI view files.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("formal_training_allowed") is not False:
        raise RuntimeError("native pilot runner only accepts blocked pre-promotion scenes")
    trace_count = int(manifest["grid"]["trace_count"])
    if trace_count != 256:
        raise RuntimeError(f"native pilot requires 256 traces, got {trace_count}")
    runtime = load_runtime()
    try:
        configured_python, configured_root = require_gprmax(runtime)
    except RuntimeConfigError as exc:
        if args.gprmax_python and args.gprmax_root:
            configured_python, configured_root = Path(args.gprmax_python), args.gprmax_root
        else:
            raise RuntimeError(str(exc)) from exc
    gprmax_python = Path(args.gprmax_python).resolve() if args.gprmax_python else configured_python
    root = args.gprmax_root.resolve() if args.gprmax_root else configured_root
    if not gprmax_python.is_file():
        raise FileNotFoundError(f"gprMax Python does not exist: {gprmax_python}")
    if not (root / "gprMax").is_dir():
        raise FileNotFoundError(f"not a gprMax source root: {root}")
    gpu = int(args.gpu) if args.gpu is not None else runtime.gpu_index
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    log_dir = case_dir / "run_logs"
    log_dir.mkdir(exist_ok=True)
    inputs = ["full_scene.in"]
    if bool(manifest["target_presence"]):
        inputs.append("no_basal_contrast_control.in")
    inputs.append("air_reference.in")
    for input_name in inputs:
        stem = Path(input_name).stem
        if args.geometry_only:
            _run(
                [str(gprmax_python), "-m", "gprMax", input_name, "--geometry-only"],
                cwd=case_dir,
                env=env,
                execute=args.execute,
            )
            if args.execute:
                _remove_geometry_views(case_dir)
            continue
        # gprMax names individual moving-source outputs ``<stem>1.out``;
        # capture them before outputfiles_merge removes the evidence files.
        prefix = stem
        trace_contract = log_dir / f"{stem}_trace_contract.json"
        _run(
            [str(gprmax_python), "-m", "gprMax", input_name, "-n", str(trace_count), "--geometry-fixed", "-gpu", str(gpu)],
            cwd=case_dir,
            env=env,
            execute=args.execute,
        )
        _run(
            [str(gprmax_python), str(ROOT / "scripts" / "capture_gprmax_trace_contract.py"), str(case_dir), "--prefix", prefix, "--expected", str(trace_count), "--output", str(trace_contract)],
            cwd=case_dir,
            env=env,
            execute=args.execute,
        )
        _run(
            [str(gprmax_python), "-m", "tools.outputfiles_merge", stem, "--remove-files"],
            cwd=case_dir,
            env=env,
            execute=args.execute,
        )
    if args.geometry_only:
        return 0
    _run([str(gprmax_python), str(ROOT / "scripts" / "postprocess_physical_sim_v2.py"), str(case_dir)], cwd=case_dir, env=env, execute=args.execute)
    _run([str(gprmax_python), str(ROOT / "scripts" / "validate_physical_sim_v2.py"), "--root", str(case_dir.parent)], cwd=ROOT, env=env, execute=args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
