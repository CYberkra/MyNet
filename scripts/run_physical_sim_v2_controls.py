#!/usr/bin/env python3
"""Create and optionally execute an auditable run plan for V2 control scenes."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "00_controls"
DEFAULT_PLAN = ROOT / "reports" / "simulation_v2_control_stage_20260711" / "control_run_plan.json"


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def case_plan(case_dir: Path, *, gpu: int | None, geometry_only: bool, python_executable: str) -> dict[str, Any]:
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    target_presence = bool(manifest["target_presence"])
    commands: list[dict[str, Any]] = []

    geometry_inputs = ["geometry_check_full.in"]
    if target_presence:
        geometry_inputs.append("geometry_check_control.in")
    for input_name in geometry_inputs:
        cmd = [python_executable, "-m", "gprMax", input_name, "--geometry-only"]
        commands.append({"stage": "geometry_only", "command": cmd})

    if not geometry_only:
        run_inputs = ["full_scene.in"]
        if target_presence:
            run_inputs.append("no_basal_contrast_control.in")
        run_inputs.append("air_reference.in")
        for input_name in run_inputs:
            stem = Path(input_name).stem
            cmd = [python_executable, "-m", "gprMax", input_name, "-n", "256", "--geometry-fixed"]
            if gpu is not None:
                cmd.extend(["-gpu", str(gpu)])
            commands.append({"stage": "solver", "command": cmd})
            commands.append(
                {
                    "stage": "merge",
                    "command": [python_executable, "-m", "tools.outputfiles_merge", stem, "--remove-files"],
                }
            )
        # Positive controls extract a visible target phase; negative controls
        # must still postprocess their full/air pair into a confirmed-zero
        # target mask. Omitting this stage would leave a solved negative case
        # in the pre-solver state and invite an unsafe manual promotion.
        commands.append(
            {
                "stage": "postprocess",
                "command": [
                    python_executable,
                    str(ROOT / "scripts" / "postprocess_physical_sim_v2.py"),
                    str(case_dir),
                ],
            }
        )

    return {
        "case_id": manifest["case_id"],
        "case_dir": str(case_dir),
        "target_presence": target_presence,
        "commands": commands,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--gprmax-root", default="", help="Compiled gprMax source root injected into child PYTHONPATH.")
    parser.add_argument("--plan-output", default=str(DEFAULT_PLAN))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    selected = set(args.case_id)
    case_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "scene_manifest.json").is_file())
    if selected:
        case_dirs = [p for p in case_dirs if p.name in selected]
        missing = selected - {p.name for p in case_dirs}
        if missing:
            raise SystemExit(f"Unknown case IDs: {sorted(missing)}")
    python_executable = str(Path(args.python_executable).resolve())
    gprmax_root = Path(args.gprmax_root).resolve() if args.gprmax_root else None
    if gprmax_root and not (gprmax_root / "gprMax").is_dir():
        raise SystemExit(f"--gprmax-root is not a gprMax source root: {gprmax_root}")
    plans = [case_plan(p, gpu=args.gpu, geometry_only=args.geometry_only, python_executable=python_executable) for p in case_dirs]

    payload = {
        "root": str(root),
        "execute": args.execute,
        "geometry_only": args.geometry_only,
        "gpu": args.gpu,
        "python_executable": python_executable,
        "gprmax_root": str(gprmax_root) if gprmax_root else "",
        "case_count": len(plans),
        "formal_training_allowed": False,
        "important": "Solver commands use --geometry-fixed only because scene geometry is static while simple source/receiver positions move.",
        "cases": plans,
    }
    plan_output = Path(args.plan_output)
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for plan in plans:
        print(f"\n[{plan['case_id']}] {plan['case_dir']}")
        for entry in plan["commands"]:
            print(f"  {entry['stage']}: {command_text(entry['command'])}")
            if args.execute:
                env = os.environ.copy()
                if gprmax_root:
                    env["PYTHONPATH"] = str(gprmax_root) + os.pathsep + env.get("PYTHONPATH", "")
                proc = subprocess.run(entry["command"], cwd=plan["case_dir"], env=env, check=False)
                if proc.returncode != 0:
                    print(f"FAILED: {plan['case_id']} stage={entry['stage']} returncode={proc.returncode}")
                    return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
