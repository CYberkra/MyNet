#!/usr/bin/env python3
"""Check whether the local runtime can execute PGDA Simulation Contract V2.

This script performs no installation. It records the Python, gprMax, HDF5, and
optional NVIDIA runtime state so solver outputs remain auditable.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "simulation_v2_control_stage_20260711" / "gprmax_runtime_check.json"
TARGET_STABLE_VERSION = "3.1.7"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_probe(command: list[str], timeout: int = 60) -> dict[str, object]:
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "output_tail": proc.stdout[-5000:],
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"command": command, "returncode": None, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--require-exact-stable",
        action="store_true",
        help="Fail unless installed gprMax version is exactly the audited stable target.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    gprmax_module = importlib.util.find_spec("gprMax") is not None
    gprmax_version = package_version("gprMax")
    dependencies = {
        name: package_version(name) for name in ("numpy", "scipy", "h5py", "matplotlib")
    }
    help_probe = command_probe([sys.executable, "-m", "gprMax", "--help"]) if gprmax_module else None

    nvidia_smi = shutil.which("nvidia-smi")
    gpu_probe = command_probe([nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]) if nvidia_smi else None

    errors: list[str] = []
    warnings: list[str] = []
    if not gprmax_module:
        errors.append("Python module gprMax is not installed in this environment")
    if gprmax_module and help_probe and help_probe.get("returncode") != 0:
        errors.append("python -m gprMax --help failed")
    if args.require_exact_stable and gprmax_version != TARGET_STABLE_VERSION:
        errors.append(
            f"gprMax version must equal audited stable target {TARGET_STABLE_VERSION}; got {gprmax_version}"
        )
    elif gprmax_version and gprmax_version != TARGET_STABLE_VERSION:
        warnings.append(
            f"Installed gprMax version {gprmax_version} differs from audited stable target {TARGET_STABLE_VERSION}; record and revalidate before formal use"
        )
    if not nvidia_smi:
        warnings.append("NVIDIA runtime not detected; CPU execution may still be valid but slower")

    payload = {
        "ok": not errors,
        "formal_solver_run_allowed": not errors,
        "audited_target_gprmax_version": TARGET_STABLE_VERSION,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "gprmax": {
            "module_available": gprmax_module,
            "installed_version": gprmax_version,
            "help_probe": help_probe,
        },
        "dependencies": dependencies,
        "gpu": {"nvidia_smi_path": nvidia_smi, "probe": gpu_probe},
        "errors": errors,
        "warnings": warnings,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
