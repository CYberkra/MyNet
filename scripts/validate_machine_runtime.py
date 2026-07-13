#!/usr/bin/env python3
"""Validate the portable local machine profile without running a solver."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.runtime import RuntimeConfigError, load_runtime, profile_summary, require_gprmax  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--require-training", action="store_true")
    parser.add_argument("--require-gprmax", action="store_true")
    parser.add_argument(
        "--require-gpu-compiler",
        action="store_true",
        help="Require nvcc.exe for gprMax GPU kernel compilation.",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    try:
        profile = load_runtime(args.profile)
        result = profile_summary(profile)
        if args.require_training and not profile.project_python.is_file():
            errors.append(f"project Python does not exist: {profile.project_python}")
        if args.require_gprmax:
            require_gprmax(profile)
        if args.require_gpu_compiler and not shutil.which("nvcc"):
            errors.append(
                "CUDA Toolkit compiler nvcc.exe is not available on PATH; "
                "gprMax GPU runs cannot compile kernels."
            )
    except RuntimeConfigError as exc:
        result = {"profile_path": str(args.profile) if args.profile else None}
        errors.append(str(exc))
    result["ok"] = not errors
    result["errors"] = errors
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
