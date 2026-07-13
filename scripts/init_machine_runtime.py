#!/usr/bin/env python3
"""Create the Git-ignored local machine runtime profile."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "environment" / "project_runtime.example.json"
DEFAULT = ROOT / "environment" / "project_runtime.local.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--project-python")
    parser.add_argument("--gprmax-python")
    parser.add_argument("--gprmax-root")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force to replace it")
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["profile_name"] = socket.gethostname()
    for key, value in {
        "project_python": args.project_python,
        "gprmax_python": args.gprmax_python,
        "gprmax_source": args.gprmax_root,
        "gpu_index": args.gpu,
    }.items():
        if value is not None:
            payload[key] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
