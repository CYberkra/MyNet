#!/usr/bin/env python3
"""Fingerprint the installed gprMax source and matching local manual."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


FILES = (
    "gprMax/_version.py",
    "gprMax/input_cmds_geometry.py",
    "gprMax/fractals.py",
    "gprMax/fractals_generate_ext.pyx",
    "gprMax/materials.py",
    "gprMax/grid.py",
    "gprMax/pml.py",
    "gprMax/model_build_run.py",
    "gprMax/utilities.py",
    "tools/outputfiles_merge.py",
    "docs/source/input.rst",
    "docs/source/gprmodelling.rst",
    "docs/source/gpu.rst",
    "docs/source/output.rst",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(root: Path) -> dict:
    root = root.resolve()
    version_path = root / "gprMax" / "_version.py"
    version_text = version_path.read_text(encoding="utf-8") if version_path.is_file() else ""
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)", version_text, re.MULTILINE)
    codename_match = re.search(r"^codename\s*=\s*['\"]([^'\"]+)", version_text, re.MULTILINE)
    files = {}
    missing = []
    for relative in FILES:
        path = root / relative
        if path.is_file():
            files[relative] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        else:
            missing.append(relative)
    return {
        "root": str(root),
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "version": version_match.group(1) if version_match else None,
        "codename": codename_match.group(1) if codename_match else None,
        "files": files,
        "missing": missing,
        "ok": version_match is not None and not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = fingerprint(args.root)
    output = json.dumps(result, indent=2)
    print(output)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
