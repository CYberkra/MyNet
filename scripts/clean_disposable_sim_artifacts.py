#!/usr/bin/env python3
"""Report or remove disposable raw gprMax artifacts in approved local roots.

This utility never touches source decks, labels, canonical releases, measured
data, or manifests. It is dry-run by default and requires ``--execute`` for
deletion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    ROOT / "data" / "simulations" / "v2" / "01_solver_runs",
    ROOT / "workspace",
]
DISPOSABLE_SUFFIXES = {".out", ".h5", ".hdf5", ".vti", ".log"}


def find_disposable(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in DISPOSABLE_SUFFIXES:
                found.append(path)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    roots = [path.resolve() for path in (args.root or DEFAULT_ROOTS)]
    files = find_disposable(roots)
    bytes_total = sum(path.stat().st_size for path in files)
    if args.execute:
        for path in files:
            path.unlink()
    report = {
        "mode": "execute" if args.execute else "dry_run",
        "roots": [str(path) for path in roots],
        "file_count": len(files),
        "bytes_total": bytes_total,
        "files": [str(path) for path in files],
        "protected": [
            "source decks", "labels", "canonical releases", "manifests",
            "original measured data",
        ],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
