#!/usr/bin/env python3
"""Audit and optionally delete transient gprMax VTI geometry views."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cleanup(
    scan_root: Path,
    report_path: Path,
    *,
    delete: bool,
    repo_root: Path = ROOT,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    scan_root = scan_root.resolve()
    report_path = report_path.resolve()
    if not scan_root.is_dir() or not _inside(scan_root, repo_root):
        raise ValueError("scan root must be an existing directory inside the repository")
    if not _inside(report_path, repo_root):
        raise ValueError("report path must stay inside the repository")

    artifacts: list[dict[str, object]] = []
    for path in sorted(scan_root.rglob("*.vti")):
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_file() or not _inside(resolved, scan_root):
            raise ValueError(f"unsafe VTI path: {path}")
        record: dict[str, object] = {
            "path": resolved.relative_to(repo_root).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
            "deleted": False,
        }
        if delete:
            resolved.unlink()
            record["deleted"] = not resolved.exists()
        artifacts.append(record)

    payload: dict[str, object] = {
        "schema": "gprmax_vti_cleanup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scan_root": scan_root.relative_to(repo_root).as_posix(),
        "policy": "VTI is transient geometry visualization, not solver or training data",
        "delete_requested": delete,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(item["bytes"]) for item in artifacts),
        "all_deleted": bool(delete and all(bool(item["deleted"]) for item in artifacts)),
        "artifacts": artifacts,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=ROOT / "data" / "simulations" / "v2",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    result = cleanup(args.scan_root, args.report, delete=args.delete)
    print(json.dumps({key: result[key] for key in ("artifact_count", "total_bytes", "all_deleted")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
