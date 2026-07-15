from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.cleanup_gprmax_geometry_views import cleanup


def test_cleanup_records_hashes_before_deleting(tmp_path: Path) -> None:
    scan_root = tmp_path / "data"
    scan_root.mkdir()
    first = scan_root / "case_a" / "geometry_full.vti"
    second = scan_root / "case_b" / "geometry_control.vti"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"full geometry")
    second.write_bytes(b"control geometry")
    report = tmp_path / "reports" / "cleanup.json"

    result = cleanup(scan_root, report, delete=True, repo_root=tmp_path)

    assert result["artifact_count"] == 2
    assert result["all_deleted"] is True
    assert not first.exists() and not second.exists()
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["artifacts"][0]["sha256"] == hashlib.sha256(b"full geometry").hexdigest()


def test_cleanup_dry_run_preserves_views(tmp_path: Path) -> None:
    scan_root = tmp_path / "data"
    scan_root.mkdir()
    view = scan_root / "geometry.vti"
    view.write_bytes(b"view")
    result = cleanup(
        scan_root,
        tmp_path / "reports" / "dry_run.json",
        delete=False,
        repo_root=tmp_path,
    )
    assert result["all_deleted"] is False
    assert view.is_file()
