from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uavgpr_simlab.core.history import HistoryRecord, delete_history_record, export_history_csv, scan_simulation_history
from uavgpr_simlab.core.visual_history import (
    HistoryPreview,
    build_history_preview,
    find_label_json_for_input,
    load_bscan_for_history,
)
from uavgpr_simlab.services.easy_batch_service import count_manifest_cases
from uavgpr_simlab.core.runner import resolve_manifest_path

ALL_STATUS_FILTER = "全部"


@dataclass(frozen=True)
class HistoryListEntry:
    """One row of data needed by the easy GUI history list."""

    record: HistoryRecord
    preview: HistoryPreview
    marker: dict[str, Any]


@dataclass(frozen=True)
class HistoryDetail:
    """Data for the selected history preview panel."""

    label_json: Path | None
    bscan: np.ndarray | None
    bscan_meta: dict[str, Any]
    detail_text: str


@dataclass(frozen=True)
class ProjectSummary:
    """Home-page project status summary independent of Qt widgets."""

    models: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "models": self.models,
            "running": self.running,
            "done": self.done,
            "failed": self.failed,
            "skipped": self.skipped,
        }



_SCENEWORLD_VARIANT_BSCAN_FIELD = {
    "raw": "raw_bscan_npy",
    "target_only": "target_bscan_npy",
    "background_only": "background_bscan_npy",
    "clutter_only": "clutter_bscan_npy",
    "air_only": "air_bscan_npy",
    "clutter_gt": "clutter_gt_bscan_npy",
}


def _scan_sceneworld_manifest_records(workspace: str | Path) -> list[tuple[HistoryRecord, dict[str, Any]]]:
    """Build history-like records from SceneWorld manifest/QC outputs.

    SceneWorld batch datasets are run by the unified runner and do not use the
    old job-marker registry.  This adapter makes completed/failed variant
    outputs visible in the same history page without introducing a database.
    """
    root = Path(workspace).expanduser()
    manifests = sorted((root / "datasets").glob("*_manifest.csv"))
    if not manifests:
        return []
    manifest = manifests[-1]
    records: list[tuple[HistoryRecord, dict[str, Any]]] = []
    try:
        with manifest.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cid = row.get("case_id", "")
        variant = row.get("variant", "")
        if not cid or not variant or (cid, variant) in seen:
            continue
        seen.add((cid, variant))
        qc_path = resolve_manifest_path(row.get("bscan_qc_report_json", f"models/{cid}/bscan_qc_report.json"), manifest)
        qc = read_marker_json(qc_path) if qc_path.exists() else {}
        variant_qc = (qc.get("variants") or {}).get(variant, {}) if isinstance(qc, dict) else {}
        status_raw = row.get("bscan_status") or qc.get("status") or "not_run"
        if variant_qc.get("status") in {"success", "failed"}:
            status_raw = variant_qc.get("status")
        if status_raw == "success":
            status = "done"
        elif status_raw == "failed":
            status = "failed"
        elif status_raw in {"running", "stale_running"}:
            status = str(status_raw)
        else:
            status = "pending"
        input_file = resolve_manifest_path(row.get("input_file", ""), manifest)
        field = _SCENEWORLD_VARIANT_BSCAN_FIELD.get(variant, "bscan_npy")
        bscan_path = resolve_manifest_path(row.get(field) or row.get("bscan_npy", ""), manifest)
        marker = {
            "schema": "uavgpr_simlab.history.sceneworld_virtual.v1",
            "source": "sceneworld_manifest",
            "manifest": str(manifest),
            "workspace": str(root),
            "case_dir": str((root / "models" / cid).resolve()),
            "family": row.get("family", ""),
            "bscan_npy": str(bscan_path),
            "qc_report_json": str(qc_path),
            "variant_qc": variant_qc,
            "bscan_error": row.get("bscan_error", ""),
        }
        n_traces = int(float(row.get("n_traces") or row.get("trace_count") or 0))
        rec = HistoryRecord(
            job_id=f"sceneworld_{cid}_{variant}",
            status=status,
            case_id=cid,
            variant=variant,
            n_traces=n_traces,
            input_file=str(input_file),
            marker_file=str(qc_path),
            time_local="SceneWorld",
            output_dir=str(bscan_path.parent) if str(bscan_path) else "",
            available_traces=n_traces if status == "done" else 0,
            has_bscan_preview=bscan_path.exists(),
        )
        records.append((rec, marker))

    # Add one derived clutter_gt record per case when the contract file exists or
    # the case QC reports it.  This gives users a direct history target for the
    # PGDA-CSNet label candidate instead of hiding it behind raw/target rows.
    case_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        cid = row.get("case_id", "")
        if cid and cid not in case_rows:
            case_rows[cid] = row
    for cid, row in case_rows.items():
        cdir = root / "models" / cid
        qc_path = resolve_manifest_path(row.get("bscan_qc_report_json", f"models/{cid}/bscan_qc_report.json"), manifest)
        qc = read_marker_json(qc_path) if qc_path.exists() else {}
        cg_path = resolve_manifest_path(row.get("clutter_gt_bscan_npy", f"models/{cid}/outputs/clutter_gt_bscan.npy"), manifest)
        if not cg_path.exists() and not qc.get("clutter_gt_generated"):
            continue
        status = "done" if qc.get("clutter_gt_generated") and qc.get("status") == "success" else ("failed" if qc.get("status") == "failed" else "running")
        marker = {
            "schema": "uavgpr_simlab.history.sceneworld_virtual.v1",
            "source": "sceneworld_manifest",
            "manifest": str(manifest),
            "workspace": str(root),
            "case_dir": str(cdir.resolve()),
            "family": row.get("family", ""),
            "bscan_npy": str(cg_path),
            "qc_report_json": str(qc_path),
            "variant_qc": qc.get("clutter_gt", {}),
            "bscan_error": qc.get("clutter_gt_error", ""),
        }
        rec = HistoryRecord(
            job_id=f"sceneworld_{cid}_clutter_gt",
            status=status,
            case_id=cid,
            variant="clutter_gt",
            n_traces=int(float(row.get("n_traces") or row.get("trace_count") or 0)),
            input_file=str(resolve_manifest_path(row.get("input_file", ""), manifest)),
            marker_file=str(qc_path),
            time_local="SceneWorld",
            output_dir=str(cg_path.parent),
            available_traces=int(float(row.get("n_traces") or row.get("trace_count") or 0)) if status == "done" else 0,
            has_bscan_preview=cg_path.exists(),
        )
        records.append((rec, marker))
    return records

def read_marker_json(path: str | Path) -> dict[str, Any]:
    """Best-effort marker reader for GUI history previews."""

    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}




def load_case_variant_bscan(case_dir: str | Path, variant: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load a canonical SceneWorld B-scan for a case/variant.

    This is used by the history page comparison modes.  It reads only files
    inside a case directory and never infers training readiness.
    """
    cdir = Path(case_dir)
    alias = {
        "raw": "raw_bscan.npy",
        "target_only": "target_only_bscan.npy",
        "background_only": "background_only_bscan.npy",
        "clutter_only": "clutter_only_bscan.npy",
        "air_only": "air_only_bscan.npy",
        "clutter_gt": "clutter_gt_bscan.npy",
    }
    p = cdir / "outputs" / alias.get(variant, f"{variant}_bscan.npy")
    meta: dict[str, Any] = {"variant": variant, "path": str(p), "exists": p.exists()}
    if not p.exists():
        return None, meta
    try:
        arr = np.asarray(np.load(p), dtype=float)
        meta.update({"shape": list(arr.shape), "finite_count": int(np.isfinite(arr).sum())})
        return arr, meta
    except Exception as exc:
        meta["error"] = repr(exc)
        return None, meta


def summarize_project(workspace: str | Path, manifest: str | Path | None = None) -> ProjectSummary:
    """Compute the status-card counts shown on the home page."""

    counts = {"models": count_manifest_cases(manifest), "running": 0, "done": 0, "failed": 0, "skipped": 0}
    try:
        for record in scan_simulation_history(workspace):
            if record.status == "running":
                counts["running"] += 1
            elif record.status == "done":
                counts["done"] += 1
            elif record.status == "failed":
                counts["failed"] += 1
            elif record.status == "stale_running":
                counts["failed"] += 1
            elif record.status == "skipped":
                counts["skipped"] += 1
    except Exception:
        pass
    return ProjectSummary(**counts)


def scan_history_entries(
    workspace: str | Path,
    *,
    status_filter: str = ALL_STATUS_FILTER,
    limit: int = 300,
    make_png: bool = True,
) -> list[HistoryListEntry]:
    """Scan history records and build preview metadata for list rendering."""

    root = Path(workspace).expanduser()
    registry_records = [(record, read_marker_json(record.marker_file)) for record in scan_simulation_history(root)]
    sceneworld_records = _scan_sceneworld_manifest_records(root)
    all_records = registry_records + sceneworld_records
    if status_filter != ALL_STATUS_FILTER:
        all_records = [(record, marker) for record, marker in all_records if record.status == status_filter]
    entries: list[HistoryListEntry] = []
    for record, marker in all_records[: max(0, int(limit))]:
        preview = build_history_preview(record, root, marker_data=marker, make_png=make_png)
        entries.append(HistoryListEntry(record=record, preview=preview, marker=marker))
    return entries


def build_history_detail(entry: HistoryListEntry, workspace: str | Path, *, time_window_ns: float = 700.0) -> HistoryDetail:
    """Load detail-panel inputs for the selected history list entry."""

    record = entry.record
    preview = entry.preview
    label = Path(preview.label_json) if preview.label_json else find_label_json_for_input(record.input_file, workspace, record.case_id)
    bscan, meta = load_bscan_for_history(record.input_file, marker_data=entry.marker)
    available_traces = int(bscan.shape[1]) if bscan is not None and getattr(bscan, "ndim", 0) == 2 else int(preview.available_traces)
    detail_text = (
        f"状态：{record.status}    模型：{record.case_id}    仿真内容：{record.variant}\n"
        f"完成道数：{available_traces} / {record.n_traces} 道    时间：{record.time_local or '—'}\n"
        f"输入文件：{record.input_file}"
    )
    meta = dict(meta)
    meta.setdefault("time_window_ns", float(time_window_ns))
    return HistoryDetail(label_json=label, bscan=bscan, bscan_meta=meta, detail_text=detail_text)


def export_history_report(workspace: str | Path) -> dict[str, Any]:
    """Export history records to CSV."""

    return export_history_csv(Path(workspace).expanduser())


def delete_history_entry(workspace: str | Path, marker_file: str | Path, *, delete_outputs: bool) -> dict[str, Any]:
    """Delete one history record through the core history API."""

    return delete_history_record(Path(workspace).expanduser(), marker_file=marker_file, delete_outputs=delete_outputs)
