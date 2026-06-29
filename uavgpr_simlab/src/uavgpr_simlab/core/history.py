from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .job_registry import registry_dir, safe_name, STATUS_DONE, STATUS_FAILED, STATUS_RUNNING

STATUS_STALE_RUNNING = "stale_running"


@dataclass
class HistoryRecord:
    job_id: str
    status: str
    case_id: str
    variant: str
    n_traces: int
    input_file: str
    marker_file: str
    time_local: str
    geometry_only: bool = False
    fingerprint: str = ""
    returncode: str = ""
    output_dir: str = ""
    log_hint: str = ""
    available_traces: int = 0
    has_model_preview: bool = False
    has_bscan_preview: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pid_alive(pid: object) -> bool:
    try:
        p = int(pid)
    except Exception:
        return False
    if p <= 0:
        return False
    try:
        os.kill(p, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _resolved_status(marker_status: str, path: Path, data: Dict[str, Any], stale_after_s: float = 24 * 3600.0) -> str:
    status = str(data.get("status") or marker_status)
    if status != STATUS_RUNNING:
        return status
    pid = data.get("supervisor_pid") or data.get("pid")
    marker_host = str(data.get("host") or "")
    current_host = socket.gethostname()
    # PID checks are only reliable on the host where the wrapper process runs.
    # For shared HPC filesystems, a GUI on a login node must not mark a compute-node
    # task stale merely because its PID is not visible locally.
    if pid is not None and (not marker_host or marker_host == current_host) and not _pid_alive(pid):
        return STATUS_STALE_RUNNING
    if pid is None or (marker_host and marker_host != current_host):
        try:
            age = time.time() - path.stat().st_mtime
            if age > stale_after_s:
                return STATUS_STALE_RUNNING
        except Exception:
            pass
    return status


def scan_simulation_history(workspace: str | Path, include_failed: bool = True, include_done: bool = True, include_running: bool = True) -> List[HistoryRecord]:
    """Scan the resumable registry and return all known simulation runs.

    This is intentionally filesystem based, so users can move/copy a workspace
    and still recover its run history without a database server.
    """
    root = Path(workspace)
    records: List[HistoryRecord] = []
    candidates: List[tuple[str, Path]] = []
    if include_done:
        candidates.extend((STATUS_DONE, p) for p in (root / "jobs" / "done").glob("*.json"))
    if include_running:
        candidates.extend((STATUS_RUNNING, p) for p in (root / "jobs" / "running").glob("*.json"))
    if include_failed:
        candidates.extend((STATUS_FAILED, p) for p in (root / "jobs" / "failed").glob("*.json"))
    for status, path in candidates:
        data = _read_json(path)
        post = data.get("postprocess") or {}
        if isinstance(post, dict):
            output_dir = str(post.get("output_dir") or post.get("out") or "")
        else:
            output_dir = ""
        records.append(HistoryRecord(
            job_id=str(data.get("job_id") or path.stem),
            status=_resolved_status(status, path, data),
            case_id=str(data.get("case_id") or ""),
            variant=str(data.get("variant") or ""),
            n_traces=int(data.get("n_traces") or 0),
            input_file=str(data.get("input_file") or ""),
            marker_file=str(path.resolve()),
            time_local=str(data.get("time_local") or ""),
            geometry_only=bool(data.get("geometry_only") or False),
            fingerprint=str(data.get("fingerprint") or ""),
            returncode=str(data.get("returncode") if data.get("returncode") is not None else ""),
            output_dir=output_dir,
            log_hint=str(data.get("log") or ""),
        ))
    records.sort(key=lambda r: (r.time_local, r.job_id), reverse=True)
    return records


def export_history_csv(workspace: str | Path, out_csv: str | Path | None = None) -> Dict[str, Any]:
    records = scan_simulation_history(workspace)
    out = Path(out_csv) if out_csv else Path(workspace) / "reports" / "simulation_history.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(HistoryRecord("", "", "", "", 0, "", "", "").to_dict().keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in records:
            wr.writerow(r.to_dict())
    return {"history_csv": str(out.resolve()), "count": len(records)}


def _is_within_workspace(path: Path, workspace: Path) -> bool:
    try:
        path.resolve().relative_to(workspace.resolve())
        return True
    except ValueError:
        return False


def _safe_remove_path(path: Path, workspace: Path) -> Optional[str]:
    try:
        rp = path.resolve()
        wr = workspace.resolve()
        if not _is_within_workspace(rp, wr):
            return f"skip outside workspace: {path}"
        if rp == wr:
            return f"refuse to remove workspace root: {rp}"
        if rp.is_dir():
            shutil.rmtree(rp)
            return f"removed dir: {rp}"
        if rp.exists():
            rp.unlink()
            return f"removed file: {rp}"
        return f"not exists: {rp}"
    except Exception as exc:
        return f"error removing {path}: {exc}"


def delete_history_record(
    workspace: str | Path,
    job_id: str | None = None,
    marker_file: str | Path | None = None,
    delete_outputs: bool = False,
    delete_case_model: bool = False,
) -> Dict[str, Any]:
    """Delete one run record and optionally its generated outputs/model files.

    The function refuses to delete paths outside the selected workspace.
    """
    root = Path(workspace)
    marker: Optional[Path] = Path(marker_file) if marker_file else None
    if marker is not None:
        try:
            marker_resolved = marker.resolve()
        except Exception:
            marker_resolved = marker
        if not _is_within_workspace(marker_resolved, root):
            raise PermissionError(f"history marker is outside workspace: {marker}")
        marker = marker_resolved
    if marker is None and job_id:
        for sub in ["done", "failed", "running"]:
            p = root / "jobs" / sub / f"{safe_name(job_id)}.json"
            if p.exists():
                marker = p
                break
    if marker is None or not marker.exists():
        raise FileNotFoundError(f"history marker not found: job_id={job_id}, marker={marker_file}")
    data = _read_json(marker)
    messages: List[str] = []
    input_file = Path(str(data.get("input_file") or ""))
    if delete_outputs:
        jid = safe_name(str(data.get("job_id") or marker.stem))
        for suffix in ["model.png", "bscan.png"]:
            msg = _safe_remove_path(root / "previews" / "history" / f"{jid}_{suffix}", root)
            if msg:
                messages.append(msg)
        post = data.get("postprocess") or {}
        if isinstance(post, dict):
            for key in ["output_dir", "out"]:
                if post.get(key):
                    msg = _safe_remove_path(Path(str(post[key])), root)
                    if msg:
                        messages.append(msg)
        # Common QC output path fallback.
        case = safe_name(str(data.get("case_id") or ""))
        variant = safe_name(str(data.get("variant") or ""))
        if case and variant:
            msg = _safe_remove_path(root / "outputs" / "gprmax_qc" / f"{case}_{variant}", root)
            if msg:
                messages.append(msg)
    if delete_case_model and input_file.exists():
        # Remove only the case directory containing .in files, never the whole workspace.
        case_dir = input_file.parent
        msg = _safe_remove_path(case_dir, root)
        if msg:
            messages.append(msg)
    messages.append(_safe_remove_path(marker, root) or "")
    return {"deleted_marker": str(marker), "messages": messages, "time_local": time.strftime("%Y-%m-%d %H:%M:%S")}


def delete_history_bulk(workspace: str | Path, statuses: Sequence[str] = (STATUS_FAILED,), delete_outputs: bool = False) -> Dict[str, Any]:
    root = Path(workspace)
    records = scan_simulation_history(root, include_done=True, include_failed=True, include_running=True)
    selected = [r for r in records if r.status in set(statuses)]
    reps = []
    for r in selected:
        reps.append(delete_history_record(root, marker_file=r.marker_file, delete_outputs=delete_outputs))
    return {"deleted": len(reps), "reports": reps}
