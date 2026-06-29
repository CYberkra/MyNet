from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import AppConfig
from .postprocess import export_gprmax_bscan_for_input
from .runner import GprMaxTask, run_task, tasks_from_manifest

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_RUNNING = "running"
STATUS_SKIPPED = "skipped"


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text)).strip("_") or "job"


def job_fingerprint(input_file: str | Path, n_traces: int, variant: str = "", extra: Optional[Dict[str, Any]] = None) -> str:
    p = Path(input_file)
    if p.exists():
        content_hash = sha256_file(p)
    else:
        content_hash = sha256_text(str(p.resolve()))
    payload = {
        "input_file_name": p.name,
        "input_sha256": content_hash,
        "n_traces": int(n_traces or 1),
        "variant": str(variant or ""),
        "extra": extra or {},
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def job_id_for(input_file: str | Path, case_id: str = "", variant: str = "", n_traces: int = 1) -> str:
    fp = job_fingerprint(input_file, n_traces, variant)
    prefix = safe_name(f"{case_id}_{variant}" if case_id else f"{Path(input_file).stem}_{variant}")
    return f"{prefix}_{fp[:12]}"


def registry_dir(workspace: str | Path) -> Path:
    d = Path(workspace) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "done").mkdir(parents=True, exist_ok=True)
    (d / "failed").mkdir(parents=True, exist_ok=True)
    (d / "running").mkdir(parents=True, exist_ok=True)
    return d


def marker_path(workspace: str | Path, job_id: str, status: str = STATUS_DONE) -> Path:
    sub = "done" if status == STATUS_DONE else "failed" if status == STATUS_FAILED else "running" if status == STATUS_RUNNING else status
    return registry_dir(workspace) / sub / f"{safe_name(job_id)}.json"




def remove_marker_if_exists(workspace: str | Path, job_id: str, status: str) -> None:
    """Best-effort removal of one registry marker.

    Used when a later successful attempt supersedes an older failed/running marker.
    """
    p = marker_path(workspace, job_id, status)
    try:
        p.unlink(missing_ok=True)
    except TypeError:
        if p.exists():
            p.unlink()

def read_marker(workspace: str | Path, job_id: str) -> Optional[Dict[str, Any]]:
    p = marker_path(workspace, job_id, STATUS_DONE)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_done(workspace: str | Path, job_id: str, fingerprint: str | None = None) -> bool:
    data = read_marker(workspace, job_id)
    if not data:
        return False
    if data.get("status") != STATUS_DONE:
        return False
    if fingerprint and data.get("fingerprint") != fingerprint:
        return False
    return True


def write_marker(workspace: str | Path, job_id: str, status: str, payload: Dict[str, Any]) -> Path:
    p = marker_path(workspace, job_id, status)
    payload = dict(payload)
    payload.setdefault("job_id", job_id)
    payload.setdefault("status", status)
    payload.setdefault("time_local", time.strftime("%Y-%m-%d %H:%M:%S"))
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def expected_output_files(input_file: str | Path, n_traces: int) -> List[str]:
    """Return common gprMax output candidates for status display.

    gprMax produces separate .out files for B-scan traces. Many workflows then
    merge them into one file. We therefore return candidates instead of treating
    this as a strict completion test; the registry marker is the primary source
    of truth for resume/skip behavior.
    """

    p = Path(input_file)
    stem = p.with_suffix("")
    n = max(1, int(n_traces or 1))
    candidates: List[Path] = []
    if n == 1:
        candidates.append(stem.with_suffix(".out"))
    else:
        candidates.append(stem.with_name(stem.name + "_merged.out"))
        candidates.append(stem.with_suffix(".out"))
        candidates.extend(stem.with_name(f"{stem.name}{i}.out") for i in range(1, min(n, 3) + 1))
    return [str(x) for x in candidates]


@dataclass
class JobRecord:
    job_id: str
    fingerprint: str
    status: str
    case_id: str
    variant: str
    split: str
    input_file: str
    n_traces: int
    expected_outputs: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _write_job_csv(records: Sequence[JobRecord], out_csv: str | Path) -> Path:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(JobRecord("", "", "", "", "", "", "", 1, "").to_dict().keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in records:
            wr.writerow(r.to_dict())
    return out


def build_job_records(
    manifest_csv: str | Path,
    workspace: str | Path,
    variants: Optional[Sequence[str]] = None,
    max_tasks: int = 0,
    skip_completed: bool = True,
) -> List[JobRecord]:
    tasks = tasks_from_manifest(manifest_csv, variants=variants, limit=max_tasks)
    records: List[JobRecord] = []
    for task in tasks:
        fp = job_fingerprint(task.input_file, task.n_traces, task.variant)
        jid = job_id_for(task.input_file, str(task.case_id), task.variant, task.n_traces)
        done = is_done(workspace, jid, fp)
        status = STATUS_SKIPPED if (skip_completed and done) else STATUS_PENDING
        reason = "done marker exists" if done else "not completed yet"
        records.append(JobRecord(
            job_id=jid,
            fingerprint=fp,
            status=status,
            case_id=str(task.case_id),
            variant=str(task.variant),
            split="",
            input_file=str(task.input_file),
            n_traces=int(task.n_traces),
            expected_outputs=";".join(expected_output_files(task.input_file, task.n_traces)),
            reason=reason,
        ))
    return records


def write_job_plan(
    manifest_csv: str | Path,
    workspace: str | Path,
    out_csv: str | Path,
    variants: Optional[Sequence[str]] = None,
    max_tasks: int = 0,
    skip_completed: bool = True,
) -> Dict[str, Any]:
    records = build_job_records(manifest_csv, workspace, variants=variants, max_tasks=max_tasks, skip_completed=skip_completed)
    out = _write_job_csv(records, out_csv)
    counts: Dict[str, int] = {}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = {"job_plan_csv": str(out.resolve()), "total": len(records), "counts": counts}
    summary_path = Path(out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path.resolve())
    return summary


def run_registered_task(
    cfg: AppConfig,
    task: GprMaxTask,
    workspace: str | Path,
    skip_completed: bool = True,
    force: bool = False,
    postprocess: bool = True,
    log_dir: str | Path | None = None,
    postprocess_out_dir: str | Path | None = None,
) -> Dict[str, Any]:
    run_extra = {"geometry_only": True} if task.geometry_only else None
    fp = job_fingerprint(task.input_file, task.n_traces, task.variant, extra=run_extra)
    jid = job_id_for(task.input_file, str(task.case_id), task.variant, task.n_traces)
    if task.geometry_only:
        jid = f"{jid}_geometry"
    old = read_marker(workspace, jid)
    if skip_completed and old and old.get("fingerprint") == fp and not force:
        return {"status": STATUS_SKIPPED, "job_id": jid, "fingerprint": fp, "marker": str(marker_path(workspace, jid, STATUS_DONE).resolve()), "reason": "already completed"}

    running_marker = write_marker(workspace, jid, STATUS_RUNNING, {
        "status": STATUS_RUNNING,
        "fingerprint": fp,
        "input_file": str(Path(task.input_file).resolve()),
        "case_id": task.case_id,
        "variant": task.variant,
        "n_traces": int(task.n_traces),
        "geometry_only": bool(task.geometry_only),
        "expected_outputs": expected_output_files(task.input_file, task.n_traces),
        "pid": None,
        "supervisor_pid": os.getpid(),
        "host": socket.gethostname(),
    })

    log_root = Path(log_dir) if log_dir else Path(workspace) / "logs"
    try:
        rc = run_task(cfg, task, log_root)
        post_rep: Dict[str, Any] | None = None
        post_error: str | None = None
        if rc == 0 and postprocess and (not task.geometry_only):
            out = Path(postprocess_out_dir) if postprocess_out_dir else Path(workspace) / "outputs" / "gprmax_qc" / f"{safe_name(str(task.case_id))}_{safe_name(task.variant)}"
            try:
                post_rep = export_gprmax_bscan_for_input(task.input_file, out, stem=f"{safe_name(str(task.case_id))}_{safe_name(task.variant)}", time_window_ns=float(cfg.radar.time_window_ns))
            except Exception as exc:
                post_error = repr(exc)
        marker = write_marker(workspace, jid, STATUS_DONE, {
            "status": STATUS_DONE,
            "fingerprint": fp,
            "input_file": str(Path(task.input_file).resolve()),
            "case_id": task.case_id,
            "variant": task.variant,
            "n_traces": int(task.n_traces),
            "geometry_only": bool(task.geometry_only),
            "returncode": rc,
            "expected_outputs": expected_output_files(task.input_file, task.n_traces),
            "postprocess": post_rep,
            "postprocess_error": post_error,
            "host": socket.gethostname(),
        })
        remove_marker_if_exists(workspace, jid, STATUS_FAILED)
        try:
            marker_path(workspace, jid, STATUS_RUNNING).unlink(missing_ok=True)
        except TypeError:
            rp = marker_path(workspace, jid, STATUS_RUNNING)
            if rp.exists():
                rp.unlink()
        return {"status": STATUS_DONE, "job_id": jid, "fingerprint": fp, "marker": str(marker.resolve()), "returncode": rc, "postprocess": post_rep, "postprocess_error": post_error}
    except Exception as exc:
        marker = write_marker(workspace, jid, STATUS_FAILED, {
            "status": STATUS_FAILED,
            "fingerprint": fp,
            "input_file": str(Path(task.input_file).resolve()),
            "case_id": task.case_id,
            "variant": task.variant,
            "n_traces": int(task.n_traces),
            "geometry_only": bool(task.geometry_only),
            "error": repr(exc),
            "host": socket.gethostname(),
        })
        try:
            marker_path(workspace, jid, STATUS_RUNNING).unlink(missing_ok=True)
        except TypeError:
            rp = marker_path(workspace, jid, STATUS_RUNNING)
            if rp.exists():
                rp.unlink()
        raise RuntimeError(f"registered gprMax task failed; marker={marker}; error={exc}") from exc
