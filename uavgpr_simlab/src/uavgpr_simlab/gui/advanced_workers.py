from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import QThread, Signal

from uavgpr_simlab.core.config import AppConfig
from uavgpr_simlab.core.job_registry import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_RUNNING,
    is_done,
    job_fingerprint,
    job_id_for,
    marker_path,
    remove_marker_if_exists,
    write_marker,
)
from uavgpr_simlab.core.postprocess import export_gprmax_bscan_for_input, merge_available_bscan_for_input
from uavgpr_simlab.core.runner import (
    GprMaxTask,
    build_gprmax_command,
    command_to_string,
    options_from_config_task,
)


class GenericWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.done.emit(self.func(*self.args, **self.kwargs))
        except Exception:
            self.failed.emit(traceback.format_exc())


class LiveQueueWorker(QThread):
    log = Signal(str)
    preview = Signal(object, str, float)
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: AppConfig, tasks: Sequence[GprMaxTask], log_dir: str | Path, geometry_only: bool = False, auto_export: bool = True, skip_completed: bool = True, force: bool = False):
        super().__init__()
        self.cfg = cfg
        self.tasks = list(tasks)
        self.log_dir = Path(log_dir)
        self.geometry_only = geometry_only
        self.auto_export = auto_export
        self.skip_completed = bool(skip_completed)
        self.force = bool(force)
        self._cancel = False
        self._proc: subprocess.Popen | None = None

    def cancel(self) -> None:
        self._cancel = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _emit_preview(self, task: GprMaxTask) -> None:
        merged = merge_available_bscan_for_input(task.input_file)
        if merged is None:
            return
        bscan, _ = merged
        self.preview.emit(bscan, f"{task.case_id} | {task.variant} | {Path(task.input_file).name}", float(self.cfg.radar.time_window_ns))

    def _run_task(self, task: GprMaxTask) -> int:
        task.geometry_only = bool(self.geometry_only)
        workspace = Path(self.cfg.runtime.project_root)
        fp = job_fingerprint(task.input_file, task.n_traces, task.variant, extra={"geometry_only": True} if task.geometry_only else None)
        jid = job_id_for(task.input_file, str(task.case_id), task.variant, task.n_traces)
        if task.geometry_only:
            jid = f"{jid}_geometry"
        if self.skip_completed and (not self.force) and is_done(workspace, jid, fp):
            self.log.emit(f"[SKIP] {task.case_id} {task.variant} already completed: {jid}")
            return 0
        running_marker = write_marker(workspace, jid, STATUS_RUNNING, {
            "status": STATUS_RUNNING,
            "fingerprint": fp,
            "input_file": str(Path(task.input_file).resolve()),
            "case_id": task.case_id,
            "variant": task.variant,
            "n_traces": int(task.n_traces),
            "geometry_only": bool(task.geometry_only),
            "log": "",
            "supervisor_pid": os.getpid(),
            "host": socket.gethostname(),
        })
        options = options_from_config_task(self.cfg, task)
        cmd = build_gprmax_command(options)
        env = os.environ.copy()
        if options.openmp_threads and int(options.openmp_threads) > 0:
            env["OMP_NUM_THREADS"] = str(int(options.openmp_threads))
        cwd = options.gprmax_root or str(Path(options.input_file).parent)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"gprmax_{task.case_id}_{task.variant}_{stamp}.log"
        self.log.emit(f"\n[TASK] {task.case_id} {task.variant}")
        self.log.emit(f"[CMD] {command_to_string(cmd)}")
        self.log.emit(f"[CWD] {cwd}")
        self.log.emit(f"[LOG] {log_path}")
        write_marker(workspace, jid, STATUS_RUNNING, {
            "status": STATUS_RUNNING,
            "fingerprint": fp,
            "input_file": str(Path(task.input_file).resolve()),
            "case_id": task.case_id,
            "variant": task.variant,
            "n_traces": int(task.n_traces),
            "geometry_only": bool(task.geometry_only),
            "log": str(log_path.resolve()),
            "supervisor_pid": os.getpid(),
            "host": socket.gethostname(),
        })
        last_preview = time.monotonic()
        with log_path.open("w", encoding="utf-8", errors="replace") as lf:
            lf.write(command_to_string(cmd) + "\n")
            self._proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            write_marker(workspace, jid, STATUS_RUNNING, {
                "status": STATUS_RUNNING,
                "fingerprint": fp,
                "input_file": str(Path(task.input_file).resolve()),
                "case_id": task.case_id,
                "variant": task.variant,
                "n_traces": int(task.n_traces),
                "geometry_only": bool(task.geometry_only),
                "log": str(log_path.resolve()),
                "pid": int(self._proc.pid),
                "supervisor_pid": os.getpid(),
                "host": socket.gethostname(),
            })
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                lf.write(line)
                self.log.emit(line.rstrip("\n"))
                now = time.monotonic()
                if now - last_preview > 1.25:
                    self._emit_preview(task)
                    last_preview = now
                if self._cancel:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                    break
            rc = self._proc.wait()
            lf.write(f"\n[EXIT] returncode={rc}\n")
        self._emit_preview(task)
        post_rep = None
        if rc == 0 and (not self.geometry_only) and self.auto_export:
            try:
                out = Path(self.cfg.runtime.project_root) / "outputs" / "gprmax_qc" / f"{task.case_id}_{task.variant}"
                post_rep = export_gprmax_bscan_for_input(task.input_file, out, stem=f"{task.case_id}_{task.variant}", time_window_ns=float(self.cfg.radar.time_window_ns))
                self.log.emit(f"[EXPORT] {json.dumps({'out': str(out), 'shape': post_rep.get('bscan_shape')}, ensure_ascii=False)}")
            except Exception as exc:
                self.log.emit(f"[EXPORT-WARN] {exc}")
        if rc != 0:
            marker = write_marker(Path(self.cfg.runtime.project_root), jid, STATUS_FAILED, {
                "status": STATUS_FAILED,
                "fingerprint": fp,
                "input_file": str(Path(task.input_file).resolve()),
                "case_id": task.case_id,
                "variant": task.variant,
                "n_traces": int(task.n_traces),
                "geometry_only": bool(task.geometry_only),
                "returncode": rc,
                "log": str(log_path.resolve()),
                "host": socket.gethostname(),
            })
            try:
                marker_path(Path(self.cfg.runtime.project_root), jid, STATUS_RUNNING).unlink(missing_ok=True)
            except TypeError:
                rp = marker_path(Path(self.cfg.runtime.project_root), jid, STATUS_RUNNING)
                if rp.exists():
                    rp.unlink()
            self.log.emit(f"[MARK] failed marker written: {marker}")
        if rc == 0:
            marker = write_marker(Path(self.cfg.runtime.project_root), jid, STATUS_DONE, {
                "status": STATUS_DONE,
                "fingerprint": fp,
                "input_file": str(Path(task.input_file).resolve()),
                "case_id": task.case_id,
                "variant": task.variant,
                "n_traces": int(task.n_traces),
                "geometry_only": bool(task.geometry_only),
                "returncode": rc,
                "log": str(log_path.resolve()),
                "postprocess": post_rep,
                "host": socket.gethostname(),
            })
            remove_marker_if_exists(Path(self.cfg.runtime.project_root), jid, STATUS_FAILED)
            try:
                marker_path(Path(self.cfg.runtime.project_root), jid, STATUS_RUNNING).unlink(missing_ok=True)
            except TypeError:
                rp = marker_path(Path(self.cfg.runtime.project_root), jid, STATUS_RUNNING)
                if rp.exists():
                    rp.unlink()
            self.log.emit(f"[MARK] done marker written: {marker}")
        return rc

    def run(self) -> None:
        try:
            results = []
            total = len(self.tasks)
            for i, task in enumerate(self.tasks, start=1):
                if self._cancel:
                    break
                self.progress.emit(i - 1, total)
                rc = self._run_task(task)
                results.append({"task": task.to_dict(), "returncode": rc})
                if rc != 0:
                    raise RuntimeError(f"gprMax returned {rc} for {task.input_file}")
            self.progress.emit(len(results), total)
            self.done.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())

