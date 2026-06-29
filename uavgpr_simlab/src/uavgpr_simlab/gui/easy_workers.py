from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QObject, Signal, Slot

from uavgpr_simlab.services.gprmax_smoke_service import (
    GprMaxSourceSmokeReport,
    run_gprmax_source_smoke,
)


class GprMaxSourceSmokeWorker(QObject):
    """Background worker for the easy GUI gprMax minimal CPU smoke test.

    The worker owns only the long-running diagnostic call. It deliberately does
    not touch normal batch execution, fingerprints, history markers or B-scan
    post-processing. Results are returned to the main thread through Qt signals.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        gprmax_root: str | Path,
        work_dir: str | Path,
        build: bool = False,
        omp_threads: int = 1,
        timeout: int = 180,
    ) -> None:
        super().__init__()
        self.gprmax_root = str(gprmax_root)
        self.work_dir = str(work_dir)
        self.build = bool(build)
        self.omp_threads = max(1, int(omp_threads or 1))
        self.timeout = max(1, int(timeout or 180))

    @Slot()
    def run(self) -> None:
        """Run the diagnostic and emit the result back to the GUI thread."""

        try:
            report: GprMaxSourceSmokeReport = run_gprmax_source_smoke(
                self.gprmax_root,
                self.work_dir,
                build=self.build,
                omp_threads=self.omp_threads,
                timeout=self.timeout,
            )
            self.finished.emit(report)
        except Exception:
            self.failed.emit(traceback.format_exc())

class SceneWorldFullChainWorker(QObject):
    """Background worker for GUI SceneWorld full-chain verification.

    It runs the same service used by scripts/run_all_gprmax.py, but exposes
    progress messages to the GUI so the user can see current case/family/variant
    and gprMax output without opening a separate BAT window.
    """

    log = Signal(str)
    event = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        workspace: str | Path,
        gprmax_root: str | Path,
        python_executable: str | Sequence[str] = "python",
        omp_threads: int = 1,
        timeout: int = 600,
        allow_resample: bool = True,
        one_case_per_family: bool = False,
        max_cases: int = 1,
        variants: list[str] | tuple[str, ...] | None = None,
        no_gpu: bool = True,
        gpu_ids: list[int] | None = None,
        run_label: str = "SceneWorld",
        force: bool = False,
        skip_completed: bool = True,
        rerun_failed_only: bool = False,
    ) -> None:
        super().__init__()
        self.workspace = Path(workspace)
        self.gprmax_root = Path(gprmax_root)
        self.python_executable = list(python_executable) if isinstance(python_executable, (list, tuple)) else str(python_executable or "python")
        self.omp_threads = max(1, int(omp_threads or 1))
        self.timeout = max(1, int(timeout or 600))
        self.allow_resample = bool(allow_resample)
        self.one_case_per_family = bool(one_case_per_family)
        self.max_cases = max(0, int(max_cases or 0))
        self.variants = list(variants or ["raw", "target_only", "background_only", "clutter_only", "air_only"])
        self.no_gpu = bool(no_gpu)
        self.gpu_ids = list(gpu_ids or [])
        self.run_label = str(run_label or "SceneWorld")
        self.force = bool(force)
        self.skip_completed = bool(skip_completed)
        self.rerun_failed_only = bool(rerun_failed_only)
        self._cancel_event = threading.Event()


    @Slot()
    def cancel(self) -> None:
        """Request cancellation and terminate the current gprMax child process when possible."""
        self._cancel_event.set()
        self.log.emit("[停止] 已请求停止；当前 gprMax 子进程会被终止，已完成结果保留。")

    def _handle_progress(self, payload: object) -> None:
        if isinstance(payload, dict):
            self.event.emit(payload)
            message = payload.get("message") or payload.get("event") or ""
            if message:
                self.log.emit(str(message))
        else:
            self.log.emit(str(payload))

    @Slot()
    def run(self) -> None:
        try:
            from uavgpr_simlab.services.sceneworld_bscan_service import run_sceneworld_bscan_from_manifest

            manifests = sorted((self.workspace / "datasets").glob("*_manifest.csv"))
            if not manifests:
                raise FileNotFoundError(f"未找到 manifest: {self.workspace / 'datasets'}")
            manifest = manifests[-1]
            self.log.emit(f"[start] {self.run_label}")
            self.log.emit(f"[start] workspace={self.workspace}")
            self.log.emit(f"[start] manifest={manifest}")
            self.log.emit(f"[start] gprMax={self.gprmax_root}")
            report = run_sceneworld_bscan_from_manifest(
                manifest,
                gprmax_root=self.gprmax_root,
                variants=self.variants,
                one_case_per_family=self.one_case_per_family,
                max_cases=self.max_cases,
                python_executable=self.python_executable,
                omp_threads=self.omp_threads,
                timeout_sec=self.timeout,
                no_gpu=self.no_gpu,
                gpu_ids=self.gpu_ids,
                force=self.force,
                allow_resample=self.allow_resample,
                progress_callback=self._handle_progress,
                cancel_event=self._cancel_event,
                skip_completed=self.skip_completed,
                rerun_failed_only=self.rerun_failed_only,
            )
            self.finished.emit(report)
        except Exception:
            self.failed.emit(traceback.format_exc())


# Backward-compatible alias used by older imports.
SceneWorldUltraTinyWorker = SceneWorldFullChainWorker
