from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from uavgpr_simlab.core.runner import resolve_manifest_path
from uavgpr_simlab.core.dataset_contract import STANDARD_SCENEWORLD_VARIANTS
from uavgpr_simlab.core.config import read_combined_simlab_env

STATUS_ALIASES = {
    "success": "done",
    "done": "done",
    "completed": "done",
    "failed": "failed",
    "error": "failed",
    "running": "running",
    "stale_running": "stale_running",
    "skipped": "skipped",
    "skip": "skipped",
    "not_run": "pending",
    "pending": "pending",
    "": "pending",
}

ORDERED_STATUSES = ("pending", "running", "done", "failed", "skipped", "stale_running")


@dataclass(frozen=True)
class DatasetRunItem:
    """One case/variant row prepared for GUI run dashboards."""

    case_id: str
    variant: str
    status: str
    n_traces: int
    input_file: str
    bscan_path: str = ""
    error: str = ""
    family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetRunDashboard:
    """Operator-facing snapshot of one imported dataset skeleton/run."""

    manifest: str
    workspace: str
    dataset_name: str
    case_count: int
    row_count: int
    variants: list[str]
    counts: dict[str, int]
    next_items: list[DatasetRunItem] = field(default_factory=list)
    running_items: list[DatasetRunItem] = field(default_factory=list)
    failed_items: list[DatasetRunItem] = field(default_factory=list)
    latest_done_items: list[DatasetRunItem] = field(default_factory=list)
    training_ready: bool = False
    summary_json: str = ""
    average_variant_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    runtime_profile: dict[str, str] = field(default_factory=dict)

    @property
    def pending(self) -> int:
        return int(self.counts.get("pending", 0))

    @property
    def running(self) -> int:
        return int(self.counts.get("running", 0) + self.counts.get("stale_running", 0))

    @property
    def done(self) -> int:
        return int(self.counts.get("done", 0))

    @property
    def failed(self) -> int:
        return int(self.counts.get("failed", 0))

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["next_items"] = [x.to_dict() for x in self.next_items]
        data["running_items"] = [x.to_dict() for x in self.running_items]
        data["failed_items"] = [x.to_dict() for x in self.failed_items]
        data["latest_done_items"] = [x.to_dict() for x in self.latest_done_items]
        return data

    def format_for_operator(self, *, max_items: int = 4) -> str:
        """Return a compact Chinese status block for the batch page."""

        bits = [
            f"数据集：{self.dataset_name or Path(self.workspace).name}",
            f"case={self.case_count}，任务={self.row_count}",
            f"等待={self.pending}，运行中={self.running}，完成={self.done}，失败={self.failed}",
            f"训练就绪={'是' if self.training_ready else '否/未完成'}",
            f"预计剩余={format_duration_seconds(self.estimated_remaining_seconds) if self.estimated_remaining_seconds else '待估算'}",
        ]
        lines = [" | ".join(bits)]
        if self.running_items:
            lines.append("正在跑：" + ", ".join(f"{x.case_id}/{x.variant}" for x in self.running_items[:max_items]))
        elif self.next_items:
            lines.append("即将跑：" + ", ".join(f"{x.case_id}/{x.variant}" for x in self.next_items[:max_items]))
        if self.failed_items:
            lines.append("失败待处理：" + ", ".join(f"{x.case_id}/{x.variant}" for x in self.failed_items[:max_items]))
        return "\n".join(lines)


def format_duration_seconds(seconds: float | int | None) -> str:
    """Format seconds as a compact Chinese duration for operator dashboards."""

    try:
        total = int(max(0, round(float(seconds or 0))))
    except Exception:
        total = 0
    if total <= 0:
        return "—"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分"
    if minutes:
        return f"{minutes}分{secs:02d}秒"
    return f"{secs}秒"


def _gpu_summary_short() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "未检测到 nvidia-smi"
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
        line = (proc.stdout or "").strip().splitlines()[0] if proc.stdout else ""
        return line if proc.returncode == 0 and line else "nvidia-smi 查询失败"
    except Exception:
        return "GPU 信息暂不可用"


def _runtime_profile_snapshot() -> dict[str, str]:
    env = read_combined_simlab_env()

    def pick(key: str, default: str = "") -> str:
        return str(env.get(key) or os.environ.get(key, default) or default).strip()

    python_exe = pick("UAVGPR_PYTHON_EXE")
    conda_prefix = pick("UAVGPR_CONDA_ENV_PREFIX")
    if not python_exe and conda_prefix:
        candidate = Path(conda_prefix) / ("python.exe" if os.name == "nt" else "bin/python")
        python_exe = str(candidate)
    return {
        "machine_profile": pick("UAVGPR_MACHINE_PROFILE", "未设置"),
        "runtime_root": pick("UAVGPR_RUNTIME_ROOT"),
        "gpu_runtime_env": pick("UAVGPR_GPU_RUNTIME_ENV"),
        "python_exe": python_exe,
        "conda_env_prefix": conda_prefix,
        "gprmax_root": pick("UAVGPR_GPRMAX_ROOT") or pick("GPRMAX_SOURCE_DIR"),
        "use_gpu": pick("UAVGPR_USE_GPU", "0"),
        "gpu_ids": pick("UAVGPR_GPU_IDS", "0"),
        "gpu_name": _gpu_summary_short(),
        "omp_threads": pick("UAVGPR_OMP_THREADS", ""),
        "run_scale": pick("UAVGPR_RUN_SCALE"),
    }


def _elapsed_seconds_from_report(workspace: Path) -> list[float]:
    report_path = workspace / "reports" / "sceneworld_bscan_run_report.json"
    if not report_path.exists():
        return []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    values: list[float] = []
    for case in (report.get("cases") or {}).values():
        runs = case.get("runs") if isinstance(case, dict) else {}
        for run in (runs or {}).values():
            if not isinstance(run, dict):
                continue
            if run.get("skipped") or run.get("cancelled"):
                continue
            try:
                sec = float(run.get("elapsed_sec") or 0)
            except Exception:
                sec = 0.0
            if sec > 0:
                values.append(sec)
    return values


def _read_manifest(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", newline="") as f:
        return [{str(k): str(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _normalize_status(value: str) -> str:
    return STATUS_ALIASES.get(str(value or "").strip().lower(), str(value or "pending").strip().lower() or "pending")


def _variant_bscan_field(variant: str) -> str:
    return {
        "raw": "raw_bscan_npy",
        "target_only": "target_bscan_npy",
        "background_only": "background_bscan_npy",
        "clutter_only": "clutter_bscan_npy",
        "air_only": "air_bscan_npy",
        "clutter_gt": "clutter_gt_bscan_npy",
    }.get(variant, "bscan_npy")


def _item_from_row(row: dict[str, str], manifest: Path) -> DatasetRunItem:
    variant = row.get("variant", "") or "raw"
    status = _normalize_status(row.get("bscan_status", ""))
    bscan_value = row.get(_variant_bscan_field(variant), "") or row.get("bscan_npy", "")
    bscan_path = str(resolve_manifest_path(bscan_value, manifest)) if bscan_value else ""
    try:
        n_traces = int(float(row.get("n_traces") or row.get("trace_count") or 0))
    except Exception:
        n_traces = 0
    return DatasetRunItem(
        case_id=row.get("case_id", ""),
        variant=variant,
        status=status,
        n_traces=n_traces,
        input_file=str(resolve_manifest_path(row.get("input_file", ""), manifest)) if row.get("input_file") else "",
        bscan_path=bscan_path,
        error=row.get("bscan_error", ""),
        family=row.get("family", ""),
    )


def summarize_dataset_run_dashboard(
    manifest_csv: str | Path,
    *,
    expected_variants: Sequence[str] = STANDARD_SCENEWORLD_VARIANTS,
    write_report: bool = False,
) -> DatasetRunDashboard:
    """Summarize imported, queued, running and completed models for operators.

    This is intentionally manifest-first: the manifest and per-case QC files are
    the shared contract between machines and software versions.  It lets the GUI
    show "already ran / running / next to run" without requiring a database.
    """

    manifest = Path(manifest_csv).expanduser().resolve()
    workspace = manifest.parent.parent
    rows = _read_manifest(manifest) if manifest.exists() and manifest.is_file() else []
    items = [_item_from_row(row, manifest) for row in rows]
    counts = {status: 0 for status in ORDERED_STATUSES}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1

    case_ids = {x.case_id for x in items if x.case_id}
    variants = sorted({x.variant for x in items if x.variant}) or list(expected_variants)
    next_items = [x for x in items if x.status == "pending"][:8]
    running_items = [x for x in items if x.status in {"running", "stale_running"}][:8]
    failed_items = [x for x in items if x.status == "failed"][:12]
    latest_done = [x for x in items if x.status == "done"][-8:]

    summary_json = ""
    training_ready = False
    summary_path = workspace / "reports" / "dataset_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            training_ready = bool(summary.get("training_ready") or summary.get("bscan_run_status", {}).get("training_ready"))
        except Exception:
            training_ready = False

    elapsed_values = _elapsed_seconds_from_report(workspace)
    average_variant_seconds = round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else 0.0
    # Running items still occupy GPU time, so count them as remaining unless
    # the operator refreshes after they complete.  This is deliberately
    # conservative rather than optimistic.
    remaining_items = counts.get("pending", 0) + counts.get("running", 0) + counts.get("stale_running", 0)
    estimated_remaining_seconds = round(average_variant_seconds * remaining_items, 3) if average_variant_seconds else 0.0
    runtime_profile = _runtime_profile_snapshot()

    report = DatasetRunDashboard(
        manifest=str(manifest),
        workspace=str(workspace),
        dataset_name=workspace.name,
        case_count=len(case_ids),
        row_count=len(items),
        variants=variants,
        counts=counts,
        next_items=next_items,
        running_items=running_items,
        failed_items=failed_items,
        latest_done_items=latest_done,
        training_ready=training_ready,
        summary_json=summary_json,
        average_variant_seconds=average_variant_seconds,
        estimated_remaining_seconds=estimated_remaining_seconds,
        runtime_profile=runtime_profile,
    )
    if write_report:
        out = workspace / "reports" / "run_dashboard_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data["summary_json"] = str(out)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        report = replace(report, summary_json=str(out))
    return report
