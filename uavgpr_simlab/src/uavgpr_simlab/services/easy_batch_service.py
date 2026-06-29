from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from uavgpr_simlab.core.job_registry import JobRecord, build_job_records, write_job_plan
from uavgpr_simlab.core.runner import GprMaxTask, tasks_from_manifest
from uavgpr_simlab.core.visual_history import find_label_json_for_input, render_model_preview


@dataclass(frozen=True)
class BatchPlanResult:
    """Preview result for the easy GUI batch precheck table."""

    manifest: Path
    workspace: Path
    records: list[JobRecord]
    counts: dict[str, int]
    job_plan_csv: Path
    summary_json: Path | None = None


@dataclass(frozen=True)
class BatchPreviewAsset:
    """Resolved thumbnail inputs for one batch-table row."""

    label_json: Path | None
    preview_png: Path


def workspace_from_manifest(manifest: str | Path) -> Path:
    """Infer the workspace root from a generated manifest.csv path."""

    return Path(manifest).expanduser().resolve().parent.parent


def parse_variants(text: str) -> list[str]:
    """Parse the variant text box used by the easy GUI."""

    return [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]


def read_manifest_rows(manifest: str | Path) -> list[dict[str, str]]:
    """Read a model manifest as normalized string dictionaries."""

    path = Path(manifest).expanduser()
    rows: list[dict[str, str]] = []
    if not path.exists() or not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({str(key): str(value) for key, value in row.items()})
    return rows


def require_manifest_file(manifest: str | Path) -> Path:
    """Return a valid manifest CSV path or raise a clear operator-facing error."""

    text = str(manifest).strip()
    if not text:
        raise ValueError("manifest.csv 路径为空，请先生成模型或选择模型清单。")
    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"manifest.csv 不存在：{path}")
    if not path.is_file():
        raise IsADirectoryError(f"当前路径不是 manifest.csv 文件，而是目录：{path}")
    return path


def unique_case_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Return one manifest row per case_id, preserving manifest order."""

    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        cid = str(row.get("case_id", ""))
        if cid and cid not in seen:
            seen.add(cid)
            out.append({str(key): str(value) for key, value in row.items()})
    return out


def count_manifest_cases(manifest: str | Path | None) -> int:
    """Count unique case_id values in a manifest, returning 0 on missing file."""

    if not manifest:
        return 0
    return len({row.get("case_id") for row in read_manifest_rows(manifest) if row.get("case_id")})


def build_batch_plan(
    manifest: str | Path,
    *,
    variants: Sequence[str],
    max_tasks: int,
    skip_completed: bool,
) -> BatchPlanResult:
    """Build records and persist the job plan used by resume/skip workflows."""

    manifest_path = require_manifest_file(manifest).resolve()
    workspace = workspace_from_manifest(manifest_path)
    records = build_job_records(
        manifest_path,
        workspace,
        variants=variants,
        max_tasks=max_tasks,
        skip_completed=skip_completed,
    )
    report = write_job_plan(
        manifest_path,
        workspace,
        workspace / "jobs" / "job_plan.csv",
        variants=variants,
        max_tasks=max_tasks,
        skip_completed=skip_completed,
    )
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    summary_json = report.get("summary_json")
    return BatchPlanResult(
        manifest=manifest_path,
        workspace=workspace,
        records=records,
        counts=counts,
        job_plan_csv=Path(str(report.get("job_plan_csv") or workspace / "jobs" / "job_plan.csv")),
        summary_json=Path(str(summary_json)) if summary_json else None,
    )


def build_pending_tasks(manifest: str | Path, *, variants: Sequence[str], max_tasks: int) -> list[GprMaxTask]:
    """Build runnable tasks from a manifest for the live queue worker."""

    limit = int(max_tasks) or None
    manifest_path = require_manifest_file(manifest)
    return tasks_from_manifest(manifest_path, variants=variants, limit=limit or 0)


def prepare_batch_preview_asset(record: JobRecord, workspace: str | Path, *, width: int = 300, height: int = 160) -> BatchPreviewAsset:
    """Resolve and render the model thumbnail used by the batch plan table."""

    root = Path(workspace)
    label = find_label_json_for_input(record.input_file, root, record.case_id)
    preview = root / "previews" / "batch" / f"{record.case_id}.png"
    if label:
        render_model_preview(label, preview, title=record.case_id, width=width, height=height)
    return BatchPreviewAsset(label_json=label, preview_png=preview)
