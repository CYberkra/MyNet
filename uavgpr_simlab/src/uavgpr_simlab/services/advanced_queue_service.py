from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from uavgpr_simlab.core.runner import GprMaxTask, tasks_from_manifest, resolve_manifest_path
from uavgpr_simlab.core.scenario import write_manifest_commands_bat


@dataclass(frozen=True)
class AdvancedQueueManifestRow:
    """One manifest row prepared for display in the advanced queue list."""

    display_text: str
    data: dict[str, str]


@dataclass(frozen=True)
class AdvancedQueueManifestPreview:
    """Manifest preview data for the advanced queue tab."""

    manifest: Path
    rows: list[AdvancedQueueManifestRow]
    displayed_count: int
    total_read: int
    truncated: bool
    max_display: int


@dataclass(frozen=True)
class AdvancedQueueBatResult:
    """Result metadata for a generated gprMax batch file."""

    manifest: Path
    workspace: Path
    bat_path: Path
    geometry_only: bool
    variant: str


def workspace_from_manifest(manifest: str | Path) -> Path:
    """Infer workspace root from a generated manifest path."""

    return Path(manifest).expanduser().resolve().parent.parent


def _stringify_manifest_row(row: dict[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in row.items()}


def _runtime_row_paths(row: dict[str, str], manifest: Path) -> dict[str, str]:
    out = dict(row)
    if out.get("input_file"):
        out["input_file"] = str(resolve_manifest_path(out["input_file"], manifest))
    return out


def _row_display_text(row: dict[str, str]) -> str:
    case_id = row.get("case_id", "")
    variant = row.get("variant", "")
    n_traces = row.get("n_traces") or row.get("trace_count") or row.get("trace_count_per_input") or ""
    input_file = row.get("input_file", "")
    return f"{case_id} | {variant} | n={n_traces} | {input_file}"


def read_queue_manifest_preview(manifest: str | Path, *, max_display: int = 1500) -> AdvancedQueueManifestPreview:
    """Read a manifest for queue-list display without touching Qt widgets."""

    manifest_path = Path(manifest).expanduser()
    rows: list[AdvancedQueueManifestRow] = []
    total_read = 0
    if not manifest_path.exists():
        return AdvancedQueueManifestPreview(
            manifest=manifest_path,
            rows=[],
            displayed_count=0,
            total_read=0,
            truncated=False,
            max_display=max_display,
        )
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for raw_row in csv.DictReader(f):
            row = _stringify_manifest_row(raw_row)
            total_read += 1
            if len(rows) < max_display:
                rows.append(AdvancedQueueManifestRow(display_text=_row_display_text(row), data=_runtime_row_paths(row, manifest_path)))
    return AdvancedQueueManifestPreview(
        manifest=manifest_path,
        rows=rows,
        displayed_count=len(rows),
        total_read=total_read,
        truncated=total_read > len(rows),
        max_display=max_display,
    )


def write_advanced_queue_bat(
    manifest: str | Path,
    *,
    conda_env: str,
    use_gpu: bool,
    gpu_ids: Sequence[int],
    geometry_only: bool,
    variant: str,
) -> AdvancedQueueBatResult:
    """Generate the advanced queue BAT file through the existing core scenario helper."""

    manifest_path = Path(manifest).expanduser().resolve()
    workspace = workspace_from_manifest(manifest_path)
    bat_name = "run_geometry_only.bat" if geometry_only else "run_full.bat"
    bat_path = workspace / "logs" / bat_name
    written = write_manifest_commands_bat(
        manifest_path,
        bat_path,
        conda_env=conda_env or "gprMax",
        gpu=use_gpu,
        gpu_ids=list(gpu_ids),
        geometry_only=geometry_only,
        variants=[variant],
        max_tasks=0,
    )
    return AdvancedQueueBatResult(
        manifest=manifest_path,
        workspace=workspace,
        bat_path=Path(written),
        geometry_only=geometry_only,
        variant=variant,
    )


def task_from_manifest_row(row: dict[str, Any], *, fallback_variant: str) -> GprMaxTask | None:
    """Create a single queue task from a manifest row stored on a selected list item."""

    if not row or not row.get("input_file"):
        return None
    n_text = row.get("n_traces") or row.get("trace_count") or row.get("trace_count_per_input") or 1
    try:
        n_val = int(float(str(n_text)))
    except Exception:
        n_val = 1
    return GprMaxTask(
        input_file=str(row.get("input_file")),
        case_id=str(row.get("case_id", "")),
        variant=str(row.get("variant", fallback_variant) or fallback_variant),
        n_traces=max(1, n_val),
    )


def build_advanced_queue_tasks(
    manifest: str | Path,
    *,
    variant: str,
    limit: int,
    selected_row: dict[str, Any] | None = None,
) -> list[GprMaxTask]:
    """Build tasks for the advanced queue tab, either from selection or manifest."""

    if selected_row is not None:
        task = task_from_manifest_row(selected_row, fallback_variant=variant)
        return [task] if task is not None else []
    return tasks_from_manifest(Path(manifest).expanduser(), variants=[variant], limit=int(limit))


def summarize_queue_tasks(tasks: Sequence[GprMaxTask]) -> dict[str, int | list[str]]:
    """Return a compact, UI-safe summary of prepared queue tasks."""

    variants = sorted({task.variant for task in tasks})
    return {
        "tasks": len(tasks),
        "variants": variants,
        "traces": sum(int(task.n_traces) for task in tasks),
    }
