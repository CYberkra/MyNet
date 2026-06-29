from __future__ import annotations

import csv
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Sequence

from uavgpr_simlab.core.dataset_contract import PATH_COLUMNS, validate_dataset_skeleton
from uavgpr_simlab.core.run_dashboard import summarize_dataset_run_dashboard

_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")
_UNC_ABS_RE = re.compile(r"^\\\\[^\\/]+[\\/][^\\/]+")
_WINDOWS_ABS_ANY_RE = re.compile(r"([A-Za-z]:[\\/][^\"'\r\n,;|<>]*)")

JSON_FILE_GLOBS = (
    # reports/*.json are derived diagnostics and can be regenerated after relocation.
    # Scanning them makes the relocator chase its own reports and stale audit paths.
    "models/**/*.json",
    "jobs/**/*.json",
)
TEXT_FILE_GLOBS = (
    "logs/*.bat",
    "configs/*.yaml",
    "configs/*.yml",
)


@dataclass(frozen=True)
class WorkspacePathFinding:
    file: str
    location: str
    value: str
    suggested_value: str = ""
    exists_after_relocation: bool = False
    level: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkspaceRelocationChange:
    file: str
    location: str
    old_value: str
    new_value: str
    kind: str = "path"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkspaceRelocationReport:
    ok: bool
    workspace: str
    manifest: str
    dry_run: bool
    to_relative: bool
    old_roots: list[str]
    new_root: str
    absolute_path_count: int
    change_count: int
    changed_file_count: int
    warning_count: int
    error_count: int
    findings: list[WorkspacePathFinding] = field(default_factory=list)
    changes: list[WorkspaceRelocationChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    backup_dir: str = ""
    dataset_contract_ok: bool = False
    run_dashboard_json: str = ""
    report_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["findings"] = [x.to_dict() for x in self.findings]
        data["changes"] = [x.to_dict() for x in self.changes]
        return data



def is_windows_absolute_path(value: str) -> bool:
    text = str(value).strip().strip('"')
    return bool(_WINDOWS_ABS_RE.match(text) or _UNC_ABS_RE.match(text))



def _normalise_sep(value: str) -> str:
    return str(value).replace("\\", "/")



def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()



def _workspace_from_manifest(manifest_csv: str | Path) -> tuple[Path, Path]:
    manifest = Path(manifest_csv).expanduser()
    if not manifest.exists() or not manifest.is_file():
        raise FileNotFoundError(f"manifest.csv 不存在或不是文件：{manifest}")
    return manifest, manifest.resolve().parent.parent



def _split_windows_parts(value: str) -> list[str]:
    return [p for p in PureWindowsPath(str(value).replace("/", "\\")).parts if p not in {"\\", "/"}]



def _candidate_from_workspace_name(value: str, workspace: Path) -> Path | None:
    parts = _split_windows_parts(value)
    ws_name = workspace.name
    if ws_name not in parts:
        return None
    # If a path contains the dataset/workspace folder name, everything after it
    # should be portable across machines.
    idx = len(parts) - 1 - list(reversed(parts)).index(ws_name)
    suffix = parts[idx + 1 :]
    if not suffix:
        return workspace
    return workspace.joinpath(*suffix)



def _replace_old_root(value: str, old_root: str, new_root: Path) -> str | None:
    if not old_root:
        return None
    old_norm = _normalise_sep(old_root).rstrip("/")
    val_norm = _normalise_sep(value)
    if val_norm.lower() == old_norm.lower():
        return str(new_root)
    prefix = old_norm + "/"
    if val_norm.lower().startswith(prefix.lower()):
        suffix = val_norm[len(prefix) :]
        return str(new_root / PureWindowsPath(suffix))
    return None



def suggest_relocated_path(value: str, workspace: Path, *, old_roots: Sequence[str] = (), to_relative: bool = True) -> tuple[str, bool]:
    """Return a portable replacement for a path-like value.

    Absolute paths under an old workspace root are rewritten under the current
    workspace.  If ``to_relative`` is true and the candidate is inside the current
    workspace, the manifest-safe relative path is returned.
    """

    text = str(value).strip()
    if not text:
        return text, False
    if not is_windows_absolute_path(text) and not Path(text).is_absolute():
        return text.replace("\\", "/"), False

    candidate: Path | None = None
    for old in old_roots:
        repl = _replace_old_root(text, old, workspace)
        if repl:
            candidate = Path(repl)
            break
    if candidate is None:
        candidate = _candidate_from_workspace_name(text, workspace)
    if candidate is None and Path(text).is_absolute():
        try:
            p = Path(text)
            p.resolve().relative_to(workspace.resolve())
            candidate = p
        except Exception:
            candidate = None
    if candidate is None:
        return text, False
    if to_relative:
        return _safe_rel(candidate, workspace), True
    return str(candidate), True



def _read_manifest(manifest: Path) -> tuple[list[str], list[dict[str, str]]]:
    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [{str(k): str(v or "") for k, v in row.items()} for row in reader]
    return fields, rows



def _write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})



def _backup_file(path: Path, workspace: Path, backup_dir: Path) -> None:
    try:
        rel = path.resolve().relative_to(workspace.resolve())
    except Exception:
        rel = Path(path.name)
    target = backup_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)



def _iter_files(workspace: Path, manifest: Path) -> Iterable[Path]:
    yielded: set[Path] = set()
    for p in [manifest]:
        rp = p.resolve()
        if rp not in yielded:
            yielded.add(rp)
            yield p
    for pattern in [*JSON_FILE_GLOBS, *TEXT_FILE_GLOBS]:
        for p in workspace.glob(pattern):
            if not p.is_file():
                continue
            rel_text = p.as_posix()
            if "/reports/relocation_backups/" in rel_text or rel_text.endswith("/reports/workspace_relocation_report.json"):
                continue
            rp = p.resolve()
            if rp not in yielded:
                yielded.add(rp)
                yield p



def _scan_text_absolute_paths(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    out: list[str] = []
    for match in _WINDOWS_ABS_ANY_RE.finditer(text):
        value = match.group(1).strip().rstrip(').,;"\'')
        if value and value not in out:
            out.append(value)
    return out



def _relocate_manifest(
    manifest: Path,
    workspace: Path,
    *,
    old_roots: Sequence[str],
    to_relative: bool,
) -> tuple[list[WorkspacePathFinding], list[WorkspaceRelocationChange], list[str], list[str], list[str], list[dict[str, str]]]:
    findings: list[WorkspacePathFinding] = []
    changes: list[WorkspaceRelocationChange] = []
    warnings: list[str] = []
    errors: list[str] = []
    fields, rows = _read_manifest(manifest)
    path_cols = [c for c in PATH_COLUMNS if c in fields]

    for row_idx, row in enumerate(rows, start=2):
        cid = row.get("case_id", "")
        variant = row.get("variant", "")
        for col in path_cols:
            value = row.get(col, "").strip()
            if not value:
                continue
            if is_windows_absolute_path(value) or Path(value).is_absolute():
                suggested, changed = suggest_relocated_path(value, workspace, old_roots=old_roots, to_relative=to_relative)
                exists = (workspace / suggested).exists() if changed and to_relative else Path(suggested).exists() if changed else False
                findings.append(
                    WorkspacePathFinding(
                        file=str(manifest),
                        location=f"row={row_idx}, case={cid}, variant={variant}, column={col}",
                        value=value,
                        suggested_value=suggested if changed else "",
                        exists_after_relocation=bool(exists),
                        level="warning" if changed else "error",
                    )
                )
                if changed and suggested != value:
                    row[col] = suggested.replace("\\", "/")
                    changes.append(
                        WorkspaceRelocationChange(
                            file=str(manifest),
                            location=f"row={row_idx}, column={col}",
                            old_value=value,
                            new_value=row[col],
                        )
                    )
                elif not changed:
                    errors.append(f"无法自动重定位 manifest 第 {row_idx} 行 {col}: {value}")
    return findings, changes, warnings, errors, fields, rows



def _relocate_json_file(
    path: Path,
    workspace: Path,
    *,
    old_roots: Sequence[str],
    to_relative: bool,
) -> tuple[list[WorkspacePathFinding], list[WorkspaceRelocationChange], Any, bool]:
    findings: list[WorkspacePathFinding] = []
    changes: list[WorkspaceRelocationChange] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return findings, changes, None, False

    changed_any = False

    def visit(obj: Any, loc: str) -> Any:
        nonlocal changed_any
        if isinstance(obj, dict):
            return {k: visit(v, f"{loc}.{k}" if loc else str(k)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [visit(v, f"{loc}[{i}]") for i, v in enumerate(obj)]
        if isinstance(obj, str) and (is_windows_absolute_path(obj) or Path(obj).is_absolute()):
            suggested, changed = suggest_relocated_path(obj, workspace, old_roots=old_roots, to_relative=to_relative)
            exists = (workspace / suggested).exists() if changed and to_relative else Path(suggested).exists() if changed else False
            findings.append(WorkspacePathFinding(file=str(path), location=loc, value=obj, suggested_value=suggested if changed else "", exists_after_relocation=bool(exists), level="warning" if changed else "error"))
            if changed and suggested != obj:
                changes.append(WorkspaceRelocationChange(file=str(path), location=loc, old_value=obj, new_value=suggested, kind="json_string"))
                changed_any = True
                return suggested.replace("\\", "/")
        return obj

    new_data = visit(data, "")
    return findings, changes, new_data, changed_any



def _relocate_text_file(
    path: Path,
    workspace: Path,
    *,
    old_roots: Sequence[str],
    to_relative: bool,
) -> tuple[list[WorkspacePathFinding], list[WorkspaceRelocationChange], str | None, bool]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return [], [], None, False
    findings: list[WorkspacePathFinding] = []
    changes: list[WorkspaceRelocationChange] = []
    new_text = text
    for value in _scan_text_absolute_paths(path):
        # Runtime solver locations such as E:\gprMax are intentionally machine-local
        # and must not be rewritten as workspace paths.  Workspace relocation only
        # handles dataset/workspace artifacts.
        if "gprmax" in value.lower():
            continue
        suggested, changed = suggest_relocated_path(value, workspace, old_roots=old_roots, to_relative=False)
        exists = Path(suggested).exists() if changed else False
        findings.append(WorkspacePathFinding(file=str(path), location="text", value=value, suggested_value=suggested if changed else "", exists_after_relocation=bool(exists), level="warning" if changed else "error"))
        if changed and suggested != value:
            new_text = new_text.replace(value, suggested)
            changes.append(WorkspaceRelocationChange(file=str(path), location="text", old_value=value, new_value=suggested, kind="text"))
    return findings, changes, new_text, new_text != text



def relocate_workspace_paths(
    manifest_csv: str | Path,
    *,
    old_root: str | None = None,
    old_roots: Sequence[str] = (),
    new_root: str | Path | None = None,
    to_relative: bool = True,
    dry_run: bool = True,
    write_report: bool = True,
    make_backup: bool = True,
    validate_after: bool = True,
) -> WorkspaceRelocationReport:
    """Inspect and optionally rewrite a moved dataset workspace.

    The primary operation is converting manifest path columns from generation-host
    absolute paths to portable workspace-relative paths.  Related JSON markers and
    generated BAT/YAML files are scanned and rewritten only when an automatic,
    workspace-safe replacement can be inferred.
    """

    manifest, detected_workspace = _workspace_from_manifest(manifest_csv)
    workspace = Path(new_root).expanduser().resolve() if new_root else detected_workspace.resolve()
    all_old_roots = [x for x in [old_root, *old_roots] if x]
    findings: list[WorkspacePathFinding] = []
    changes: list[WorkspaceRelocationChange] = []
    warnings: list[str] = []
    errors: list[str] = []
    changed_files: set[str] = set()
    backup_dir = ""

    if not workspace.exists() or not workspace.is_dir():
        errors.append(f"目标 workspace 不存在或不是目录：{workspace}")
    if manifest.resolve().parent.parent.resolve() != workspace.resolve():
        warnings.append(f"manifest 所在 workspace 与 new_root 不一致：manifest={manifest.parent.parent}, new_root={workspace}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = workspace / "reports" / "relocation_backups" / ts

    mf_findings, mf_changes, mf_warnings, mf_errors, mf_fields, mf_rows = _relocate_manifest(manifest, workspace, old_roots=all_old_roots, to_relative=to_relative)
    findings.extend(mf_findings)
    changes.extend(mf_changes)
    warnings.extend(mf_warnings)
    errors.extend(mf_errors)
    if mf_changes:
        changed_files.add(str(manifest))
        if not dry_run:
            if make_backup:
                _backup_file(manifest, workspace, backup_path)
            _write_manifest(manifest, mf_fields, mf_rows)

    for path in _iter_files(workspace, manifest):
        if path.resolve() == manifest.resolve():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            jf, jc, new_data, changed = _relocate_json_file(path, workspace, old_roots=all_old_roots, to_relative=to_relative)
            findings.extend(jf)
            changes.extend(jc)
            if changed:
                changed_files.add(str(path))
                if not dry_run:
                    if make_backup:
                        _backup_file(path, workspace, backup_path)
                    path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
        elif suffix in {".bat", ".yaml", ".yml"}:
            tf, tc, new_text, changed = _relocate_text_file(path, workspace, old_roots=all_old_roots, to_relative=to_relative)
            findings.extend(tf)
            changes.extend(tc)
            if changed and new_text is not None:
                changed_files.add(str(path))
                if not dry_run:
                    if make_backup:
                        _backup_file(path, workspace, backup_path)
                    path.write_text(new_text, encoding="utf-8")

    dataset_contract_ok = False
    run_dashboard_json = ""
    if validate_after:
        try:
            contract = validate_dataset_skeleton(manifest, write_report=not dry_run)
            dataset_contract_ok = bool(contract.ok)
            if not contract.ok:
                errors.append(f"迁移后 dataset contract 仍未通过：{contract.error_count} error(s)")
            dash = summarize_dataset_run_dashboard(manifest, write_report=not dry_run)
            run_dashboard_json = dash.summary_json
        except Exception as exc:
            errors.append(f"迁移后验证失败：{exc}")

    if changed_files and not dry_run and make_backup:
        backup_dir = str(backup_path)

    absolute_count = len(findings)
    report = WorkspaceRelocationReport(
        ok=not errors,
        workspace=str(workspace),
        manifest=str(manifest),
        dry_run=bool(dry_run),
        to_relative=bool(to_relative),
        old_roots=list(all_old_roots),
        new_root=str(workspace),
        absolute_path_count=absolute_count,
        change_count=len(changes),
        changed_file_count=len(changed_files),
        warning_count=len(warnings) + sum(1 for f in findings if f.level == "warning"),
        error_count=len(errors) + sum(1 for f in findings if f.level == "error"),
        findings=findings,
        changes=changes,
        warnings=warnings,
        errors=errors,
        backup_dir=backup_dir,
        dataset_contract_ok=dataset_contract_ok,
        run_dashboard_json=run_dashboard_json,
    )
    if write_report:
        out = workspace / "reports" / "workspace_relocation_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data["report_json"] = str(out)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        report = replace(report, report_json=str(out))
    return report


__all__ = [
    "WorkspacePathFinding",
    "WorkspaceRelocationChange",
    "WorkspaceRelocationReport",
    "is_windows_absolute_path",
    "suggest_relocated_path",
    "relocate_workspace_paths",
]
