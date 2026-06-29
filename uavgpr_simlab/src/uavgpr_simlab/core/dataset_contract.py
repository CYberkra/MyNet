from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from uavgpr_simlab.core.runner import resolve_manifest_path

STANDARD_SCENEWORLD_VARIANTS = ("raw", "target_only", "background_only", "clutter_only", "air_only")

REQUIRED_MANIFEST_COLUMNS = (
    "case_id",
    "variant",
    "input_file",
    "n_traces",
)

REQUIRED_SCENEWORLD_COLUMNS = (
    "scene_world_json",
    "metadata_summary_json",
    "label_json",
    "interface_gt_npy",
    "layer_gt_npy",
    "time_axis_ns_npy",
    "distance_axis_m_npy",
    "bscan_qc_report_json",
)

PATH_COLUMNS = (
    "input_file",
    "label_json",
    "interface_csv",
    "mask_npy",
    "scene_world_json",
    "metadata_summary_json",
    "interface_gt_npy",
    "layer_gt_npy",
    "model_preview_png",
    "variant_preview_png",
    "bscan_npy",
    "raw_bscan_npy",
    "target_bscan_npy",
    "background_bscan_npy",
    "clutter_bscan_npy",
    "air_bscan_npy",
    "clutter_gt_bscan_npy",
    "time_axis_ns_npy",
    "distance_axis_m_npy",
    "tx_x_axis_m_npy",
    "rx_x_axis_m_npy",
    "midpoint_x_axis_m_npy",
    "layer_gt_x_axis_m_npy",
    "layer_gt_y_axis_m_npy",
    "interface_mask_bscan_npy",
    "layer_mask_bscan_npy",
    "bscan_placeholder_status_json",
    "bscan_qc_report_json",
)

# Skeletons are allowed to be pre-run: output B-scan/QC products may be placeholders
# or not-ready.  These files are still checked when present, but they are not hard
# requirements for an importable ready-to-run skeleton.
PRE_RUN_OUTPUT_COLUMNS = (
    "bscan_npy",
    "raw_bscan_npy",
    "target_bscan_npy",
    "background_bscan_npy",
    "clutter_bscan_npy",
    "air_bscan_npy",
    "clutter_gt_bscan_npy",
    "bscan_placeholder_status_json",
    "bscan_qc_report_json",
)

REQUIRED_EXISTING_COLUMNS = tuple(c for c in PATH_COLUMNS if c not in PRE_RUN_OUTPUT_COLUMNS)


@dataclass(frozen=True)
class DatasetContractIssue:
    level: str
    code: str
    message: str
    path: str = ""
    case_id: str = ""
    variant: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetContractReport:
    ok: bool
    manifest: str
    workspace: str
    schema: str
    case_count: int
    row_count: int
    variants: list[str]
    imported_ready_to_run: bool
    training_ready: bool
    issue_count: int
    error_count: int
    warning_count: int
    issues: list[DatasetContractIssue] = field(default_factory=list)
    summary_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [x.to_dict() for x in self.issues]
        return data


def _read_rows(manifest: Path) -> tuple[list[str], list[dict[str, str]]]:
    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [{str(k): str(v) for k, v in row.items()} for row in reader]
    return fields, rows


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _is_sceneworld(fields: Iterable[str]) -> bool:
    field_set = set(fields)
    return bool({"scene_world_json", "bscan_qc_report_json", "raw_bscan_npy", "target_bscan_npy"} & field_set)


def validate_dataset_skeleton(
    manifest_csv: str | Path,
    *,
    expected_variants: Sequence[str] = STANDARD_SCENEWORLD_VARIANTS,
    require_relative_paths: bool = True,
    write_report: bool = False,
) -> DatasetContractReport:
    """Validate a generated/imported dataset skeleton before it is handed to batch run.

    The check is intentionally side-effect free unless ``write_report`` is true.
    It validates the portable manifest contract that lets users design a dataset
    skeleton on one machine, import it into UavGPR-SimLab, then run gprMax on a
    different GPU workstation.
    """

    manifest = Path(manifest_csv).expanduser()
    issues: list[DatasetContractIssue] = []

    def add(level: str, code: str, message: str, *, path: str | Path = "", case_id: str = "", variant: str = "") -> None:
        issues.append(DatasetContractIssue(level=level, code=code, message=message, path=str(path) if path else "", case_id=case_id, variant=variant))

    if not str(manifest).strip():
        add("error", "manifest_empty", "manifest.csv 路径为空。")
        return _final_report(False, manifest, Path("."), "unknown", 0, 0, [], issues, write_report=False)
    if not manifest.exists():
        add("error", "manifest_missing", f"manifest.csv 不存在：{manifest}", path=manifest)
        return _final_report(False, manifest, manifest.parent.parent, "unknown", 0, 0, [], issues, write_report=False)
    if not manifest.is_file():
        add("error", "manifest_not_file", f"manifest.csv 路径不是文件：{manifest}", path=manifest)
        return _final_report(False, manifest, manifest.parent.parent, "unknown", 0, 0, [], issues, write_report=False)

    workspace = manifest.resolve().parent.parent
    fields, rows = _read_rows(manifest)
    field_set = set(fields)
    schema = "sceneworld" if _is_sceneworld(fields) else "generic_manifest"

    if not rows:
        add("error", "manifest_empty_rows", "manifest.csv 没有数据行。", path=manifest)
    for col in REQUIRED_MANIFEST_COLUMNS:
        if col not in field_set:
            add("error", "missing_required_column", f"manifest 缺少必需列：{col}", path=manifest)
    if schema == "sceneworld":
        for col in REQUIRED_SCENEWORLD_COLUMNS:
            if col not in field_set:
                add("error", "missing_sceneworld_column", f"SceneWorld manifest 缺少关键列：{col}", path=manifest)

    cases: dict[str, set[str]] = {}
    row_ids: set[tuple[str, str, str]] = set()
    variants_seen: set[str] = set()
    expected = set(expected_variants or [])

    for idx, row in enumerate(rows, start=2):
        cid = row.get("case_id", "").strip()
        variant = row.get("variant", "").strip() or "raw"
        inp = row.get("input_file", "").strip()
        if not cid:
            add("error", "missing_case_id", f"第 {idx} 行 case_id 为空。", path=manifest, variant=variant)
        if not variant:
            add("error", "missing_variant", f"第 {idx} 行 variant 为空。", path=manifest, case_id=cid)
        if not inp:
            add("error", "missing_input_file", f"第 {idx} 行 input_file 为空。", path=manifest, case_id=cid, variant=variant)
        if cid:
            cases.setdefault(cid, set()).add(variant)
        variants_seen.add(variant)
        key = (cid, variant, inp)
        if key in row_ids:
            add("warning", "duplicate_manifest_row", f"第 {idx} 行与前面行重复：case={cid}, variant={variant}, input={inp}", path=manifest, case_id=cid, variant=variant)
        row_ids.add(key)

        for col in PATH_COLUMNS:
            value = row.get(col, "").strip()
            if not value:
                continue
            p = Path(value)
            if require_relative_paths and p.is_absolute():
                add("warning", "absolute_path", f"列 {col} 使用绝对路径，不利于两台电脑迁移：{value}", path=value, case_id=cid, variant=variant)
            if col in REQUIRED_EXISTING_COLUMNS:
                resolved = resolve_manifest_path(value, manifest)
                if not resolved.exists():
                    add("error", "referenced_file_missing", f"列 {col} 引用的文件不存在：{value}", path=resolved, case_id=cid, variant=variant)
                elif col == "input_file" and not resolved.is_file():
                    add("error", "input_not_file", f"input_file 不是文件：{value}", path=resolved, case_id=cid, variant=variant)

    if schema == "sceneworld" and expected:
        for cid, got in cases.items():
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            if missing:
                add("error", "case_missing_variants", f"{cid} 缺少变体：{','.join(missing)}", case_id=cid)
            if extra:
                add("warning", "case_extra_variants", f"{cid} 包含非标准变体：{','.join(extra)}", case_id=cid)

    summary_path = workspace / "reports" / "dataset_summary.json"
    training_ready = False
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            training_ready = bool(summary.get("training_ready") or summary.get("bscan_run_status", {}).get("training_ready"))
        except Exception as exc:
            add("warning", "summary_unreadable", f"dataset_summary.json 无法读取：{exc}", path=summary_path)
    else:
        add("warning", "summary_missing", "未找到 reports/dataset_summary.json；可运行，但建议骨架包含摘要。", path=summary_path)

    run_bat = workspace / "logs" / "run_all_gprmax.bat"
    if not run_bat.exists():
        add("warning", "run_bat_missing", "未找到 logs/run_all_gprmax.bat；GUI 仍可运行，但独立批处理入口缺失。", path=run_bat)
    else:
        try:
            text = run_bat.read_text(encoding="utf-8", errors="replace")
            required_tokens = ("windows_runtime_bootstrap.bat", "run_all_gprmax.py", "%PY_RUN%")
            for token in required_tokens:
                if token not in text:
                    add("error", "run_bat_contract_broken", f"run_all_gprmax.bat 缺少运行时合同片段：{token}", path=run_bat)
        except Exception as exc:
            add("error", "run_bat_unreadable", f"run_all_gprmax.bat 无法读取：{exc}", path=run_bat)

    return _final_report(
        error_count(issues) == 0,
        manifest,
        workspace,
        schema,
        len(cases),
        len(rows),
        sorted(variants_seen),
        issues,
        write_report=write_report,
        training_ready=training_ready,
    )


def error_count(issues: Sequence[DatasetContractIssue]) -> int:
    return sum(1 for x in issues if x.level == "error")


def warning_count(issues: Sequence[DatasetContractIssue]) -> int:
    return sum(1 for x in issues if x.level == "warning")


def _final_report(
    ok: bool,
    manifest: Path,
    workspace: Path,
    schema: str,
    case_count: int,
    row_count: int,
    variants: list[str],
    issues: list[DatasetContractIssue],
    *,
    write_report: bool,
    training_ready: bool = False,
) -> DatasetContractReport:
    report_json = ""
    report = DatasetContractReport(
        ok=bool(ok),
        manifest=str(manifest.resolve()) if str(manifest) else "",
        workspace=str(workspace.resolve()) if str(workspace) else "",
        schema=schema,
        case_count=case_count,
        row_count=row_count,
        variants=variants,
        imported_ready_to_run=bool(ok),
        training_ready=bool(training_ready),
        issue_count=len(issues),
        error_count=error_count(issues),
        warning_count=warning_count(issues),
        issues=issues,
        summary_json="",
    )
    if write_report and str(workspace):
        out = workspace / "reports" / "dataset_contract_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data["summary_json"] = str(out)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        report_json = str(out)
        report = DatasetContractReport(**{**report.to_dict(), "summary_json": report_json, "issues": issues})
    return report
