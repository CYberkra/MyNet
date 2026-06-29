from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc), "_path": str(path)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields or ["empty"])
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def summarize_manifest(manifest: str | Path) -> dict[str, Any]:
    p = Path(manifest)
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    variants = Counter(row.get("variant", "") for row in rows)
    splits = Counter(row.get("split", "") for row in rows)
    case_ids = {row.get("case_id", "") for row in rows if row.get("case_id")}
    workspace = p.resolve().parent.parent
    def _exists(value: str) -> bool:
        q = Path(value)
        if not q.is_absolute():
            q = workspace / q
        return q.exists()
    missing_inputs = [row.get("input_file", "") for row in rows if row.get("input_file") and not _exists(row.get("input_file", ""))]
    missing_labels = [row.get("label_json", "") for row in rows if row.get("label_json") and not _exists(row.get("label_json", ""))]
    return {
        "manifest": str(p.resolve()),
        "records": len(rows),
        "case_count": len(case_ids),
        "variants": dict(variants),
        "splits": dict(splits),
        "missing_input_files": len(missing_inputs),
        "missing_label_files": len(missing_labels),
        "missing_input_examples": missing_inputs[:10],
        "missing_label_examples": missing_labels[:10],
    }


def _collect_qc_reports(workspace: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(workspace.glob("**/qc_report.json")):
        data = _read_json(p)
        out.append({
            "path": str(p),
            "bscan_shape": data.get("bscan_shape", ""),
            "snr_raw_db": data.get("snr_raw_db_default_windows", ""),
            "snr_background_removed_db": data.get("snr_background_removed_db_default_windows", ""),
            "out_dir": data.get("out_dir", ""),
        })
    return out


def _collect_soft_mask_reports(workspace: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(workspace.glob("**/soft_mask_report.json")):
        data = _read_json(p)
        out.append({
            "path": str(p),
            "shape": data.get("shape", ""),
            "total_picks_loaded": data.get("total_picks_loaded", ""),
            "total_picks_used": data.get("total_picks_used", ""),
            "mask_nonzero_fraction": data.get("mask_nonzero_fraction", ""),
            "mask_png": data.get("mask_png", ""),
        })
    return out


def _collect_products(workspace: Path) -> dict[str, int]:
    suffixes = [".npz", ".npy", ".csv", ".png", ".json", ".h5", ".out", ".in"]
    counts: dict[str, int] = {}
    for suf in suffixes:
        counts[suf] = len(list(workspace.glob(f"**/*{suf}")))
    return counts


def build_auto_report(workspace: str | Path, out_dir: str | Path | None = None, title: str = "UavGPR-SimLab 自动实验报告") -> dict[str, Any]:
    """Collect machine-readable and paper-facing summaries for a workspace."""

    root = Path(workspace)
    out = Path(out_dir) if out_dir else root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    manifest_paths = sorted((root / "datasets").glob("*_manifest.csv")) if (root / "datasets").exists() else sorted(root.glob("**/*_manifest.csv"))
    manifest_summaries = [summarize_manifest(p) for p in manifest_paths]
    dataset_summary_path = _first_existing([root / "reports" / "dataset_summary.json", root / "dataset_summary.json"])
    environment_report_path = _first_existing([root / "reports" / "environment_report.json", root / "environment_report.json"])
    dataset_summary = _read_json(dataset_summary_path) if dataset_summary_path else {}
    environment_report = _read_json(environment_report_path) if environment_report_path else {}
    qc_reports = _collect_qc_reports(root)
    soft_mask_reports = _collect_soft_mask_reports(root)
    product_counts = _collect_products(root)

    paper_tables = root / "paper" / "tables"
    manifest_table_rows: list[dict[str, Any]] = []
    for ms in manifest_summaries:
        manifest_table_rows.append({
            "manifest": ms["manifest"],
            "case_count": ms["case_count"],
            "records": ms["records"],
            "variants": json.dumps(ms["variants"], ensure_ascii=False),
            "splits": json.dumps(ms["splits"], ensure_ascii=False),
            "missing_input_files": ms["missing_input_files"],
            "missing_label_files": ms["missing_label_files"],
        })
    _write_csv(paper_tables / "manifest_summary.csv", manifest_table_rows)
    _write_csv(paper_tables / "real_qc_summary.csv", qc_reports)
    _write_csv(paper_tables / "soft_mask_summary.csv", soft_mask_reports)

    issues: list[str] = []
    if not manifest_summaries:
        issues.append("未发现 datasets/*_manifest.csv；请先运行 generate 或 pipeline 生成仿真清单。")
    for ms in manifest_summaries:
        if ms["missing_input_files"]:
            issues.append(f"Manifest 中有 {ms['missing_input_files']} 个 input_file 不存在：{Path(ms['manifest']).name}")
        if ms["missing_label_files"]:
            issues.append(f"Manifest 中有 {ms['missing_label_files']} 个 label_json 不存在：{Path(ms['manifest']).name}")
    if not qc_reports:
        issues.append("未发现实测 CSV 质控 qc_report.json；如需论文实测指标，请运行 preview-csv --convert。")
    if not soft_mask_reports:
        issues.append("未发现 soft_mask_report.json；如需钻孔弱监督，请运行 soft-mask。")

    report = {
        "workspace": str(root.resolve()),
        "title": title,
        "dataset_summary": dataset_summary,
        "environment_report": environment_report,
        "manifests": manifest_summaries,
        "qc_reports": qc_reports,
        "soft_mask_reports": soft_mask_reports,
        "product_counts": product_counts,
        "paper_tables": {
            "manifest_summary_csv": str((paper_tables / "manifest_summary.csv").resolve()),
            "real_qc_summary_csv": str((paper_tables / "real_qc_summary.csv").resolve()),
            "soft_mask_summary_csv": str((paper_tables / "soft_mask_summary.csv").resolve()),
        },
        "issues": issues,
    }

    json_path = out / "auto_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out / "auto_report.md"
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    report["auto_report_json"] = str(json_path.resolve())
    report["auto_report_md"] = str(md_path.resolve())
    return report


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_无记录。_\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return "\n".join(lines) + "\n"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {report.get('title', 'UavGPR-SimLab 自动实验报告')}\n")
    lines.append(f"工作区：`{report.get('workspace', '')}`\n")
    lines.append("## 1. 数据集与 Manifest 概况\n")
    rows = []
    for ms in report.get("manifests", []):
        rows.append([
            Path(ms.get("manifest", "")).name,
            ms.get("case_count", 0),
            ms.get("records", 0),
            json.dumps(ms.get("variants", {}), ensure_ascii=False),
            json.dumps(ms.get("splits", {}), ensure_ascii=False),
            ms.get("missing_input_files", 0),
        ])
    lines.append(_md_table(["manifest", "case", "records", "variants", "splits", "missing inputs"], rows))

    lines.append("\n## 2. 实测 CSV 质控与传统基线\n")
    rows = []
    for qc in report.get("qc_reports", []):
        rows.append([Path(qc.get("path", "")).parent.name, qc.get("bscan_shape", ""), qc.get("snr_raw_db", ""), qc.get("snr_background_removed_db", "")])
    lines.append(_md_table(["目录", "B-scan shape", "Raw SNR(dB)", "BG-removed SNR(dB)"], rows))

    lines.append("\n## 3. 钻孔/界面弱监督 Soft Mask\n")
    rows = []
    for sm in report.get("soft_mask_reports", []):
        rows.append([Path(sm.get("path", "")).parent.name, sm.get("shape", ""), sm.get("total_picks_loaded", ""), sm.get("total_picks_used", ""), sm.get("mask_nonzero_fraction", "")])
    lines.append(_md_table(["目录", "shape", "picks loaded", "picks used", "nonzero fraction"], rows))

    lines.append("\n## 4. 文件产物统计\n")
    counts = report.get("product_counts", {})
    lines.append(_md_table(["后缀", "数量"], [[k, v] for k, v in sorted(counts.items())]))

    lines.append("\n## 5. 需要人工复核或下一步动作\n")
    issues = report.get("issues", [])
    if issues:
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- 暂未发现阻塞项。建议继续运行 gprMax 正式仿真、PGDA-CSNet 训练与按线留出评估。")
    lines.append("")

    lines.append("## 6. 论文可直接引用的表格文件\n")
    for key, value in report.get("paper_tables", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)
