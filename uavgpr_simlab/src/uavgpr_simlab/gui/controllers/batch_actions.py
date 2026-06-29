from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from uavgpr_simlab.core.dataset_contract import validate_dataset_skeleton
from uavgpr_simlab.core.run_dashboard import summarize_dataset_run_dashboard
from uavgpr_simlab.core.workspace_relocator import relocate_workspace_paths
from uavgpr_simlab.services.easy_batch_service import parse_variants, require_manifest_file


def relocate_imported_workspace_paths(win: Any) -> None:
    """GUI action: dry-run and optionally apply workspace path relocation."""

    text = win.batch_manifest_edit.text().strip() or (str(win.current_manifest) if win.current_manifest else "")
    if not text:
        text, _ = QFileDialog.getOpenFileName(win, "选择需要迁移/修复的 manifest.csv", str(win.root_dir / "workspace"), "Manifest CSV (*.csv);;All Files (*)")
    if not text:
        return
    try:
        manifest = require_manifest_file(text)
        old_root, ok = QInputDialog.getText(win, "旧 workspace 根目录（可留空）", "填旧数据集根目录；留空则按当前数据集文件夹名推断。")
        if not ok:
            return
        old_arg = old_root.strip() or None
        dry = relocate_workspace_paths(manifest, old_root=old_arg, dry_run=True, write_report=True)
    except Exception:
        QMessageBox.critical(win, "迁移预检失败", traceback.format_exc())
        return
    win.batch_log.setPlainText(json.dumps(dry.to_dict(), ensure_ascii=False, indent=2))
    if dry.error_count:
        QMessageBox.warning(win, "存在无法自动修复的路径", f"发现 {dry.error_count} 个问题，请查看日志和报告：\n{dry.report_json or ''}")
        return
    if dry.change_count == 0:
        QMessageBox.information(win, "无需迁移", f"未发现需要自动修复的绝对路径。\n报告：{dry.report_json or '未写入'}")
        return
    msg = f"发现 {dry.change_count} 处可修复路径，涉及 {dry.changed_file_count} 个文件。\n将创建备份并改为可迁移路径。\n\n是否应用？"
    if QMessageBox.question(win, "确认写入路径修复", msg) != QMessageBox.Yes:
        return
    try:
        rep = relocate_workspace_paths(manifest, old_root=old_arg, dry_run=False, write_report=True)
    except Exception:
        QMessageBox.critical(win, "路径修复失败", traceback.format_exc())
        return
    win.batch_log.setPlainText(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    win.current_manifest = manifest
    win.batch_manifest_edit.setText(str(manifest))
    win.refresh_batch_plan(); win.refresh_history()
    QMessageBox.information(win, "路径修复完成", f"已修复 {rep.change_count} 处路径。\n备份：{rep.backup_dir or '未创建'}\n报告：{rep.report_json or '未写入'}")


def import_dataset_skeleton(win: Any) -> None:
    """GUI action: import a ready-to-run dataset skeleton and refresh dashboards."""

    path, _ = QFileDialog.getOpenFileName(win, "导入数据集骨架 manifest.csv", str(win.root_dir / "workspace"), "Manifest CSV (*.csv);;All Files (*)")
    if not path:
        return
    try:
        manifest = require_manifest_file(path)
        variants = parse_variants(win.batch_variants_edit.text())
        contract = validate_dataset_skeleton(manifest, expected_variants=variants, write_report=True)
        dashboard = summarize_dataset_run_dashboard(manifest, expected_variants=variants, write_report=True)
    except Exception:
        QMessageBox.critical(win, "导入失败", traceback.format_exc())
        return
    win.current_manifest = manifest
    win.current_manifest_rows = win._load_manifest_rows(manifest)
    win.batch_manifest_edit.setText(str(manifest))
    if hasattr(win, "model_manifest_edit"):
        win.model_manifest_edit.setText(str(manifest))
    if hasattr(win, "workspace_edit"):
        win.workspace_edit.setText(str(manifest.parent.parent))
    win.batch_dataset_status_label.setText(dashboard.format_for_operator())
    if contract.ok:
        QMessageBox.information(win, "数据集骨架已导入", f"合同检查通过，可进入预检/一键运行。\n\n{dashboard.format_for_operator()}\n\n合同报告：\n{contract.summary_json or '未写入'}")
    else:
        details = "\n".join(f"- {x.code}: {x.message}" for x in contract.issues[:10] if x.level == "error")
        QMessageBox.warning(win, "数据集骨架未通过", f"已导入但不能启动仿真，请先修复：\n\n{details}")
    win.load_model_manifest(silent=True)
    win.refresh_batch_plan()
    win.refresh_history()
