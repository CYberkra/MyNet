from __future__ import annotations

import json
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QListWidgetItem, QMenu, QMessageBox, QTreeWidgetItem

from uavgpr_simlab.gui.pages.history_page import build_history_page
from uavgpr_simlab.gui.easy_ui import _friendly_variant_name
from uavgpr_simlab.gui.easy_cards import history_record_card
from uavgpr_simlab.services.easy_history_service import (
    ALL_STATUS_FILTER,
    build_history_detail,
    delete_history_entry,
    export_history_report,
    load_case_variant_bscan,
    scan_history_entries,
)


class HistoryControllerMixin:
    # --- history page ---
    def _build_history_page(self) -> None:
        widgets = build_history_page(
            status_filters=[ALL_STATUS_FILTER, "pending", "running", "done", "failed", "stale_running"],
            on_refresh=self.refresh_history,
            on_export=self.export_history,
            on_rerun=lambda: self.show_page(2),
            on_delete=self.delete_selected_history,
            on_preview_selection=self.preview_selected_history,
            on_show_failure=self.show_selected_history_failure_location,
            on_rerun_failed=self.prepare_rerun_failed_from_history,
            on_context_menu=self.show_history_context_menu,
        )
        self.history_status_filter = widgets.history_status_filter
        self.history_list = widgets.history_list
        self.history_tree = widgets.history_tree
        self.history_bscan_mode = widgets.history_bscan_mode
        self.history_failure_box = widgets.history_failure_box
        self.history_model_canvas_easy = widgets.history_model_canvas_easy
        self.history_bscan_canvas_easy = widgets.history_bscan_canvas_easy
        self.history_detail_easy = widgets.history_detail_easy
        self.stack.addWidget(widgets.page)

    def refresh_history(self) -> None:
        workspace = self._workspace_from_manifest()
        self.history_list.clear()
        if hasattr(self, "history_tree"):
            self.history_tree.clear()
        filt = self.history_status_filter.currentText() if hasattr(self, "history_status_filter") else ALL_STATUS_FILTER
        try:
            entries = scan_history_entries(workspace, status_filter=filt, limit=300, make_png=True)
        except Exception as exc:
            self.history_detail_easy.setText(f"读取历史失败：{exc}")
            return
        self._history_entry_by_job = {}
        for entry in entries:
            self._history_entry_by_job[entry.record.job_id] = entry
            item = QListWidgetItem()
            item.setSizeHint(QSize(390, 132))
            item.setData(Qt.UserRole, entry)
            self.history_list.addItem(item)
            self.history_list.setItemWidget(item, history_record_card(entry))
        self._populate_history_tree(entries)
        if self.history_tree.topLevelItemCount() > 0:
            self.history_tree.expandToDepth(1)
            first_dataset = self.history_tree.topLevelItem(0)
            if first_dataset and first_dataset.childCount() > 0 and first_dataset.child(0).childCount() > 0:
                self.history_tree.setCurrentItem(first_dataset.child(0).child(0))
        elif self.history_list.count() > 0:
            self.history_list.setCurrentRow(0)
        else:
            self.history_detail_easy.setText("当前 workspace 没有已导入的数据集或历史仿真记录。导入骨架后，这里也会显示即将要跑的模型。")
            if hasattr(self, "history_failure_box"):
                self.history_failure_box.clear()

    def _populate_history_tree(self, entries: list[object]) -> None:
        """Populate dataset/case/variant tree without introducing a database."""
        if not hasattr(self, "history_tree"):
            return
        grouped: dict[str, dict[str, list[object]]] = {}
        for entry in entries:
            marker = getattr(entry, "marker", {}) or {}
            dataset = Path(str(marker.get("manifest") or self._workspace_from_manifest())).parent.parent.name if marker.get("manifest") else "legacy_jobs"
            cid = entry.record.case_id or "unknown_case"
            grouped.setdefault(dataset, {}).setdefault(cid, []).append(entry)
        for dataset, cases in grouped.items():
            total = sum(len(v) for v in cases.values())
            failed = sum(1 for variants in cases.values() for e in variants if e.record.status == "failed")
            done = sum(1 for variants in cases.values() for e in variants if e.record.status == "done")
            top = QTreeWidgetItem([dataset, f"done={done} failed={failed}", "dataset", f"{len(cases)} cases / {total} records"])
            top.setData(0, Qt.UserRole, "")
            self.history_tree.addTopLevelItem(top)
            for cid, variants in sorted(cases.items()):
                fam = ""
                for e in variants:
                    fam = str((e.marker or {}).get("family") or fam)
                statuses = {str(e.record.status) for e in variants}
                if "failed" in statuses:
                    case_status = "failed"
                elif "running" in statuses or "stale_running" in statuses:
                    case_status = "running"
                elif statuses and statuses <= {"done", "skipped"}:
                    case_status = "done"
                elif statuses and statuses <= {"pending"}:
                    case_status = "pending"
                else:
                    case_status = "mixed"
                case_item = QTreeWidgetItem([cid, case_status, fam, f"{len(variants)} variants"])
                case_item.setData(0, Qt.UserRole, "")
                top.addChild(case_item)
                order = {"raw": 0, "target_only": 1, "background_only": 2, "clutter_only": 3, "air_only": 4, "clutter_gt": 5}
                for entry in sorted(variants, key=lambda e: order.get(e.record.variant, 99)):
                    rec = entry.record
                    note = str((entry.marker or {}).get("bscan_error") or "")
                    node = QTreeWidgetItem([_friendly_variant_name(rec.variant), rec.status, rec.variant, note[:120]])
                    node.setData(0, Qt.UserRole, rec.job_id)
                    case_item.addChild(node)
        self.history_tree.resizeColumnToContents(0)

    def _selected_history_entry(self):
        if hasattr(self, "history_tree"):
            item = self.history_tree.currentItem()
            if item is not None:
                job_id = item.data(0, Qt.UserRole)
                if job_id and job_id in self._history_entry_by_job:
                    return self._history_entry_by_job[job_id]
        items = self.history_list.selectedItems()
        if items:
            return items[0].data(Qt.UserRole)
        return None

    def select_history_entry_by_case_variant(self, case_id: str, variant: str = "") -> bool:
        """Select a history tree/list entry from an operator queue jump."""

        target = None
        for entry in getattr(self, "_history_entry_by_job", {}).values():
            rec = entry.record
            if rec.case_id == case_id and (not variant or rec.variant == variant):
                target = entry
                break
        if target is None:
            return False
        target_job = target.record.job_id

        def walk(node):
            if node.data(0, Qt.UserRole) == target_job:
                return node
            for idx in range(node.childCount()):
                found = walk(node.child(idx))
                if found is not None:
                    return found
            return None

        if hasattr(self, "history_tree"):
            for idx in range(self.history_tree.topLevelItemCount()):
                found = walk(self.history_tree.topLevelItem(idx))
                if found is not None:
                    self.history_tree.setCurrentItem(found)
                    self.preview_selected_history()
                    return True
        if hasattr(self, "history_list"):
            for idx in range(self.history_list.count()):
                item = self.history_list.item(idx)
                entry = item.data(Qt.UserRole)
                if entry is not None and entry.record.job_id == target_job:
                    self.history_list.setCurrentItem(item)
                    self.preview_selected_history()
                    return True
        return False

    def preview_selected_history(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            return
        rec = entry.record
        pv = entry.preview
        detail = build_history_detail(entry, self._workspace_from_manifest())
        if detail.label_json:
            self.history_model_canvas_easy.show_label_json(detail.label_json)
        mode = self.history_bscan_mode.currentText() if hasattr(self, "history_bscan_mode") else "当前记录"
        if mode.startswith("对比"):
            case_dir = (entry.marker or {}).get("case_dir")
            bscans = []
            if case_dir:
                for variant in ["raw", "target_only", "clutter_gt"]:
                    arr, _meta = load_case_variant_bscan(case_dir, variant)
                    if arr is not None:
                        bscans.append((_friendly_variant_name(variant), arr))
            if bscans:
                self.history_bscan_canvas_easy.show_bscan_grid(bscans, f"{rec.case_id} | raw / target / clutter_gt", float(detail.bscan_meta.get("time_window_ns", 700.0)))
        elif mode != "当前记录":
            case_dir = (entry.marker or {}).get("case_dir")
            arr = None
            meta = {}
            if case_dir:
                arr, meta = load_case_variant_bscan(case_dir, mode)
            if arr is not None:
                self.history_bscan_canvas_easy.show_bscan(arr, f"{rec.case_id} | {_friendly_variant_name(mode)}", float(detail.bscan_meta.get("time_window_ns", 700.0)))
        elif detail.bscan is not None:
            self.history_bscan_canvas_easy.show_bscan(
                detail.bscan,
                f"{rec.case_id} | {_friendly_variant_name(rec.variant)} | 已完成 {detail.bscan.shape[1]} / {rec.n_traces} 道",
                float(detail.bscan_meta.get("time_window_ns", 700.0)),
            )
        self.history_detail_easy.setText(detail.detail_text)
        if hasattr(self, "history_failure_box"):
            marker = entry.marker or {}
            variant_qc = marker.get("variant_qc", {}) if isinstance(marker, dict) else {}
            paths = [
                f"manifest: {marker.get('manifest', '—')}",
                f"case_dir: {marker.get('case_dir', '—')}",
                f"bscan: {marker.get('bscan_npy', '—')}",
                f"qc: {marker.get('qc_report_json', '—')}",
                f"status: {rec.status}",
                f"reason: {marker.get('bscan_error') or (variant_qc.get('reason') if isinstance(variant_qc, dict) else '') or '—'}",
            ]
            self.history_failure_box.setPlainText("\n".join(paths))


    def show_history_context_menu(self, global_pos: object) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            return
        menu = QMenu(self)
        open_case = menu.addAction("打开 case 文件夹")
        open_qc = menu.addAction("打开 QC JSON")
        copy_reason = menu.addAction("复制失败原因")
        copy_paths = menu.addAction("复制路径摘要")
        menu.addSeparator()
        locate = menu.addAction("显示失败定位")
        rerun_failed = menu.addAction("只重跑 failed")
        chosen = menu.exec(global_pos)
        if chosen == open_case:
            self.open_selected_history_case_folder()
        elif chosen == open_qc:
            self.open_selected_history_qc_json()
        elif chosen == copy_reason:
            self.copy_selected_history_failure_reason()
        elif chosen == copy_paths:
            self.copy_selected_history_path_summary()
        elif chosen == locate:
            self.show_selected_history_failure_location()
        elif chosen == rerun_failed:
            self.prepare_rerun_failed_from_history()

    def _open_history_path(self, raw: object, title: str) -> None:
        if not raw:
            QMessageBox.information(self, title, "当前记录没有对应路径。")
            return
        path = Path(str(raw)).expanduser()
        if not path.exists():
            QMessageBox.warning(self, title, f"路径不存在：\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def open_selected_history_case_folder(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            return
        marker = entry.marker or {}
        self._open_history_path(marker.get("case_dir") or Path(entry.record.input_file).parent, "打开 case 文件夹")

    def open_selected_history_qc_json(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            return
        marker = entry.marker or {}
        self._open_history_path(marker.get("qc_report_json") or entry.record.marker_file, "打开 QC JSON")

    def _selected_history_path_summary(self) -> str:
        entry = self._selected_history_entry()
        if entry is None:
            return ""
        marker = entry.marker or {}
        variant_qc = marker.get("variant_qc", {}) if isinstance(marker, dict) else {}
        payload = {
            "case_id": entry.record.case_id,
            "variant": entry.record.variant,
            "status": entry.record.status,
            "reason": marker.get("bscan_error") or (variant_qc.get("reason") if isinstance(variant_qc, dict) else "") or "",
            "manifest": marker.get("manifest"),
            "case_dir": marker.get("case_dir"),
            "bscan_npy": marker.get("bscan_npy"),
            "qc_report_json": marker.get("qc_report_json"),
            "input_file": entry.record.input_file,
            "marker_file": entry.record.marker_file,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def copy_selected_history_failure_reason(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            return
        marker = entry.marker or {}
        variant_qc = marker.get("variant_qc", {}) if isinstance(marker, dict) else {}
        reason = marker.get("bscan_error") or (variant_qc.get("reason") if isinstance(variant_qc, dict) else "") or "无失败原因。"
        QApplication.clipboard().setText(str(reason))
        if hasattr(self, "history_failure_box"):
            self.history_failure_box.setPlainText(str(reason))

    def copy_selected_history_path_summary(self) -> None:
        text = self._selected_history_path_summary()
        if not text:
            return
        QApplication.clipboard().setText(text)
        if hasattr(self, "history_failure_box"):
            self.history_failure_box.setPlainText(text)

    def show_selected_history_failure_location(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            QMessageBox.information(self, "无选择", "请先选择一个 case / variant。")
            return
        marker = entry.marker or {}
        text = json.dumps({
            "case_id": entry.record.case_id,
            "variant": entry.record.variant,
            "status": entry.record.status,
            "manifest": marker.get("manifest"),
            "case_dir": marker.get("case_dir"),
            "bscan_npy": marker.get("bscan_npy"),
            "qc_report_json": marker.get("qc_report_json"),
            "bscan_error": marker.get("bscan_error"),
            "variant_qc": marker.get("variant_qc"),
        }, ensure_ascii=False, indent=2)
        if hasattr(self, "history_failure_box"):
            self.history_failure_box.setPlainText(text)
        QMessageBox.information(self, "失败定位 / 路径", text[:3500])

    def prepare_rerun_failed_from_history(self) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            QMessageBox.information(self, "无选择", "请先选择一个 SceneWorld 记录。")
            return
        manifest = (entry.marker or {}).get("manifest")
        if not manifest:
            QMessageBox.information(self, "不支持", "该记录不是 SceneWorld manifest 记录，不能用统一任务只重跑 failed。")
            return
        self.batch_manifest_edit.setText(str(manifest))
        self.current_manifest = Path(str(manifest))
        self.batch_failed_only.setChecked(True)
        self.batch_skip_done.setChecked(True)
        self.batch_force_rerun.setChecked(False)
        self.show_page(2)
        self.refresh_batch_plan()
        self.batch_log.appendPlainText("[history] 已切换到批量页，只重跑 failed。")

    def _tick_history_live(self) -> None:
        if self.stack.currentIndex() == 3:
            self.preview_selected_history()

    def export_history(self) -> None:
        try:
            rep = export_history_report(self._workspace_from_manifest())
            QMessageBox.information(self, "已导出", f"历史记录已导出：\n{rep.get('history_csv')}")
        except Exception:
            QMessageBox.critical(self, "导出失败", traceback.format_exc())

    def delete_selected_history(self, delete_outputs: bool) -> None:
        entry = self._selected_history_entry()
        if entry is None:
            return
        rec = entry.record
        marker_path = Path(str(rec.marker_file)) if rec.marker_file else None
        if marker_path is None or not marker_path.exists():
            QMessageBox.information(self, "不能删除", "当前记录是 manifest 中的 pending 虚拟记录，还没有独立历史 marker。请在批量页通过 manifest 管理该任务。")
            return
        if QMessageBox.question(self, "确认删除", f"删除记录 {rec.case_id} / {rec.variant}？") != QMessageBox.Yes:
            return
        try:
            rep = delete_history_entry(self._workspace_from_manifest(), marker_file=rec.marker_file, delete_outputs=delete_outputs)
            self.refresh_history()
            QMessageBox.information(self, "已删除", json.dumps(rep, ensure_ascii=False, indent=2))
        except Exception:
            QMessageBox.critical(self, "删除失败", traceback.format_exc())

