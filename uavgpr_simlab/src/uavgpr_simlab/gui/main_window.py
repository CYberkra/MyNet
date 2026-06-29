from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from uavgpr_simlab import __display_version__
from uavgpr_simlab.cli import _cfg_from_plan
from uavgpr_simlab.core.config import (
    AppConfig,
    load_yaml,
    read_simlab_env,
    repo_root,
    save_config,
    write_simlab_env,
)
from uavgpr_simlab.core.environment import run_environment_checks, save_report
from uavgpr_simlab.core.postprocess import export_gprmax_bscan_for_input
from uavgpr_simlab.core.real_data import robust_normalize
from uavgpr_simlab.core.runner import (
    GprMaxRunOptions,
    GprMaxTask,
    build_gprmax_command,
    command_to_string,
    tasks_from_manifest,
)
from uavgpr_simlab.core.scenario import generate_cases, write_manifest_commands_bat
from uavgpr_simlab.core.job_registry import build_job_records, write_job_plan
from uavgpr_simlab.core.history import scan_simulation_history, export_history_csv, delete_history_record, _read_json
from uavgpr_simlab.core.visual_history import build_history_preview, load_bscan_for_history
from uavgpr_simlab.gui.advanced_pages import (
    build_advanced_dashboard_tab,
    build_advanced_env_tab,
    build_advanced_generation_tab,
    build_advanced_history_tab,
    build_advanced_model_preview_tab,
    build_advanced_preflight_tab,
    build_advanced_qc_tab,
    build_advanced_queue_tab,
    build_advanced_real_csv_tab,
    build_advanced_train_tab,
)
from uavgpr_simlab.gui.advanced_widgets import MplCanvas, Model3DCanvas
from uavgpr_simlab.gui.advanced_workers import GenericWorker, LiveQueueWorker
from uavgpr_simlab.services.advanced_queue_service import (
    build_advanced_queue_tasks,
    read_queue_manifest_preview,
    summarize_queue_tasks,
    write_advanced_queue_bat,
)
from uavgpr_simlab.services.real_csv_service import export_real_csv_qc, load_real_csv_preview


APP_STYLE = """
QMainWindow, QWidget { background: #f5f7fb; color: #172033; }
QTabWidget::pane { border: 1px solid #d6deeb; border-radius: 8px; background: #ffffff; }
QTabBar::tab { background: #e9eef7; padding: 10px 16px; margin-right: 2px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { background: #164a7a; color: white; font-weight: 600; }
QGroupBox { border: 1px solid #cfdae9; border-radius: 10px; margin-top: 14px; padding: 12px; background: #ffffff; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #164a7a; }
QLineEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QTableWidget { background: #ffffff; border: 1px solid #c7d2e3; border-radius: 6px; padding: 4px; }
QPushButton { background: #1f6aa5; color: white; border: none; border-radius: 7px; padding: 7px 12px; font-weight: 600; }
QPushButton:hover { background: #155986; }
QPushButton:disabled { background: #9badbf; }
QLabel#banner { background: #102c4a; color: white; border-radius: 10px; padding: 12px 16px; font-size: 17px; font-weight: 700; }
QLabel#subtle { color: #576579; }
QProgressBar { border: 1px solid #c7d2e3; border-radius: 6px; background: #ffffff; text-align: center; }
QProgressBar::chunk { background: #1f6aa5; border-radius: 6px; }
"""


def _parse_gpu_ids(text: str) -> list[int]:
    out: list[int] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out or [0]


class MainWindow(QMainWindow):
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle(f"UavGPR-SimLab {__display_version__} - 高级工程界面 · 可视化历史与自动化批量仿真工作台")
        self.resize(1380, 860)
        self.setMinimumSize(1180, 740)
        self.config_path = config_path
        self.cfg = AppConfig()
        env = read_simlab_env()
        if env.get("UAVGPR_GPRMAX_ROOT"):
            self.cfg.runtime.gprmax_source_dir = env["UAVGPR_GPRMAX_ROOT"]
        if env.get("UAVGPR_CONDA_ENV"):
            self.cfg.runtime.conda_env_gprmax = env["UAVGPR_CONDA_ENV"]
        if env.get("UAVGPR_GPU_IDS"):
            self.cfg.runtime.gpu_ids = env["UAVGPR_GPU_IDS"]
        self.current_manifest: Optional[Path] = None
        self.worker: Optional[GenericWorker] = None
        self.live_worker: Optional[LiveQueueWorker] = None
        self.last_preview: Optional[np.ndarray] = None

        QApplication.instance().setStyleSheet(APP_STYLE)  # type: ignore[union-attr]
        root = QWidget()
        root_layout = QVBoxLayout(root)
        banner = QLabel(f"UavGPR-SimLab {__display_version__}：高级工程界面 · 3D模型预览 · 实时B-scan历史 · 预检去重 · 批量运行 · 自动报告")
        banner.setObjectName("banner")
        root_layout.addWidget(banner)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self._build_dashboard_tab()
        self._build_env_tab()
        self._build_generation_tab()
        self._build_model_preview_tab()
        self._build_preflight_tab()
        self._build_queue_tab()
        self._build_history_tab()
        self._build_real_csv_tab()
        self._build_qc_tab()
        self._build_train_tab()
        self.statusBar().showMessage("Ready")


    # ---------- v0.5 dashboard / guide ----------
    def _build_dashboard_tab(self) -> None:
        widgets = build_advanced_dashboard_tab(
            on_jump_env=self._jump_env,
            on_jump_generation=self._jump_generation,
            on_jump_preview=self._jump_preview,
            on_jump_preflight=self._jump_preflight,
            on_jump_queue=self._jump_queue,
            on_jump_history=self._jump_history,
        )
        self.advanced_dashboard_widgets = widgets
        self.dashboard_summary = widgets.summary
        self.dashboard_env_button = widgets.env_button
        self.dashboard_generation_button = widgets.generation_button
        self.dashboard_preview_button = widgets.preview_button
        self.dashboard_preflight_button = widgets.preflight_button
        self.dashboard_queue_button = widgets.queue_button
        self.dashboard_history_button = widgets.history_button
        self.tabs.addTab(widgets.page, "0 工作台")

    def _jump_env(self): self.tabs.setCurrentIndex(1)
    def _jump_generation(self): self.tabs.setCurrentIndex(2)
    def _jump_preview(self): self.tabs.setCurrentIndex(3)
    def _jump_preflight(self): self.tabs.setCurrentIndex(4)
    def _jump_queue(self): self.tabs.setCurrentWidget(self.queue_tab)
    def _jump_history(self): self.tabs.setCurrentIndex(6)

    # ---------- v0.5 3D model preview ----------
    def _build_model_preview_tab(self) -> None:
        widgets = build_advanced_model_preview_tab(
            choose_manifest=self._choose_file,
            on_load_manifest=self.load_preview_manifest,
            on_selection_changed=self.preview_selected_case,
            canvas_factory=Model3DCanvas,
        )
        self.advanced_model_preview_widgets = widgets
        self.preview_manifest_edit = widgets.manifest_edit
        self.preview_case_table = widgets.case_table
        self.preview_info_box = widgets.info_box
        self.model3d_canvas = widgets.model_canvas
        self.preview_manifest_pick_button = widgets.pick_button
        self.preview_manifest_load_button = widgets.load_button
        self.preview_note_label = widgets.note_label
        self.tabs.addTab(widgets.page, "3 3D预览")

    def load_preview_manifest(self) -> None:
        path = Path(self.preview_manifest_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "无 manifest", "请先生成或选择 manifest.csv")
            return
        unique = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id") or ""
                if cid and cid not in unique:
                    unique[cid] = row
        self.preview_case_table.setRowCount(0)
        for row in unique.values():
            r = self.preview_case_table.rowCount(); self.preview_case_table.insertRow(r)
            label = row.get("label_json", "")
            if label and not Path(label).is_absolute():
                label = str((path.parent.parent / label).resolve())
            vals = [row.get("case_id", ""), row.get("split", ""), row.get("variant", ""), row.get("interface_depth_mean_m", ""), row.get("slope_deg", ""), label]
            for c, v in enumerate(vals):
                self.preview_case_table.setItem(r, c, QTableWidgetItem(str(v)))
        self.preview_info_box.setPlainText(f"已加载 {len(unique)} 个唯一 case。双击/选中任一行查看模型预览。")

    def preview_selected_case(self) -> None:
        rows = self.preview_case_table.selectionModel().selectedRows() if self.preview_case_table.selectionModel() else []
        if not rows:
            return
        r = rows[0].row()
        item = self.preview_case_table.item(r, 5)
        if not item:
            return
        rep = self.model3d_canvas.show_label_json(item.text())
        self.preview_info_box.setPlainText(json.dumps(rep, ensure_ascii=False, indent=2))

    # ---------- v0.5 preflight / dedup ----------
    def _build_preflight_tab(self) -> None:
        widgets = build_advanced_preflight_tab(
            choose_manifest=self._choose_file,
            on_run_preflight=self.run_preflight,
            on_sync_to_queue=self.sync_preflight_to_queue,
        )
        self.advanced_preflight_widgets = widgets
        self.preflight_manifest_edit = widgets.manifest_edit
        self.preflight_variants_edit = widgets.variants_edit
        self.preflight_skip_check = widgets.skip_check
        self.preflight_limit_spin = widgets.limit_spin
        self.preflight_summary = widgets.summary
        self.preflight_table = widgets.table
        self.preflight_pick_button = widgets.pick_button
        self.preflight_plan_button = widgets.plan_button
        self.preflight_sync_button = widgets.sync_button
        self.tabs.addTab(widgets.page, "4 预检去重")

    def _variant_list_from_text(self, text: str) -> list[str]:
        return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]

    def run_preflight(self) -> None:
        path = Path(self.preflight_manifest_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "无 manifest", "请先生成或选择 manifest.csv")
            return
        workspace = path.parent.parent
        variants = self._variant_list_from_text(self.preflight_variants_edit.text())
        records = build_job_records(path, workspace, variants=variants, max_tasks=int(self.preflight_limit_spin.value()), skip_completed=self.preflight_skip_check.isChecked())
        out = workspace / "jobs" / "job_plan.csv"
        rep = write_job_plan(path, workspace, out, variants=variants, max_tasks=int(self.preflight_limit_spin.value()), skip_completed=self.preflight_skip_check.isChecked())
        self.preflight_table.setRowCount(0)
        counts = {}
        for rec in records:
            counts[rec.status] = counts.get(rec.status, 0) + 1
            r = self.preflight_table.rowCount(); self.preflight_table.insertRow(r)
            vals = [rec.status, rec.case_id, rec.variant, str(rec.n_traces), rec.job_id, rec.reason, rec.input_file, rec.fingerprint[:16]]
            for c, v in enumerate(vals):
                self.preflight_table.setItem(r, c, QTableWidgetItem(str(v)))
        self.preflight_summary.setText(f"预检完成：总任务 {len(records)} | 待运行 {counts.get('pending',0)} | 将跳过 {counts.get('skipped',0)} | job_plan: {rep.get('job_plan_csv')}")
        self.statusBar().showMessage("Preflight finished")

    def sync_preflight_to_queue(self) -> None:
        if self.preflight_manifest_edit.text().strip():
            self.manifest_edit.setText(self.preflight_manifest_edit.text().strip())
            self.load_manifest()
            QMessageBox.information(self, "已同步", "manifest 已同步到批量运行页。")

    # ---------- v0.5.2 visual history ----------
    def _build_history_tab(self) -> None:
        widgets = build_advanced_history_tab(
            choose_dir=self._choose_dir,
            on_refresh=self.refresh_history,
            on_export=self.export_history,
            on_delete=lambda: self.delete_selected_history(False),
            on_delete_outputs=lambda: self.delete_selected_history(True),
            on_filter_changed=self.refresh_history,
            on_selection_changed=self.preview_history_selected,
            model_canvas_factory=Model3DCanvas,
            bscan_canvas_factory=lambda: MplCanvas("历史 / 实时 B-scan 预览"),
            workspace_default=repo_root() / "workspace",
        )
        self.advanced_history_widgets = widgets
        self.history_workspace_edit = widgets.workspace_edit
        self.history_filter = widgets.filter_combo
        self.history_thumb_check = widgets.thumb_check
        self.history_autorefresh_check = widgets.autorefresh_check
        self.history_limit_spin = widgets.limit_spin
        self.history_summary = widgets.summary
        self.history_table = widgets.table
        self.history_log = widgets.log
        self.history_model_canvas = widgets.model_canvas
        self.history_bscan_canvas = widgets.bscan_canvas
        self.history_detail_box = widgets.detail_box
        self.history_choose_button = widgets.choose_button
        self.history_refresh_button = widgets.refresh_button
        self.history_export_button = widgets.export_button
        self.history_delete_button = widgets.delete_button
        self.history_delete_outputs_button = widgets.delete_outputs_button
        self.history_records = []
        self.history_timer = QTimer(self)
        self.history_timer.setInterval(2500)
        self.history_timer.timeout.connect(self._history_auto_refresh_tick)
        self.history_timer.start()
        self.tabs.addTab(widgets.page, "6 历史记录")

    def _history_auto_refresh_tick(self) -> None:
        if not hasattr(self, "history_autorefresh_check") or not self.history_autorefresh_check.isChecked():
            return
        # Keep the selected running task live without forcing the user to click.
        try:
            self.preview_history_selected()
        except Exception:
            pass
        # Periodically refresh the table so new running/done markers appear.
        now = time.monotonic()
        last = getattr(self, "_history_last_full_refresh", 0.0)
        if now - last > 8.0:
            self.refresh_history(preserve_selection=True)
            self._history_last_full_refresh = now

    def _set_thumb_cell(self, row: int, col: int, image_path: str, fallback: str) -> None:
        lab = QLabel(); lab.setAlignment(Qt.AlignCenter); lab.setWordWrap(True)
        if image_path and Path(image_path).exists():
            pix = QPixmap(image_path)
            if not pix.isNull():
                lab.setPixmap(pix.scaled(140, 82, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                lab.setText(fallback)
        else:
            lab.setText(fallback)
        self.history_table.setCellWidget(row, col, lab)
        self.history_table.setRowHeight(row, 92)

    def _history_marker_data(self, marker_file: str) -> dict:
        try:
            return _read_json(Path(marker_file))
        except Exception:
            return {}

    def refresh_history(self, preserve_selection: bool = False) -> None:
        workspace = Path(self.history_workspace_edit.text())
        selected_job = ""
        if preserve_selection and hasattr(self, "history_table"):
            rows = self.history_table.selectionModel().selectedRows() if self.history_table.selectionModel() else []
            if rows and self.history_table.item(rows[0].row(), 10):
                selected_job = self.history_table.item(rows[0].row(), 10).text()
        records = scan_simulation_history(workspace)
        filt = self.history_filter.currentText() if hasattr(self, "history_filter") else "全部"
        if filt == "running": records = [r for r in records if r.status == "running"]
        elif filt == "done": records = [r for r in records if r.status == "done"]
        elif filt == "failed": records = [r for r in records if r.status == "failed"]
        elif filt == "stale_running": records = [r for r in records if r.status == "stale_running"]
        elif filt == "geometry-only": records = [r for r in records if r.geometry_only]
        elif filt == "full simulation": records = [r for r in records if not r.geometry_only]
        max_rows = int(self.history_limit_spin.value()) if hasattr(self, "history_limit_spin") else 300
        records = records[:max_rows]
        self.history_records = records
        self.history_table.blockSignals(True)
        self.history_table.setRowCount(0)
        counts = {}
        thumb = self.history_thumb_check.isChecked() if hasattr(self, "history_thumb_check") else True
        for rec in records:
            counts[rec.status] = counts.get(rec.status, 0) + 1
            marker_data = self._history_marker_data(rec.marker_file)
            pv = build_history_preview(rec, workspace, marker_data=marker_data, time_window_ns=float(self.cfg.radar.time_window_ns), make_png=thumb)
            r = self.history_table.rowCount(); self.history_table.insertRow(r)
            vals = [rec.status, "", "", str(pv.available_traces), rec.time_local, rec.case_id, rec.variant, str(rec.n_traces), str(rec.geometry_only), rec.returncode, rec.job_id, rec.input_file, rec.marker_file]
            for c, v in enumerate(vals):
                if c in (1, 2):
                    continue
                self.history_table.setItem(r, c, QTableWidgetItem(str(v)))
            if thumb:
                self._set_thumb_cell(r, 1, pv.model_preview_png, "无模型标签" if not pv.label_json else "模型")
                self._set_thumb_cell(r, 2, pv.bscan_preview_png, "运行中等待 .out" if rec.status == "running" else "无 B-scan")
            else:
                self.history_table.setItem(r, 1, QTableWidgetItem("点击行查看"))
                self.history_table.setItem(r, 2, QTableWidgetItem("点击行查看"))
        self.history_table.blockSignals(False)
        if selected_job:
            for r in range(self.history_table.rowCount()):
                item = self.history_table.item(r, 10)
                if item and item.text() == selected_job:
                    self.history_table.selectRow(r)
                    break
        self.history_summary.setText(f"历史记录：总 {len(records)} | running {counts.get('running',0)} | done {counts.get('done',0)} | failed {counts.get('failed',0)} | stale {counts.get('stale_running',0)}")
        self.history_log.setPlainText("已加载历史记录。缩略图来自 labels.json 和已完成/实时 .out；删除操作只作用于当前 workspace 内的 marker/输出目录。")

    def preview_history_selected(self) -> None:
        if not hasattr(self, "history_table"):
            return
        rows = self.history_table.selectionModel().selectedRows() if self.history_table.selectionModel() else []
        if not rows:
            return
        r = rows[0].row()
        marker_item = self.history_table.item(r, 12)
        input_item = self.history_table.item(r, 11)
        case_item = self.history_table.item(r, 5)
        variant_item = self.history_table.item(r, 6)
        status_item = self.history_table.item(r, 0)
        if not marker_item or not input_item:
            return
        marker = marker_item.text()
        marker_data = self._history_marker_data(marker)
        workspace = Path(self.history_workspace_edit.text())
        # Build a lightweight record-like object from the table row.
        class _R: pass
        rec = _R()
        rec.job_id = self.history_table.item(r, 10).text() if self.history_table.item(r, 10) else ""
        rec.case_id = case_item.text() if case_item else ""
        rec.variant = variant_item.text() if variant_item else ""
        rec.status = status_item.text() if status_item else ""
        rec.input_file = input_item.text()
        pv = build_history_preview(rec, workspace, marker_data=marker_data, time_window_ns=float(self.cfg.radar.time_window_ns), make_png=False)
        if pv.label_json:
            rep = self.history_model_canvas.show_label_json(pv.label_json)
        else:
            self.history_model_canvas.show_empty(); rep = {"warning": "没有找到 labels.json，无法显示模型几何。"}
        arr, meta = load_bscan_for_history(pv.input_file, marker_data=marker_data)
        if arr is not None:
            self.history_bscan_canvas.show_bscan(robust_normalize(arr), f"{pv.case_id} | {pv.variant} | {pv.status} | {arr.shape[1]} traces", float(self.cfg.radar.time_window_ns))
            meta["bscan_shape"] = list(arr.shape)
        else:
            self.history_bscan_canvas.ax.clear()
            self.history_bscan_canvas.ax.set_title("暂无可读 B-scan：geometry-only 或 .out 尚未写出")
            self.history_bscan_canvas.draw_idle()
        detail = {
            "job_id": pv.job_id,
            "status": pv.status,
            "case_id": pv.case_id,
            "variant": pv.variant,
            "input_file": pv.input_file,
            "label_json": pv.label_json,
            "marker_file": marker,
            "model_preview": rep,
            "bscan_meta": meta,
        }
        self.history_detail_box.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2)[:6000])
        # update trace cell in-place for running tasks
        if arr is not None and self.history_table.item(r, 3):
            self.history_table.item(r, 3).setText(str(int(arr.shape[1] if arr.ndim == 2 else 0)))

    def export_history(self) -> None:
        rep = export_history_csv(Path(self.history_workspace_edit.text()))
        self.history_log.setPlainText(json.dumps(rep, ensure_ascii=False, indent=2))

    def delete_selected_history(self, delete_outputs: bool) -> None:
        rows = self.history_table.selectionModel().selectedRows() if self.history_table.selectionModel() else []
        if not rows:
            QMessageBox.warning(self, "未选择", "请先选择一条或多条历史记录。")
            return
        msg = "将删除选中历史记录"
        if delete_outputs:
            msg += "，并尝试删除对应输出目录、QC 预览和 marker"
        msg += "。该操作不可撤销，是否继续？"
        if QMessageBox.question(self, "确认删除", msg) != QMessageBox.Yes:
            return
        reps = []
        for idx in rows:
            marker_item = self.history_table.item(idx.row(), 12)
            if marker_item:
                reps.append(delete_history_record(Path(self.history_workspace_edit.text()), marker_file=marker_item.text(), delete_outputs=delete_outputs))
        self.history_log.setPlainText(json.dumps(reps, ensure_ascii=False, indent=2))
        self.refresh_history()

    # ---------- shared helpers ----------
    def _choose_dir(self, line: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择目录", line.text() or str(repo_root()))
        if d:
            line.setText(d)

    def _choose_file(self, line: QLineEdit, pattern: str) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", line.text() or str(repo_root()), pattern)
        if f:
            line.setText(f)

    def _gpu_ids(self) -> list[int]:
        return _parse_gpu_ids(self.gpu_ids_edit.text())

    def _runtime_cfg_from_ui(self) -> AppConfig:
        cfg = self.cfg
        cfg.runtime.project_root = str(Path(self.workspace_edit.text()))
        cfg.runtime.conda_env_gprmax = self.conda_env_edit.text().strip() or "gprMax"
        cfg.runtime.gprmax_source_dir = self.gprmax_root_edit.text().strip()
        cfg.runtime.gpu_enabled = self.use_gpu_check.isChecked()
        cfg.runtime.gpu_ids = self.gpu_ids_edit.text().strip() or "0"
        cfg.runtime.use_conda_run = self.use_conda_check.isChecked()
        cfg.runtime.omp_threads = int(self.omp_spin.value())
        return cfg

    def _run_worker(self, func: Callable, on_done: Callable[[object], None], log_box: QPlainTextEdit | None = None) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "任务运行中", "请等待当前任务结束。")
            return
        self.worker = GenericWorker(func)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(lambda err: self._worker_failed(err, log_box))
        self.worker.start()

    def _worker_failed(self, err: str, log_box: QPlainTextEdit | None = None) -> None:
        if log_box is not None:
            log_box.appendPlainText(err)
        QMessageBox.critical(self, "任务失败", err)
        self.statusBar().showMessage("Failed")

    # ---------- tab 1 ----------
    def _build_env_tab(self) -> None:
        widgets = build_advanced_env_tab(
            cfg=self.cfg,
            choose_dir=self._choose_dir,
            on_save=self.save_local_env,
            on_check=self.check_env,
            on_show_smoke_command=self.show_smoke_command,
            on_open_setup_script=self.open_setup_script,
        )
        self.advanced_env_widgets = widgets
        self.workspace_edit = widgets.workspace_edit
        self.gprmax_root_edit = widgets.gprmax_root_edit
        self.conda_env_edit = widgets.conda_env_edit
        self.gpu_ids_edit = widgets.gpu_ids_edit
        self.omp_spin = widgets.omp_spin
        self.use_gpu_check = widgets.use_gpu_check
        self.use_conda_check = widgets.use_conda_check
        self.log_box = widgets.log_box
        self.env_save_button = widgets.save_button
        self.env_check_button = widgets.check_button
        self.env_smoke_command_button = widgets.smoke_command_button
        self.env_setup_script_button = widgets.setup_script_button
        self.tabs.addTab(widgets.page, "1 环境检查")

    def save_local_env(self) -> None:
        p = write_simlab_env({
            "UAVGPR_GPRMAX_ROOT": self.gprmax_root_edit.text().strip(),
            "UAVGPR_CONDA_ENV": self.conda_env_edit.text().strip() or "gprMax",
            "UAVGPR_GPU_IDS": self.gpu_ids_edit.text().strip() or "0",
        })
        self.log_box.appendPlainText(f"[OK] saved {p}")
        QMessageBox.information(self, "保存成功", f"已写入 {p}")

    def check_env(self) -> None:
        def job():
            rep = run_environment_checks(self.conda_env_edit.text().strip() or "gprMax", self.use_conda_check.isChecked(), self.gprmax_root_edit.text().strip())
            out = Path(self.workspace_edit.text()) / "reports" / "environment_report.json"
            save_report(rep, out)
            return rep.to_dict(), str(out)
        self._run_worker(job, lambda result: self._show_env_report(result), self.log_box)

    def _show_env_report(self, result: object) -> None:
        rep, out = result  # type: ignore[misc]
        self.log_box.appendPlainText(json.dumps(rep, ensure_ascii=False, indent=2))
        self.log_box.appendPlainText(f"[REPORT] {out}")
        self.statusBar().showMessage("Environment report written")

    def open_setup_script(self) -> None:
        p = repo_root() / "scripts" / "OneClick_Install_GPRMAX_Windows.bat"
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            self.log_box.appendPlainText(f"Windows script path: {p}")

    def show_smoke_command(self) -> None:
        example = repo_root() / "sample_data" / "gprmax_smoke_Ascan_2D.in"
        opts = GprMaxRunOptions(
            str(example),
            conda_env=self.conda_env_edit.text().strip() or "gprMax",
            use_conda_run=self.use_conda_check.isChecked(),
            use_gpu=self.use_gpu_check.isChecked(),
            gpu_ids=self._gpu_ids(),
            geometry_only=True,
        )
        self.log_box.appendPlainText(command_to_string(build_gprmax_command(opts)))

    # ---------- tab 2 ----------
    def _build_real_csv_tab(self) -> None:
        widgets = build_advanced_real_csv_tab(
            default_csv=str(repo_root() / "sample_data" / "Line9origin36_first16traces.csv"),
            choose_csv=self._choose_file,
            on_load_preview=self.load_csv_preview,
            on_show_fk=self.show_real_fk,
            on_export_qc=self.export_csv_qc,
            canvas_factory=lambda: MplCanvas("Real CSV B-scan preview"),
        )
        self.csv_edit = widgets.csv_edit
        self.max_traces_spin = widgets.max_traces_spin
        self.csv_canvas = widgets.csv_canvas
        self.csv_info = widgets.csv_info
        self.csv_pick_button = widgets.pick_button
        self.csv_load_button = widgets.load_button
        self.csv_fk_button = widgets.fk_button
        self.csv_export_button = widgets.export_button
        self.tabs.addTab(widgets.page, "7 实测/弱监督")

    def load_csv_preview(self) -> None:
        try:
            preview = load_real_csv_preview(self.csv_edit.text(), max_traces=self.max_traces_spin.value())
            self.last_preview = preview.normalized_bscan
            self.csv_canvas.show_bscan(
                preview.normalized_bscan,
                "Line CSV: mean background removed + robust normalized",
                preview.time_window_ns,
            )
            self.csv_info.setPlainText(json.dumps(preview.info, ensure_ascii=False, indent=2))
        except Exception:
            QMessageBox.critical(self, "CSV 读取失败", traceback.format_exc())

    def show_real_fk(self) -> None:
        if self.last_preview is None:
            self.load_csv_preview()
        if self.last_preview is not None:
            self.csv_canvas.show_fk(self.last_preview, "Line CSV f-k preview")

    def export_csv_qc(self) -> None:
        def job():
            return export_real_csv_qc(
                self.csv_edit.text(),
                self.workspace_edit.text(),
                max_traces=self.max_traces_spin.value(),
                make_baselines=True,
            )
        self._run_worker(job, lambda rep: self.csv_info.setPlainText(json.dumps(rep, ensure_ascii=False, indent=2)), self.csv_info)

    # ---------- tab 3 ----------
    def _build_generation_tab(self) -> None:
        widgets = build_advanced_generation_tab(
            default_plan=repo_root() / "configs" / "run_plan_3060_quick.yaml",
            choose_plan=self._choose_file,
            on_preview_plan=self.preview_plan,
            on_generate_dataset=self.generate_dataset,
        )
        self.advanced_generation_widgets = widgets
        self.plan_edit = widgets.plan_edit
        self.case_override_spin = widgets.case_override_spin
        self.antenna_combo = widgets.antenna_combo
        self.component_combo = widgets.component_combo
        self.gen_log = widgets.gen_log
        self.plan_choose_button = widgets.choose_button
        self.plan_preview_button = widgets.preview_button
        self.plan_generate_button = widgets.generate_button
        self.tabs.addTab(widgets.page, "2 仿真计划")

    def preview_plan(self) -> None:
        try:
            plan = load_yaml(self.plan_edit.text())
            self.gen_log.setPlainText(json.dumps(plan, ensure_ascii=False, indent=2))
        except Exception:
            self.gen_log.setPlainText(traceback.format_exc())

    def generate_dataset(self) -> None:
        plan = Path(self.plan_edit.text())
        workspace = Path(self.workspace_edit.text())
        def job():
            cfg = _cfg_from_plan(plan, workspace)
            if self.case_override_spin.value() > 0:
                cfg.dataset.cases = int(self.case_override_spin.value())
            cfg.radar.antenna_model = self.antenna_combo.currentText()
            comp = self.component_combo.currentText()
            if comp == "raw_only":
                cfg.dataset.variants = ["raw"]
            elif comp == "raw_target_clutter":
                cfg.dataset.variants = ["raw", "target_only", "clutter_only"]
            models, manifest = generate_cases(cfg, cfg.runtime.project_root, cfg.dataset.cases)
            cfg.runtime.conda_env_gprmax = self.conda_env_edit.text().strip() or "gprMax"
            cfg.runtime.gprmax_source_dir = self.gprmax_root_edit.text().strip()
            cfg.runtime.gpu_enabled = self.use_gpu_check.isChecked()
            cfg.runtime.gpu_ids = self.gpu_ids_edit.text().strip() or "0"
            save_config(cfg, Path(cfg.runtime.project_root) / "configs" / "generated_config.yaml")
            bat = write_manifest_commands_bat(
                manifest,
                Path(cfg.runtime.project_root) / "logs" / "run_all_gprmax_raw.bat",
                conda_env=cfg.runtime.conda_env_gprmax,
                gpu=cfg.runtime.gpu_enabled,
                gpu_ids=self._gpu_ids(),
                variants=["raw"],
            )
            return {"models": str(models), "manifest": str(manifest), "run_bat": str(bat), "project_root": str(cfg.runtime.project_root), "cases": cfg.dataset.cases, "variants": cfg.dataset.variants}
        self._run_worker(job, self._dataset_generated, self.gen_log)

    def _dataset_generated(self, rep: object) -> None:
        data = rep  # type: ignore[assignment]
        self.current_manifest = Path(data["manifest"])  # type: ignore[index]
        self.manifest_edit.setText(str(self.current_manifest))
        if hasattr(self, "preview_manifest_edit"):
            self.preview_manifest_edit.setText(str(self.current_manifest))
            self.load_preview_manifest()
        if hasattr(self, "preflight_manifest_edit"):
            self.preflight_manifest_edit.setText(str(self.current_manifest))
        self.gen_log.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
        self.load_manifest()
        self.tabs.setCurrentWidget(self.queue_tab)
        self.statusBar().showMessage("Dataset manifest generated")

    # ---------- tab 4 ----------
    def _build_queue_tab(self) -> None:
        widgets = build_advanced_queue_tab(
            choose_manifest=self._choose_file,
            on_load_manifest=self.load_manifest,
            on_write_geometry_bat=lambda: self.write_bat(True),
            on_write_full_bat=lambda: self.write_bat(False),
            on_run_selected=self.run_selected_task,
            on_run_batch=self.run_batch_tasks,
            on_stop=self.stop_live_run,
            canvas_factory=lambda: MplCanvas("Realtime gprMax B-scan preview"),
        )
        self.advanced_queue_widgets = widgets
        self.manifest_edit = widgets.manifest_edit
        self.variant_combo = widgets.variant_combo
        self.limit_spin = widgets.limit_spin
        self.geometry_check = widgets.geometry_check
        self.skip_completed_check = widgets.skip_completed_check
        self.force_rerun_check = widgets.force_rerun_check
        self.task_list = widgets.task_list
        self.progress = widgets.progress
        self.queue_log = widgets.queue_log
        self.queue_canvas = widgets.queue_canvas
        self.preview_info = widgets.preview_info
        self.queue_pick_button = widgets.pick_button
        self.queue_load_button = widgets.load_button
        self.queue_write_geometry_button = widgets.write_geometry_button
        self.queue_write_full_button = widgets.write_full_button
        self.queue_run_selected_button = widgets.run_selected_button
        self.queue_run_batch_button = widgets.run_batch_button
        self.queue_stop_button = widgets.stop_button
        self.queue_tab = widgets.page
        self.tabs.addTab(widgets.page, "5 批量运行")

    def load_manifest(self) -> None:
        self.task_list.clear()
        path = Path(self.manifest_edit.text())
        if not path.exists():
            self.queue_log.appendPlainText("manifest 不存在。")
            return
        preview = read_queue_manifest_preview(path, max_display=1500)
        for row in preview.rows:
            item = QListWidgetItem(row.display_text)
            item.setData(Qt.UserRole, row.data)
            self.task_list.addItem(item)
        if preview.truncated:
            self.task_list.addItem(f"... only first {preview.max_display} records shown")
        suffix = f" / total {preview.total_read}" if preview.truncated else ""
        self.queue_log.appendPlainText(f"[LOAD] {preview.manifest} ({preview.displayed_count} displayed{suffix})")

    def write_bat(self, geometry_only: bool) -> None:
        path = Path(self.manifest_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "无 manifest", "请先生成或选择 manifest.csv")
            return
        result = write_advanced_queue_bat(
            path,
            conda_env=self.conda_env_edit.text().strip() or "gprMax",
            use_gpu=self.use_gpu_check.isChecked(),
            gpu_ids=self._gpu_ids(),
            geometry_only=geometry_only,
            variant=self.variant_combo.currentText(),
        )
        self.queue_log.appendPlainText(f"[BAT] {result.bat_path}")

    def _tasks_for_ui(self, selected_only: bool) -> list[GprMaxTask]:
        path = Path(self.manifest_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "无 manifest", "请先生成或选择 manifest.csv")
            return []
        variant = self.variant_combo.currentText()
        selected_row = None
        if selected_only:
            item = self.task_list.currentItem()
            if item is None:
                return []
            row = item.data(Qt.UserRole)
            selected_row = row if isinstance(row, dict) else None
        tasks = build_advanced_queue_tasks(
            path,
            variant=variant,
            limit=self.limit_spin.value(),
            selected_row=selected_row,
        )
        summary = summarize_queue_tasks(tasks)
        self.queue_log.appendPlainText(f"[PLAN] tasks={summary['tasks']} traces={summary['traces']} variants={','.join(summary['variants']) if summary['variants'] else '-'}")
        return tasks

    def _start_live_tasks(self, tasks: list[GprMaxTask]) -> None:
        if not tasks:
            QMessageBox.warning(self, "无任务", "没有可运行的任务。")
            return
        if self.live_worker and self.live_worker.isRunning():
            QMessageBox.warning(self, "任务运行中", "请先停止或等待当前任务结束。")
            return
        cfg = self._runtime_cfg_from_ui()
        # Keep project_root aligned with the manifest parent if possible.
        m = Path(self.manifest_edit.text())
        if m.exists():
            cfg.runtime.project_root = str(m.parent.parent)
        cfg.geometry.trace_count = int(tasks[0].n_traces)
        self.progress.setValue(0)
        self.live_worker = LiveQueueWorker(
            cfg,
            tasks,
            Path(cfg.runtime.project_root) / "logs",
            geometry_only=self.geometry_check.isChecked(),
            auto_export=True,
            skip_completed=self.skip_completed_check.isChecked(),
            force=self.force_rerun_check.isChecked(),
        )
        self.live_worker.log.connect(lambda s: self.queue_log.appendPlainText(s))
        self.live_worker.preview.connect(self._update_live_preview)
        self.live_worker.progress.connect(self._update_progress)
        self.live_worker.done.connect(self._live_done)
        self.live_worker.failed.connect(lambda err: self._worker_failed(err, self.queue_log))
        self.live_worker.start()
        self.statusBar().showMessage("gprMax queue running")

    def run_selected_task(self) -> None:
        self._start_live_tasks(self._tasks_for_ui(selected_only=True))

    def run_batch_tasks(self) -> None:
        self._start_live_tasks(self._tasks_for_ui(selected_only=False))

    def stop_live_run(self) -> None:
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.cancel()
            self.queue_log.appendPlainText("[CANCEL] stop requested")

    def _update_live_preview(self, arr: object, title: str, time_window_ns: float) -> None:
        bscan = np.asarray(arr, dtype=float)
        self.last_preview = bscan
        self.queue_canvas.show_bscan(robust_normalize(bscan), title, time_window_ns)
        self.preview_info.setText(f"Realtime preview: {title} | shape={bscan.shape}")

    def _update_progress(self, i: int, total: int) -> None:
        self.progress.setValue(int(100 * i / max(1, total)))

    def _live_done(self, rep: object) -> None:
        self.queue_log.appendPlainText("[DONE] " + json.dumps(rep, ensure_ascii=False)[:2000])
        self.progress.setValue(100)
        self.statusBar().showMessage("gprMax queue finished")

    # ---------- tab 5 ----------
    def _build_qc_tab(self) -> None:
        widgets = build_advanced_qc_tab(
            choose_input=self._choose_file,
            on_export_products=self.export_out_products,
        )
        self.advanced_qc_widgets = widgets
        self.out_input_edit = widgets.out_input_edit
        self.qc_text = widgets.qc_text
        self.qc_choose_button = widgets.choose_button
        self.qc_export_button = widgets.export_button
        self.tabs.addTab(widgets.page, "8 结果/报告")

    def export_out_products(self) -> None:
        inp = Path(self.out_input_edit.text())
        if not inp.exists():
            QMessageBox.warning(self, "缺少 .in", "请选择已经运行过的 input .in 文件。")
            return
        out = Path(self.workspace_edit.text()) / "manual_gprmax_qc" / inp.stem
        def job():
            return export_gprmax_bscan_for_input(inp, out, stem=inp.stem)
        self._run_worker(job, lambda rep: self.qc_text.setPlainText(json.dumps(rep, ensure_ascii=False, indent=2)), self.qc_text)

    # ---------- tab 6 ----------
    def _build_train_tab(self) -> None:
        widgets = build_advanced_train_tab()
        self.advanced_train_widgets = widgets
        self.train_text = widgets.text
        self.tabs.addTab(widgets.page, "9 高级/PGDA")


def run_app(config_path: Optional[Path] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    win = MainWindow(config_path=config_path)
    win.show()
    return app.exec()
