from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, QThread
from PySide6.QtWidgets import QMessageBox, QProgressBar

from uavgpr_simlab.gui.main_window import LiveQueueWorker
from uavgpr_simlab.core.dataset_contract import validate_dataset_skeleton
from uavgpr_simlab.core.run_dashboard import summarize_dataset_run_dashboard, format_duration_seconds
from uavgpr_simlab.gui.easy_workers import SceneWorldFullChainWorker
from uavgpr_simlab.gui.pages.batch_page import build_batch_page
from uavgpr_simlab.gui.controllers.batch_actions import (
    import_dataset_skeleton as _import_dataset_skeleton_action,
    relocate_imported_workspace_paths as _relocate_imported_workspace_paths_action,
)
from uavgpr_simlab.gui.controllers.batch_recent_preview import append_recent_bscan_preview, reset_recent_bscan_preview
from uavgpr_simlab.gui.controllers.batch_queue_panel import activate_batch_queue_item, refresh_batch_queue_panel
from uavgpr_simlab.gui.easy_ui import _friendly_variant_name, _pix_label, _set_item, _status_chip
from uavgpr_simlab.gui.easy_cards import set_metric_value
from uavgpr_simlab.services.easy_batch_service import (
    build_batch_plan,
    build_pending_tasks,
    parse_variants,
    prepare_batch_preview_asset,
    read_manifest_rows,
    require_manifest_file,
)
from uavgpr_simlab.services.environment_service import check_gprmax_runtime_preflight, python_command_from_easy_settings
from uavgpr_simlab.services.simulation_job_service import (
    default_simulation_run_profiles,
    find_dataset_manifest,
    manifest_is_sceneworld,
    normalize_variants,
    profile_by_key,
)


class BatchControllerMixin:
    # --- batch page ---
    def _build_batch_page(self) -> None:
        widgets = build_batch_page(
            on_choose_manifest=lambda edit: self._choose_file(edit, "CSV (*.csv);;All Files (*)"),
            on_import_skeleton=self.import_dataset_skeleton,
            on_relocate_workspace=self.relocate_imported_workspace_paths,
            on_apply_profile=self.apply_batch_run_profile,
            on_refresh_plan=self.refresh_batch_plan,
            on_run_pending=self.run_pending_batch,
            on_stop=self.stop_batch,
            on_queue_activated=self.on_batch_queue_item_activated,
        )
        self.batch_profile_combo = widgets.batch_profile_combo
        self._populate_batch_run_profiles()
        self.batch_manifest_edit = widgets.batch_manifest_edit
        self.batch_variants_edit = widgets.batch_variants_edit
        self.batch_limit_spin = widgets.batch_limit_spin
        self.batch_skip_done = widgets.batch_skip_done
        self.batch_failed_only = widgets.batch_failed_only
        self.batch_force_rerun = widgets.batch_force_rerun
        self.batch_dataset_status_label = widgets.batch_dataset_status_label
        self.batch_runtime_summary_label = widgets.batch_runtime_summary_label
        self.batch_total_card = widgets.batch_total_card
        self.batch_pending_card = widgets.batch_pending_card
        self.batch_running_card = widgets.batch_running_card
        self.batch_done_card = widgets.batch_done_card
        self.batch_failed_card = widgets.batch_failed_card
        self.batch_queue_tree = widgets.batch_queue_tree
        self.batch_failure_box = widgets.batch_failure_box
        self.batch_table = widgets.batch_table
        self.batch_live_canvas = widgets.batch_live_canvas
        self.batch_recent_canvas = widgets.batch_recent_canvas
        self.batch_log = widgets.batch_log
        reset_recent_bscan_preview(self)
        self.stack.addWidget(widgets.page)

    def _populate_batch_run_profiles(self) -> None:
        self.batch_profile_combo.clear()
        for profile in default_simulation_run_profiles():
            self.batch_profile_combo.addItem(profile.title, profile.key)

    def _selected_batch_profile(self):
        key = self.batch_profile_combo.currentData() if hasattr(self, "batch_profile_combo") else "custom"
        return profile_by_key(str(key or "custom"))

    def apply_batch_run_profile(self) -> None:
        profile = self._selected_batch_profile()
        if profile.workspace_name:
            manifest = find_dataset_manifest(self.root_dir, profile.workspace_name)
            if manifest is None:
                QMessageBox.warning(self, "缺少数据集骨架", f"运行配置 {profile.key} 需要工作区：\nworkspace/{profile.workspace_name}\n但未找到 manifest。")
                return
            self.batch_manifest_edit.setText(str(manifest))
            self.current_manifest = manifest
        self.batch_variants_edit.setText(",".join(profile.variants))
        if profile.max_tasks_hint > 0:
            self.batch_limit_spin.setValue(profile.max_tasks_hint)
        self.batch_log.appendPlainText(f"[profile] {profile.title}\n{profile.description}")
        self.refresh_batch_plan()

    def relocate_imported_workspace_paths(self) -> None:
        _relocate_imported_workspace_paths_action(self)

    def import_dataset_skeleton(self) -> None:
        _import_dataset_skeleton_action(self)

    def _variants(self) -> list[str]:
        return parse_variants(self.batch_variants_edit.text())

    def refresh_batch_plan(self) -> None:
        text = self.batch_manifest_edit.text().strip() or (str(self.current_manifest) if self.current_manifest else "")
        if not text:
            QMessageBox.warning(self, "无清单", "请先在模型预览页生成模型，或在批量仿真页选择 manifest.csv。")
            return
        manifest = Path(text).expanduser()
        if not manifest.exists() or not manifest.is_file():
            reason = "不存在" if not manifest.exists() else "不是文件，而是目录"
            QMessageBox.warning(self, "无清单", f"当前 manifest.csv 路径{reason}：\n{manifest}\n\n请先生成模型，或选择正确的 manifest.csv。")
            return
        self.current_manifest = manifest
        self.batch_manifest_edit.setText(str(manifest))
        if manifest_is_sceneworld(manifest):
            contract = validate_dataset_skeleton(manifest, expected_variants=self._variants(), write_report=True)
            if not contract.ok:
                details = "\n".join(f"- {x.code}: {x.message}" for x in contract.issues[:8] if x.level == "error")
                QMessageBox.warning(self, "数据集骨架合同未通过", f"当前 manifest 不能作为 ready-to-run 数据集导入：\n{manifest}\n\n{details}")
                self.batch_log.setPlainText(json.dumps(contract.to_dict(), ensure_ascii=False, indent=2))
                return
            if contract.warning_count:
                self.batch_log.appendPlainText(f"[dataset-contract] 通过，但有 {contract.warning_count} 个警告；报告：{contract.summary_json or '未写入'}")
        try:
            plan = build_batch_plan(
                manifest,
                variants=self._variants(),
                max_tasks=int(self.batch_limit_spin.value()),
                skip_completed=self.batch_skip_done.isChecked(),
            )
        except Exception:
            QMessageBox.critical(self, "预检失败", traceback.format_exc())
            return
        if manifest_is_sceneworld(manifest):
            row_status_map: dict[tuple[str, str], tuple[str, str]] = {}
            for row in read_manifest_rows(manifest):
                status_raw = (row.get("bscan_status") or "not_run").strip()
                status = "done" if status_raw == "success" else "failed" if status_raw == "failed" else "running" if status_raw in {"running", "stale_running"} else "pending"
                row_status_map[(row.get("case_id", ""), row.get("variant", ""))] = (status, row.get("bscan_error", ""))
            for rec in plan.records:
                mapped = row_status_map.get((rec.case_id, rec.variant))
                if mapped:
                    rec.status, rec.reason = mapped
            plan.counts.clear()
            for rec in plan.records:
                plan.counts[rec.status] = plan.counts.get(rec.status, 0) + 1
            dashboard = summarize_dataset_run_dashboard(manifest, expected_variants=self._variants(), write_report=True)
            self.batch_dataset_status_label.setText(dashboard.format_for_operator())
            refresh_batch_queue_panel(self, dashboard)
            self._update_batch_eta_cells(dashboard)
        else:
            self.batch_dataset_status_label.setText(f"普通 manifest：{manifest.name}\n等待={plan.counts.get('pending', 0)}，完成/跳过={plan.counts.get('skipped', 0)}，失败={plan.counts.get('failed', 0)}")
            if hasattr(self, "batch_queue_tree"):
                self.batch_queue_tree.clear()
            if hasattr(self, "batch_failure_box"):
                self.batch_failure_box.setPlainText("普通 manifest 暂不生成 SceneWorld 队列树。")

        self.batch_table.setRowCount(0)
        self._batch_row_by_case_variant = {}
        for rec in plan.records:
            r = self.batch_table.rowCount()
            self.batch_table.insertRow(r)
            asset = prepare_batch_preview_asset(rec, plan.workspace)
            self.batch_table.setCellWidget(r, 0, _pix_label(asset.preview_png, QSize(118, 68)))
            _set_item(self.batch_table, r, 1, rec.case_id)
            self.batch_table.setCellWidget(r, 2, _status_chip(rec.status))
            prog = QProgressBar()
            prog.setRange(0, 100)
            prog.setValue(100 if rec.status in {"skipped", "done"} else (30 if rec.status == "running" else 0))
            self.batch_table.setCellWidget(r, 3, prog)
            _set_item(self.batch_table, r, 4, f"0 / {rec.n_traces} 道" if rec.status == "pending" else f"{rec.n_traces} / {rec.n_traces} 道")
            _set_item(self.batch_table, r, 5, "—")
            _set_item(self.batch_table, r, 6, _friendly_variant_name(rec.variant))
            _set_item(self.batch_table, r, 7, rec.reason or ("可运行" if rec.status == "pending" else "已处理"))
            self._batch_row_by_case_variant[(rec.case_id, rec.variant)] = r
        self._update_batch_cards(len(plan.records), plan.counts)
        if dashboard is not None:
            self._update_batch_eta_cells(dashboard)

    def on_batch_queue_item_activated(self, item: object, column: int = 0) -> None:
        activate_batch_queue_item(self, item, column)

    def _update_batch_cards(self, total: int, counts: dict[str,int]) -> None:
        set_metric_value(self.batch_total_card, f"{total} 个")
        set_metric_value(self.batch_pending_card, f"{counts.get('pending',0)} 个")
        set_metric_value(self.batch_running_card, f"{counts.get('running',0) + counts.get('stale_running',0)} 个")
        set_metric_value(self.batch_done_card, f"{counts.get('done',0) + counts.get('skipped',0)} 个")
        set_metric_value(self.batch_failed_card, f"{counts.get('failed',0)} 个")

    def _update_batch_eta_cells(self, dashboard: object) -> None:
        avg = float(getattr(dashboard, "average_variant_seconds", 0.0) or 0.0)
        if avg <= 0 or not hasattr(self, "batch_table"):
            return
        for (case_id, variant), row in getattr(self, "_batch_row_by_case_variant", {}).items():
            # Only show ETA for rows that may still consume runtime. Completed, failed
            # and skipped rows are already terminal and should not look pending.
            status_widget = self.batch_table.cellWidget(row, 2)
            current_text = ""
            if status_widget is not None:
                current_text = status_widget.toolTip() or status_widget.text() if hasattr(status_widget, "text") else ""
            old_hint = self.batch_table.item(row, 7).text() if self.batch_table.item(row, 7) else ""
            if any(token in (old_hint + current_text) for token in ["已生成", "完成", "失败", "跳过"]):
                continue
            _set_item(self.batch_table, row, 5, f"≈ {format_duration_seconds(avg)}")

    def run_pending_batch(self) -> None:
        text = self.batch_manifest_edit.text().strip() or (str(self.current_manifest) if self.current_manifest else "")
        if not text:
            QMessageBox.warning(self, "无清单", "请先在模型预览页生成模型，或在批量仿真页选择 manifest.csv。")
            return
        try:
            manifest = require_manifest_file(text)
        except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
            QMessageBox.warning(self, "无清单", f"无法启动批量仿真：\n{exc}")
            return
        self.batch_manifest_edit.setText(str(manifest))
        self.current_manifest = manifest
        if manifest_is_sceneworld(manifest):
            contract = validate_dataset_skeleton(manifest, expected_variants=self._variants(), write_report=True)
            if not contract.ok:
                details = "\n".join(f"- {x.code}: {x.message}" for x in contract.issues[:10] if x.level == "error")
                self.batch_log.setPlainText(json.dumps(contract.to_dict(), ensure_ascii=False, indent=2))
                QMessageBox.warning(self, "数据集骨架合同未通过", f"已阻止启动仿真。请先修复 manifest / 文件结构：\n\n{details}")
                return
            self.run_sceneworld_profile_from_batch(manifest)
            return
        try:
            tasks = build_pending_tasks(manifest, variants=self._variants(), max_tasks=int(self.batch_limit_spin.value()))
            if not tasks:
                QMessageBox.information(self, "无任务", "没有找到可运行任务。")
                return
            cfg = self._cfg_for_current()
            self.live_worker = LiveQueueWorker(cfg, tasks, Path(cfg.runtime.project_root)/"logs", geometry_only=False, auto_export=True, skip_completed=True, force=False)
            self.live_worker.log.connect(lambda s: self.batch_log.appendPlainText(str(s)))
            self.live_worker.preview.connect(lambda arr, title, tw: self.batch_live_canvas.show_bscan(arr, title, tw))
            self.live_worker.progress.connect(lambda a,b: self.batch_log.appendPlainText(f"[进度] {a}/{b}"))
            self.live_worker.done.connect(lambda _: (self.batch_log.appendPlainText("[完成] 批量运行结束"), self.refresh_batch_plan(), self.refresh_history()))
            self.live_worker.failed.connect(lambda s: self.batch_log.appendPlainText("[失败]\n"+s))
            self.live_worker.start()
            self.batch_log.appendPlainText("[开始] 已启动普通批量任务。")
        except Exception:
            QMessageBox.critical(self, "启动失败", traceback.format_exc())

    def run_sceneworld_profile_from_batch(self, manifest: Path) -> None:
        settings = self._env_settings_from_ui()
        if not settings.gprmax_root.strip():
            QMessageBox.warning(self, "缺少 gprMax 源码目录", "请先在设置页填写 gprMax 源码目录，例如 E:\\gprMax\\gprMax-v.3.1.7。")
            return
        if self._scene_job_thread is not None and self._scene_job_thread.isRunning():
            QMessageBox.information(self, "任务正在运行", "统一 SceneWorld 任务已经在后台运行。")
            return
        profile = self._selected_batch_profile()
        workspace = manifest.resolve().parent.parent
        variants = normalize_variants(self._variants())
        if profile.key == "custom":
            max_cases = 0
            one_case_per_family = False
            allow_resample = False
            timeout = 3600
        else:
            max_cases = profile.max_cases
            one_case_per_family = profile.one_case_per_family
            allow_resample = profile.allow_resample
            timeout = profile.timeout_sec
        gpu_ids: list[int] = []
        if settings.use_gpu:
            for item in settings.gpu_ids.replace(";", ",").split(","):
                item = item.strip()
                if item:
                    try:
                        gpu_ids.append(int(item))
                    except ValueError:
                        QMessageBox.warning(self, "GPU ID 无效", f"GPU ID 必须是整数，当前值：{item}")
                        return
        python_cmd: str | list[str] = python_command_from_easy_settings(settings)

        preflight = check_gprmax_runtime_preflight(settings, require_gpu=bool(settings.use_gpu), timeout=45)
        if not preflight.ok:
            msg = preflight.format_for_user()
            self.batch_log.setPlainText(msg)
            QMessageBox.warning(
                self,
                "运行前环境检查未通过",
                msg + "\n\n已停止启动本次批量任务，避免 25-run 重复失败。",
            )
            return

        reset_recent_bscan_preview(self)
        self.batch_log.setPlainText(
            f"[统一任务] {profile.title}\n"
            f"workspace={workspace}\nmanifest={manifest}\n"
            f"variants={','.join(variants)}\n"
            f"python={python_cmd}\n"
            f"gpu={'on ' + ','.join(str(x) for x in gpu_ids) if settings.use_gpu else 'off'}\n"
            f"allow_resample={allow_resample} strict_qc={not allow_resample}\n"
        )
        self._scene_job_thread = QThread(self)
        self._scene_job_worker = SceneWorldFullChainWorker(
            workspace=workspace,
            gprmax_root=settings.gprmax_root,
            python_executable=python_cmd,
            omp_threads=max(1, int(settings.omp_threads or 1)),
            timeout=timeout,
            allow_resample=allow_resample,
            one_case_per_family=one_case_per_family,
            max_cases=max_cases,
            variants=variants,
            no_gpu=not bool(settings.use_gpu),
            gpu_ids=gpu_ids,
            run_label=profile.key,
            force=bool(getattr(self, "batch_force_rerun", None) and self.batch_force_rerun.isChecked()),
            skip_completed=bool(getattr(self, "batch_skip_done", None) is None or self.batch_skip_done.isChecked()),
            rerun_failed_only=bool(getattr(self, "batch_failed_only", None) and self.batch_failed_only.isChecked()),
        )
        self._scene_job_worker.moveToThread(self._scene_job_thread)
        self._scene_job_thread.started.connect(self._scene_job_worker.run)
        self._scene_job_worker.log.connect(lambda msg: self.batch_log.appendPlainText(str(msg)))
        self._scene_job_worker.event.connect(self._on_scene_job_event)
        self._scene_job_worker.finished.connect(self._on_scene_job_finished)
        self._scene_job_worker.failed.connect(self._on_scene_job_failed)
        self._scene_job_worker.finished.connect(self._scene_job_thread.quit)
        self._scene_job_worker.failed.connect(self._scene_job_thread.quit)
        self._scene_job_thread.finished.connect(self._cleanup_scene_job_worker)
        self._scene_job_thread.start()

    def _on_scene_job_event(self, event: object) -> None:
        if not isinstance(event, dict):
            return
        name = str(event.get("event") or "")
        cid = str(event.get("case_id") or "")
        variant = str(event.get("variant") or "")
        if name == "variant_started" and cid and variant:
            self._set_batch_row_status(cid, variant, "running", "运行中")
            self._refresh_run_dashboard_label()
        elif name == "variant_skipped" and cid and variant:
            self._set_batch_row_status(cid, variant, "skipped", str(event.get("reason") or "已跳过"))
            self._refresh_run_dashboard_label()
        elif name in {"variant_done", "variant_output_ready"} and cid and variant:
            status = "done" if event.get("status") == "success" else "failed"
            self._set_batch_row_status(cid, variant, status, "已生成 B-scan" if status == "done" else str(event.get("error") or "失败"))
            self._refresh_run_dashboard_label()
            bscan_path = event.get("bscan_path")
            if bscan_path and Path(str(bscan_path)).exists():
                try:
                    arr = np.load(str(bscan_path))
                    tw = float(event.get("time_window_ns") or 450.0)
                    label = _friendly_variant_name(variant)
                    self.batch_live_canvas.show_bscan(arr, f"{cid} | {label}", tw)
                    append_recent_bscan_preview(self, cid, variant, arr, tw, label)
                except Exception as exc:
                    self.batch_log.appendPlainText(f"[preview failed] {bscan_path}: {exc}")
        elif name == "case_qc_done":
            self.refresh_history()
        elif name == "job_cancelled":
            self.batch_log.appendPlainText("[停止] 统一任务已中断。")
            self.refresh_batch_plan()
            self.refresh_history()
        elif name == "job_done":
            self.refresh_batch_plan()
            self.refresh_history()


    def _refresh_run_dashboard_label(self) -> None:
        if not getattr(self, "current_manifest", None):
            return
        try:
            dashboard = summarize_dataset_run_dashboard(self.current_manifest, expected_variants=self._variants(), write_report=True)
            self.batch_dataset_status_label.setText(dashboard.format_for_operator())
            refresh_batch_queue_panel(self, dashboard)
            self._update_batch_eta_cells(dashboard)
        except Exception:
            pass

    def _set_batch_row_status(self, case_id: str, variant: str, status: str, note: str = "") -> None:
        row = self._batch_row_by_case_variant.get((case_id, variant))
        if row is None:
            return
        self.batch_table.setCellWidget(row, 2, _status_chip(status))
        prog = self.batch_table.cellWidget(row, 3)
        if isinstance(prog, QProgressBar):
            prog.setValue(100 if status in {"done", "failed", "skipped"} else 30)
        _set_item(self.batch_table, row, 4, "完成" if status == "done" else ("失败" if status == "failed" else ("跳过" if status == "skipped" else "运行中")))
        _set_item(self.batch_table, row, 7, note or status)

    def _on_scene_job_finished(self, report: object) -> None:
        rep = report if isinstance(report, dict) else {}
        self.batch_log.appendPlainText("\n--- 统一仿真任务完成 ---")
        self.batch_log.appendPlainText(json.dumps({
            "ok": rep.get("ok"),
            "case_count": rep.get("case_count"),
            "report_json": rep.get("report_json"),
        }, ensure_ascii=False, indent=2))
        self.refresh_batch_plan()
        self.refresh_history()

    def _on_scene_job_failed(self, detail: str) -> None:
        self.batch_log.appendPlainText("\n--- 统一仿真任务异常 ---\n" + detail)

    def _cleanup_scene_job_worker(self) -> None:
        if self._scene_job_worker is not None:
            self._scene_job_worker.deleteLater()
            self._scene_job_worker = None
        if self._scene_job_thread is not None:
            self._scene_job_thread.deleteLater()
            self._scene_job_thread = None

    def stop_batch(self) -> None:
        if self.live_worker:
            self.live_worker.cancel(); self.batch_log.appendPlainText("[停止] 已请求停止当前普通批量运行。")
        if self._scene_job_thread is not None and self._scene_job_thread.isRunning():
            if self._scene_job_worker is not None:
                self._scene_job_worker.cancel()
            self.batch_log.appendPlainText("[停止] 已请求停止 SceneWorld 统一任务；当前 gprMax 子进程会被终止。")

