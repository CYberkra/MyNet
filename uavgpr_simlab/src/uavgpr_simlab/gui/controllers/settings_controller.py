from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox

from uavgpr_simlab.gui.main_window import MainWindow
from uavgpr_simlab.gui.easy_workers import GprMaxSourceSmokeWorker, SceneWorldFullChainWorker
from uavgpr_simlab.gui.pages.settings_page import build_settings_help_page
from uavgpr_simlab.services.environment_service import (
    format_easy_environment_report,
    inspect_gprmax_source,
    run_easy_environment_diagnostics,
    save_easy_environment_settings,
)
from uavgpr_simlab.services.gprmax_smoke_service import format_gprmax_source_smoke_report


class SettingsControllerMixin:
    # --- help/settings page ---
    def _build_help_page(self) -> None:
        widgets = build_settings_help_page(
            env_settings=self.env_settings,
            cfg=self.cfg,
            on_save=self.save_easy_env,
            on_check=self.check_easy_env,
            on_smoke_test=self.run_easy_gprmax_source_smoke,
            on_link_check=self.run_ultra_tiny_full_chain_from_gui,
            on_open_advanced=self.open_advanced_window,
        )
        self.gprmax_root_edit = widgets.gprmax_root_edit
        self.conda_env_edit = widgets.conda_env_edit
        self.gpu_ids_edit = widgets.gpu_ids_edit
        self.omp_spin = widgets.omp_spin
        self.use_gpu_check = widgets.use_gpu_check
        self.use_conda_check = widgets.use_conda_check
        self.smoke_test_button = widgets.smoke_test_button
        self.ultra_tiny_button = widgets.link_check_button
        self.help_log = widgets.help_log
        self.stack.addWidget(widgets.page)

    def save_easy_env(self) -> None:
        settings = self._env_settings_from_ui()
        self.env_settings = settings
        p = save_easy_environment_settings(settings)
        info = inspect_gprmax_source(settings.gprmax_root)
        message = f"设置已保存：\n{p}\n\n{info.message}"
        QMessageBox.information(self, "保存成功", message)

    def check_easy_env(self) -> None:
        try:
            report = run_easy_environment_diagnostics(
                self._env_settings_from_ui(),
                report_dir=Path(self.workspace_edit.text()) / "reports",
            )
            summary = format_easy_environment_report(report, self._env_settings_from_ui())
            raw_json = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
            self.help_log.setPlainText(f"{summary}\n\n--- 原始 JSON ---\n{raw_json}")
        except Exception:
            self.help_log.setPlainText(traceback.format_exc())

    def run_easy_gprmax_source_smoke(self) -> None:
        settings = self._env_settings_from_ui()
        if not settings.gprmax_root.strip():
            QMessageBox.warning(self, "缺少 gprMax 源码目录", "请先填写 gprMax 源码目录，再运行最小 CPU 测试。")
            return
        if self._smoke_thread is not None and self._smoke_thread.isRunning():
            QMessageBox.information(self, "测试正在运行", "最小 CPU 测试已经在后台运行，请等待当前测试结束。")
            return

        work_dir = Path(self.workspace_edit.text()).expanduser() / "gprmax_source_smoke"
        self.help_log.setPlainText(
            "正在后台运行 gprMax 最小 CPU 测试...\n"
            "说明：此测试使用当前 Python 和本地 gprMax 源码，不启用 GPU，也不改变正式批量任务语义。\n"
            "测试期间可以继续查看界面，按钮会在完成后自动恢复。"
        )
        self.smoke_test_button.setEnabled(False)
        self.smoke_test_button.setText("测试中...")

        self._smoke_thread = QThread(self)
        self._smoke_worker = GprMaxSourceSmokeWorker(
            gprmax_root=settings.gprmax_root,
            work_dir=work_dir,
            build=False,
            omp_threads=max(1, int(settings.omp_threads or 1)),
            timeout=180,
        )
        self._smoke_worker.moveToThread(self._smoke_thread)
        self._smoke_thread.started.connect(self._smoke_worker.run)
        self._smoke_worker.finished.connect(self._on_gprmax_smoke_finished)
        self._smoke_worker.failed.connect(self._on_gprmax_smoke_failed)
        self._smoke_worker.finished.connect(self._smoke_thread.quit)
        self._smoke_worker.failed.connect(self._smoke_thread.quit)
        self._smoke_thread.finished.connect(self._cleanup_gprmax_smoke_worker)
        self._smoke_thread.start()

    def _on_gprmax_smoke_finished(self, report: object) -> None:
        summary = format_gprmax_source_smoke_report(report)  # type: ignore[arg-type]
        raw_json = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)  # type: ignore[attr-defined]
        self.help_log.setPlainText(f"{summary}\n\n--- 原始 JSON ---\n{raw_json}")

    def _on_gprmax_smoke_failed(self, detail: str) -> None:
        self.help_log.setPlainText(detail)

    def _cleanup_gprmax_smoke_worker(self) -> None:
        if self._smoke_worker is not None:
            self._smoke_worker.deleteLater()
            self._smoke_worker = None
        if self._smoke_thread is not None:
            self._smoke_thread.deleteLater()
            self._smoke_thread = None
        if hasattr(self, "smoke_test_button"):
            self.smoke_test_button.setEnabled(True)
            self.smoke_test_button.setText("最小 CPU 测试")


    def run_ultra_tiny_full_chain_from_gui(self) -> None:
        """Run the ultra-tiny SceneWorld full-chain validation inside the GUI."""
        settings = self._env_settings_from_ui()
        if not settings.gprmax_root.strip():
            QMessageBox.warning(self, "缺少 gprMax 源码目录", "请先填写 gprMax 源码目录，例如 E:\\gprMax\\gprMax-v.3.1.7。")
            return
        workspace = self.root_dir / "workspace" / "yingshan_sceneworld_ultra_tiny_v080a14"
        if not workspace.exists():
            for name in ("yingshan_sceneworld_ultra_tiny_v080a14", "yingshan_sceneworld_ultra_tiny_v080a12", "yingshan_sceneworld_ultra_tiny_v080a10", "yingshan_sceneworld_ultra_tiny_v080a9", "yingshan_sceneworld_ultra_tiny_v080a8", "yingshan_sceneworld_ultra_tiny_v080a7"):
                fallback = self.root_dir / "workspace" / name
                if fallback.exists():
                    workspace = fallback
                    break
        if not (workspace / "datasets").exists():
            QMessageBox.warning(self, "缺少 ultra tiny 骨架", f"未找到验证骨架：\n{workspace}\n请确认软件包解压完整。")
            return
        if self._ultra_thread is not None and self._ultra_thread.isRunning():
            QMessageBox.information(self, "验证正在运行", "ultra tiny 全链路验证已经在后台运行。")
            return
        self.help_log.setPlainText(
            "正在运行 ultra tiny 全链路验证...\n"
            "目标：1 case × 5 variants，验证 .in → gprMax → .out → .npy → QC → clutter_gt。\n"
            "说明：ultra tiny 与 25-run smoke 属于链路验证，可显式重采样并记录原生 shape；pilot/formal 仍保持严格 QC。\n"
        )
        self.ultra_tiny_button.setEnabled(False)
        self.ultra_tiny_button.setText("验证中...")
        self._ultra_thread = QThread(self)
        self._ultra_worker = SceneWorldFullChainWorker(
            workspace=workspace,
            gprmax_root=settings.gprmax_root,
            python_executable=sys.executable,
            omp_threads=max(1, int(settings.omp_threads or 1)),
            timeout=600,
            allow_resample=True,
            one_case_per_family=False,
            max_cases=1,
            run_label="ultra tiny full-chain",
        )
        self._ultra_worker.moveToThread(self._ultra_thread)
        self._ultra_thread.started.connect(self._ultra_worker.run)
        self._ultra_worker.log.connect(self._append_ultra_tiny_log)
        self._ultra_worker.finished.connect(self._on_ultra_tiny_finished)
        self._ultra_worker.failed.connect(self._on_ultra_tiny_failed)
        self._ultra_worker.finished.connect(self._ultra_thread.quit)
        self._ultra_worker.failed.connect(self._ultra_thread.quit)
        self._ultra_thread.finished.connect(self._cleanup_ultra_tiny_worker)
        self._ultra_thread.start()

    def _append_ultra_tiny_log(self, text: str) -> None:
        self.help_log.appendPlainText(str(text))

    def _on_ultra_tiny_finished(self, report: object) -> None:
        rep = report if isinstance(report, dict) else {}
        status = "成功" if rep.get("ok") else "失败"
        summary = {
            "status": status,
            "ok": rep.get("ok"),
            "case_count": rep.get("case_count"),
            "report_json": rep.get("report_json"),
        }
        self.help_log.appendPlainText("\n--- ultra tiny 验证完成 ---")
        self.help_log.appendPlainText(json.dumps(summary, ensure_ascii=False, indent=2))
        if not rep.get("ok"):
            self.help_log.appendPlainText("\n提示：若仍失败，请把 reports/sceneworld_bscan_run_report.json 和 models/case_000001/bscan_qc_report.json 发给我。")

    def _on_ultra_tiny_failed(self, detail: str) -> None:
        self.help_log.appendPlainText("\n--- ultra tiny 验证异常 ---\n" + detail)

    def _cleanup_ultra_tiny_worker(self) -> None:
        if self._ultra_worker is not None:
            self._ultra_worker.deleteLater()
            self._ultra_worker = None
        if self._ultra_thread is not None:
            self._ultra_thread.deleteLater()
            self._ultra_thread = None
        if hasattr(self, "ultra_tiny_button"):
            self.ultra_tiny_button.setEnabled(True)
            self.ultra_tiny_button.setText("运行最小链路验证")

    def open_advanced_window(self) -> None:
        win = MainWindow()
        win.show()
        self.advanced_windows.append(win)
