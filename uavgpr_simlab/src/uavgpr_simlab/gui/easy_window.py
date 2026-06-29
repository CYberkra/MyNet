from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from uavgpr_simlab import __display_version__
from uavgpr_simlab.core.config import AppConfig, repo_root
from uavgpr_simlab.services.easy_batch_service import read_manifest_rows, unique_case_rows, workspace_from_manifest
from uavgpr_simlab.services.easy_history_service import summarize_project
from uavgpr_simlab.services.environment_service import (
    EasyEnvironmentSettings,
    build_runtime_config_for_easy,
    load_easy_environment_settings,
)
from uavgpr_simlab.services.project_service import discover_model_plan_presets
from uavgpr_simlab.gui.controllers import (
    BatchControllerMixin,
    HistoryControllerMixin,
    HomeProjectControllerMixin,
    ModelPreviewControllerMixin,
    SettingsControllerMixin,
)
from uavgpr_simlab.gui.easy_ui import EASY_STYLE
from uavgpr_simlab.gui.main_window import LiveQueueWorker, MainWindow
from uavgpr_simlab.gui.easy_workers import GprMaxSourceSmokeWorker, SceneWorldFullChainWorker


class EasyMainWindow(
    HomeProjectControllerMixin,
    ModelPreviewControllerMixin,
    BatchControllerMixin,
    HistoryControllerMixin,
    SettingsControllerMixin,
    QMainWindow,
):
    """Human-readable GUI focused on model pictures, batch state and results.

    The advanced engineering GUI is still available from 设置与帮助 -> 打开高级工程界面.
    """

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle(f"UavGPR-SimLab {__display_version__} 产品化易用版 - 看模型、看进度、看结果")
        self.resize(1480, 900)
        self.setMinimumSize(1240, 760)
        self.config_path = config_path
        self.cfg = AppConfig()
        self.env_settings = load_easy_environment_settings()
        if self.env_settings.gprmax_root:
            self.cfg.runtime.gprmax_source_dir = self.env_settings.gprmax_root
        self.cfg.runtime.conda_env_gprmax = self.env_settings.conda_env
        self.cfg.runtime.gpu_ids = self.env_settings.gpu_ids
        self.cfg.runtime.omp_threads = self.env_settings.omp_threads
        self.cfg.runtime.gpu_enabled = self.env_settings.use_gpu
        self.cfg.runtime.use_conda_run = self.env_settings.use_conda_run
        self.root_dir = repo_root()
        self.workspace = self.root_dir / "workspace" / "easy_project"
        self.plan_path = self.root_dir / "configs" / "run_plan_3060_quick.yaml"
        self.plan_presets = discover_model_plan_presets(self.root_dir)
        self.current_manifest: Optional[Path] = None
        self.current_manifest_rows: list[dict[str, str]] = []
        self.live_worker: Optional[LiveQueueWorker] = None
        self.advanced_windows: list[MainWindow] = []
        self._smoke_thread: Optional[QThread] = None
        self._smoke_worker: Optional[GprMaxSourceSmokeWorker] = None
        self._ultra_thread: Optional[QThread] = None
        self._ultra_worker: Optional[SceneWorldFullChainWorker] = None
        self._scene_job_thread: Optional[QThread] = None
        self._scene_job_worker: Optional[SceneWorldFullChainWorker] = None
        self._batch_row_by_case_variant: dict[tuple[str, str], int] = {}
        self._history_entry_by_job: dict[str, object] = {}

        QApplication.instance().setStyleSheet(EASY_STYLE)  # type: ignore[union-attr]
        root = QWidget(); main = QHBoxLayout(root); main.setContentsMargins(14, 14, 14, 14); main.setSpacing(14)
        self.sidebar = QFrame(); self.sidebar.setObjectName("sidebar"); self.sidebar.setFixedWidth(185)
        side = QVBoxLayout(self.sidebar); side.setSpacing(6)
        logo = QLabel("UavGPR\nSimLab"); logo.setObjectName("logo"); side.addWidget(logo)
        cap = QLabel("看模型 · 跑批量 · 看结果"); cap.setObjectName("sideCaption"); cap.setWordWrap(True); side.addWidget(cap)
        self.nav_buttons: list[QPushButton] = []
        navs = [
            ("首页", "当前项目状态"),
            ("模型预览", "看模型长什么样"),
            ("批量仿真", "哪些在跑，跑到哪了"),
            ("历史与结果", "看每次仿真效果"),
            ("项目管理", "工作目录和计划"),
            ("设置与帮助", "环境和高级入口"),
        ]
        for idx, (name, tip) in enumerate(navs):
            icons = ["🏠", "🖼", "▶", "🗂", "📁", "⚙"]
            b = QPushButton(f" {icons[idx]}  {name}"); b.setObjectName("nav"); b.setCheckable(True); b.setToolTip(tip)
            b.clicked.connect(lambda _=False, i=idx: self.show_page(i))
            side.addWidget(b)
            self.nav_buttons.append(b)
        side.addStretch(1)
        self.sidebar_hint = QLabel("少术语，多图像\n先看模型，再批量跑")
        self.sidebar_hint.setStyleSheet("color:#c7dcf4;padding:12px;font-weight:700;")
        self.sidebar_hint.setWordWrap(True); side.addWidget(self.sidebar_hint)
        main.addWidget(self.sidebar)

        self.stack = QStackedWidget(); main.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self._build_home_page()
        self._build_model_page()
        self._build_batch_page()
        self._build_history_page()
        self._build_project_page()
        self._build_help_page()
        self.show_page(0)

        self.home_timer = QTimer(self); self.home_timer.setInterval(3500); self.home_timer.timeout.connect(self.refresh_home)
        self.home_timer.start()
        self.history_timer = QTimer(self); self.history_timer.setInterval(3000); self.history_timer.timeout.connect(self._tick_history_live)
        self.history_timer.start()

    # --- layout helpers ---
    def show_page(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        for i, b in enumerate(self.nav_buttons):
            b.setChecked(i == idx)
        if idx == 0:
            self.refresh_home()
        elif idx == 1:
            self.load_model_manifest(silent=True)
        elif idx == 2:
            self.refresh_batch_plan()
        elif idx == 3:
            self.refresh_history()

    def _choose_file(self, edit: QLineEdit, filt: str) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "选择文件", str(self.root_dir), filt)
        if p:
            edit.setText(p)

    def _choose_dir(self, edit: QLineEdit) -> None:
        p = QFileDialog.getExistingDirectory(self, "选择目录", str(self.root_dir))
        if p:
            edit.setText(p)

    def _workspace_from_manifest(self) -> Path:
        if self.current_manifest and self.current_manifest.exists():
            return workspace_from_manifest(self.current_manifest)
        return Path(self.workspace_edit.text()).expanduser()

    def _resolve_workspace_path(self, value: str | Path) -> Path:
        p = Path(str(value)).expanduser()
        if p.is_absolute():
            return p
        return (self._workspace_from_manifest() / p).resolve()

    def _summary_counts(self) -> dict[str, int]:
        workspace = Path(self.workspace_edit.text()).expanduser() if hasattr(self, "workspace_edit") else self.workspace
        manifest = self.current_manifest if self.current_manifest and self.current_manifest.exists() else None
        return summarize_project(workspace, manifest).to_dict()

    def _load_manifest_rows(self, manifest: Path) -> list[dict[str, str]]:
        return read_manifest_rows(manifest)

    def _unique_case_rows(self) -> list[dict[str, str]]:
        return unique_case_rows(self.current_manifest_rows)

    def _env_settings_from_ui(self) -> EasyEnvironmentSettings:
        base = getattr(self, "env_settings", EasyEnvironmentSettings())
        return EasyEnvironmentSettings(
            runtime_root=base.runtime_root,
            gprmax_root=self.gprmax_root_edit.text().strip(),
            conda_env=self.conda_env_edit.text().strip(),
            conda_env_prefix=base.conda_env_prefix,
            conda_exe=base.conda_exe,
            gpu_ids=self.gpu_ids_edit.text().strip() or "0",
            omp_threads=int(self.omp_spin.value()),
            use_gpu=self.use_gpu_check.isChecked(),
            use_conda_run=self.use_conda_check.isChecked(),
        )

    def _cfg_for_current(self) -> AppConfig:
        return build_runtime_config_for_easy(
            plan_path=self.plan_edit.text(),
            workspace=self.workspace_edit.text(),
            current_manifest=self.current_manifest,
            settings=self._env_settings_from_ui(),
        )



def run_easy_app(config_path: Optional[Path] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    win = EasyMainWindow(config_path=config_path)
    win.show()
    return app.exec()
