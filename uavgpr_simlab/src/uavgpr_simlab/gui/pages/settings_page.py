from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from uavgpr_simlab.core.config import AppConfig
from uavgpr_simlab.gui.easy_cards import help_step, page_header
from uavgpr_simlab.services.environment_service import EasyEnvironmentSettings


@dataclass(frozen=True)
class SettingsPageWidgets:
    """Widget references the main window needs after building the settings page."""

    page: QWidget
    gprmax_root_edit: QLineEdit
    conda_env_edit: QLineEdit
    gpu_ids_edit: QLineEdit
    omp_spin: QSpinBox
    use_gpu_check: QCheckBox
    use_conda_check: QCheckBox
    smoke_test_button: QPushButton
    link_check_button: QPushButton
    help_log: QPlainTextEdit


def build_settings_help_page(
    *,
    env_settings: EasyEnvironmentSettings,
    cfg: AppConfig,
    on_save: Callable[[], None],
    on_check: Callable[[], None],
    on_smoke_test: Callable[[], None],
    on_link_check: Callable[[], None],
    on_open_advanced: Callable[[], None],
) -> SettingsPageWidgets:
    """Build the settings/help page for the easy GUI.

    The page builder owns only widget construction. Reading, saving and checking
    environment values stay in services/environment_service.py and the main
    window remains responsible for wiring callbacks into application state.
    """

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(
        page_header(
            "设置与帮助：少数必要环境设置 + 高级入口",
            "首次使用先检查 gprMax 和 GPU；日常使用只需要从首页、模型预览、批量仿真、历史结果四个页面走。",
        )
    )

    env = QGroupBox("环境设置")
    form = QFormLayout(env)
    gprmax_root_edit = QLineEdit(str(env_settings.gprmax_root or cfg.runtime.gprmax_source_dir or ""))
    conda_env_edit = QLineEdit(str(env_settings.conda_env or cfg.runtime.conda_env_gprmax or "gprMax"))
    gpu_ids_edit = QLineEdit(str(env_settings.gpu_ids or cfg.runtime.gpu_ids or "0"))
    omp_spin = QSpinBox()
    omp_spin.setRange(0, 128)
    omp_spin.setValue(int(env_settings.omp_threads or cfg.runtime.omp_threads or 4))
    use_gpu_check = QCheckBox("使用 GPU")
    use_gpu_check.setChecked(bool(env_settings.use_gpu))
    use_conda_check = QCheckBox("使用 conda run 调用 gprMax")
    use_conda_check.setChecked(bool(env_settings.use_conda_run))

    form.addRow("gprMax 源码目录", gprmax_root_edit)
    form.addRow("conda 环境名", conda_env_edit)
    form.addRow("GPU ID", gpu_ids_edit)
    form.addRow("OpenMP 线程", omp_spin)
    form.addRow("GPU", use_gpu_check)
    form.addRow("Conda", use_conda_check)
    layout.addWidget(env)

    btns = QHBoxLayout()
    save = QPushButton("保存设置")
    save.clicked.connect(on_save)
    check = QPushButton("检查环境")
    check.clicked.connect(on_check)
    smoke = QPushButton("最小 CPU 测试")
    smoke.setObjectName("light")
    smoke.setToolTip("使用当前 Python 对本地 gprMax 源码运行极小 CPU A-scan，不启用 GPU。")
    smoke.clicked.connect(on_smoke_test)
    link_check = QPushButton("运行最小链路验证")
    link_check.setObjectName("primary")
    link_check.setToolTip("只运行内置 ultra tiny：验证 .in → gprMax → .out → .npy → QC → clutter_gt。正式任务请到批量仿真页选择运行配置。")
    link_check.clicked.connect(on_link_check)
    advanced = QPushButton("打开高级工程界面")
    advanced.setObjectName("light")
    advanced.clicked.connect(on_open_advanced)
    btns.addWidget(save)
    btns.addWidget(check)
    btns.addWidget(smoke)
    btns.addWidget(link_check)
    btns.addWidget(advanced)
    btns.addStretch(1)
    layout.addLayout(btns)

    help_text = QGroupBox("怎么用：五步就够")
    help_layout = QVBoxLayout(help_text)
    for text in [
        "1. 在“项目管理”确认工作目录和模型数量。",
        "2. 在“模型预览”点击“生成一批模型”，先看模型是否合理。",
        "3. 在“批量仿真”点击“预检任务”，确认哪些会跑、哪些会自动跳过。",
        "4. 在“批量仿真”选择运行配置，点击“开始运行统一任务”。",
        "5. 批量页看实时日志/B-scan，历史页自动归档已完成和失败结果。",
    ]:
        help_layout.addWidget(help_step(text))
    layout.addWidget(help_text)

    help_log = QPlainTextEdit()
    help_log.setReadOnly(True)
    layout.addWidget(help_log, 1)

    return SettingsPageWidgets(
        page=page,
        gprmax_root_edit=gprmax_root_edit,
        conda_env_edit=conda_env_edit,
        gpu_ids_edit=gpu_ids_edit,
        omp_spin=omp_spin,
        use_gpu_check=use_gpu_check,
        use_conda_check=use_conda_check,
        smoke_test_button=smoke,
        link_check_button=link_check,
        help_log=help_log,
    )
