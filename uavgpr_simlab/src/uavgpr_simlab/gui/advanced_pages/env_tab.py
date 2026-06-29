from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from uavgpr_simlab.core.config import AppConfig, repo_root


@dataclass(frozen=True)
class AdvancedEnvTabWidgets:
    """Widget references needed by the advanced main window after tab construction."""

    page: QWidget
    workspace_edit: QLineEdit
    gprmax_root_edit: QLineEdit
    conda_env_edit: QLineEdit
    gpu_ids_edit: QLineEdit
    omp_spin: QSpinBox
    use_gpu_check: QCheckBox
    use_conda_check: QCheckBox
    log_box: QPlainTextEdit
    save_button: QPushButton
    check_button: QPushButton
    smoke_command_button: QPushButton
    setup_script_button: QPushButton


def _directory_row(widget: QLineEdit, choose_dir: Callable[[QLineEdit], None]) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(widget)
    browse = QPushButton("浏览")
    browse.clicked.connect(lambda _checked=False, w=widget: choose_dir(w))
    row.addWidget(browse)
    return row


def build_advanced_env_tab(
    *,
    cfg: AppConfig,
    choose_dir: Callable[[QLineEdit], None],
    on_save: Callable[[], None],
    on_check: Callable[[], None],
    on_show_smoke_command: Callable[[], None],
    on_open_setup_script: Callable[[], None],
) -> AdvancedEnvTabWidgets:
    """Build the advanced environment tab without owning environment logic.

    This builder only creates widgets and wires callbacks supplied by the main
    window. Environment persistence, report generation and smoke-command
    construction remain in ``main_window.py`` / core services for this step.
    """

    tab = QWidget()
    layout = QVBoxLayout(tab)
    grid = QGridLayout()

    env_box = QGroupBox("1. 本机环境与单卡 GPU 设置")
    form = QFormLayout(env_box)
    workspace_edit = QLineEdit(str(repo_root() / "workspace"))
    gprmax_root_edit = QLineEdit(cfg.runtime.gprmax_source_dir)
    conda_env_edit = QLineEdit(cfg.runtime.conda_env_gprmax)
    gpu_ids_edit = QLineEdit(cfg.runtime.gpu_ids)
    omp_spin = QSpinBox()
    omp_spin.setRange(0, 128)
    omp_spin.setValue(4)
    use_gpu_check = QCheckBox("使用 gprMax -gpu 0（3060/4090 单机单卡）")
    use_gpu_check.setChecked(True)
    use_conda_check = QCheckBox("使用 conda run -n gprMax 调用")
    use_conda_check.setChecked(True)

    form.addRow("工作目录", _directory_row(workspace_edit, choose_dir))
    form.addRow("gprMax 源码目录", _directory_row(gprmax_root_edit, choose_dir))
    form.addRow("conda 环境名", conda_env_edit)
    form.addRow("GPU ID", gpu_ids_edit)
    form.addRow("OpenMP 线程数", omp_spin)
    form.addRow("GPU", use_gpu_check)
    form.addRow("Conda", use_conda_check)
    grid.addWidget(env_box, 0, 0)

    ops_box = QGroupBox("2. 一键操作")
    ops = QVBoxLayout(ops_box)
    save_button = QPushButton("保存本地设置")
    save_button.clicked.connect(on_save)
    check_button = QPushButton("环境检查并写 report")
    check_button.clicked.connect(on_check)
    smoke_command_button = QPushButton("显示 gprMax smoke test 命令")
    smoke_command_button.clicked.connect(on_show_smoke_command)
    setup_script_button = QPushButton("打开 4090 无人值守安装脚本")
    setup_script_button.clicked.connect(on_open_setup_script)
    for button in [save_button, check_button, smoke_command_button, setup_script_button]:
        ops.addWidget(button)
    hint = QLabel("建议：3060 先跑 1-3 个 geometry-only；4090 安装完成后先点环境检查，再跑正式 batch。")
    hint.setObjectName("subtle")
    hint.setWordWrap(True)
    ops.addWidget(hint)
    grid.addWidget(ops_box, 0, 1)
    grid.setColumnStretch(0, 3)
    grid.setColumnStretch(1, 2)
    layout.addLayout(grid)

    log_box = QPlainTextEdit()
    log_box.setReadOnly(True)
    log_box.setPlaceholderText("环境检查、安装路径、smoke-test 命令会显示在这里。")
    layout.addWidget(log_box, 1)

    return AdvancedEnvTabWidgets(
        page=tab,
        workspace_edit=workspace_edit,
        gprmax_root_edit=gprmax_root_edit,
        conda_env_edit=conda_env_edit,
        gpu_ids_edit=gpu_ids_edit,
        omp_spin=omp_spin,
        use_gpu_check=use_gpu_check,
        use_conda_check=use_conda_check,
        log_box=log_box,
        save_button=save_button,
        check_button=check_button,
        smoke_command_button=smoke_command_button,
        setup_script_button=setup_script_button,
    )
