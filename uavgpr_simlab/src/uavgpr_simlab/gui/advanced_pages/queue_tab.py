from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class AdvancedQueueTabWidgets:
    """Widget references needed by the advanced main window after queue tab construction."""

    page: QWidget
    manifest_edit: QLineEdit
    variant_combo: QComboBox
    limit_spin: QSpinBox
    geometry_check: QCheckBox
    skip_completed_check: QCheckBox
    force_rerun_check: QCheckBox
    task_list: QListWidget
    progress: QProgressBar
    queue_log: QPlainTextEdit
    queue_canvas: QWidget
    preview_info: QLabel
    pick_button: QPushButton
    load_button: QPushButton
    write_geometry_button: QPushButton
    write_full_button: QPushButton
    run_selected_button: QPushButton
    run_batch_button: QPushButton
    stop_button: QPushButton


def build_advanced_queue_tab(
    *,
    choose_manifest: Callable[[QLineEdit, str], None],
    on_load_manifest: Callable[[], None],
    on_write_geometry_bat: Callable[[], None],
    on_write_full_bat: Callable[[], None],
    on_run_selected: Callable[[], None],
    on_run_batch: Callable[[], None],
    on_stop: Callable[[], None],
    canvas_factory: Callable[[], QWidget],
) -> AdvancedQueueTabWidgets:
    """Build the advanced batch-queue tab without owning run logic.

    This builder creates queue controls and wires callbacks supplied by the
    advanced main window. Manifest parsing, BAT generation, live gprMax
    execution, progress updates, preview loading, markers and task semantics
    remain in ``main_window.py`` / core runner modules.
    """

    tab = QWidget()
    layout = QVBoxLayout(tab)

    top = QGridLayout()
    manifest_edit = QLineEdit("")
    pick_button = QPushButton("选择 manifest")
    pick_button.clicked.connect(lambda _checked=False: choose_manifest(manifest_edit, "CSV (*.csv);;All Files (*)"))
    load_button = QPushButton("加载任务")
    load_button.clicked.connect(on_load_manifest)

    variant_combo = QComboBox()
    variant_combo.addItems(["raw", "target_only", "clutter_only", "background_only", "air_only"])
    limit_spin = QSpinBox()
    limit_spin.setRange(1, 100000)
    limit_spin.setValue(1)
    geometry_check = QCheckBox("geometry-only")
    geometry_check.setChecked(True)
    skip_completed_check = QCheckBox("跳过已完成")
    skip_completed_check.setChecked(True)
    force_rerun_check = QCheckBox("强制重跑")
    force_rerun_check.setChecked(False)

    top.addWidget(QLabel("manifest"), 0, 0)
    top.addWidget(manifest_edit, 0, 1, 1, 5)
    top.addWidget(pick_button, 0, 6)
    top.addWidget(load_button, 0, 7)
    top.addWidget(QLabel("variant"), 1, 0)
    top.addWidget(variant_combo, 1, 1)
    top.addWidget(QLabel("批量数量"), 1, 2)
    top.addWidget(limit_spin, 1, 3)
    top.addWidget(geometry_check, 1, 4)
    top.addWidget(skip_completed_check, 2, 1)
    top.addWidget(force_rerun_check, 2, 2)

    write_geometry_button = QPushButton("写 geometry-only BAT")
    write_geometry_button.clicked.connect(on_write_geometry_bat)
    write_full_button = QPushButton("写完整 raw BAT")
    write_full_button.clicked.connect(on_write_full_bat)
    run_selected_button = QPushButton("运行选中任务")
    run_selected_button.clicked.connect(on_run_selected)
    run_batch_button = QPushButton("运行前 N 个任务")
    run_batch_button.clicked.connect(on_run_batch)
    stop_button = QPushButton("停止")
    stop_button.clicked.connect(on_stop)

    top.addWidget(write_geometry_button, 1, 5)
    top.addWidget(write_full_button, 1, 6)
    top.addWidget(run_selected_button, 2, 5)
    top.addWidget(run_batch_button, 2, 6)
    top.addWidget(stop_button, 2, 7)
    layout.addLayout(top)

    splitter = QSplitter(Qt.Horizontal)

    left = QWidget()
    left_layout = QVBoxLayout(left)
    task_list = QListWidget()
    left_layout.addWidget(task_list, 3)
    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    left_layout.addWidget(progress)
    queue_log = QPlainTextEdit()
    queue_log.setReadOnly(True)
    left_layout.addWidget(queue_log, 2)

    right = QWidget()
    right_layout = QVBoxLayout(right)
    queue_canvas = canvas_factory()
    preview_info = QLabel("Waiting for readable .out files; geometry-only does not produce waveforms.")
    preview_info.setObjectName("subtle")
    right_layout.addWidget(queue_canvas, 1)
    right_layout.addWidget(preview_info)

    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 4)
    layout.addWidget(splitter, 1)

    return AdvancedQueueTabWidgets(
        page=tab,
        manifest_edit=manifest_edit,
        variant_combo=variant_combo,
        limit_spin=limit_spin,
        geometry_check=geometry_check,
        skip_completed_check=skip_completed_check,
        force_rerun_check=force_rerun_check,
        task_list=task_list,
        progress=progress,
        queue_log=queue_log,
        queue_canvas=queue_canvas,
        preview_info=preview_info,
        pick_button=pick_button,
        load_button=load_button,
        write_geometry_button=write_geometry_button,
        write_full_button=write_full_button,
        run_selected_button=run_selected_button,
        run_batch_button=run_batch_button,
        stop_button=stop_button,
    )
