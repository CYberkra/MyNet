from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from uavgpr_simlab.core.config import repo_root


@dataclass(frozen=True)
class AdvancedHistoryTabWidgets:
    """Widget references needed by the advanced main window after history tab construction."""

    page: QWidget
    workspace_edit: QLineEdit
    filter_combo: QComboBox
    thumb_check: QCheckBox
    autorefresh_check: QCheckBox
    limit_spin: QSpinBox
    summary: QLabel
    table: QTableWidget
    log: QPlainTextEdit
    model_canvas: QWidget
    bscan_canvas: QWidget
    detail_box: QPlainTextEdit
    choose_button: QPushButton
    refresh_button: QPushButton
    export_button: QPushButton
    delete_button: QPushButton
    delete_outputs_button: QPushButton


def build_advanced_history_tab(
    *,
    choose_dir: Callable[[QLineEdit], None],
    on_refresh: Callable[..., None],
    on_export: Callable[[], None],
    on_delete: Callable[[], None],
    on_delete_outputs: Callable[[], None],
    on_filter_changed: Callable[..., None],
    on_selection_changed: Callable[[], None],
    model_canvas_factory: Callable[[], QWidget],
    bscan_canvas_factory: Callable[[], QWidget],
    workspace_default: Path | None = None,
) -> AdvancedHistoryTabWidgets:
    """Build the advanced visual-history tab without owning history logic.

    This builder creates controls and wires callbacks supplied by the advanced
    main window. History scanning, thumbnail generation, detail preview,
    export and deletion remain in ``main_window.py`` / core history modules.
    """

    tab = QWidget()
    layout = QVBoxLayout(tab)
    header = QLabel("历史仿真可视化：每条记录同时显示模型视图、B-scan 预览和实时 trace 数。")
    header.setObjectName("banner")
    layout.addWidget(header)

    top = QGridLayout()
    workspace_edit = QLineEdit(str(workspace_default or (repo_root() / "workspace")))
    choose_button = QPushButton("选择 workspace")
    choose_button.clicked.connect(lambda _checked=False: choose_dir(workspace_edit))
    refresh_button = QPushButton("刷新历史记录")
    refresh_button.clicked.connect(on_refresh)
    export_button = QPushButton("导出 CSV")
    export_button.clicked.connect(on_export)
    delete_button = QPushButton("删除选中记录")
    delete_button.clicked.connect(on_delete)
    delete_outputs_button = QPushButton("彻底删除选中记录+输出")
    delete_outputs_button.clicked.connect(on_delete_outputs)

    filter_combo = QComboBox()
    filter_combo.addItems(["全部", "running", "done", "failed", "stale_running", "geometry-only", "full simulation"])
    filter_combo.currentIndexChanged.connect(on_filter_changed)
    thumb_check = QCheckBox("显示模型/B-scan缩略图")
    thumb_check.setChecked(True)
    autorefresh_check = QCheckBox("运行中自动刷新")
    autorefresh_check.setChecked(True)
    limit_spin = QSpinBox()
    limit_spin.setRange(10, 5000)
    limit_spin.setValue(300)

    top.addWidget(QLabel("workspace"), 0, 0)
    top.addWidget(workspace_edit, 0, 1, 1, 6)
    top.addWidget(choose_button, 0, 7)
    top.addWidget(QLabel("筛选"), 1, 0)
    top.addWidget(filter_combo, 1, 1)
    top.addWidget(QLabel("最多显示"), 1, 2)
    top.addWidget(limit_spin, 1, 3)
    top.addWidget(thumb_check, 1, 4)
    top.addWidget(autorefresh_check, 1, 5)
    top.addWidget(refresh_button, 1, 6)
    top.addWidget(export_button, 1, 7)
    top.addWidget(delete_button, 2, 6)
    top.addWidget(delete_outputs_button, 2, 7)
    layout.addLayout(top)

    splitter = QSplitter(Qt.Horizontal)

    left = QWidget()
    left_layout = QVBoxLayout(left)
    summary = QLabel("尚未加载历史记录。")
    summary.setObjectName("banner")
    left_layout.addWidget(summary)
    table = QTableWidget(0, 13)
    table.setHorizontalHeaderLabels(["状态", "模型视图", "B-scan", "trace", "时间", "case_id", "variant", "n", "geometry", "return", "job_id", "input", "marker"])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.itemSelectionChanged.connect(on_selection_changed)
    table.setColumnWidth(1, 150)
    table.setColumnWidth(2, 150)
    table.setColumnWidth(3, 70)
    table.setColumnWidth(4, 130)
    table.setColumnHidden(11, True)
    table.setColumnHidden(12, True)
    left_layout.addWidget(table, 1)
    log = QPlainTextEdit()
    log.setReadOnly(True)
    log.setMaximumHeight(150)
    left_layout.addWidget(log)

    right = QWidget()
    right_layout = QVBoxLayout(right)
    right_title = QLabel("选中记录详情：模型几何 + 已完成/实时 B-scan")
    right_title.setObjectName("banner")
    right_layout.addWidget(right_title)
    model_canvas = model_canvas_factory()
    bscan_canvas = bscan_canvas_factory()
    right_layout.addWidget(model_canvas, 1)
    right_layout.addWidget(bscan_canvas, 1)
    detail_box = QPlainTextEdit()
    detail_box.setReadOnly(True)
    detail_box.setMaximumHeight(170)
    right_layout.addWidget(detail_box)

    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 5)
    splitter.setStretchFactor(1, 4)
    layout.addWidget(splitter, 1)

    return AdvancedHistoryTabWidgets(
        page=tab,
        workspace_edit=workspace_edit,
        filter_combo=filter_combo,
        thumb_check=thumb_check,
        autorefresh_check=autorefresh_check,
        limit_spin=limit_spin,
        summary=summary,
        table=table,
        log=log,
        model_canvas=model_canvas,
        bscan_canvas=bscan_canvas,
        detail_box=detail_box,
        choose_button=choose_button,
        refresh_button=refresh_button,
        export_button=export_button,
        delete_button=delete_button,
        delete_outputs_button=delete_outputs_button,
    )
