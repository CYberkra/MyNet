from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QHeaderView, QVBoxLayout, QWidget


@dataclass(frozen=True)
class AdvancedModelPreviewTabWidgets:
    """Widget references needed by the advanced model-preview tab owner."""

    page: QWidget
    manifest_edit: QLineEdit
    case_table: QTableWidget
    info_box: QPlainTextEdit
    model_canvas: QWidget
    pick_button: QPushButton
    load_button: QPushButton
    note_label: QLabel


def build_advanced_model_preview_tab(
    *,
    choose_manifest: Callable[[QLineEdit, str], None],
    on_load_manifest: Callable[[], None],
    on_selection_changed: Callable[[], None],
    canvas_factory: Callable[[], QWidget],
) -> AdvancedModelPreviewTabWidgets:
    """Build the advanced 3D model-preview tab without loading manifest data."""

    tab = QWidget()
    layout = QVBoxLayout(tab)
    top = QHBoxLayout()
    manifest_edit = QLineEdit("")
    pick_button = QPushButton("选择 manifest")
    pick_button.clicked.connect(lambda _checked=False: choose_manifest(manifest_edit, "CSV (*.csv);;All Files (*)"))
    load_button = QPushButton("加载 case 列表")
    load_button.clicked.connect(on_load_manifest)
    top.addWidget(QLabel("manifest"))
    top.addWidget(manifest_edit, 1)
    top.addWidget(pick_button)
    top.addWidget(load_button)
    layout.addLayout(top)

    splitter = QSplitter(Qt.Horizontal)
    left = QWidget()
    left_layout = QVBoxLayout(left)
    hint = QLabel("选择任一 case 后，右侧会显示 3D/2.5D 地形与基覆界面预览。")
    hint.setWordWrap(True)
    left_layout.addWidget(hint)
    case_table = QTableWidget(0, 6)
    case_table.setHorizontalHeaderLabels(["case_id", "split", "variant", "界面均深(m)", "坡度", "label_json"])
    case_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    case_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    case_table.itemSelectionChanged.connect(on_selection_changed)
    left_layout.addWidget(case_table, 1)
    info_box = QPlainTextEdit()
    info_box.setReadOnly(True)
    left_layout.addWidget(info_box, 1)

    right = QWidget()
    right_layout = QVBoxLayout(right)
    model_canvas = canvas_factory()
    right_layout.addWidget(model_canvas, 1)
    note_label = QLabel("说明：该预览来自 labels.json，可在 gprMax 正式运行前完成质量检查；它不是 FDTD 波场结果。")
    note_label.setObjectName("subtle")
    note_label.setWordWrap(True)
    right_layout.addWidget(note_label)

    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 5)
    layout.addWidget(splitter, 1)

    return AdvancedModelPreviewTabWidgets(
        page=tab,
        manifest_edit=manifest_edit,
        case_table=case_table,
        info_box=info_box,
        model_canvas=model_canvas,
        pick_button=pick_button,
        load_button=load_button,
        note_label=note_label,
    )
