from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class AdvancedRealCsvTabWidgets:
    """Widget references needed by the advanced main window after real-CSV tab construction."""

    page: QWidget
    csv_edit: QLineEdit
    max_traces_spin: QSpinBox
    csv_canvas: QWidget
    csv_info: QPlainTextEdit
    pick_button: QPushButton
    load_button: QPushButton
    fk_button: QPushButton
    export_button: QPushButton


def build_advanced_real_csv_tab(
    *,
    default_csv: str,
    choose_csv: Callable[[QLineEdit, str], None],
    on_load_preview: Callable[[], None],
    on_show_fk: Callable[[], None],
    on_export_qc: Callable[[], None],
    canvas_factory: Callable[[], QWidget],
) -> AdvancedRealCsvTabWidgets:
    """Build the advanced real-CSV / weak-supervision tab.

    The builder owns only widget construction and callback wiring. CSV parsing,
    background removal, robust normalization, f-k plotting data and QC export
    remain in ``main_window.py`` / ``core.real_data``.
    """

    tab = QWidget()
    layout = QVBoxLayout(tab)

    top = QHBoxLayout()
    csv_edit = QLineEdit(default_csv)
    pick_button = QPushButton("选择 CSV")
    pick_button.clicked.connect(lambda _checked=False: choose_csv(csv_edit, "CSV (*.csv);;All Files (*)"))
    load_button = QPushButton("加载并预览")
    load_button.clicked.connect(on_load_preview)
    export_button = QPushButton("导出 NPZ/PNG 质控")
    export_button.clicked.connect(on_export_qc)
    fk_button = QPushButton("显示 f-k 图")
    fk_button.clicked.connect(on_show_fk)
    max_traces_spin = QSpinBox()
    max_traces_spin.setRange(1, 200000)
    max_traces_spin.setValue(300)

    for widget in [
        QLabel("CSV"),
        csv_edit,
        pick_button,
        QLabel("最多道数"),
        max_traces_spin,
        load_button,
        fk_button,
        export_button,
    ]:
        top.addWidget(widget)
    layout.addLayout(top)

    splitter = QSplitter(Qt.Vertical)
    csv_canvas = canvas_factory()
    csv_info = QPlainTextEdit()
    csv_info.setReadOnly(True)
    csv_info.setMaximumHeight(210)
    csv_info.setPlaceholderText("CSV 解析信息、弱监督导出和 QC 结果会显示在这里。")
    splitter.addWidget(csv_canvas)
    splitter.addWidget(csv_info)
    splitter.setStretchFactor(0, 4)
    splitter.setStretchFactor(1, 1)
    layout.addWidget(splitter, 1)

    return AdvancedRealCsvTabWidgets(
        page=tab,
        csv_edit=csv_edit,
        max_traces_spin=max_traces_spin,
        csv_canvas=csv_canvas,
        csv_info=csv_info,
        pick_button=pick_button,
        load_button=load_button,
        fk_button=fk_button,
        export_button=export_button,
    )
