from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


@dataclass(frozen=True)
class AdvancedQcTabWidgets:
    """Widget references needed by the advanced QC/export tab owner."""

    page: QWidget
    out_input_edit: QLineEdit
    qc_text: QPlainTextEdit
    choose_button: QPushButton
    export_button: QPushButton


def build_advanced_qc_tab(
    *,
    choose_input: Callable[[QLineEdit, str], None],
    on_export_products: Callable[[], None],
) -> AdvancedQcTabWidgets:
    """Build the advanced .out QC/export tab without exporting products."""

    tab = QWidget()
    layout = QVBoxLayout(tab)
    box = QGroupBox("仿真 .out 后处理与 ML 数据导出")
    form = QFormLayout(box)
    out_input_edit = QLineEdit("")
    row = QHBoxLayout()
    row.addWidget(out_input_edit)
    choose_button = QPushButton("选择对应 .in")
    choose_button.clicked.connect(lambda _checked=False: choose_input(out_input_edit, "gprMax input (*.in);;All Files (*)"))
    row.addWidget(choose_button)
    form.addRow("input .in", row)
    export_button = QPushButton("合并可读 .out 并导出传统基线/NPZ/PNG")
    export_button.clicked.connect(on_export_products)
    form.addRow(export_button)
    layout.addWidget(box)

    qc_text = QPlainTextEdit()
    qc_text.setReadOnly(True)
    qc_text.setPlainText(
        "流程说明：raw/target_only/clutter_only/background_only/air_only 用同一几何标签；raw 运行结束后会自动尝试读取 .out，"
        "生成 raw、dewow、mean_subtract、gain、svd_clean、fk_clean 等产品，供 PGDA-CSNet 数据加载器使用。"
    )
    layout.addWidget(qc_text, 1)

    return AdvancedQcTabWidgets(
        page=tab,
        out_input_edit=out_input_edit,
        qc_text=qc_text,
        choose_button=choose_button,
        export_button=export_button,
    )
