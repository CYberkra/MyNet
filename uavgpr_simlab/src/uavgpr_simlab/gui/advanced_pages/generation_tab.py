from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget


@dataclass(frozen=True)
class AdvancedGenerationTabWidgets:
    """Widget references needed by the advanced generation tab owner."""

    page: QWidget
    plan_edit: QLineEdit
    case_override_spin: QSpinBox
    antenna_combo: QComboBox
    component_combo: QComboBox
    gen_log: QPlainTextEdit
    choose_button: QPushButton
    preview_button: QPushButton
    generate_button: QPushButton


def build_advanced_generation_tab(
    *,
    default_plan: Path,
    choose_plan: Callable[[QLineEdit, str], None],
    on_preview_plan: Callable[[], None],
    on_generate_dataset: Callable[[], None],
) -> AdvancedGenerationTabWidgets:
    """Build the advanced simulation-plan tab without generating data."""

    tab = QWidget()
    layout = QVBoxLayout(tab)
    box = QGroupBox("动态生成论文仿真计划：2D 主训练 + 高保真验证预留")
    form = QFormLayout(box)

    plan_edit = QLineEdit(str(default_plan))
    plan_row = QHBoxLayout()
    plan_row.addWidget(plan_edit)
    choose_button = QPushButton("选择 YAML")
    choose_button.clicked.connect(lambda _checked=False: choose_plan(plan_edit, "YAML (*.yaml *.yml);;All Files (*)"))
    plan_row.addWidget(choose_button)
    form.addRow("run plan", plan_row)

    case_override_spin = QSpinBox()
    case_override_spin.setRange(0, 200000)
    case_override_spin.setValue(0)
    case_override_spin.setToolTip("0 表示使用 YAML 中的 scene_count")
    antenna_combo = QComboBox()
    antenna_combo.addItems(["wire_dipole_equivalent", "bowtie_placeholder"])
    component_combo = QComboBox()
    component_combo.addItems(["five_labels_raw_target_clutter_background_air", "raw_only", "raw_target_clutter"])
    form.addRow("覆盖场景数", case_override_spin)
    form.addRow("天线模式", antenna_combo)
    form.addRow("标签组件", component_combo)

    actions = QHBoxLayout()
    preview_button = QPushButton("预览计划摘要")
    preview_button.clicked.connect(on_preview_plan)
    generate_button = QPushButton("生成 gprMax 输入 + 标签 + manifest")
    generate_button.clicked.connect(on_generate_dataset)
    actions.addWidget(preview_button)
    actions.addWidget(generate_button)
    form.addRow(actions)

    layout.addWidget(box)
    gen_log = QPlainTextEdit()
    gen_log.setReadOnly(True)
    layout.addWidget(gen_log, 1)

    return AdvancedGenerationTabWidgets(
        page=tab,
        plan_edit=plan_edit,
        case_override_spin=case_override_spin,
        antenna_combo=antenna_combo,
        component_combo=component_combo,
        gen_log=gen_log,
        choose_button=choose_button,
        preview_button=preview_button,
        generate_button=generate_button,
    )
