from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
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

from uavgpr_simlab.gui.easy_cards import page_header
from uavgpr_simlab.services.project_service import ModelPlanPreset


@dataclass(frozen=True)
class ProjectPageWidgets:
    """Widget references the main window needs after building the project page."""

    page: QWidget
    workspace_edit: QLineEdit
    plan_preset_combo: QComboBox
    plan_edit: QLineEdit
    case_count_spin: QSpinBox
    project_plan_text: QPlainTextEdit


def build_project_page(
    *,
    workspace: Path,
    plan_path: Path,
    plan_presets: Sequence[ModelPlanPreset],
    on_choose_workspace: Callable[[QLineEdit], None],
    on_choose_plan: Callable[[QLineEdit], None],
    on_select_plan_preset: Callable[[int], None],
    on_preview_plan: Callable[[], None],
) -> ProjectPageWidgets:
    """Build the project-management page for the easy GUI.

    The page builder owns only widget construction. Project-plan parsing and
    model generation remain in services/project_service.py; the main window
    wires callbacks and keeps cross-page state such as the active manifest.
    """

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(
        page_header(
            "项目管理：路径、计划和数据都放在这里",
            "先选择模型配置，再确认工作目录和模型数量。模型预览页的“生成一批模型”会使用这里的配置。",
        )
    )

    project_box = QGroupBox("当前项目")
    form = QFormLayout(project_box)

    workspace_edit = QLineEdit(str(workspace))
    plan_edit = QLineEdit(str(plan_path))
    case_count_spin = QSpinBox()
    case_count_spin.setRange(1, 200000)
    case_count_spin.setValue(20)

    plan_preset_combo = QComboBox()
    plan_preset_combo.setObjectName("planPresetCombo")
    if plan_presets:
        for preset in plan_presets:
            plan_preset_combo.addItem(preset.label, str(preset.path))
        default_index = next((i for i, preset in enumerate(plan_presets) if preset.path.resolve() == plan_path.resolve()), 0)
        plan_preset_combo.setCurrentIndex(default_index)
    else:
        plan_preset_combo.addItem("未发现 configs/run_plan*.yaml，请手动选择", "")
        plan_preset_combo.setEnabled(False)
    plan_preset_combo.currentIndexChanged.connect(on_select_plan_preset)

    workspace_row = QHBoxLayout()
    workspace_row.addWidget(workspace_edit)
    choose_workspace = QPushButton("选择")
    choose_workspace.setObjectName("light")
    choose_workspace.clicked.connect(lambda: on_choose_workspace(workspace_edit))
    workspace_row.addWidget(choose_workspace)

    plan_row = QHBoxLayout()
    plan_row.addWidget(plan_edit)
    choose_plan = QPushButton("选择自定义 YAML")
    choose_plan.setObjectName("light")
    choose_plan.clicked.connect(lambda: on_choose_plan(plan_edit))
    plan_row.addWidget(choose_plan)

    hint = QLabel("模型配置来自 configs/run_plan*.yaml；选择后会自动回填下方仿真计划路径。")
    hint.setObjectName("pageHint")
    hint.setWordWrap(True)

    form.addRow("模型配置", plan_preset_combo)
    form.addRow("配置说明", hint)
    form.addRow("工作目录", workspace_row)
    form.addRow("仿真计划", plan_row)
    form.addRow("生成模型数量", case_count_spin)
    layout.addWidget(project_box)

    project_plan_text = QPlainTextEdit()
    project_plan_text.setReadOnly(True)
    load_plan = QPushButton("预览计划内容")
    load_plan.clicked.connect(on_preview_plan)
    layout.addWidget(load_plan)
    layout.addWidget(project_plan_text, 1)

    return ProjectPageWidgets(
        page=page,
        workspace_edit=workspace_edit,
        plan_preset_combo=plan_preset_combo,
        plan_edit=plan_edit,
        case_count_spin=case_count_spin,
        project_plan_text=project_plan_text,
    )
