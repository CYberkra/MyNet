from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


@dataclass(frozen=True)
class AdvancedDashboardTabWidgets:
    """Widget references for the advanced dashboard / guide tab."""

    page: QWidget
    summary: QPlainTextEdit
    env_button: QPushButton
    generation_button: QPushButton
    preview_button: QPushButton
    preflight_button: QPushButton
    queue_button: QPushButton
    history_button: QPushButton


def _card(title: str, body: str, on_enter: Callable[[], None]) -> tuple[QGroupBox, QPushButton]:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    label = QLabel(body)
    label.setWordWrap(True)
    label.setObjectName("subtle")
    layout.addWidget(label)
    button = QPushButton("进入")
    button.clicked.connect(on_enter)
    layout.addWidget(button)
    return box, button


def build_advanced_dashboard_tab(
    *,
    on_jump_env: Callable[[], None],
    on_jump_generation: Callable[[], None],
    on_jump_preview: Callable[[], None],
    on_jump_preflight: Callable[[], None],
    on_jump_queue: Callable[[], None],
    on_jump_history: Callable[[], None],
) -> AdvancedDashboardTabWidgets:
    """Build the advanced dashboard tab without owning navigation state."""

    tab = QWidget()
    layout = QVBoxLayout(tab)
    title = QLabel("从这里开始：按 1→5 的顺序即可批量生成、预检、运行并复盘仿真。")
    title.setObjectName("banner")
    layout.addWidget(title)

    grid = QGridLayout()
    card_specs = [
        ("1 环境检查", "确认 conda、gprMax、GPU 和工作目录。\n建议先保存本地设置，再运行环境检查。", on_jump_env),
        ("2 仿真计划", "选择 YAML 计划，设置 case 数量，一键生成 raw/target/clutter 等输入。", on_jump_generation),
        ("3 3D 模型预览", "在正式 FDTD 前查看地形、飞行高度和基覆界面，提前发现异常模型。", on_jump_preview),
        ("4 预检与去重", "自动计算总任务、已完成、将跳过、待运行，防止重复跑同一模型。", on_jump_preflight),
        ("5 批量运行", "本机或 SLURM 安全运行；默认跳过已完成；中断后可恢复。", on_jump_queue),
        ("6 历史记录", "查看所有历史仿真，筛选 done/failed，导出或彻底删除某次仿真。", on_jump_history),
    ]
    buttons: list[QPushButton] = []
    for i, (heading, body, callback) in enumerate(card_specs):
        box, button = _card(heading, body, callback)
        buttons.append(button)
        grid.addWidget(box, i // 3, i % 3)
    layout.addLayout(grid)

    summary = QPlainTextEdit()
    summary.setReadOnly(True)
    summary.setPlainText(
        "自动化原则：\n"
        "- 不重复跑：以 .in 内容 SHA256 + variant + trace 数生成任务指纹。\n"
        "- 可恢复：成功任务写入 jobs/done/*.json；失败任务写入 jobs/failed/*.json。\n"
        "- 可读性：用户只需关注计划、预检、运行、历史；高级参数默认隐藏在模板中。\n"
        "- 安全删除：历史页删除操作会限制在当前 workspace 内，避免误删外部文件。"
    )
    layout.addWidget(summary, 1)

    return AdvancedDashboardTabWidgets(
        page=tab,
        summary=summary,
        env_button=buttons[0],
        generation_button=buttons[1],
        preview_button=buttons[2],
        preflight_button=buttons[3],
        queue_button=buttons[4],
        history_button=buttons[5],
    )
