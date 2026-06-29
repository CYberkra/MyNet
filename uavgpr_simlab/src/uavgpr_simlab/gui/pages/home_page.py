from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from uavgpr_simlab.gui.easy_cards import flow_step, metric_card, page_header
from uavgpr_simlab.gui.easy_ui import _make_title
from uavgpr_simlab.gui.main_window import MplCanvas


@dataclass(frozen=True)
class HomePageWidgets:
    """Widget references owned by the easy GUI home page."""

    page: QWidget
    home_project_card: QFrame
    home_models_card: QFrame
    home_running_card: QFrame
    home_done_card: QFrame
    home_check_card: QFrame
    home_bscan: MplCanvas
    next_step_label: QLabel


def build_home_page(
    *,
    on_go_model_preview: Callable[[], None],
    on_go_batch: Callable[[], None],
) -> HomePageWidgets:
    """Build the easy GUI home/dashboard page.

    This module only owns static UI construction and signal wiring. Runtime
    counters, recent B-scan refresh, history scanning and project summary logic
    remain in the main window/services layer.
    """

    page = QWidget()
    page.setObjectName("page")
    layout = QVBoxLayout(page)
    layout.setSpacing(14)
    layout.addWidget(
        page_header(
            "首页：一眼看懂当前项目",
            "不需要先理解专业参数。先看模型、再跑批量、最后在历史结果里复盘每一次仿真。",
        )
    )

    top = QHBoxLayout()
    top.setSpacing(12)
    home_project_card = metric_card("我现在在做什么", "未选择项目", "当前批次、工作目录和下一步建议。", "📁")
    home_models_card = metric_card("已生成模型", "0 个", "每个模型都有缩略图。", "🖼")
    home_running_card = metric_card("正在仿真", "0 个", "运行中可看到 B-scan 增长。", "▶")
    home_done_card = metric_card("已完成", "0 个", "完成后可导出、重跑、删除。", "✓")
    home_check_card = metric_card("需检查", "0 个", "失败或中断集中处理。", "⚠")
    for widget in [home_project_card, home_models_card, home_running_card, home_done_card, home_check_card]:
        top.addWidget(widget)
    layout.addLayout(top)

    mid = QHBoxLayout()
    mid.setSpacing(14)
    left = QFrame()
    left.setObjectName("heroCard")
    left_layout = QVBoxLayout(left)
    left_layout.setContentsMargins(16, 14, 16, 14)
    left_layout.addWidget(
        _make_title(
            "最近结果预览（B-scan）",
            "跑完或运行中都会在这里给你一个可视化结果，不用翻文件夹。",
        )
    )
    home_bscan = MplCanvas("最近 B-scan")
    left_layout.addWidget(home_bscan, 1)
    mid.addWidget(left, 4)

    tip = QFrame()
    tip.setObjectName("tip")
    tip_layout = QVBoxLayout(tip)
    tip_layout.setContentsMargins(26, 24, 26, 24)
    tip_layout.setSpacing(12)
    title = QLabel("下一步")
    title.setStyleSheet("font-size:24px;font-weight:950;color:#0e3158;")
    tip_layout.addWidget(title)
    next_step_label = QLabel("先看模型预览，确认模型长什么样；再去批量仿真。")
    next_step_label.setObjectName("hintStrong")
    next_step_label.setWordWrap(True)
    tip_layout.addWidget(next_step_label)
    btn_model = QPushButton("去看模型长什么样")
    btn_model.clicked.connect(on_go_model_preview)
    tip_layout.addWidget(btn_model)
    btn_batch = QPushButton("查看仿真跑到哪了")
    btn_batch.setObjectName("light")
    btn_batch.clicked.connect(on_go_batch)
    tip_layout.addWidget(btn_batch)
    tip_layout.addStretch(1)
    mid.addWidget(tip, 2)
    layout.addLayout(mid, 1)

    steps = QHBoxLayout()
    steps.setSpacing(10)
    for icon, title_text, subtitle in [
        ("🧱", "生成模型", "先造一批"),
        ("👀", "看模型", "确认形状"),
        ("▶", "批量仿真", "自动跳过重复"),
        ("📡", "看 B-scan", "实时增长"),
        ("🗂", "历史复盘", "导出/删除/对比"),
    ]:
        steps.addWidget(flow_step(icon, title_text, subtitle))
    layout.addLayout(steps)

    return HomePageWidgets(
        page=page,
        home_project_card=home_project_card,
        home_models_card=home_models_card,
        home_running_card=home_running_card,
        home_done_card=home_done_card,
        home_check_card=home_check_card,
        home_bscan=home_bscan,
        next_step_label=next_step_label,
    )
