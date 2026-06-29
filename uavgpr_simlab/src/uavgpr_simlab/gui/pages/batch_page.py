from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QFrame,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTreeWidget,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from uavgpr_simlab.gui.easy_cards import metric_card, page_header
from uavgpr_simlab.gui.main_window import MplCanvas


@dataclass(frozen=True)
class BatchPageWidgets:
    """Widget references the main window needs after building the batch page."""

    page: QWidget
    batch_profile_combo: QComboBox
    batch_manifest_edit: QLineEdit
    batch_variants_edit: QLineEdit
    batch_limit_spin: QSpinBox
    batch_skip_done: QCheckBox
    batch_failed_only: QCheckBox
    batch_force_rerun: QCheckBox
    batch_dataset_status_label: QLabel
    batch_runtime_summary_label: QLabel
    batch_total_card: QWidget
    batch_pending_card: QWidget
    batch_running_card: QWidget
    batch_done_card: QWidget
    batch_failed_card: QWidget
    batch_queue_tree: QTreeWidget
    batch_failure_box: QPlainTextEdit
    batch_table: QTableWidget
    batch_live_canvas: MplCanvas
    batch_recent_canvas: MplCanvas
    batch_log: QPlainTextEdit


def _make_card_frame(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(6)
    label = QLabel(title)
    label.setStyleSheet("font-weight:900;color:#1f3b57;")
    layout.addWidget(label)
    return card, layout


def build_batch_page(
    *,
    on_choose_manifest: Callable[[QLineEdit], None],
    on_import_skeleton: Callable[[], None],
    on_relocate_workspace: Callable[[], None],
    on_apply_profile: Callable[[], None],
    on_refresh_plan: Callable[[], None],
    on_run_pending: Callable[[], None],
    on_stop: Callable[[], None],
    on_queue_activated: Callable[[object, int], None],
) -> BatchPageWidgets:
    """Build the operator-oriented batch-simulation page.

    The front page deliberately keeps only the controls needed during routine
    use: import a skeleton, fix paths, precheck, start and stop.  Runtime
    profiles, manifest paths, variant tags, queues, logs and failure details are
    still available, but are hidden by default under the details panel so the
    page stays focused for a single expert operator.
    """

    page = QWidget()
    page.setObjectName("page")
    layout = QVBoxLayout(page)
    layout.setSpacing(12)
    layout.addWidget(
        page_header(
            "批量仿真：导入骨架，一键运行，实时看结果",
            "日常使用只保留核心动作；环境、队列、日志和失败明细放在“运行细节/高级诊断”中，需要排查时再展开。",
        )
    )

    batch_manifest_edit = QLineEdit("")
    batch_variants_edit = QLineEdit("raw,target_only,clutter_only,background_only,air_only")
    batch_limit_spin = QSpinBox()
    batch_limit_spin.setRange(0, 999999)
    batch_limit_spin.setValue(60)
    batch_skip_done = QCheckBox("自动跳过已完成")
    batch_skip_done.setChecked(True)
    batch_failed_only = QCheckBox("只重跑 failed")
    batch_failed_only.setChecked(False)
    batch_force_rerun = QCheckBox("强制重跑")
    batch_force_rerun.setChecked(False)
    batch_profile_combo = QComboBox()

    action_bar = QHBoxLayout()
    action_bar.setSpacing(10)
    import_skeleton = QPushButton("导入数据集骨架")
    import_skeleton.setObjectName("green")
    import_skeleton.clicked.connect(on_import_skeleton)
    relocate_workspace = QPushButton("迁移/修复路径")
    relocate_workspace.setObjectName("light")
    relocate_workspace.clicked.connect(on_relocate_workspace)
    refresh = QPushButton("预检")
    refresh.clicked.connect(on_refresh_plan)
    run = QPushButton("一键开始")
    run.setObjectName("green")
    run.clicked.connect(on_run_pending)
    stop = QPushButton("停止")
    stop.setObjectName("red")
    stop.clicked.connect(on_stop)
    advanced_toggle = QPushButton("显示运行细节/高级诊断")
    advanced_toggle.setCheckable(True)
    advanced_toggle.setObjectName("light")
    for widget in [import_skeleton, relocate_workspace, refresh, run, stop, advanced_toggle]:
        action_bar.addWidget(widget)
    action_bar.addStretch(1)
    layout.addLayout(action_bar)

    batch_dataset_status_label = QLabel("尚未导入数据集骨架。导入后这里会显示：已跑、正在跑、即将跑、失败待处理。")
    batch_dataset_status_label.setObjectName("pageHint")
    batch_dataset_status_label.setWordWrap(True)
    batch_dataset_status_label.setStyleSheet("background:#eef6ff;border:1px solid #c6daf1;border-radius:14px;padding:10px;color:#284b6b;font-weight:800;")
    layout.addWidget(batch_dataset_status_label)

    cards = QHBoxLayout()
    cards.setSpacing(12)
    batch_total_card = metric_card("本批次任务数", "0 个", "所有要检查的任务", "🧱")
    batch_pending_card = metric_card("即将运行", "0 个", "下一批要跑", "○")
    batch_running_card = metric_card("正在运行", "0 个", "实时刷新", "●")
    batch_done_card = metric_card("历史完成", "0 个", "可直接看结果", "✓")
    batch_failed_card = metric_card("失败待处理", "0 个", "可定位/重跑", "!")
    for widget in [batch_total_card, batch_pending_card, batch_running_card, batch_done_card, batch_failed_card]:
        cards.addWidget(widget)
    layout.addLayout(cards)

    batch_table = QTableWidget(0, 8)
    batch_table.setHorizontalHeaderLabels(["模型缩略图", "模型名称", "状态", "进度", "已完成道数", "预计剩余时间", "仿真内容", "操作提示"])
    batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    batch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    batch_table.verticalHeader().setDefaultSectionSize(92)
    layout.addWidget(batch_table, 1)

    preview_stack = QWidget()
    preview_layout = QVBoxLayout(preview_stack)
    preview_layout.setContentsMargins(0, 0, 0, 0)
    preview_layout.setSpacing(8)
    batch_live_canvas = MplCanvas("运行中 B-scan 实时预览")
    batch_recent_canvas = MplCanvas("最近完成的 B-scan（最多 5 个）")
    preview_layout.addWidget(batch_live_canvas, 2)
    preview_layout.addWidget(batch_recent_canvas, 1)
    layout.addWidget(preview_stack, 2)

    advanced_panel = QFrame()
    advanced_panel.setObjectName("card")
    advanced_panel.setVisible(False)
    advanced_layout = QVBoxLayout(advanced_panel)
    advanced_layout.setContentsMargins(12, 12, 12, 12)
    advanced_layout.setSpacing(10)

    profile_bar = QHBoxLayout()
    apply_profile = QPushButton("应用运行配置")
    apply_profile.setObjectName("light")
    apply_profile.clicked.connect(on_apply_profile)
    profile_bar.addWidget(QLabel("运行配置"))
    profile_bar.addWidget(batch_profile_combo, 1)
    profile_bar.addWidget(apply_profile)
    profile_bar.addStretch(1)
    advanced_layout.addLayout(profile_bar)

    manifest_bar = QHBoxLayout()
    pick = QPushButton("选择模型清单")
    pick.setObjectName("light")
    pick.clicked.connect(lambda: on_choose_manifest(batch_manifest_edit))
    for widget in [QLabel("清单"), batch_manifest_edit, pick, QLabel("标签"), batch_variants_edit, QLabel("最多"), batch_limit_spin, batch_skip_done, batch_failed_only, batch_force_rerun]:
        manifest_bar.addWidget(widget)
    advanced_layout.addLayout(manifest_bar)

    batch_runtime_summary_label = QLabel("当前运行环境：未检测。导入骨架或预检后会显示 profile / GPU / Python / gprMax。")
    batch_runtime_summary_label.setObjectName("pageHint")
    batch_runtime_summary_label.setWordWrap(True)
    batch_runtime_summary_label.setStyleSheet("background:#f6fbf4;border:1px solid #cfe7c9;border-radius:14px;padding:8px;color:#365c34;font-weight:800;")
    advanced_layout.addWidget(batch_runtime_summary_label)

    monitor = QSplitter(Qt.Horizontal)
    monitor.setMaximumHeight(220)
    queue_card, queue_layout = _make_card_frame("运行队列：双击 case / variant 可跳转历史")
    batch_queue_tree = QTreeWidget()
    batch_queue_tree.setHeaderLabels(["Case / Variant", "状态", "内容", "提示"])
    batch_queue_tree.itemDoubleClicked.connect(on_queue_activated)
    queue_layout.addWidget(batch_queue_tree)

    failure_card, failure_layout = _make_card_frame("失败原因聚合：failed-only 重跑前先看这里")
    batch_failure_box = QPlainTextEdit()
    batch_failure_box.setReadOnly(True)
    batch_failure_box.setPlaceholderText("导入或预检数据集后，这里会按失败原因聚合显示 case / variant，便于定位环境错误、模型错误或后处理错误。")
    failure_layout.addWidget(batch_failure_box)
    monitor.addWidget(queue_card)
    monitor.addWidget(failure_card)
    monitor.setStretchFactor(0, 3)
    monitor.setStretchFactor(1, 2)
    advanced_layout.addWidget(monitor)

    batch_log = QPlainTextEdit()
    batch_log.setReadOnly(True)
    batch_log.setMaximumHeight(170)
    batch_log.setPlaceholderText("统一任务日志会显示在这里；每完成一个 variant 会刷新实时 B-scan，并写回 QC / dataset_summary / history。")
    advanced_layout.addWidget(batch_log)
    layout.addWidget(advanced_panel)

    def _toggle_advanced(checked: bool) -> None:
        advanced_panel.setVisible(checked)
        advanced_toggle.setText("隐藏运行细节/高级诊断" if checked else "显示运行细节/高级诊断")

    advanced_toggle.toggled.connect(_toggle_advanced)

    return BatchPageWidgets(
        page=page,
        batch_profile_combo=batch_profile_combo,
        batch_manifest_edit=batch_manifest_edit,
        batch_variants_edit=batch_variants_edit,
        batch_limit_spin=batch_limit_spin,
        batch_skip_done=batch_skip_done,
        batch_failed_only=batch_failed_only,
        batch_force_rerun=batch_force_rerun,
        batch_dataset_status_label=batch_dataset_status_label,
        batch_runtime_summary_label=batch_runtime_summary_label,
        batch_total_card=batch_total_card,
        batch_pending_card=batch_pending_card,
        batch_running_card=batch_running_card,
        batch_done_card=batch_done_card,
        batch_failed_card=batch_failed_card,
        batch_queue_tree=batch_queue_tree,
        batch_failure_box=batch_failure_box,
        batch_table=batch_table,
        batch_live_canvas=batch_live_canvas,
        batch_recent_canvas=batch_recent_canvas,
        batch_log=batch_log,
    )
