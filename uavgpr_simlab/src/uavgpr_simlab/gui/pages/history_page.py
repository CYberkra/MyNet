from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from uavgpr_simlab.gui.easy_cards import mode_tab, page_header
from uavgpr_simlab.gui.easy_ui import _make_title
from uavgpr_simlab.gui.main_window import MplCanvas, Model3DCanvas


@dataclass(frozen=True)
class HistoryPageWidgets:
    """Widget references the main window needs after building the history page."""

    page: QWidget
    history_status_filter: QComboBox
    history_list: QListWidget
    history_tree: QTreeWidget
    history_bscan_mode: QComboBox
    history_failure_box: QPlainTextEdit
    history_model_canvas_easy: Model3DCanvas
    history_bscan_canvas_easy: MplCanvas
    history_detail_easy: QLabel


def build_history_page(
    *,
    status_filters: Sequence[str],
    on_refresh: Callable[[], None],
    on_export: Callable[[], None],
    on_rerun: Callable[[], None],
    on_delete: Callable[[bool], None],
    on_preview_selection: Callable[[], None],
    on_show_failure: Callable[[], None],
    on_rerun_failed: Callable[[], None],
    on_context_menu: Callable[[object], None],
) -> HistoryPageWidgets:
    """Build the history/results page for the easy GUI.

    This builder owns only widget construction and signal wiring. History
    scanning, preview detail construction, B-scan loading, CSV export and
    deletion remain in services/easy_history_service.py plus the main window's
    event handlers.
    """

    page = QWidget()
    page.setObjectName("page")
    layout = QVBoxLayout(page)
    layout.setSpacing(12)
    layout.addWidget(
        page_header(
            "历史与结果：直观看每次仿真效果",
            "左边每条记录像档案卡一样显示模型缩略图和 B-scan 小图；右边显示大图、完成道数和操作。",
        )
    )

    bar = QHBoxLayout()
    history_status_filter = QComboBox()
    history_status_filter.addItems(list(status_filters))
    history_status_filter.currentIndexChanged.connect(on_refresh)
    refresh = QPushButton("刷新")
    refresh.clicked.connect(on_refresh)
    export = QPushButton("导出历史 CSV")
    export.setObjectName("light")
    export.clicked.connect(on_export)
    rerun = QPushButton("重新运行")
    rerun.clicked.connect(on_rerun)
    delete = QPushButton("删除本次记录")
    delete.setObjectName("red")
    delete.clicked.connect(lambda: on_delete(True))
    for widget in [QLabel("筛选"), history_status_filter, refresh, export, rerun, delete]:
        bar.addWidget(widget)
    bar.addStretch(1)
    layout.addLayout(bar)

    splitter = QSplitter(Qt.Horizontal)

    left = QFrame()
    left.setObjectName("card")
    left_layout = QVBoxLayout(left)
    left_layout.setContentsMargins(12, 12, 12, 12)
    left_layout.addWidget(_make_title("数据集 / Case / Variant", "按数据集树状查看 SceneWorld 结果；下方保留卡片列表。"))
    history_tree = QTreeWidget()
    history_tree.setHeaderLabels(["对象", "状态", "Family / Variant", "说明"])
    history_tree.itemSelectionChanged.connect(on_preview_selection)
    history_tree.setContextMenuPolicy(Qt.CustomContextMenu)

    def _tree_context_menu(pos: object) -> None:
        item = history_tree.itemAt(pos)
        if item is not None:
            history_tree.setCurrentItem(item)
        on_context_menu(history_tree.mapToGlobal(pos))

    history_tree.customContextMenuRequested.connect(_tree_context_menu)
    left_layout.addWidget(history_tree, 2)
    left_layout.addWidget(_make_title("历史卡片", "兼容旧任务记录。"))
    history_list = QListWidget()
    history_list.setIconSize(QSize(96, 56))
    history_list.setSpacing(8)
    history_list.itemSelectionChanged.connect(on_preview_selection)
    history_list.setContextMenuPolicy(Qt.CustomContextMenu)

    def _list_context_menu(pos: object) -> None:
        item = history_list.itemAt(pos)
        if item is not None:
            history_list.setCurrentItem(item)
        on_context_menu(history_list.mapToGlobal(pos))

    history_list.customContextMenuRequested.connect(_list_context_menu)
    left_layout.addWidget(history_list, 1)

    right = QFrame()
    right.setObjectName("heroCard")
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(14, 14, 14, 14)
    right_layout.setSpacing(10)

    top_tabs = QHBoxLayout()
    for title in ["模型样子", "仿真过程", "结果图像"]:
        top_tabs.addWidget(mode_tab(title))
    top_tabs.addStretch(1)
    right_layout.addLayout(top_tabs)

    action_bar = QHBoxLayout()
    history_bscan_mode = QComboBox()
    history_bscan_mode.addItems(["当前记录", "raw", "target_only", "background_only", "clutter_only", "air_only", "clutter_gt", "对比：raw / target / clutter_gt"])
    history_bscan_mode.currentIndexChanged.connect(on_preview_selection)
    locate = QPushButton("失败定位 / 打开路径")
    locate.setObjectName("light")
    locate.clicked.connect(on_show_failure)
    rerun_failed = QPushButton("只重跑 failed")
    rerun_failed.clicked.connect(on_rerun_failed)
    action_bar.addWidget(QLabel("B-scan 视图"))
    action_bar.addWidget(history_bscan_mode, 1)
    action_bar.addWidget(locate)
    action_bar.addWidget(rerun_failed)
    right_layout.addLayout(action_bar)

    history_model_canvas_easy = Model3DCanvas()
    right_layout.addWidget(history_model_canvas_easy, 1)
    history_bscan_canvas_easy = MplCanvas("B-scan 结果（实时/完成）")
    right_layout.addWidget(history_bscan_canvas_easy, 1)
    history_failure_box = QPlainTextEdit()
    history_failure_box.setReadOnly(True)
    history_failure_box.setMaximumHeight(110)
    history_failure_box.setPlaceholderText("失败定位、QC 路径、输入文件和输出文件会显示在这里。")
    right_layout.addWidget(history_failure_box)

    history_detail_easy = QLabel("选择左侧某次仿真后，这里显示状态、完成道数和路径。")
    history_detail_easy.setWordWrap(True)
    history_detail_easy.setStyleSheet(
        "background:#fff;border:1px solid #d7e2f0;border-radius:14px;"
        "padding:12px;color:#213a54;font-weight:800;"
    )
    right_layout.addWidget(history_detail_easy)

    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 6)
    layout.addWidget(splitter, 1)

    return HistoryPageWidgets(
        page=page,
        history_status_filter=history_status_filter,
        history_list=history_list,
        history_tree=history_tree,
        history_bscan_mode=history_bscan_mode,
        history_failure_box=history_failure_box,
        history_model_canvas_easy=history_model_canvas_easy,
        history_bscan_canvas_easy=history_bscan_canvas_easy,
        history_detail_easy=history_detail_easy,
    )
