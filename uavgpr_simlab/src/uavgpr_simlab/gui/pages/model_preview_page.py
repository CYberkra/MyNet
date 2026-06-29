from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from uavgpr_simlab.gui.easy_cards import model_info_row, page_header
from uavgpr_simlab.gui.easy_ui import _make_title
from uavgpr_simlab.gui.main_window import Model3DCanvas


@dataclass(frozen=True)
class ModelPreviewPageWidgets:
    """Widget references the main window needs after building the model page."""

    page: QWidget
    model_manifest_edit: QLineEdit
    model_list: QListWidget
    easy_model_canvas: Model3DCanvas
    model_preview_stack: QStackedWidget
    model_2d_preview: QLabel
    view_3d_button: QPushButton
    view_2d_button: QPushButton
    model_info_labels: dict[str, QLabel]


def build_model_preview_page(
    *,
    on_choose_manifest: Callable[[QLineEdit], None],
    on_load_manifest: Callable[[], None],
    on_generate_models: Callable[[], None],
    on_preview_selection: Callable[[], None],
    on_open_model_png: Callable[[], None],
    on_set_preview_mode: Callable[[str], None],
    on_sync_to_batch: Callable[[], None],
) -> ModelPreviewPageWidgets:
    """Build the model-preview page for the easy GUI.

    This builder owns only widget construction and signal wiring. Manifest
    loading, preview image rendering, 3D/2D view switching and batch synchronization
    remain in the main window and service layer so the page stays presentation
    oriented.
    """

    page = QWidget()
    page.setObjectName("page")
    layout = QVBoxLayout(page)
    layout.setSpacing(12)
    layout.addWidget(
        page_header(
            "模型预览：看懂自己做的模型长什么样",
            "左边是模型图库；中间看地形、覆盖层、基岩面和飞行高度；右边用普通话解释模型特点。",
        )
    )

    bar = QHBoxLayout()
    model_manifest_edit = QLineEdit("")
    pick = QPushButton("选择已有模型清单")
    pick.setObjectName("light")
    pick.clicked.connect(lambda: on_choose_manifest(model_manifest_edit))
    load = QPushButton("加载模型图库")
    load.clicked.connect(on_load_manifest)
    generate = QPushButton("生成一批模型")
    generate.setObjectName("green")
    generate.clicked.connect(on_generate_models)
    for widget in [QLabel("模型清单"), model_manifest_edit, pick, load, generate]:
        bar.addWidget(widget)
    layout.addLayout(bar)

    splitter = QSplitter(Qt.Horizontal)

    left = QFrame()
    left.setObjectName("card")
    left_layout = QVBoxLayout(left)
    left_layout.setContentsMargins(12, 12, 12, 12)
    left_layout.addWidget(_make_title("本批次模型", "像图库一样点一下就看大图。"))
    model_list = QListWidget()
    model_list.setIconSize(QSize(170, 96))
    model_list.setMinimumWidth(285)
    model_list.itemSelectionChanged.connect(on_preview_selection)
    left_layout.addWidget(model_list, 1)

    center = QFrame()
    center.setObjectName("heroCard")
    center_layout = QVBoxLayout(center)
    center_layout.setContentsMargins(14, 14, 14, 14)
    mode_bar = QHBoxLayout()
    mode_bar.setSpacing(8)
    view_3d_button = QPushButton("3D 视图")
    view_3d_button.setObjectName("green")
    view_3d_button.setCheckable(True)
    view_3d_button.setChecked(True)
    view_2d_button = QPushButton("2D 剖面")
    view_2d_button.setObjectName("light")
    view_2d_button.setCheckable(True)
    view_group = QButtonGroup(center)
    view_group.setExclusive(True)
    view_group.addButton(view_3d_button)
    view_group.addButton(view_2d_button)
    view_3d_button.clicked.connect(lambda: on_set_preview_mode("3d"))
    view_2d_button.clicked.connect(lambda: on_set_preview_mode("2d"))
    mode_bar.addWidget(QLabel("预览模式"))
    mode_bar.addWidget(view_3d_button)
    mode_bar.addWidget(view_2d_button)
    mode_bar.addStretch(1)
    center_layout.addLayout(mode_bar)

    easy_model_canvas = Model3DCanvas()
    model_2d_preview = QLabel("请选择模型后查看 2D 剖面")
    model_2d_preview.setAlignment(Qt.AlignCenter)
    model_2d_preview.setMinimumSize(QSize(520, 360))
    model_2d_preview.setStyleSheet("background:#f1f6fc;border:1px solid #d4e1ef;border-radius:14px;color:#61758d;font-weight:700;")
    model_preview_stack = QStackedWidget()
    model_preview_stack.addWidget(easy_model_canvas)
    model_preview_stack.addWidget(model_2d_preview)
    center_layout.addWidget(model_preview_stack, 1)
    center_hint = QLabel("点击上方 3D / 2D 切换视图；正式跑 gprMax 前就能检查地形和基岩面是否合理。")
    center_hint.setObjectName("pageHint")
    center_layout.addWidget(center_hint)

    right = QFrame()
    right.setObjectName("infoCard")
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(16, 16, 16, 16)
    right_layout.setSpacing(10)
    right_layout.addWidget(_make_title("模型信息", "不用懂参数，也能判断这个模型难不难。"))

    model_info_labels: dict[str, QLabel] = {}
    for key, icon in [("地表起伏", "〰"), ("覆盖层厚度", "▰"), ("是否有障碍物", "🌳"), ("模型难度", "📊"), ("推荐用途", "🎯")]:
        label = model_info_row(key, icon)
        right_layout.addWidget(label)
        model_info_labels[key] = label

    show_3d = QPushButton("切换到 3D 视图")
    show_3d.clicked.connect(lambda: on_set_preview_mode("3d"))
    right_layout.addWidget(show_3d)
    show_png = QPushButton("打开 2D 剖面图路径")
    show_png.setObjectName("light")
    show_png.clicked.connect(on_open_model_png)
    right_layout.addWidget(show_png)
    sync = QPushButton("加入批量仿真")
    sync.setObjectName("green")
    sync.clicked.connect(on_sync_to_batch)
    right_layout.addWidget(sync)
    right_layout.addStretch(1)

    splitter.addWidget(left)
    splitter.addWidget(center)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 2)
    splitter.setStretchFactor(1, 6)
    splitter.setStretchFactor(2, 2)
    layout.addWidget(splitter, 1)

    return ModelPreviewPageWidgets(
        page=page,
        model_manifest_edit=model_manifest_edit,
        model_list=model_list,
        easy_model_canvas=easy_model_canvas,
        model_preview_stack=model_preview_stack,
        model_2d_preview=model_2d_preview,
        view_3d_button=view_3d_button,
        view_2d_button=view_2d_button,
        model_info_labels=model_info_labels,
    )
