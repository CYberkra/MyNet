from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QGridLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QTableWidget, QHeaderView, QVBoxLayout, QWidget


@dataclass(frozen=True)
class AdvancedPreflightTabWidgets:
    """Widget references needed by the advanced preflight tab owner."""

    page: QWidget
    manifest_edit: QLineEdit
    variants_edit: QLineEdit
    skip_check: QCheckBox
    limit_spin: QSpinBox
    summary: QLabel
    table: QTableWidget
    pick_button: QPushButton
    plan_button: QPushButton
    sync_button: QPushButton


def build_advanced_preflight_tab(
    *,
    choose_manifest: Callable[[QLineEdit, str], None],
    on_run_preflight: Callable[[], None],
    on_sync_to_queue: Callable[[], None],
) -> AdvancedPreflightTabWidgets:
    """Build the advanced preflight/dedup tab without computing jobs."""

    tab = QWidget()
    layout = QVBoxLayout(tab)
    top = QGridLayout()

    manifest_edit = QLineEdit("")
    pick_button = QPushButton("选择 manifest")
    pick_button.clicked.connect(lambda _checked=False: choose_manifest(manifest_edit, "CSV (*.csv);;All Files (*)"))
    variants_edit = QLineEdit("raw,target_only,clutter_only,background_only,air_only")
    skip_check = QCheckBox("跳过已完成")
    skip_check.setChecked(True)
    limit_spin = QSpinBox()
    limit_spin.setRange(0, 1000000)
    limit_spin.setValue(0)
    plan_button = QPushButton("生成预检任务表")
    plan_button.clicked.connect(on_run_preflight)
    sync_button = QPushButton("同步到批量运行页")
    sync_button.clicked.connect(on_sync_to_queue)

    top.addWidget(QLabel("manifest"), 0, 0)
    top.addWidget(manifest_edit, 0, 1, 1, 5)
    top.addWidget(pick_button, 0, 6)
    top.addWidget(QLabel("variants"), 1, 0)
    top.addWidget(variants_edit, 1, 1, 1, 3)
    top.addWidget(QLabel("最大任务数(0=全部)"), 1, 4)
    top.addWidget(limit_spin, 1, 5)
    top.addWidget(skip_check, 1, 6)
    top.addWidget(plan_button, 2, 5)
    top.addWidget(sync_button, 2, 6)
    layout.addLayout(top)

    summary = QLabel("尚未预检。")
    summary.setObjectName("banner")
    layout.addWidget(summary)
    table = QTableWidget(0, 8)
    table.setHorizontalHeaderLabels(["状态", "case_id", "variant", "n", "job_id", "原因", "input", "fingerprint"])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    layout.addWidget(table, 1)

    return AdvancedPreflightTabWidgets(
        page=tab,
        manifest_edit=manifest_edit,
        variants_edit=variants_edit,
        skip_check=skip_check,
        limit_spin=limit_spin,
        summary=summary,
        table=table,
        pick_button=pick_button,
        plan_button=plan_button,
        sync_button=sync_button,
    )
