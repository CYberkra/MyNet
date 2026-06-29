from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from uavgpr_simlab.gui.easy_ui import _friendly_variant_name, _pix_label, _status_chip


def page_header(title: str, hint: str) -> QWidget:
    """Build a consistent page title block for the simplified GUI."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 4)
    title_label = QLabel(title)
    title_label.setObjectName("pageTitle")
    layout.addWidget(title_label)
    hint_label = QLabel(hint)
    hint_label.setObjectName("pageHint")
    hint_label.setWordWrap(True)
    layout.addWidget(hint_label)
    return widget


def metric_card(title: str, value: str, subtitle: str = "", icon: str = "") -> QFrame:
    """Build a compact dashboard metric card."""
    frame = QFrame()
    frame.setObjectName("card")
    frame.setMinimumHeight(128)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(5)

    title_label = QLabel(f"{icon}  {title}" if icon else title)
    title_label.setObjectName("cardTitle")
    title_label.setWordWrap(True)
    layout.addWidget(title_label)

    value_label = QLabel(value)
    value_label.setObjectName("bigNumber")
    value_label.setWordWrap(True)
    layout.addWidget(value_label)

    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("metricSub")
    subtitle_label.setWordWrap(True)
    layout.addWidget(subtitle_label)
    return frame


def set_metric_value(card: QFrame, value: str) -> None:
    """Update the main value label inside a metric card."""
    label = card.findChild(QLabel, "bigNumber")
    if label is not None:
        label.setText(value)


def flow_step(icon: str, title: str, subtitle: str = "") -> QFrame:
    """Build one step in the visual workflow strip."""
    frame = QFrame()
    frame.setObjectName("flowStep")
    frame.setMinimumHeight(86)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(2)

    icon_label = QLabel(icon)
    icon_label.setAlignment(Qt.AlignCenter)
    icon_label.setStyleSheet("font-size:22px;")
    layout.addWidget(icon_label)

    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet("font-size:13px;font-weight:900;color:#123a62;")
    layout.addWidget(title_label)

    if subtitle:
        subtitle_label = QLabel(subtitle)
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setObjectName("metricSub")
        layout.addWidget(subtitle_label)
    return frame


def model_info_row(key: str, icon: str) -> QLabel:
    """Build a consistent right-panel model information row."""
    label = QLabel(f"{icon}  {key}：—")
    label.setStyleSheet(
        "font-size:15px;font-weight:800;color:#213a54;"
        "padding:8px;background:#f7fbff;border-radius:10px;"
    )
    label.setWordWrap(True)
    return label


def mode_tab(text: str) -> QLabel:
    """Build the small chip tabs used above the history detail preview."""
    label = QLabel(text)
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet(
        "background:#eaf4ff;color:#104d83;border:1px solid #c8def7;"
        "border-radius:12px;padding:7px 12px;font-weight:900;"
    )
    return label


def help_step(text: str) -> QLabel:
    """Build one readable step in the help page."""
    label = QLabel(text)
    label.setStyleSheet("font-size:15px;font-weight:750;color:#263f59;padding:5px;")
    label.setWordWrap(True)
    return label


def history_record_card(entry: Any) -> QFrame:
    """Build a list-widget card for one history entry.

    The function intentionally accepts a loose object because the history service
    owns the concrete dataclass. GUI code only needs the public record/preview
    attributes returned by that service.
    """
    record = entry.record
    preview = entry.preview

    card = QFrame()
    card.setObjectName("historyCard")
    layout = QHBoxLayout(card)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)

    layout.addWidget(_pix_label(preview.model_preview_png or "", QSize(100, 64), "模型图"))
    layout.addWidget(_pix_label(preview.bscan_preview_png or "", QSize(100, 64), "B-scan"))

    info = QVBoxLayout()
    name = QLabel(f"{record.case_id}")
    name.setStyleSheet("font-size:14px;font-weight:950;color:#123a62;")
    info.addWidget(name)
    info.addWidget(_status_chip(record.status))

    desc = QLabel(
        f"{_friendly_variant_name(record.variant)} · "
        f"{preview.available_traces}/{record.n_traces} 道\n"
        f"{record.time_local or '无时间'}"
    )
    desc.setStyleSheet("color:#5c7188;font-weight:700;")
    desc.setWordWrap(True)
    info.addWidget(desc)

    layout.addLayout(info, 1)
    return card
