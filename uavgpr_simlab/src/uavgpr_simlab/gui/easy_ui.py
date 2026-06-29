from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

EASY_STYLE = """
QMainWindow, QWidget {
    background: #f5f8fc;
    color: #142033;
    font-family: 'Microsoft YaHei UI', 'Noto Sans CJK SC', 'Arial';
    font-size: 12px;
}
QFrame#sidebar {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0b2d55, stop:1 #114878);
    border-radius: 22px;
}
QLabel#logo { color: white; font-size: 21px; font-weight: 900; padding: 18px 14px 8px 14px; }
QLabel#sideCaption { color: #bdd7f2; font-size: 12px; font-weight: 650; padding: 0 14px 14px 14px; }
QPushButton#nav { background: transparent; color: #e6f2ff; border: none; border-radius: 14px; padding: 13px 14px; text-align: left; font-size: 15px; font-weight: 850; }
QPushButton#nav:hover { background: rgba(255,255,255,0.10); }
QPushButton#nav:checked { background: #1f7ed0; color: white; border-left: 4px solid #9ed7ff; }
QFrame#page { background: transparent; }
QLabel#pageTitle { color: #0d3159; font-size: 26px; font-weight: 900; }
QLabel#pageHint { color: #60738b; font-size: 13px; font-weight: 650; }
QLabel#sectionTitle { color: #102f52; font-size: 17px; font-weight: 900; }
QLabel#cardTitle { color: #2a4059; font-size: 13px; font-weight: 850; }
QLabel#bigNumber { color: #0c3b6d; font-size: 34px; font-weight: 950; }
QLabel#metricSub { color: #66788d; font-size: 12px; font-weight: 600; }
QLabel#hintStrong { color: #104d83; font-size: 20px; font-weight: 900; }
QFrame#card, QFrame#heroCard, QFrame#taskCard, QFrame#historyCard, QFrame#infoCard {
    background: white;
    border: 1px solid #dbe6f2;
    border-radius: 20px;
}
QFrame#heroCard { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ffffff, stop:1 #eaf5ff); }
QFrame#tip { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #eaf5ff, stop:1 #f3fbff); border: 1px solid #bad8f6; border-radius: 20px; }
QFrame#flowStep { background: #ffffff; border: 1px solid #d8e5f4; border-radius: 18px; }
QLabel#statusDone { color: #138044; background: #e9f8ef; border: 1px solid #b9e7c8; border-radius: 12px; padding: 4px 8px; font-weight: 900; }
QLabel#statusRun { color: #0e5fa8; background: #e9f3ff; border: 1px solid #b6d9ff; border-radius: 12px; padding: 4px 8px; font-weight: 900; }
QLabel#statusWait { color: #5b6470; background: #f0f3f7; border: 1px solid #d4dce6; border-radius: 12px; padding: 4px 8px; font-weight: 900; }
QLabel#statusBad { color: #b62727; background: #fff0f0; border: 1px solid #f4b7b7; border-radius: 12px; padding: 4px 8px; font-weight: 900; }
QGroupBox { background: white; border: 1px solid #dbe6f2; border-radius: 20px; margin-top: 14px; padding: 14px; font-size: 15px; font-weight: 900; color: #0e3158; }
QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 8px; }
QLineEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QTableWidget { background: white; border: 1px solid #c8d6e8; border-radius: 12px; padding: 7px; }
QListWidget::item { border-radius: 12px; margin: 5px; padding: 8px; }
QListWidget::item:selected { background: #e4f2ff; color: #142033; border: 1px solid #66aee8; }
QTableWidget { gridline-color: #e7eef7; selection-background-color: #e8f3ff; selection-color: #142033; }
QHeaderView::section { background: #eef5fc; color: #27445f; border: none; padding: 10px; font-weight: 900; }
QPushButton { background: #176db3; color: white; border: none; border-radius: 12px; padding: 10px 17px; font-size: 13px; font-weight: 900; }
QPushButton:hover { background: #0d5e9e; }
QPushButton:disabled { background: #a7b8ca; }
QPushButton#green { background: #24a45a; }
QPushButton#green:hover { background: #168c48; }
QPushButton#red { background: #db4545; }
QPushButton#red:hover { background: #b93030; }
QPushButton#light { background: #eef6ff; color: #17588e; border: 1px solid #c6daf1; }
QPushButton#ghost { background: white; color: #17588e; border: 1px solid #c6daf1; }
QProgressBar { border: 1px solid #cad8e8; border-radius: 9px; background: #edf2f8; height: 20px; text-align: center; font-weight: 900; }
QProgressBar::chunk { background: #1e80d0; border-radius: 9px; }
"""



def _status_text(status: str) -> str:
    mapping = {
        "pending": "等待中",
        "skipped": "自动跳过",
        "done": "已完成",
        "running": "运行中",
        "stale_running": "中断待检查",
        "failed": "失败待检查",
    }
    return mapping.get(status, status or "未知")


def _status_icon(status: str) -> str:
    if status in {"done", "skipped"}:
        return "✓"
    if status in {"running"}:
        return "●"
    if status in {"failed", "stale_running"}:
        return "!"
    return "○"


def _set_item(table: QTableWidget, row: int, col: int, text: str, align: Qt.AlignmentFlag = Qt.AlignCenter) -> None:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    table.setItem(row, col, item)


def _pix_label(path: str | Path, size: QSize = QSize(120, 70), fallback: str = "暂无图") -> QLabel:
    label = QLabel()
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumSize(size)
    label.setMaximumSize(size)
    p = Path(path) if path else Path()
    if p.exists():
        pix = QPixmap(str(p))
        if not pix.isNull():
            label.setPixmap(pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return label
    label.setText(fallback)
    label.setStyleSheet("background:#eef4fb;border:1px solid #d1dde9;border-radius:8px;color:#61758d;")
    return label


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v)))
    except Exception:
        return default



def _status_style_name(status: str) -> str:
    if status in {"done", "skipped"}:
        return "statusDone"
    if status == "running":
        return "statusRun"
    if status in {"failed", "stale_running"}:
        return "statusBad"
    return "statusWait"


def _status_chip(status: str) -> QLabel:
    lab = QLabel(f"{_status_icon(status)} {_status_text(status)}")
    lab.setObjectName(_status_style_name(status))
    lab.setAlignment(Qt.AlignCenter)
    return lab


def _make_title(text: str, subtitle: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    t = QLabel(text)
    t.setObjectName("sectionTitle")
    lay.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("pageHint")
        s.setWordWrap(True)
        lay.addWidget(s)
    return w


def _friendly_variant_name(v: str) -> str:
    mapping = {
        "raw": "完整模型",
        "target_only": "地下有效反射",
        "clutter_only": "杂波模型",
        "background_only": "背景模型",
        "air_only": "空中干扰",
        "clutter_gt": "杂波标签",
    }
    return mapping.get(v, v or "默认模型")
