from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


@dataclass(frozen=True)
class AdvancedTrainTabWidgets:
    """Widget references for the advanced PGDA/ML tab."""

    page: QWidget
    text: QPlainTextEdit


def build_advanced_train_tab() -> AdvancedTrainTabWidgets:
    """Build the reserved PGDA-CSNet training guidance tab."""

    tab = QWidget()
    layout = QVBoxLayout(tab)
    text = QPlainTextEdit()
    text.setReadOnly(True)
    text.setPlainText(
        "PGDA-CSNet 训练入口已预留。当前 v0.7 仍以仿真数据生产、实时预览和 ML 导出稳定性为优先。\n\n"
        "推荐顺序：\n"
        "1) 3060：run_plan_3060_quick.yaml -> 1-3 个 geometry-only；\n"
        "2) 3060：少量 full raw，检查 .out、B-scan 和导出 NPZ；\n"
        "3) 4090：run_plan_4090_validation_hifi.yaml；\n"
        "4) 4090：run_plan_4090_formal.yaml 分批生成正式训练集。\n\n"
        "训练配置模板：configs/ml_pgda_csnet.yaml。"
    )
    layout.addWidget(text)
    return AdvancedTrainTabWidgets(page=tab, text=text)
