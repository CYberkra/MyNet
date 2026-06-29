from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QMessageBox

from uavgpr_simlab.gui.pages.home_page import build_home_page
from uavgpr_simlab.gui.pages.project_page import build_project_page
from uavgpr_simlab.gui.easy_cards import set_metric_value
from uavgpr_simlab.services.project_service import preview_plan_yaml


class HomeProjectControllerMixin:
    # --- home ---
    def _build_home_page(self) -> None:
        widgets = build_home_page(
            on_go_model_preview=lambda: self.show_page(1),
            on_go_batch=lambda: self.show_page(2),
        )
        self.home_project_card = widgets.home_project_card
        self.home_models_card = widgets.home_models_card
        self.home_running_card = widgets.home_running_card
        self.home_done_card = widgets.home_done_card
        self.home_check_card = widgets.home_check_card
        self.home_bscan = widgets.home_bscan
        self.next_step_label = widgets.next_step_label
        self.stack.addWidget(widgets.page)

    def refresh_home(self) -> None:
        counts = self._summary_counts()
        ws = Path(self.workspace_edit.text()).name if hasattr(self, "workspace_edit") else "easy_project"
        set_metric_value(self.home_project_card, ws)
        set_metric_value(self.home_models_card, f"{counts['models']} 个")
        set_metric_value(self.home_running_card, f"{counts['running']} 个")
        set_metric_value(self.home_done_card, f"{counts['done']} 个")
        set_metric_value(self.home_check_card, f"{counts['failed']} 个")
        # Show a generic preview that is readable even before simulation exists.
        t = np.linspace(0, 1, 500)[:, None]
        x = np.linspace(0, 1, 140)[None, :]
        mock = 0.25 * np.sin(2*np.pi*(36*t + 3*x))*np.exp(-2.7*t)
        mock += np.exp(-((t - (0.42 + 0.11*np.sin(2*np.pi*x)))**2)/0.0012)*np.sin(2*np.pi*75*t)
        self.home_bscan.show_bscan(mock, "示例/最近 B-scan：完成后这里会显示真实结果", 700.0)

    # --- project page ---
    def _build_project_page(self) -> None:
        widgets = build_project_page(
            workspace=self.workspace,
            plan_path=self.plan_path,
            plan_presets=self.plan_presets,
            on_choose_workspace=self._choose_dir,
            on_choose_plan=lambda edit: self._choose_file(edit, "YAML (*.yaml *.yml);;All Files (*)"),
            on_select_plan_preset=self.select_model_plan_preset,
            on_preview_plan=self.preview_easy_plan,
        )
        self.workspace_edit = widgets.workspace_edit
        self.plan_preset_combo = widgets.plan_preset_combo
        self.plan_edit = widgets.plan_edit
        self.case_count_spin = widgets.case_count_spin
        self.project_plan_text = widgets.project_plan_text
        self.stack.addWidget(widgets.page)


    def select_model_plan_preset(self, index: int) -> None:
        """Use the selected preset YAML as the active model-generation plan."""

        if index < 0 or not hasattr(self, "plan_preset_combo"):
            return
        path_text = self.plan_preset_combo.itemData(index)
        if path_text:
            self.plan_edit.setText(str(path_text))
            self.current_manifest = None
            if hasattr(self, "model_manifest_edit"):
                self.model_manifest_edit.clear()
            if hasattr(self, "batch_manifest_edit"):
                self.batch_manifest_edit.clear()
            self.preview_easy_plan()
            self.statusBar().showMessage(f"已选择模型配置：{Path(path_text).name}")

    def preview_easy_plan(self) -> None:
        try:
            data = preview_plan_yaml(self.plan_edit.text())
            self.project_plan_text.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            self.project_plan_text.setPlainText(traceback.format_exc())

