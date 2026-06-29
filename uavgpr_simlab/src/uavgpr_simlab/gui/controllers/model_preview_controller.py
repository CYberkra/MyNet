from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QListWidgetItem, QMessageBox

from uavgpr_simlab.core.visual_history import render_model_preview
from uavgpr_simlab.gui.pages.model_preview_page import build_model_preview_page
from uavgpr_simlab.services.project_service import find_latest_manifest, generate_model_batch


class ModelPreviewControllerMixin:
    # --- model page ---
    def _build_model_page(self) -> None:
        widgets = build_model_preview_page(
            on_choose_manifest=lambda edit: self._choose_file(edit, "CSV (*.csv);;All Files (*)"),
            on_load_manifest=lambda: self.load_model_manifest(silent=False),
            on_generate_models=self.generate_easy_models,
            on_preview_selection=self.preview_selected_model,
            on_open_model_png=self.open_selected_model_png,
            on_set_preview_mode=self.set_model_preview_mode,
            on_sync_to_batch=self.sync_models_to_batch,
        )
        self.model_manifest_edit = widgets.model_manifest_edit
        self.model_list = widgets.model_list
        self.easy_model_canvas = widgets.easy_model_canvas
        self.model_preview_stack = widgets.model_preview_stack
        self.model_2d_preview = widgets.model_2d_preview
        self.view_3d_button = widgets.view_3d_button
        self.view_2d_button = widgets.view_2d_button
        self.model_preview_mode = "3d"
        self.model_info_labels = widgets.model_info_labels
        self.stack.addWidget(widgets.page)

    def generate_easy_models(self) -> None:
        try:
            result = generate_model_batch(
                self.plan_edit.text(),
                self.workspace_edit.text(),
                int(self.case_count_spin.value()),
            )
            self.current_manifest = result.manifest
            self.model_manifest_edit.setText(str(result.manifest))
            self.batch_manifest_edit.setText(str(result.manifest))
            self.current_manifest_rows = result.rows
            self.load_model_manifest(silent=True)
            plan_name = Path(self.plan_edit.text()).name
            self.statusBar().showMessage(f"已使用 {plan_name} 生成 {len(result.models)} 个模型，manifest={result.manifest}")
            QMessageBox.information(
                self,
                "生成完成",
                f"已使用模型配置 {plan_name} 生成 {len(result.models)} 个模型。\n\n"
                "下一步可以先查看模型预览，再去批量仿真。",
            )
        except Exception:
            QMessageBox.critical(self, "生成失败", traceback.format_exc())

    def load_model_manifest(self, silent: bool = False) -> None:
        text = self.model_manifest_edit.text().strip()
        if not text and self.current_manifest:
            text = str(self.current_manifest)
            self.model_manifest_edit.setText(text)
        if not text:
            discovered = find_latest_manifest(self.workspace_edit.text())
            if discovered is not None:
                text = str(discovered)
                self.model_manifest_edit.setText(text)
                self.batch_manifest_edit.setText(text)
        if not text:
            if not silent:
                ws = Path(self.workspace_edit.text()).expanduser()
                QMessageBox.warning(
                    self,
                    "没有模型清单",
                    "未找到模型清单。\n\n"
                    "请先点击“生成一批模型”，或手动选择 manifest.csv。\n\n"
                    f"默认查找位置：\n{ws / 'datasets'}",
                )
            return
        manifest = Path(text).expanduser()
        if not manifest.exists() or not manifest.is_file():
            if not silent:
                reason = "不存在" if not manifest.exists() else "不是文件，而是目录"
                QMessageBox.warning(
                    self,
                    "没有模型清单",
                    f"当前填写的 manifest.csv {reason}。\n\n"
                    f"当前路径：\n{manifest}\n\n"
                    "请先生成模型，或选择正确的 manifest.csv。",
                )
            return
        self.current_manifest = manifest
        self.model_manifest_edit.setText(str(manifest))
        self.batch_manifest_edit.setText(str(manifest))
        self.current_manifest_rows = self._load_manifest_rows(manifest)
        self.model_list.clear()
        self.model_list.setSpacing(6)
        workspace = manifest.parent.parent
        unique_rows = self._unique_case_rows()
        if not unique_rows:
            if not silent:
                QMessageBox.warning(self, "空模型清单", f"manifest 中没有可显示的模型记录：\n{manifest}")
            return
        for row in unique_rows:
            cid = row.get("case_id", "")
            label = self._resolve_workspace_path(row.get("label_json") or (workspace / "datasets" / f"{cid}_labels.json"))
            preview = workspace / "previews" / "models" / f"{cid}.png"
            render_model_preview(label, preview, title=f"{cid} 模型预览", width=420, height=240)
            item = QListWidgetItem(f"🧱 {cid}\n地形 + 覆盖层 + 基岩面")
            if preview.exists():
                item.setIcon(QPixmap(str(preview)))
            item.setData(Qt.UserRole, row)
            item.setSizeHint(QSize(260, 112))
            self.model_list.addItem(item)
        if self.model_list.count() > 0:
            self.model_list.setCurrentRow(0)
            self.statusBar().showMessage(f"已加载模型图库：{self.model_list.count()} 个模型，manifest={manifest}")

    def set_model_preview_mode(self, mode: str) -> None:
        """Switch the easy model preview between 3D and generated 2D cross-section."""
        normalized = "2d" if str(mode).lower() == "2d" else "3d"
        self.model_preview_mode = normalized
        if hasattr(self, "model_preview_stack"):
            self.model_preview_stack.setCurrentIndex(1 if normalized == "2d" else 0)
        if hasattr(self, "view_3d_button"):
            self.view_3d_button.setChecked(normalized == "3d")
        if hasattr(self, "view_2d_button"):
            self.view_2d_button.setChecked(normalized == "2d")
        self.preview_selected_model()

    def _selected_model_row(self) -> dict:
        items = self.model_list.selectedItems()
        if not items:
            return {}
        return items[0].data(Qt.UserRole) or {}

    def _model_preview_png_for_row(self, row: dict) -> Path:
        cid = row.get("case_id", "")
        label = self._resolve_workspace_path(row.get("label_json") or (self._workspace_from_manifest() / "datasets" / f"{cid}_labels.json"))
        out = self._workspace_from_manifest() / "previews" / "models" / f"{cid}.png"
        render_model_preview(label, out, title=f"{cid} 2D 剖面预览", width=760, height=430)
        return out

    def _show_selected_model_2d(self, row: dict) -> None:
        if not hasattr(self, "model_2d_preview"):
            return
        if not row:
            self.model_2d_preview.setText("请选择模型后查看 2D 剖面")
            self.model_2d_preview.setPixmap(QPixmap())
            return
        p = self._model_preview_png_for_row(row)
        if not p.exists():
            self.model_2d_preview.setText(f"暂无 2D 剖面图\n{p}")
            self.model_2d_preview.setPixmap(QPixmap())
            return
        pix = QPixmap(str(p))
        if pix.isNull():
            self.model_2d_preview.setText(f"2D 剖面图无法读取\n{p}")
            self.model_2d_preview.setPixmap(QPixmap())
            return
        target = self.model_2d_preview.size()
        if target.width() < 100 or target.height() < 100:
            target = QSize(760, 430)
        self.model_2d_preview.setText("")
        self.model_2d_preview.setPixmap(pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def preview_selected_model(self) -> None:
        row = self._selected_model_row()
        if not row:
            return
        label = row.get("label_json") or ""
        rep = self.easy_model_canvas.show_label_json(self._resolve_workspace_path(label) if label else "")
        if getattr(self, "model_preview_mode", "3d") == "2d":
            self._show_selected_model_2d(row)
        depth = rep.get("interface_depth_mean_m") if isinstance(rep, dict) else None
        try:
            depth_text = f"约 {float(depth):.1f} m" if depth is not None else "—"
        except Exception:
            depth_text = "—"
        slope = row.get("slope_deg") or ""
        try:
            slope_v = abs(float(slope)); terrain = "平缓" if slope_v < 5 else ("中等" if slope_v < 18 else "较陡")
        except Exception:
            terrain = "中等"
        self.model_info_labels["地表起伏"].setText(f"〰  地表起伏：{terrain}")
        self.model_info_labels["覆盖层厚度"].setText(f"▰  覆盖层厚度：{depth_text}")
        self.model_info_labels["是否有障碍物"].setText("🌳  是否有障碍物：可能有（树木 / 电线等按计划随机生成）")
        self.model_info_labels["模型难度"].setText(f"📊  模型难度：{'较高' if terrain == '较陡' else '中等'}")
        self.model_info_labels["推荐用途"].setText("🎯  推荐用途：基覆界面探测、杂波抑制训练、批量仿真")

    def open_selected_model_png(self) -> None:
        row = self._selected_model_row()
        if not row:
            QMessageBox.information(self, "模型剖面图", "请先在左侧选择一个模型。")
            return
        p = self._model_preview_png_for_row(row)
        self.set_model_preview_mode("2d")
        QMessageBox.information(self, "模型 2D 剖面图", f"2D 剖面图路径：\n{p}")

    def sync_models_to_batch(self) -> None:
        if self.current_manifest:
            self.batch_manifest_edit.setText(str(self.current_manifest))
            self.show_page(2)
            self.refresh_batch_plan()

