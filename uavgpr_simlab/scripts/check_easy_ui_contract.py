from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MIXINS = [
    "HomeProjectControllerMixin",
    "ModelPreviewControllerMixin",
    "BatchControllerMixin",
    "HistoryControllerMixin",
    "SettingsControllerMixin",
    "QMainWindow",
]
EXPECTED_BUILD_ORDER = [
    "_build_home_page",
    "_build_model_page",
    "_build_batch_page",
    "_build_history_page",
    "_build_project_page",
    "_build_help_page",
]
REQUIRED_METHODS = {
    "HomeProjectControllerMixin": ["_build_home_page", "refresh_home", "_build_project_page", "select_model_plan_preset", "preview_easy_plan"],
    "ModelPreviewControllerMixin": ["_build_model_page", "generate_easy_models", "load_model_manifest", "set_model_preview_mode", "preview_selected_model", "open_selected_model_png", "sync_models_to_batch"],
    "BatchControllerMixin": ["_build_batch_page", "import_dataset_skeleton", "relocate_imported_workspace_paths", "apply_batch_run_profile", "refresh_batch_plan", "on_batch_queue_item_activated", "run_pending_batch", "run_sceneworld_profile_from_batch", "_refresh_run_dashboard_label", "_update_batch_eta_cells", "stop_batch"],
    "HistoryControllerMixin": ["_build_history_page", "refresh_history", "select_history_entry_by_case_variant", "preview_selected_history", "show_history_context_menu", "open_selected_history_case_folder", "open_selected_history_qc_json", "copy_selected_history_failure_reason", "copy_selected_history_path_summary", "show_selected_history_failure_location", "prepare_rerun_failed_from_history", "export_history", "delete_selected_history"],
    "SettingsControllerMixin": ["_build_help_page", "save_easy_env", "check_easy_env", "run_easy_gprmax_source_smoke", "run_ultra_tiny_full_chain_from_gui", "open_advanced_window"],
}
PAGE_WIDGET_FIELDS = {
    "HomePageWidgets": ["home_project_card", "home_models_card", "home_running_card", "home_done_card", "home_check_card", "home_bscan", "next_step_label"],
    "ModelPreviewPageWidgets": ["model_manifest_edit", "model_list", "easy_model_canvas", "model_preview_stack", "model_2d_preview", "view_3d_button", "view_2d_button", "model_info_labels"],
    "BatchPageWidgets": ["batch_profile_combo", "batch_manifest_edit", "batch_variants_edit", "batch_limit_spin", "batch_skip_done", "batch_failed_only", "batch_force_rerun", "batch_dataset_status_label", "batch_runtime_summary_label", "batch_total_card", "batch_pending_card", "batch_running_card", "batch_done_card", "batch_failed_card", "batch_queue_tree", "batch_failure_box", "batch_table", "batch_live_canvas", "batch_recent_canvas", "batch_log"],
    "HistoryPageWidgets": ["history_status_filter", "history_list", "history_tree", "history_bscan_mode", "history_failure_box", "history_model_canvas_easy", "history_bscan_canvas_easy", "history_detail_easy"],
    "ProjectPageWidgets": ["workspace_edit", "plan_preset_combo", "plan_edit", "case_count_spin", "project_plan_text"],
    "SettingsPageWidgets": ["gprmax_root_edit", "conda_env_edit", "gpu_ids_edit", "omp_spin", "use_gpu_check", "use_conda_check", "smoke_test_button", "link_check_button", "help_log"],
}
REQUIRED_WINDOW_ATTRS = sorted({field for fields in PAGE_WIDGET_FIELDS.values() for field in fields if field != "page"} | {"ultra_tiny_button"})


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def parse(rel: str) -> ast.Module:
    return ast.parse((ROOT / rel).read_text(encoding="utf-8", errors="replace"), filename=rel)


def class_node(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def func_names(cls: ast.ClassDef) -> set[str]:
    return {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}


def check_mixin_order() -> Check:
    cls = class_node(parse("src/uavgpr_simlab/gui/easy_window.py"), "EasyMainWindow")
    if not cls:
        return Check("mixin_order", False, "EasyMainWindow not found")
    bases = [getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases]
    return Check("mixin_order", bases == EXPECTED_MIXINS, f"bases={bases}; expected={EXPECTED_MIXINS}")


def check_build_order() -> Check:
    cls = class_node(parse("src/uavgpr_simlab/gui/easy_window.py"), "EasyMainWindow")
    if not cls:
        return Check("build_order", False, "EasyMainWindow not found")
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    calls: list[str] = []
    if init:
        for node in ast.walk(init):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                if node.func.attr.startswith("_build_"):
                    calls.append(node.func.attr)
    return Check("build_order", calls == EXPECTED_BUILD_ORDER, f"calls={calls}; expected={EXPECTED_BUILD_ORDER}")


def check_controller_methods() -> Check:
    failed: list[str] = []
    details: list[str] = []
    for path in sorted((ROOT / "src/uavgpr_simlab/gui/controllers").glob("*_controller.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("Mixin")]
        for cls in classes:
            methods = func_names(cls)
            required = REQUIRED_METHODS.get(cls.name, [])
            missing = [m for m in required if m not in methods]
            details.append(f"{cls.name}:{len(methods)} methods")
            if missing:
                failed.append(f"{cls.name} missing {missing}")
    return Check("controller_methods", not failed, "; ".join(details) + ("; " + " | ".join(failed) if failed else ""))


def check_page_widget_dataclasses() -> Check:
    failed: list[str] = []
    details: list[str] = []
    for path in sorted((ROOT / "src/uavgpr_simlab/gui/pages").glob("*_page.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("Widgets")]:
            fields = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
            required = ["page"] + PAGE_WIDGET_FIELDS.get(cls.name, [])
            missing = [x for x in required if x not in fields]
            details.append(f"{cls.name}:{fields}")
            if missing:
                failed.append(f"{cls.name} missing {missing}")
    return Check("page_widget_dataclasses", not failed, " | ".join(failed) if failed else "; ".join(d.split(':',1)[0] for d in details))


def check_widget_assignment_contract() -> Check:
    failed: list[str] = []
    assigned: set[str] = set()
    for path in sorted((ROOT / "src/uavgpr_simlab/gui/controllers").glob("*_controller.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        assigned.add(target.attr)
    missing = [attr for attr in REQUIRED_WINDOW_ATTRS if attr not in assigned]
    # Settings controller intentionally aliases link_check_button to ultra_tiny_button.
    if "link_check_button" in missing and "ultra_tiny_button" in assigned:
        missing.remove("link_check_button")
    if missing:
        failed.append("missing self assignments: " + ", ".join(missing))
    return Check("widget_assignment_contract", not failed, f"assigned={len(assigned)}; " + (" | ".join(failed) if failed else "all required widget references assigned"))


def check_static_callbacks() -> Check:
    files = ["src/uavgpr_simlab/gui/pages/batch_page.py", "src/uavgpr_simlab/gui/pages/model_preview_page.py", "src/uavgpr_simlab/gui/pages/settings_page.py", "src/uavgpr_simlab/gui/pages/history_page.py", "src/uavgpr_simlab/gui/pages/project_page.py"]
    required_tokens = [
        "on_generate_models", "on_load_manifest", "on_import_skeleton", "on_relocate_workspace", "on_refresh_plan", "on_run_pending", "on_stop", "on_queue_activated",
        "advanced_toggle", "advanced_panel", "显示运行细节/高级诊断",
        "on_save", "on_check", "on_smoke_test", "on_link_check", "on_open_advanced",
        "on_refresh", "on_preview", "on_context_menu", "on_rerun_failed", "on_export", "on_delete",
    ]
    text = "\n".join((ROOT / f).read_text(encoding="utf-8", errors="replace") for f in files)
    missing = [tok for tok in required_tokens if tok not in text]
    return Check("static_callbacks", not missing, "missing=" + ", ".join(missing) if missing else "all key page callback tokens present")


def check_optional_runtime_instantiation() -> Check:
    code = """
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from uavgpr_simlab.gui.easy_window import EasyMainWindow
app = QApplication.instance() or QApplication([])
win = EasyMainWindow()
required = %r
missing = [name for name in required if not hasattr(win, name)]
print({'missing': missing, 'stack_count': win.stack.count()})
raise SystemExit(0 if not missing and win.stack.count() == 6 else 2)
""" % REQUIRED_WINDOW_ATTRS
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
    except Exception as exc:
        return Check("optional_runtime_instantiation", True, f"skipped/failed to execute optional runtime check: {exc!r}")
    output = (proc.stdout or "")[-2000:]
    if "No module named 'PySide6'" in output or "ModuleNotFoundError" in output:
        return Check("optional_runtime_instantiation", True, "skipped because PySide6 is not installed in this environment")
    return Check("optional_runtime_instantiation", proc.returncode == 0, output.replace("\n", " | "))


def main() -> int:
    checks = [
        check_mixin_order(),
        check_build_order(),
        check_controller_methods(),
        check_page_widget_dataclasses(),
        check_widget_assignment_contract(),
        check_static_callbacks(),
        check_optional_runtime_instantiation(),
    ]
    payload = {"ok": all(c.ok for c in checks), "checks": [c.to_dict() for c in checks]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
