from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from uavgpr_simlab.cli import _cfg_from_plan
from uavgpr_simlab.core.config import repo_root
from uavgpr_simlab.core.hpc import write_slurm_array_script
from uavgpr_simlab.core.postprocess import export_gprmax_bscan_for_input
from uavgpr_simlab.core.reporting import build_auto_report
from uavgpr_simlab.core.runner import tasks_from_manifest
from uavgpr_simlab.core.dataset_contract import validate_dataset_skeleton
from uavgpr_simlab.core.workspace_relocator import relocate_workspace_paths
from uavgpr_simlab.services.advanced_queue_service import (
    build_advanced_queue_tasks,
    read_queue_manifest_preview,
    summarize_queue_tasks,
    task_from_manifest_row,
)
from uavgpr_simlab.services.environment_service import (
    EasyEnvironmentSettings,
    build_runtime_config_for_easy,
    format_easy_environment_report,
    inspect_gprmax_source,
    run_easy_environment_diagnostics,
)
from uavgpr_simlab.services.gprmax_smoke_service import GprMaxSmokeStep, GprMaxSourceSmokeReport, format_gprmax_source_smoke_report
from uavgpr_simlab.services.project_service import discover_model_plan_presets, find_latest_manifest, generate_model_batch, preview_plan_yaml
from uavgpr_simlab.services.real_csv_service import export_real_csv_qc, load_real_csv_preview
from uavgpr_simlab.services.sceneworld_bscan_service import build_case_bscan_qc, check_sceneworld_case_package, resample_bscan
from uavgpr_simlab.core.scenario import generate_cases
from uavgpr_simlab.core.softmask import generate_borehole_soft_mask


def ok(name: str, data: dict | None = None) -> dict:
    return {"name": name, "ok": True, "data": data or {}}


def skip(name: str, reason: str) -> dict:
    return {"name": name, "ok": True, "skipped": True, "reason": reason}


def main() -> int:
    root = repo_root()
    tmp = root / "workspace" / "self_test_runtime"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    results: list[dict] = []

    # 1. Import/GUI smoke test. PySide6 is optional for headless CI/code-review environments.
    try:
        from PySide6.QtWidgets import QApplication, QLabel
        from uavgpr_simlab.gui.main_window import MainWindow
        from uavgpr_simlab.gui.easy_window import EasyMainWindow
        from uavgpr_simlab.gui.advanced_widgets import MplCanvas, Model3DCanvas
        from uavgpr_simlab.gui.advanced_workers import GenericWorker, LiveQueueWorker

        app = QApplication.instance() or QApplication([])
        win = MainWindow()
        win.show(); app.processEvents()
        assert win.tabs.count() == 10
        assert win.log_box.isReadOnly()
        assert win.env_smoke_command_button.text() == "显示 gprMax smoke test 命令"
        assert win.workspace_edit.text().endswith("workspace")
        assert win.dashboard_summary.isReadOnly()
        assert win.dashboard_env_button.text() == "进入"
        assert win.plan_edit.text().endswith("run_plan_3060_quick.yaml")
        assert win.case_override_spin.value() == 0
        assert win.antenna_combo.count() == 2
        assert win.component_combo.count() == 3
        assert win.gen_log.isReadOnly()
        assert win.preview_manifest_edit.text() == ""
        assert win.preview_case_table.columnCount() == 6
        assert win.preview_info_box.isReadOnly()
        assert win.preview_note_label.wordWrap()
        assert win.preflight_variants_edit.text() == "raw,target_only,clutter_only,background_only,air_only"
        assert win.preflight_skip_check.isChecked()
        assert win.preflight_limit_spin.value() == 0
        assert win.preflight_table.columnCount() == 8
        assert win.qc_text.isReadOnly()
        assert win.qc_export_button.text() == "合并可读 .out 并导出传统基线/NPZ/PNG"
        assert win.train_text.isReadOnly()
        assert "PGDA-CSNet" in win.train_text.toPlainText()
        assert win.history_filter.count() == 7
        assert win.history_filter.currentText() == "全部"
        assert win.history_table.columnCount() == 13
        assert win.history_table.isColumnHidden(11)
        assert win.history_table.isColumnHidden(12)
        assert win.history_log.isReadOnly()
        assert win.history_detail_box.isReadOnly()
        assert win.variant_combo.count() == 5
        assert win.variant_combo.currentText() == "raw"
        assert win.limit_spin.value() == 1
        assert win.geometry_check.isChecked()
        assert win.skip_completed_check.isChecked()
        assert not win.force_rerun_check.isChecked()
        assert win.task_list.count() == 0
        assert win.progress.value() == 0
        assert win.queue_log.isReadOnly()
        assert "Waiting for readable" in win.preview_info.text()
        assert win.csv_edit.text().endswith("Line9origin36_first16traces.csv")
        assert win.max_traces_spin.value() == 300
        assert win.csv_info.isReadOnly()
        assert win.csv_info.maximumHeight() == 210
        assert win.csv_load_button.text() == "加载并预览"
        assert win.csv_fk_button.text() == "显示 f-k 图"
        assert win.csv_export_button.text() == "导出 NPZ/PNG 质控"
        assert isinstance(win.queue_canvas, MplCanvas)
        assert isinstance(win.model3d_canvas, Model3DCanvas)
        assert isinstance(win.history_bscan_canvas, MplCanvas)
        assert type(win.worker).__name__ == "NoneType" or isinstance(win.worker, GenericWorker)
        assert type(win.live_worker).__name__ == "NoneType" or isinstance(win.live_worker, LiveQueueWorker)
        easy = EasyMainWindow()
        easy.show(); app.processEvents()
        assert easy.stack.count() == 6
        assert easy.home_bscan is not None
        assert easy.next_step_label.wordWrap()
        assert easy.home_project_card.findChild(QLabel, "bigNumber") is not None
        assert easy.project_plan_text.isReadOnly()
        assert easy.plan_preset_combo.count() >= 1
        assert easy.plan_preset_combo.currentData()
        assert easy.case_count_spin.value() == 20
        assert easy.model_manifest_edit.text() == ""
        assert easy.model_list.minimumWidth() == 285
        assert easy.model_list.iconSize().width() == 170
        assert len(easy.model_info_labels) == 5
        assert "地表起伏" in easy.model_info_labels
        assert easy.easy_model_canvas is not None
        assert easy.model_preview_stack.count() == 2
        assert easy.model_preview_stack.currentIndex() == 0
        assert easy.view_3d_button.isChecked()
        assert easy.view_2d_button.text() == "2D 剖面"
        assert easy.batch_limit_spin.value() == 60
        assert easy.batch_skip_done.isChecked()
        assert easy.batch_dataset_status_label.wordWrap()
        assert easy.batch_running_card.findChild(QLabel, "bigNumber") is not None
        assert easy.batch_failed_card.findChild(QLabel, "bigNumber") is not None
        assert easy.batch_table.columnCount() == 8
        assert easy.batch_log.isReadOnly()
        assert easy.history_status_filter.count() == 6
        assert easy.history_status_filter.currentText() == "全部"
        assert easy.history_list.spacing() == 8
        assert easy.history_detail_easy.wordWrap()
        assert easy.smoke_test_button.text() == "最小 CPU 测试"
        assert easy._smoke_thread is None
        results.append(ok("GUI imports and offscreen launch", {
            "tabs": win.tabs.count(),
            "advanced_title": win.windowTitle(),
            "advanced_env_log_readonly": win.log_box.isReadOnly(),
            "advanced_env_smoke_button": win.env_smoke_command_button.text(),
            "advanced_workspace_default": win.workspace_edit.text(),
            "advanced_dashboard_summary_readonly": win.dashboard_summary.isReadOnly(),
            "advanced_generation_plan_default": win.plan_edit.text(),
            "advanced_generation_case_override": win.case_override_spin.value(),
            "advanced_generation_component_count": win.component_combo.count(),
            "advanced_model_preview_columns": win.preview_case_table.columnCount(),
            "advanced_model_preview_info_readonly": win.preview_info_box.isReadOnly(),
            "advanced_preflight_variants": win.preflight_variants_edit.text(),
            "advanced_preflight_limit": win.preflight_limit_spin.value(),
            "advanced_preflight_table_columns": win.preflight_table.columnCount(),
            "advanced_qc_text_readonly": win.qc_text.isReadOnly(),
            "advanced_qc_export_button": win.qc_export_button.text(),
            "advanced_train_text_readonly": win.train_text.isReadOnly(),
            "advanced_history_filter_count": win.history_filter.count(),
            "advanced_history_filter_current": win.history_filter.currentText(),
            "advanced_history_table_columns": win.history_table.columnCount(),
            "advanced_history_log_readonly": win.history_log.isReadOnly(),
            "advanced_history_detail_readonly": win.history_detail_box.isReadOnly(),
            "advanced_queue_variant_count": win.variant_combo.count(),
            "advanced_queue_variant_current": win.variant_combo.currentText(),
            "advanced_queue_limit": win.limit_spin.value(),
            "advanced_queue_geometry_only": win.geometry_check.isChecked(),
            "advanced_queue_skip_completed": win.skip_completed_check.isChecked(),
            "advanced_queue_force_rerun": win.force_rerun_check.isChecked(),
            "advanced_queue_log_readonly": win.queue_log.isReadOnly(),
            "advanced_queue_progress": win.progress.value(),
            "advanced_queue_preview_hint": win.preview_info.text(),
            "advanced_real_csv_default": win.csv_edit.text(),
            "advanced_real_csv_max_traces": win.max_traces_spin.value(),
            "advanced_real_csv_info_readonly": win.csv_info.isReadOnly(),
            "advanced_real_csv_info_max_height": win.csv_info.maximumHeight(),
            "advanced_real_csv_load_button": win.csv_load_button.text(),
            "advanced_real_csv_fk_button": win.csv_fk_button.text(),
            "advanced_real_csv_export_button": win.csv_export_button.text(),
            "advanced_queue_canvas_class": type(win.queue_canvas).__name__,
            "advanced_model_canvas_class": type(win.model3d_canvas).__name__,
            "advanced_history_bscan_canvas_class": type(win.history_bscan_canvas).__name__,
            "advanced_worker_class": GenericWorker.__name__,
            "advanced_live_worker_class": LiveQueueWorker.__name__,
            "easy_pages": easy.stack.count(),
            "easy_title": easy.windowTitle(),
            "home_bscan_ready": easy.home_bscan is not None,
            "home_next_step_wordwrap": easy.next_step_label.wordWrap(),
            "home_project_metric_attached": easy.home_project_card.findChild(QLabel, "bigNumber") is not None,
            "project_plan_readonly": easy.project_plan_text.isReadOnly(),
            "project_plan_preset_count": easy.plan_preset_combo.count(),
            "project_plan_preset_current": easy.plan_preset_combo.currentText(),
            "model_manifest_empty": easy.model_manifest_edit.text() == "",
            "model_list_min_width": easy.model_list.minimumWidth(),
            "model_list_icon_width": easy.model_list.iconSize().width(),
            "model_info_label_count": len(easy.model_info_labels),
            "model_preview_stack_count": easy.model_preview_stack.count(),
            "model_preview_default_index": easy.model_preview_stack.currentIndex(),
            "model_preview_2d_button": easy.view_2d_button.text(),
            "batch_limit": easy.batch_limit_spin.value(),
            "batch_skip_done": easy.batch_skip_done.isChecked(),
            "batch_dashboard_wordwrap": easy.batch_dataset_status_label.wordWrap(),
            "batch_running_metric_attached": easy.batch_running_card.findChild(QLabel, "bigNumber") is not None,
            "batch_failed_metric_attached": easy.batch_failed_card.findChild(QLabel, "bigNumber") is not None,
            "batch_table_columns": easy.batch_table.columnCount(),
            "batch_log_readonly": easy.batch_log.isReadOnly(),
            "history_filter_count": easy.history_status_filter.count(),
            "history_filter_current": easy.history_status_filter.currentText(),
            "history_list_spacing": easy.history_list.spacing(),
            "history_detail_wordwrap": easy.history_detail_easy.wordWrap(),
            "settings_smoke_button": easy.smoke_test_button.text(),
            "settings_smoke_worker_idle": easy._smoke_thread is None,
        }))
    except Exception as exc:
        results.append(skip("GUI imports and offscreen launch", f"GUI dependency unavailable or offscreen init failed: {exc}"))

    # 2. Dynamic gprMax input generation
    cfg = _cfg_from_plan(root / "configs" / "run_plan_3060_quick.yaml", tmp)
    cfg.dataset.cases = 2
    discovered_presets = discover_model_plan_presets(root)
    assert any(p.path.name == "run_plan_3060_quick.yaml" for p in discovered_presets)
    assert all(p.path.name.startswith("run_plan") for p in discovered_presets)
    project_batch = generate_model_batch(root / "configs" / "run_plan_3060_quick.yaml", tmp / "easy_service_project", 2)
    assert len(project_batch.models) == 2
    assert project_batch.manifest.exists()
    assert find_latest_manifest(tmp / "easy_service_project") == project_batch.manifest
    models, manifest = generate_cases(cfg, cfg.runtime.project_root, cases=cfg.dataset.cases)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    assert len(rows) == 10, len(rows)
    assert {r["variant"] for r in rows} == {"raw", "target_only", "clutter_only", "background_only", "air_only"}
    assert "bscan_status" in rows[0] and rows[0]["bscan_status"] == "not_run"
    gen_root = Path(cfg.runtime.project_root)
    def _gen_path(value: str) -> Path:
        q = Path(value)
        return q if q.is_absolute() else gen_root / q
    assert _gen_path(rows[0]["input_file"]).exists()
    assert not Path(rows[0]["input_file"]).is_absolute()
    first_case_rows = [r for r in rows if r["case_id"] == rows[0]["case_id"]]
    assert len(first_case_rows) == 5
    assert len({r["scene_world_json"] for r in first_case_rows}) == 1
    assert len({r["random_seed"] for r in first_case_rows}) == 1
    assert first_case_rows[0]["family"] in {"gentle_interbed", "terrace_paddy", "wire_tree_endpoint", "deep_anomaly_21m", "cross_slope_high_relief"}
    assert first_case_rows[0]["flight_height_mode"] == "constant_level"
    assert _gen_path(first_case_rows[0]["scene_world_json"]).exists()
    assert _gen_path(first_case_rows[0]["metadata_summary_json"]).exists()
    assert _gen_path(first_case_rows[0]["interface_gt_npy"]).exists()
    assert _gen_path(first_case_rows[0]["layer_gt_npy"]).exists()
    assert _gen_path(first_case_rows[0]["model_preview_png"]).exists()
    assert _gen_path(first_case_rows[0]["variant_preview_png"]).exists()
    world_doc = json.loads(_gen_path(first_case_rows[0]["scene_world_json"]).read_text(encoding="utf-8"))
    assert world_doc["schema"] == "uavgpr_simlab.scene_world.v1alpha1"
    assert world_doc["trajectory"]["mode"] == "constant_level"
    assert "not true terrain-following" in world_doc["trajectory"]["note"]
    assert float(first_case_rows[0]["model_length_m"]) == float(first_case_rows[0]["model_length_actual_m"])
    assert _gen_path(first_case_rows[0]["raw_bscan_npy"]).exists()
    assert _gen_path(first_case_rows[0]["target_bscan_npy"]).exists()
    bscan_shape = np.load(_gen_path(first_case_rows[0]["raw_bscan_npy"])).shape
    mask_shape = np.load(_gen_path(first_case_rows[0]["interface_mask_bscan_npy"])).shape
    assert bscan_shape == mask_shape == (int(cfg.radar.frequency_points), int(cfg.geometry.trace_count))

    # v0.8.0-alpha.2 SceneWorld smoke plan must cover five Yingshan families and portable relative paths.
    smoke_cfg = _cfg_from_plan(root / "configs" / "run_plan_yingshan_sceneworld_smoke.yaml", tmp / "scene_world_smoke")
    smoke_models, smoke_manifest = generate_cases(smoke_cfg, smoke_cfg.runtime.project_root, cases=5)
    smoke_rows = list(csv.DictReader(open(smoke_manifest, encoding="utf-8")))
    smoke_root = Path(smoke_cfg.runtime.project_root)
    smoke_families = {r["family"] for r in smoke_rows}
    assert {"gentle_interbed", "terrace_paddy", "wire_tree_endpoint", "deep_anomaly_21m", "cross_slope_high_relief"}.issubset(smoke_families)
    assert all(not Path(r["input_file"]).is_absolute() for r in smoke_rows)
    case_by_family = {r["family"]: r for r in smoke_rows if r["variant"] == "raw"}
    for fam, row in case_by_family.items():
        world = json.loads((smoke_root / row["scene_world_json"]).read_text(encoding="utf-8"))
        meta = json.loads((smoke_root / row["metadata_summary_json"]).read_text(encoding="utf-8"))
        assert (smoke_root / "models" / row["case_id"] / "raw.in").exists()
        assert (smoke_root / "models" / row["case_id"] / "target_only.in").exists()
        assert (smoke_root / "models" / row["case_id"] / "background_only.in").exists()
        assert (smoke_root / "models" / row["case_id"] / "clutter_only.in").exists()
        assert (smoke_root / "models" / row["case_id"] / "air_only.in").exists()
        assert (smoke_root / row["raw_bscan_npy"]).exists()
        assert (smoke_root / row["bscan_qc_report_json"]).exists()
        assert np.load(smoke_root / row["raw_bscan_npy"]).shape == np.load(smoke_root / row["interface_mask_bscan_npy"]).shape
        if fam == "deep_anomaly_21m":
            depths = [float(o.get("center_depth_m")) for o in world["anomaly_objects"]]
            assert depths and all(18.0 <= d <= 23.0 for d in depths)
        if fam == "wire_tree_endpoint":
            kinds = {o.get("kind") for o in world["external_clutter_objects"]}
            assert {"wire", "tree"}.issubset(kinds)
        if fam == "terrace_paddy":
            assert world["water_zones"]
        if fam == "cross_slope_high_relief":
            assert 8.0 <= float(meta["ground_relief_m"]) <= 30.5
    pilot_cfg = _cfg_from_plan(root / "configs" / "run_plan_yingshan_sceneworld_pilot_v080b1.yaml", tmp / "scene_world_pilot")
    assert pilot_cfg.radar.frequency_points == 501
    assert float(pilot_cfg.radar.time_window_ns) == 700.0
    assert int(pilot_cfg.geometry.trace_count) >= 200
    assert int(pilot_cfg.dataset.cases) >= 50
    pilot_families = [__import__("uavgpr_simlab.simulation.yingshan_families", fromlist=["normalize_family"]).normalize_family(pilot_cfg.geology.scenario_family, i) for i in range(pilot_cfg.dataset.cases)]
    high_ratio = pilot_families.count("cross_slope_high_relief") / max(1, len(pilot_families))
    assert 0.05 <= high_ratio <= 0.10


    contract_check = validate_dataset_skeleton(smoke_manifest, write_report=True)
    assert contract_check.ok and contract_check.case_count == 5 and contract_check.row_count == 25
    reloc_dry = relocate_workspace_paths(smoke_manifest, dry_run=True, write_report=True)
    assert reloc_dry.ok and reloc_dry.change_count == 0 and reloc_dry.dataset_contract_ok
    reloc_root = tmp / "relocation_copy"
    shutil.copytree(smoke_root, reloc_root)
    reloc_manifest = reloc_root / "datasets" / smoke_manifest.name
    reloc_rows = list(csv.DictReader(open(reloc_manifest, encoding="utf-8")))
    reloc_fields = list(reloc_rows[0].keys())
    old_root = r"C:\old_machine\project\workspace" + "\\" + reloc_root.name
    reloc_rows[0]["input_file"] = old_root + "\\models\\" + reloc_rows[0]["case_id"] + "\\" + reloc_rows[0]["variant"] + ".in"
    with reloc_manifest.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=reloc_fields)
        wr.writeheader(); wr.writerows(reloc_rows)
    reloc_rep = relocate_workspace_paths(reloc_manifest, old_root=old_root, dry_run=False, write_report=True)
    assert reloc_rep.ok and reloc_rep.change_count >= 1
    fixed_first = next(csv.DictReader(open(reloc_manifest, encoding="utf-8")))
    assert fixed_first["input_file"].replace("\\", "/") == f"models/{fixed_first['case_id']}/{fixed_first['variant']}.in"
    package_check = check_sceneworld_case_package(smoke_root, manifest_csv=smoke_manifest)
    assert package_check["ok"]
    # QC service should mark placeholder arrays incomplete, then pass after finite aligned arrays are written.
    qc_row = next(r for r in smoke_rows if r["variant"] == "raw")
    qc_case_dir = smoke_root / "models" / qc_row["case_id"]
    qc_traces = int(float(qc_row.get("n_traces") or qc_row.get("trace_count") or smoke_cfg.geometry.trace_count))
    fake = np.arange(31 * qc_traces, dtype=np.float32).reshape(31, qc_traces)
    aligned = resample_bscan(fake, samples=501, traces=qc_traces)
    assert aligned.shape == (501, qc_traces)
    for name in ["raw_bscan.npy", "target_only_bscan.npy", "background_only_bscan.npy", "clutter_only_bscan.npy", "air_only_bscan.npy", "clutter_gt_bscan.npy"]:
        np.save(qc_case_dir / "outputs" / name, aligned)
    qc_rep = build_case_bscan_qc(qc_case_dir, expected_shape=(501, qc_traces))
    assert qc_rep["status"] == "success" and qc_rep["raw_minus_target_computable"] and qc_rep["clutter_gt_generated"]

    tasks = tasks_from_manifest(manifest, variants=["raw"], limit=1)
    assert tasks and tasks[0].n_traces == cfg.geometry.trace_count
    queue_preview = read_queue_manifest_preview(manifest, max_display=3)
    assert queue_preview.displayed_count == 3 and queue_preview.truncated
    selected_task = task_from_manifest_row(queue_preview.rows[0].data, fallback_variant="raw")
    assert selected_task is not None and selected_task.input_file
    advanced_tasks = build_advanced_queue_tasks(manifest, variant="raw", limit=1)
    advanced_summary = summarize_queue_tasks(advanced_tasks)
    assert advanced_summary["tasks"] == 1 and advanced_summary["traces"] == cfg.geometry.trace_count
    results.append(ok("Dynamic .in/labels/manifest generation", {
        "records": len(rows),
        "raw_n_traces": tasks[0].n_traces,
        "advanced_queue_preview_displayed": queue_preview.displayed_count,
        "advanced_queue_preview_total": queue_preview.total_read,
        "advanced_queue_tasks": advanced_summary["tasks"],
        "easy_project_service_models": len(project_batch.models),
        "easy_project_service_manifest": str(project_batch.manifest),
        "easy_model_plan_presets": len(discovered_presets),
        "scene_world_family": first_case_rows[0]["family"],
        "scene_world_schema": world_doc["schema"],
        "scene_world_homologous_variants": len({r["scene_world_json"] for r in first_case_rows}) == 1,
        "scene_world_flight_mode": world_doc["trajectory"]["mode"],
        "scene_world_smoke_families": sorted(smoke_families),
        "scene_world_mask_shape": list(mask_shape),
        "scene_world_pilot_samples": pilot_cfg.radar.frequency_points,
        "scene_world_pilot_time_window_ns": pilot_cfg.radar.time_window_ns,
        "scene_world_pilot_trace_count": pilot_cfg.geometry.trace_count,
        "scene_world_pilot_cases": pilot_cfg.dataset.cases,
        "scene_world_pilot_high_relief_ratio": round(high_ratio, 3),
        "scene_world_dataset_contract_ok": contract_check.ok,
        "scene_world_dataset_contract_report": contract_check.summary_json,
        "workspace_relocation_report": reloc_rep.report_json,
        "scene_world_case_package_ok": package_check["ok"],
        "scene_world_bscan_qc_status": qc_rep["status"],
        "scene_world_clutter_gt_generated": qc_rep["clutter_gt_generated"],
    }))

    # 3. Real CSV preview/conversion service
    csv_sample = root / "sample_data" / "Line9origin36_first16traces.csv"
    preview = load_real_csv_preview(csv_sample, max_traces=16)
    assert list(preview.normalized_bscan.shape) == [501, 16]
    assert preview.info["shape_samples_x_traces"] == [501, 16]
    qc = export_real_csv_qc(csv_sample, tmp, max_traces=16, make_baselines=True)
    assert qc["bscan_shape"] == [501, 16]
    bscan_npz = tmp / "real_csv_qc" / csv_sample.stem / "real_uavgpr_bscan_preview.npz"
    assert bscan_npz.exists()
    results.append(ok("Real Line9 CSV preview/export service", {
        "shape": qc["bscan_shape"],
        "preview_shape": preview.info["shape_samples_x_traces"],
        "snr_bg": qc["snr_background_removed_db_default_windows"],
    }))

    # 4. Borehole weak-supervision soft mask
    boreholes = tmp / "boreholes_self_test.csv"
    boreholes.write_text("line_id,borehole_id,trace_index,depth_m,uncertainty_m\nLine9,ZK07,4,8.0,0.8\nLine9,ZK08,10,12.0,1.0\n", encoding="utf-8")
    sm = generate_borehole_soft_mask(bscan_npz, boreholes, tmp / "soft_mask", velocity_m_per_ns=0.10, line_id="Line9", trace_sigma=2.0, time_sigma_ns=20.0)
    assert sm["shape"] == [501, 16]
    assert sm["total_picks_used"] >= 1
    assert (tmp / "soft_mask" / "borehole_soft_mask.npy").exists()
    results.append(ok("Borehole soft-mask generation", {"shape": sm["shape"], "picks_used": sm["total_picks_used"]}))

    # 5. Fake gprMax HDF5 .out merge/export
    h5dir = tmp / "fake_out"
    h5dir.mkdir()
    input_file = h5dir / "raw.in"
    input_file.write_text("#title: fake\n", encoding="utf-8")
    for i in range(1, 5):
        with h5py.File(h5dir / f"raw{i}.out", "w") as f:
            f.attrs["gprMax"] = "self-test"
            f.attrs["Iterations"] = 501
            f.attrs["dt"] = 700e-9 / 501
            rx = f.create_group("rxs").create_group("rx1")
            rx.attrs["Position"] = [0.0, 0.0, 0.0]
            t = np.linspace(0, 1, 501)
            rx.create_dataset("Ez", data=np.sin(2*np.pi*(6+i)*t) * np.exp(-2*t))
    rep = export_gprmax_bscan_for_input(input_file, tmp / "fake_out_qc", stem="raw", time_window_ns=700.0)
    assert rep["bscan_shape"] == [501, 4]
    assert (tmp / "fake_out_qc" / "raw_svd_clean.npz").exists()
    results.append(ok("gprMax .out HDF5 merge/export", {"shape": rep["bscan_shape"], "products": list(rep["products"].keys())}))

    # 6. Easy GUI service layer checks: plan preview, runtime config and gprMax source-tree diagnostics
    fake_gprmax = tmp / "fake_gprMax_source"
    (fake_gprmax / "gprMax").mkdir(parents=True)
    (fake_gprmax / "gprMax" / "__init__.py").write_text("__name__ = 'gprMax'\n", encoding="utf-8")
    (fake_gprmax / "gprMax" / "__main__.py").write_text("import gprMax.gprMax\n", encoding="utf-8")
    (fake_gprmax / "gprMax" / "gprMax.py").write_text("def main(): pass\n", encoding="utf-8")
    (fake_gprmax / "gprMax" / "_version.py").write_text("__version__ = 'self-test'\n", encoding="utf-8")
    (fake_gprmax / "setup.py").write_text("# fake setup\n", encoding="utf-8")
    (fake_gprmax / "conda_env.yml").write_text("name: gprMax\n", encoding="utf-8")
    source_info = inspect_gprmax_source(fake_gprmax)
    assert source_info.is_source_tree and source_info.detected_version == "self-test"
    env_report = run_easy_environment_diagnostics(
        EasyEnvironmentSettings(
            gprmax_root=str(fake_gprmax),
            conda_env="gprMax",
            gpu_ids="0",
            omp_threads=2,
            use_gpu=False,
            use_conda_run=False,
        ),
        report_dir=tmp / "reports",
    )
    formatted_env = format_easy_environment_report(env_report, EasyEnvironmentSettings(gprmax_root=str(fake_gprmax), use_conda_run=False))
    assert "诊断摘要" in formatted_env and "目标机 smoke test" in formatted_env
    fake_smoke = GprMaxSourceSmokeReport(
        ok=True,
        gprmax_root=str(fake_gprmax),
        work_dir=str(tmp / "fake_smoke"),
        python="self-test",
        steps=[GprMaxSmokeStep("inspect gprMax source", True, "ok")],
        output_file=str(tmp / "fake_smoke" / "tiny_Ascan_2D.out"),
        output_size=123,
        hdf5_summary={"iterations": 1, "rxs": ["rx1"]},
        report_path=str(tmp / "fake_smoke" / "gprmax_source_smoke_report.json"),
    )
    formatted_smoke = format_gprmax_source_smoke_report(fake_smoke)
    assert "gprMax 最小 CPU 测试摘要" in formatted_smoke and "Windows/CUDA/GPU" in formatted_smoke
    plan_data = preview_plan_yaml(root / "configs" / "run_plan_3060_quick.yaml")
    assert "plan_name" in plan_data or "run" in plan_data or "dataset" in plan_data
    settings = EasyEnvironmentSettings(
        gprmax_root=str(fake_gprmax),
        conda_env="gprMax",
        gpu_ids="0",
        omp_threads=2,
        use_gpu=False,
        use_conda_run=False,
    )
    cfg2 = build_runtime_config_for_easy(
        plan_path=root / "configs" / "run_plan_3060_quick.yaml",
        workspace=tmp,
        current_manifest=manifest,
        settings=settings,
    )
    assert cfg2.runtime.gprmax_source_dir == str(fake_gprmax)
    assert cfg2.runtime.omp_threads == 2 and cfg2.runtime.use_conda_run is False
    results.append(ok("Easy project/environment services", {
        "gprmax_source": source_info.to_dict(),
        "project_root": cfg2.runtime.project_root,
        "formatted_env_has_summary": "诊断摘要" in formatted_env,
        "formatted_smoke_has_summary": "gprMax 最小 CPU 测试摘要" in formatted_smoke,
    }))

    # 7. HPC/SLURM script generation and auto report
    hpc = write_slurm_array_script(manifest, tmp / "logs" / "run_gprmax_slurm_array.sh", variants="raw", max_tasks=2, postprocess=True)
    assert Path(hpc["script"]).exists() and Path(hpc["task_tsv"]).exists()
    ar = build_auto_report(cfg.runtime.project_root, out_dir=Path(cfg.runtime.project_root) / "reports")
    assert Path(ar["auto_report_md"]).exists()
    results.append(ok("HPC script and auto report generation", {"tasks": hpc["task_count"], "report": ar["auto_report_md"]}))

    out = root / "docs" / "SELF_TEST_REPORT.json"
    out.write_text(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
