from __future__ import annotations
import os, sys, json, csv, shutil, time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import h5py
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from uavgpr_simlab.gui.main_window import MainWindow
from uavgpr_simlab.cli import _cfg_from_plan
from uavgpr_simlab.core.config import repo_root
from uavgpr_simlab.core.scenario import generate_cases
from uavgpr_simlab.core.runner import tasks_from_manifest
from uavgpr_simlab.core.job_registry import job_fingerprint, job_id_for, write_marker, STATUS_RUNNING, STATUS_DONE
from uavgpr_simlab.core.visual_history import build_history_preview
from uavgpr_simlab.core.history import scan_simulation_history


def make_fake_out(input_file: Path, traces: int, samples: int = 501, time_window_ns: float = 700.0):
    stem = input_file.with_suffix('')
    for i in range(1, traces + 1):
        p = stem.with_name(f"{stem.name}{i}.out")
        t = np.linspace(0, 1, samples)
        x = (i - 1) / max(1, traces - 1)
        signal = 0.35*np.sin(2*np.pi*((8+i*0.3)*t + 0.4*x))*np.exp(-2.4*t)
        signal += 0.85*np.exp(-((t - (0.38 + 0.08*np.sin(2*np.pi*x)))**2)/0.0012)*np.sin(2*np.pi*45*t)
        with h5py.File(p, 'w') as f:
            f.attrs['gprMax'] = 'gui-deep-smoke-test'
            f.attrs['Iterations'] = samples
            f.attrs['dt'] = (time_window_ns*1e-9) / samples
            rx = f.create_group('rxs').create_group('rx1')
            rx.attrs['Position'] = [float(i), 0.0, 0.0]
            rx.create_dataset('Ez', data=signal.astype('float32'))


def save_widget(widget, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance(); assert app is not None
    for _ in range(8):
        app.processEvents()
    pix = widget.grab()
    ok = pix.save(str(path))
    return ok


def main() -> int:
    root = repo_root()
    report_dir = root / 'workspace' / 'gui_deep_smoke'
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True)

    # Build a small workspace with generated labels/model files.
    cfg = _cfg_from_plan(root / 'configs' / 'run_plan_3060_quick.yaml', report_dir)
    cfg.dataset.cases = 2
    models, manifest = generate_cases(cfg, cfg.runtime.project_root, cases=cfg.dataset.cases)
    tasks = tasks_from_manifest(manifest, variants=['raw'], limit=2)
    # Simulate running job with partial traces and done job with more traces.
    running_task = tasks[0]
    done_task = tasks[1]
    make_fake_out(Path(running_task.input_file), traces=5)
    make_fake_out(Path(done_task.input_file), traces=12)
    for task, status in [(running_task, STATUS_RUNNING), (done_task, STATUS_DONE)]:
        fp = job_fingerprint(task.input_file, task.n_traces, task.variant)
        jid = job_id_for(task.input_file, str(task.case_id), task.variant, task.n_traces)
        write_marker(report_dir, jid, status, {
            'status': status,
            'fingerprint': fp,
            'input_file': str(Path(task.input_file).resolve()),
            'case_id': task.case_id,
            'variant': task.variant,
            'n_traces': int(task.n_traces),
            'geometry_only': False,
            'returncode': 0 if status == STATUS_DONE else '',
            'log': str((report_dir / 'logs' / f'{jid}.log').resolve()),
            'postprocess': {'output_dir': str((report_dir / 'outputs' / 'gprmax_qc' / f'{task.case_id}_{task.variant}').resolve())}
        })

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 950)
    win.show()
    app.processEvents()

    checks = []
    expected_tabs = ['0 工作台','1 环境检查','2 仿真计划','3 3D预览','4 预检去重','5 批量运行','6 历史记录','7 实测/弱监督','8 结果/报告','9 高级/PGDA']
    actual_tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    checks.append({'name':'tab_count_and_names','ok': actual_tabs == expected_tabs, 'actual': actual_tabs})

    # Dashboard screenshot
    win.tabs.setCurrentIndex(0)
    app.processEvents()
    dashboard_png = report_dir / 'screenshots' / 'dashboard.png'
    checks.append({'name':'dashboard_screenshot','ok': save_widget(win, dashboard_png), 'path': str(dashboard_png)})

    # History view test
    win.tabs.setCurrentIndex(6)
    win.history_workspace_edit.setText(str(report_dir))
    win.history_thumb_check.setChecked(True)
    win.history_filter.setCurrentIndex(0)
    win.refresh_history()
    app.processEvents()
    row_count = win.history_table.rowCount()
    checks.append({'name':'history_rows_loaded','ok': row_count >= 2, 'rows': row_count})
    if row_count:
        win.history_table.selectRow(0)
        app.processEvents()
        win.preview_history_selected()
        app.processEvents()
    history_png = report_dir / 'screenshots' / 'visual_history.png'
    checks.append({'name':'history_screenshot','ok': save_widget(win, history_png), 'path': str(history_png)})

    records = scan_simulation_history(report_dir)
    preview_reports = []
    for rec in records:
        marker_data = json.loads(Path(rec.marker_file).read_text(encoding='utf-8'))
        pv = build_history_preview(rec, report_dir, marker_data=marker_data, make_png=True)
        preview_reports.append(pv.__dict__)
    checks.append({'name':'history_previews_generated','ok': all(p['label_json'] and p['available_traces'] > 0 for p in preview_reports), 'previews': preview_reports})

    # Preflight tab can load manifest and build table.
    win.tabs.setCurrentIndex(4)
    win.preflight_manifest_edit.setText(str(manifest))
    win.run_preflight()
    app.processEvents()
    preflight_png = report_dir / 'screenshots' / 'preflight.png'
    checks.append({'name':'preflight_screenshot','ok': save_widget(win, preflight_png), 'path': str(preflight_png)})

    # 3D preview tab can load labels.
    win.tabs.setCurrentIndex(3)
    # Try select generated label if exposed by line edit? Fall back to canvas direct.
    label_json = Path(cfg.runtime.project_root) / 'datasets' / f'{tasks[0].case_id}_labels.json'
    rep = win.model3d_canvas.show_label_json(label_json)
    app.processEvents()
    model_png = report_dir / 'screenshots' / 'model3d_preview.png'
    checks.append({'name':'model_canvas_preview','ok': ('error' not in rep) and save_widget(win, model_png), 'path': str(model_png), 'rep': rep})

    ok_all = all(c.get('ok') for c in checks)
    report = {'ok': ok_all, 'time_local': time.strftime('%Y-%m-%d %H:%M:%S'), 'workspace': str(report_dir), 'checks': checks}
    (report_dir / 'gui_deep_smoke_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok_all else 2

if __name__ == '__main__':
    raise SystemExit(main())
