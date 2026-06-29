from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from uavgpr_simlab.cli import _cfg_from_plan
from uavgpr_simlab.core.config import repo_root
from uavgpr_simlab.core.job_registry import job_fingerprint, job_id_for, write_marker, STATUS_DONE, STATUS_FAILED, STATUS_RUNNING
from uavgpr_simlab.core.scenario import generate_cases
from uavgpr_simlab.gui.easy_window import EasyMainWindow


def write_mock_outs(input_file: Path, n: int, samples: int = 425) -> None:
    parent = input_file.parent
    stem = input_file.stem
    t = np.linspace(0, 1, samples)
    for i in range(1, n + 1):
        x = i / max(n, 1)
        trace = 0.25*np.sin(2*np.pi*(30*t + 2.8*x))*np.exp(-2.5*t)
        trace += np.exp(-((t - (0.42 + 0.1*np.sin(2*np.pi*x)))**2)/0.0012)*np.sin(2*np.pi*70*t)
        p = parent / f"{stem}{i}.out"
        with h5py.File(p, "w") as f:
            f.attrs["Title"] = f"mock {stem} trace {i}"
            rxs = f.create_group("rxs")
            rx1 = rxs.create_group("rx1")
            rx1.attrs["Position"] = [float(i), 0.0, 0.0]
            rx1.create_dataset("Ez", data=trace.astype("float32"))


def prepare_demo(root: Path) -> Path:
    base_workspace = root / "workspace" / "v060_easy_demo"
    cfg = _cfg_from_plan(root / "configs" / "run_plan_3060_quick.yaml", base_workspace)
    cfg.dataset.cases = 4
    cfg.geometry.trace_count = 300
    _, manifest = generate_cases(cfg, cfg.runtime.project_root, cfg.dataset.cases)
    workspace = manifest.parent.parent
    rows = []
    import csv
    with manifest.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("variant") == "raw":
                rows.append(row)
    statuses = [STATUS_RUNNING, STATUS_DONE, STATUS_FAILED]
    traces = [86, 300, 120]
    for row, status, nready in zip(rows, statuses, traces):
        input_file = Path(row["input_file"])
        write_mock_outs(input_file, min(nready, 18))
        fp = job_fingerprint(input_file, 300, row.get("variant", "raw"))
        jid = job_id_for(input_file, row.get("case_id", ""), row.get("variant", "raw"), 300)
        write_marker(workspace, jid, status, {
            "status": status,
            "fingerprint": fp,
            "input_file": str(input_file.resolve()),
            "case_id": row.get("case_id", ""),
            "variant": row.get("variant", "raw"),
            "n_traces": 300,
            "geometry_only": False,
            "returncode": 1 if status == STATUS_FAILED else 0,
            "log": str((workspace / "logs" / f"{jid}.log").resolve()),
            "supervisor_pid": os.getpid() if status == STATUS_RUNNING else None,
        })
    return manifest


def save(win: EasyMainWindow, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance(); assert app is not None
    app.processEvents()
    p = out / name
    win.grab().save(str(p))
    return p


def make_overview(paths: list[Path], out: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    imgs = [Image.open(p).convert("RGB") for p in paths]
    # scale each to consistent width
    thumb_w = 760
    thumbs = []
    for im in imgs:
        ratio = thumb_w / im.width
        thumbs.append(im.resize((thumb_w, int(im.height * ratio))))
    pad = 30
    cols = 2
    rows = (len(thumbs) + 1) // 2
    cell_h = max(im.height for im in thumbs) + 70
    canvas = Image.new("RGB", (cols*thumb_w + (cols+1)*pad, rows*cell_h + 90), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 22), "UavGPR-SimLab v0.6 易用化真实界面截图总览", fill=(14,49,88))
    labels = ["首页", "模型预览", "批量仿真", "历史与结果", "项目管理", "设置与帮助"]
    for idx, im in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = pad + c*(thumb_w+pad)
        y = 90 + r*cell_h
        draw.text((x, y), f"{idx+1}. {labels[idx]}", fill=(25,109,179))
        canvas.paste(im, (x, y+35))
    canvas.save(out)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    root = repo_root()
    out = root / "docs" / "screenshots_v0_6_easy"
    manifest = prepare_demo(root)
    win = EasyMainWindow()
    win.workspace_edit.setText(str(manifest.parent.parent))
    win.plan_edit.setText(str(root / "configs" / "run_plan_3060_quick.yaml"))
    win.model_manifest_edit.setText(str(manifest))
    win.batch_manifest_edit.setText(str(manifest))
    win.current_manifest = manifest
    win.current_manifest_rows = win._load_manifest_rows(manifest)
    win.load_model_manifest(silent=True)
    win.refresh_batch_plan()
    win.refresh_history()
    win.resize(1480, 900)
    win.show()
    app.processEvents()
    paths = []
    names = [
        "00_home.png", "01_model_preview.png", "02_batch_simulation.png", "03_history_results.png", "04_project_manager.png", "05_settings_help.png"
    ]
    for idx, name in enumerate(names):
        win.show_page(idx)
        app.processEvents()
        paths.append(save(win, out, name))
    make_overview(paths, out / "overview.png")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
