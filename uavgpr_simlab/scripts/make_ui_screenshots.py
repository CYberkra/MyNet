from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from uavgpr_simlab.gui.main_window import MainWindow
from uavgpr_simlab.core.config import repo_root
from uavgpr_simlab.cli import _cfg_from_plan
from uavgpr_simlab.core.scenario import generate_cases


def save(win: MainWindow, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance()
    assert app is not None
    app.processEvents()
    pix = win.grab()
    pix.save(str(path))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    root = repo_root()
    out = root / "docs" / "screenshots"
    win = MainWindow()
    win.show()
    app.processEvents()
    save(win, out / "01_env_gpu.png")

    win.tabs.setCurrentIndex(1)
    try:
        win.max_traces_spin.setValue(16)
        win.load_csv_preview()
    except Exception:
        pass
    save(win, out / "02_real_csv_preview.png")

    win.tabs.setCurrentIndex(2)
    win.case_override_spin.setValue(1)
    win.preview_plan()
    save(win, out / "03_dynamic_generation.png")

    # Generate a tiny manifest for the queue screenshot without starting gprMax.
    try:
        cfg = _cfg_from_plan(root / "configs" / "run_plan_3060_quick.yaml", root / "workspace" / "screenshot_demo")
        cfg.dataset.cases = 1
        _, manifest = generate_cases(cfg, cfg.runtime.project_root, cfg.dataset.cases)
        win.manifest_edit.setText(str(manifest))
        win.load_manifest()
    except Exception as exc:
        win.queue_log.appendPlainText(str(exc))
    win.tabs.setCurrentIndex(3)
    save(win, out / "04_queue_realtime.png")

    # Show how the right-hand panel looks once .out traces have been merged.
    import numpy as np
    t = np.linspace(0, 1, 501)[:, None]
    x = np.linspace(0, 1, 72)[None, :]
    mock = 0.35*np.sin(2*np.pi*(35*t + 4*x))*np.exp(-2.7*t)
    mock += np.exp(-((t - (0.46 + 0.12*np.sin(2*np.pi*x)))**2)/0.0009)*np.sin(2*np.pi*80*t)
    win._update_live_preview(mock, "mock case_000001 raw | realtime merged .out", 700.0)
    save(win, out / "06_queue_live_bscan_mock.png")

    win.tabs.setCurrentIndex(4)
    save(win, out / "05_postprocess_export.png")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
