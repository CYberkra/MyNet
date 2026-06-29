from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Matplotlib canvas used by the advanced engineering GUI for B-scan/f-k previews."""

    def __init__(self, title: str = "B-scan"):
        self.fig = Figure(figsize=(7.5, 4.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_title(title)
        self.ax.set_xlabel("Trace")
        self.ax.set_ylabel("Time (ns)")
        self.fig.tight_layout()

    def show_bscan(self, bscan: np.ndarray, title: str, time_window_ns: float) -> None:
        arr = np.asarray(bscan, dtype=float)
        self.ax.clear()
        if arr.size == 0:
            self.ax.set_title("No data")
            self.draw_idle()
            return
        vmax = float(np.percentile(np.abs(arr), 99.0))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.max(np.abs(arr)) or 1.0)
        self.ax.imshow(
            arr,
            aspect="auto",
            cmap="gray",
            vmin=-vmax,
            vmax=vmax,
            extent=[0, arr.shape[1], float(time_window_ns), 0],
        )
        self.ax.set_title(title)
        self.ax.set_xlabel("Trace index")
        self.ax.set_ylabel("Time (ns)")
        self.fig.tight_layout()
        self.draw_idle()


    def show_bscan_grid(self, bscans: list[tuple[str, np.ndarray]], title: str, time_window_ns: float) -> None:
        """Show multiple B-scan arrays side by side for dataset QC comparison."""
        self.fig.clear()
        valid: list[tuple[str, np.ndarray]] = []
        for name, data in bscans:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 2 and arr.size:
                valid.append((str(name), arr))
        if not valid:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("No comparable B-scan data")
            self.draw_idle()
            return
        n = len(valid)
        axes = self.fig.subplots(1, n, squeeze=False)[0]
        self.ax = axes[0]
        vmax = 0.0
        for _, arr in valid:
            local = float(np.nanpercentile(np.abs(arr), 99.0)) if arr.size else 0.0
            if np.isfinite(local):
                vmax = max(vmax, local)
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1.0
        for ax, (name, arr) in zip(axes, valid):
            ax.imshow(arr, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=[0, arr.shape[1], float(time_window_ns), 0])
            ax.set_title(name)
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time (ns)")
        self.fig.suptitle(title)
        self.fig.tight_layout()
        self.draw_idle()

    def show_fk(self, bscan: np.ndarray, title: str = "f-k amplitude") -> None:
        arr = np.asarray(bscan, dtype=float)
        self.ax.clear()
        f_k = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(arr)))) if arr.size else arr
        self.ax.imshow(f_k, aspect="auto", cmap="gray")
        self.ax.set_title(title)
        self.ax.set_xlabel("k index")
        self.ax.set_ylabel("f index")
        self.fig.tight_layout()
        self.draw_idle()


class Model3DCanvas(FigureCanvas):
    """Readable 3D/2.5D preview for generated UavGPR scenes.

    It uses the saved label_json, so it works before running expensive FDTD.
    """

    def __init__(self):
        self.fig = Figure(figsize=(7.8, 5.0), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        super().__init__(self.fig)
        self.show_empty()

    def show_empty(self) -> None:
        self.ax.clear()
        self.ax.set_title("请选择 manifest 中的 case 以预览地形、基覆界面和飞行高度")
        self.ax.set_xlabel("Distance x (m)")
        self.ax.set_ylabel("Across-line z (m)")
        self.ax.set_zlabel("Elevation / model y (m)")
        self.fig.tight_layout()
        self.draw_idle()

    def show_label_json(self, label_json: str | Path) -> dict:
        p = Path(label_json)
        if not p.exists():
            self.show_empty()
            return {"error": f"label_json not found: {p}"}
        data = json.loads(p.read_text(encoding="utf-8"))
        x = np.asarray(data.get("x_m", []), dtype=float)
        ground = np.asarray(data.get("ground_y_m", []), dtype=float)
        iface = np.asarray(data.get("interface_y_m", []), dtype=float)
        if x.size < 2 or ground.size != x.size or iface.size != x.size:
            self.show_empty()
            return {"error": "invalid label_json arrays"}
        geom = data.get("geometry", {}) or {}
        radar = data.get("radar", {}) or {}
        width = float(geom.get("y_thickness_m") or 2.0)
        z = np.linspace(0, max(width, 0.5), 8)
        x_grid, z_grid = np.meshgrid(x, z)
        ground_grid = np.tile(ground, (len(z), 1))
        iface_grid = np.tile(iface, (len(z), 1))
        flight_height = float(radar.get("nominal_flight_height_m") or 8.0)
        traj = data.get("trajectory") or {}
        self.ax.clear()
        self.ax.plot_surface(x_grid, z_grid, ground_grid, alpha=0.65, linewidth=0, antialiased=True)
        self.ax.plot_surface(x_grid, z_grid, iface_grid, alpha=0.75, linewidth=0, antialiased=True)
        if traj.get("mode") == "constant_level" and traj.get("source_y_m") is not None:
            path_y = np.full_like(x, float(traj.get("source_y_m")))
            title_traj = "constant-level flight path"
        else:
            path_y = ground + flight_height
            title_traj = "UAV preview height"
        self.ax.plot(x, np.full_like(x, width / 2), path_y, linewidth=2.0)
        self.ax.set_title(f"3D 模型预览：{data.get('case_id', p.stem)}  |  地表 / 基覆界面 / {title_traj}")
        self.ax.set_xlabel("测线距离 x (m)")
        self.ax.set_ylabel("横向厚度 z (m)")
        self.ax.set_zlabel("模型高度 y (m)")
        self.ax.view_init(elev=24, azim=-62)
        self.fig.tight_layout()
        self.draw_idle()
        return {
            "case_id": data.get("case_id", p.stem),
            "points": int(x.size),
            "x_range_m": [float(np.min(x)), float(np.max(x))],
            "interface_depth_mean_m": float(
                np.mean(np.asarray(data.get("interface_depth_m", ground - iface), dtype=float))
            ),
            "flight_height_m": flight_height,
            "label_json": str(p),
        }
