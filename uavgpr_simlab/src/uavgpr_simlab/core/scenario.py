from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .config import AppConfig, ensure_project_dirs
from uavgpr_simlab.sim.materials import default_material_ranges, material_lines, sample_materials
from uavgpr_simlab.simulation.scene_world_generator import generate_scene_world
from uavgpr_simlab.simulation.scene_variant_writer import write_scene_world_case

VARIANT_ALIASES = {"target": "target_only", "clutter": "clutter_only", "background": "background_only", "air": "air_only"}


@dataclass
class CaseRecord:
    case_id: str
    variant: str
    input_file: str
    split: str
    n_traces: int
    trace_step_m: float
    model_length_m: float
    dx_m: float
    time_window_ns: float
    center_frequency_mhz: float
    flight_height_m: float
    interface_depth_mean_m: float
    interface_depth_min_m: float
    interface_depth_max_m: float
    slope_deg: float
    clutter_level: str
    label_json: str
    interface_csv: str
    mask_npy: str
    family: str = ""
    random_seed: int = 0
    model_length_config_m: float = 0.0
    model_length_actual_m: float = 0.0
    domain_x_m: float = 0.0
    scan_start_x_m: float = 0.0
    scan_end_x_m: float = 0.0
    flight_height_mode: str = "constant_level"
    scene_world_json: str = ""
    metadata_summary_json: str = ""
    interface_gt_npy: str = ""
    layer_gt_npy: str = ""
    model_preview_png: str = ""
    variant_preview_png: str = ""
    bscan_npy: str = ""
    raw_bscan_npy: str = ""
    target_bscan_npy: str = ""
    background_bscan_npy: str = ""
    clutter_bscan_npy: str = ""
    air_bscan_npy: str = ""
    clutter_gt_bscan_npy: str = ""
    time_axis_ns_npy: str = ""
    distance_axis_m_npy: str = ""
    distance_axis_role: str = "midpoint_x"
    tx_x_axis_m_npy: str = ""
    rx_x_axis_m_npy: str = ""
    midpoint_x_axis_m_npy: str = ""
    layer_gt_x_axis_m_npy: str = ""
    layer_gt_y_axis_m_npy: str = ""
    interface_mask_bscan_npy: str = ""
    layer_mask_bscan_npy: str = ""
    bscan_placeholder_status_json: str = ""
    bscan_qc_report_json: str = ""
    bscan_status: str = "not_run"
    bscan_error: str = ""
    is_ml_pair_valid: str = "true"


@dataclass
class SceneGeometry:
    x: np.ndarray
    ground_y: np.ndarray
    interface_y: np.ndarray
    interface_depth: np.ndarray
    domain_x: float
    domain_y: float
    domain_z: float
    source_x: float
    source_y: float
    rx_x: float
    rx_y: float
    source_z: float
    slope_deg: float


def _fmt(v: float) -> str:
    return f"{v:.5g}"


def _variants(vs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for v in vs:
        vv = VARIANT_ALIASES.get(str(v), str(v))
        if vv not in out:
            out.append(vv)
    return out


def _smooth_noise(rng: random.Random, n: int, scale: float, passes: int = 4) -> np.ndarray:
    arr = np.array([rng.uniform(-scale, scale) for _ in range(n)], dtype=float)
    kernel = np.array([0.15, 0.25, 0.2, 0.25, 0.15], dtype=float)
    kernel /= kernel.sum()
    for _ in range(passes):
        arr = np.convolve(arr, kernel, mode="same")
    return arr




def _dataset_grade_for_workspace(workspace: str | Path) -> tuple[str, bool]:
    name = Path(workspace).name.lower()
    if "ultra_tiny" in name:
        return "ultra_tiny_chain_check", True
    if "smoke" in name:
        return "quick_smoke", True
    if "pilot" in name or "formal" in name:
        return "pilot_or_formal", False
    return "development", True


def _family_clutter_level(family: str, default: str = "medium") -> str:
    return {
        "gentle_interbed": "medium",
        "terrace_paddy": "hard_water",
        "wire_tree_endpoint": "hard_external_clutter",
        "deep_anomaly_21m": "hard_deep_target",
        "cross_slope_high_relief": "hard_high_relief",
    }.get(str(family), str(default))


def _write_smoke_readme(workspace: str | Path, manifest: str | Path, summary: dict) -> None:
    root = Path(workspace)
    if "smoke" not in root.name.lower():
        return
    text = f"""# {root.name}

This dataset is a quick smoke / ready-to-run SceneWorld skeleton, not a training dataset.

## Purpose

Validate the full simulation chain for five Yingshan scenario families:

- gentle_interbed
- terrace_paddy
- wire_tree_endpoint
- deep_anomaly_21m
- cross_slope_high_relief

Each case has five homologous variants: raw, target_only, background_only, clutter_only, air_only.

## Current grade

- dataset_grade: {summary.get('dataset_grade')}
- ready_to_run: true
- not_ready_to_train: true
- training_ready: false until real gprMax outputs replace NaN placeholders and strict QC passes.

## Run

Place this folder under `<UavGPR-SimLab project root>/workspace/`, then use the GUI batch page or run:

```bat
logs\run_all_gprmax.bat
```

Set the gprMax source root in the GUI or environment, for example:

```bat
set "GPRMAX_SOURCE_DIR=E:\\gprMax\\gprMax-v.3.1.7"
```

## Success criteria

After a successful run, every case should contain finite arrays for:

- outputs/raw_bscan.npy
- outputs/target_only_bscan.npy
- outputs/background_only_bscan.npy
- outputs/clutter_only_bscan.npy
- outputs/air_only_bscan.npy
- outputs/clutter_gt_bscan.npy = raw - target_only

The following reports must agree:

- `{Path(manifest).as_posix()}`
- `reports/dataset_summary.json`
- `models/<case_id>/bscan_qc_report.json`

## Axis contract

- outputs/time_axis_ns.npy: B-scan time axis
- outputs/distance_axis_m.npy: canonical horizontal axis; role = midpoint_x
- outputs/tx_x_axis_m.npy: transmitter x axis
- outputs/rx_x_axis_m.npy: receiver x axis
- outputs/midpoint_x_axis_m.npy: TX/RX midpoint x axis

## Label contract

- labels/interface_mask_bscan.npy and labels/layer_mask_bscan.npy are aligned to B-scan shape.
- labels/layer_gt.npy is a model-grid label and must be used with labels/layer_gt_x_axis_m.npy and labels/layer_gt_y_axis_m.npy.
"""
    (root / "README_SMOKE.md").write_text(text, encoding="utf-8")

def _split(i: int, n: int, cfg: AppConfig) -> str:
    f = (i + 0.5) / max(1, n)
    if f < cfg.dataset.split_train:
        return "train"
    if f < cfg.dataset.split_train + cfg.dataset.split_val:
        return "val"
    return "test"


def make_scene_geometry(cfg: AppConfig, rng: random.Random) -> SceneGeometry:
    col = max(float(cfg.geometry.geometry_column_width_m), float(cfg.geometry.dx_m))
    start_x = max(2.0, float(cfg.geometry.dx_m) * 20)
    trace_span = max(0, int(cfg.geometry.trace_count) - 1) * float(cfg.geometry.trace_step_m)
    # Ensure the generated material columns and gprMax domain cover the complete
    # moving source/receiver path. This prevents silent out-of-domain traces when a
    # user picks too small a model_length_m for a large trace_count.
    min_scan_length = start_x + float(cfg.radar.tx_rx_offset_m) + trace_span + 2.0
    length = max(float(cfg.geometry.model_length_m), min_scan_length)
    x = np.arange(0.0, length, col)
    if len(x) < 4:
        x = np.linspace(0, length, 4)
    depth = float(cfg.geometry.subsurface_depth_m)
    air = float(cfg.radar.nominal_flight_height_m + cfg.geometry.air_margin_m)
    slope_deg = rng.uniform(cfg.geology.slope_deg_min, cfg.geology.slope_deg_max)
    total_relief = math.tan(math.radians(slope_deg)) * length * 0.10 * rng.choice([-1.0, 1.0])
    ground = depth + total_relief * (x / max(length, 1.0) - 0.5)
    if rng.random() < cfg.geology.terrace_probability:
        for j in range(rng.randint(2, 5)):
            center = length * (j + 1) / (rng.randint(4, 8))
            ground += rng.uniform(-0.6, 0.6) * (x > center)
    ground += _smooth_noise(rng, len(x), 0.25, passes=5)
    ground = np.clip(ground, depth - 3.0, depth + 2.0)
    base = rng.uniform(cfg.geology.interface_depth_min_m, cfg.geology.interface_depth_max_m)
    trend = rng.uniform(-2.0, 2.0) * (x / max(length, 1.0) - 0.5)
    rough = _smooth_noise(rng, len(x), cfg.geology.interface_roughness_m, passes=6)
    spoon = rng.uniform(0.0, 2.0) * np.sin(np.pi * x / max(length, 1.0))
    interface_depth = np.clip(base + trend + rough + spoon, 2.5, depth - 1.5)
    interface_y = np.clip(ground - interface_depth, 0.8, ground - 0.8)
    domain_x = length + 4.0
    domain_y = depth + air + 2.0
    domain_z = max(float(cfg.geometry.y_thickness_m), float(cfg.geometry.dz_m), 0.05)
    src_x = start_x
    rx_x = src_x + float(cfg.radar.tx_rx_offset_m)
    src_y = float(np.interp(src_x, x, ground)) + float(cfg.radar.nominal_flight_height_m)
    rx_y = float(np.interp(rx_x, x, ground)) + float(cfg.radar.nominal_flight_height_m)
    src_y = min(src_y, domain_y - 0.8)
    rx_y = min(rx_y, domain_y - 0.8)
    return SceneGeometry(x, ground, interface_y, interface_depth, domain_x, domain_y, domain_z, src_x, src_y, rx_x, rx_y, domain_z / 2.0, slope_deg)


def _antenna_lines(cfg: AppConfig, geom: SceneGeometry) -> List[str]:
    lines = [
        f"#waveform: ricker 1 {_fmt(cfg.radar.center_frequency_mhz * 1e6)} uavgpr_wavelet",
        f"#hertzian_dipole: z {_fmt(geom.source_x)} {_fmt(geom.source_y)} {_fmt(geom.source_z)} uavgpr_wavelet",
        f"#rx: {_fmt(geom.rx_x)} {_fmt(geom.rx_y)} {_fmt(geom.source_z)}",
        f"#src_steps: {_fmt(cfg.geometry.trace_step_m)} 0 0",
        f"#rx_steps: {_fmt(cfg.geometry.trace_step_m)} 0 0",
    ]
    if "bowtie" in cfg.radar.antenna_model.lower():
        sx, sy, z = geom.source_x, geom.source_y, geom.source_z
        half = min(0.35, cfg.radar.antenna_length_m / 4.0)
        lines += [
            f"#triangle: {_fmt(sx)} {_fmt(sy)} {_fmt(z)} {_fmt(sx-half)} {_fmt(sy-0.15)} {_fmt(z)} {_fmt(sx-half)} {_fmt(sy+0.15)} {_fmt(z)} 0 pec",
            f"#triangle: {_fmt(sx)} {_fmt(sy)} {_fmt(z)} {_fmt(sx+half)} {_fmt(sy-0.15)} {_fmt(z)} {_fmt(sx+half)} {_fmt(sy+0.15)} {_fmt(z)} 0 pec",
        ]
    return lines


def _column_boxes(cfg: AppConfig, geom: SceneGeometry, variant: str, rng: random.Random) -> List[str]:
    if variant == "air_only":
        return []
    lines: List[str] = []
    width = max(float(cfg.geometry.geometry_column_width_m), float(cfg.geometry.dx_m))
    for i, xi in enumerate(geom.x):
        x0, x1 = float(xi), min(float(xi + width), geom.domain_x)
        ground = float(geom.ground_y[i])
        interface = float(geom.interface_y[i])
        if variant in {"background_only", "clutter_only"}:
            interface = 0.0
        if interface > 0.05 and variant not in {"background_only", "clutter_only"}:
            bed = "sandstone_bedrock" if rng.random() > 0.30 else "weathered_mudstone"
            lines.append(f"#box: {_fmt(x0)} 0 0 {_fmt(x1)} {_fmt(interface)} {_fmt(geom.domain_z)} {bed}")
        over = "gravelly_silty_clay" if rng.random() < cfg.geology.blocky_gravel_probability else "silty_clay"
        lines.append(f"#box: {_fmt(x0)} {_fmt(interface)} 0 {_fmt(x1)} {_fmt(ground)} {_fmt(geom.domain_z)} {over}")
        if variant in {"raw", "target_only", "background_only"} and rng.random() < cfg.geology.water_table_probability / max(4, len(geom.x)):
            y0 = max(interface + 0.3, ground - rng.uniform(2.0, 6.0)); y1 = min(ground - 0.2, y0 + rng.uniform(0.4, 1.8))
            if y1 > y0:
                lines.append(f"#box: {_fmt(x0)} {_fmt(y0)} 0 {_fmt(x1)} {_fmt(y1)} {_fmt(geom.domain_z)} saturated_zone")
    return lines


def _clutter_lines(cfg: AppConfig, geom: SceneGeometry, variant: str, rng: random.Random) -> List[str]:
    if variant in {"target_only", "background_only", "air_only"}:
        return []
    lines: List[str] = []
    z = geom.domain_z / 2.0
    length = geom.domain_x
    def gy(xx: float) -> float:
        return float(np.interp(xx, geom.x, geom.ground_y))
    level = getattr(cfg.geology, "clutter_level", "medium")
    intensity = {"light": 0.6, "medium": 1.0, "hard": 1.6}.get(level, 1.0)
    if rng.random() < cfg.geology.paddy_water_probability * intensity:
        for _ in range(rng.randint(1, 4)):
            x0 = rng.uniform(2.0, max(3.0, length - 8.0)); w = rng.uniform(2.0, 8.0); y = gy(x0)
            lines.append(f"#box: {_fmt(x0)} {_fmt(y+0.02)} 0 {_fmt(min(length-0.5,x0+w))} {_fmt(y+0.12)} {_fmt(geom.domain_z)} surface_water")
    if rng.random() < cfg.geology.tree_probability * intensity:
        for _ in range(rng.randint(2, 8)):
            x0 = rng.uniform(1.0, length - 2.0); y = gy(x0); h = rng.uniform(2.0, 8.0); r = rng.uniform(0.08, 0.25)
            lines.append(f"#cylinder: {_fmt(x0)} {_fmt(y)} {_fmt(z)} {_fmt(x0)} {_fmt(min(geom.domain_y-0.2, y+h))} {_fmt(z)} {_fmt(r)} vegetation")
    if rng.random() < cfg.geology.overhead_wire_probability * intensity:
        for _ in range(rng.randint(1, 4 if level == "hard" else 2)):
            x0 = rng.uniform(0.5, max(1.0, length - 15.0)); x1 = min(length - 0.5, x0 + rng.uniform(10.0, 40.0)); y = min(gy((x0+x1)/2)+rng.uniform(5,12), geom.domain_y-0.4)
            lines.append(f"#cylinder: {_fmt(x0)} {_fmt(y)} {_fmt(z)} {_fmt(x1)} {_fmt(y)} {_fmt(z)} 0.025 pec")
    if rng.random() < cfg.geology.building_probability * intensity:
        x0 = rng.choice([rng.uniform(1, 8), rng.uniform(max(2, length-12), max(3, length-4))]); y = gy(x0)
        lines.append(f"#box: {_fmt(x0)} {_fmt(y)} 0 {_fmt(min(length-0.5,x0+rng.uniform(1.5,4.0)))} {_fmt(min(geom.domain_y-0.3,y+rng.uniform(1.0,3.0)))} {_fmt(geom.domain_z)} building_wall")
    return lines


def make_gprmax_input(cfg: AppConfig, geom: SceneGeometry, materials: Dict[str, Dict[str, float]], variant: str, case_id: str, rng: random.Random) -> str:
    lines = [
        f"#title: UavGPR-SimLab {case_id} {variant}",
        f"#domain: {_fmt(geom.domain_x)} {_fmt(geom.domain_y)} {_fmt(geom.domain_z)}",
        f"#dx_dy_dz: {_fmt(cfg.geometry.dx_m)} {_fmt(cfg.geometry.dx_m)} {_fmt(geom.domain_z)}",
        f"#time_window: {_fmt(cfg.radar.time_window_ns * 1e-9)}",
    ]
    lines += material_lines(materials)
    lines += _column_boxes(cfg, geom, variant, rng)
    lines += _clutter_lines(cfg, geom, variant, rng)
    lines += _antenna_lines(cfg, geom)
    lines.append(f"#geometry_view: 0 0 0 {_fmt(geom.domain_x)} {_fmt(geom.domain_y)} {_fmt(geom.domain_z)} {_fmt(max(cfg.geometry.dx_m,0.05))} {_fmt(max(cfg.geometry.dx_m,0.05))} {_fmt(max(geom.domain_z,cfg.geometry.dz_m))} geometry n")
    return "\n".join(lines) + "\n"


def _save_labels(dataset_dir: Path, case_id: str, cfg: AppConfig, geom: SceneGeometry, materials: Dict[str, Dict[str, float]]) -> tuple[Path, Path, Path]:
    label_json = dataset_dir / f"{case_id}_labels.json"
    iface_csv = dataset_dir / f"{case_id}_interface.csv"
    mask_npy = dataset_dir / f"{case_id}_mask.npy"
    label = {
        "case_id": case_id,
        "coordinate_note": "gprMax x horizontal, y vertical; interface_depth = ground_y - interface_y.",
        "x_m": geom.x.tolist(),
        "ground_y_m": geom.ground_y.tolist(),
        "interface_y_m": geom.interface_y.tolist(),
        "interface_depth_m": geom.interface_depth.tolist(),
        "materials": materials,
        "radar": asdict(cfg.radar),
        "geometry": asdict(cfg.geometry),
        "geology": asdict(cfg.geology),
    }
    label_json.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
    with iface_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f); wr.writerow(["x_m", "ground_y_m", "interface_y_m", "interface_depth_m"])
        for row in zip(geom.x, geom.ground_y, geom.interface_y, geom.interface_depth):
            wr.writerow([f"{float(v):.5f}" for v in row])
    nx = max(2, int(math.ceil(geom.domain_x / cfg.geometry.dx_m))); ny = max(2, int(math.ceil(geom.domain_y / cfg.geometry.dx_m)))
    mask = np.zeros((ny, nx), dtype=np.uint8)
    x_grid = np.linspace(0, geom.domain_x, nx); y_grid = np.linspace(0, geom.domain_y, ny)
    iy = np.interp(x_grid, geom.x, geom.interface_y)
    for j, y in enumerate(y_grid):
        mask[j, np.abs(y - iy) <= max(cfg.geometry.dx_m * 1.5, 0.1)] = 1
    np.save(mask_npy, mask)
    return label_json, iface_csv, mask_npy


def _rel_to_workspace(path: str | Path, workspace: str | Path) -> str:
    p = Path(path)
    root = Path(workspace).resolve()
    try:
        return p.resolve().relative_to(root).as_posix()
    except Exception:
        return p.as_posix()


def generate_cases(cfg: AppConfig, workspace: str | Path, cases: int | None = None) -> tuple[Path, Path]:
    dirs = ensure_project_dirs(workspace)
    rng = random.Random(cfg.geology.random_seed)
    n = int(cases if cases is not None else cfg.dataset.cases)
    if n <= 0:
        raise ValueError("cases must be a positive integer")
    vs = _variants(cfg.dataset.variants)
    if not vs:
        raise ValueError("dataset.variants must contain at least one variant")
    rows: List[CaseRecord] = []
    for i in range(n):
        case_id = f"case_{i+1:06d}"
        cdir = dirs["models"] / case_id; cdir.mkdir(parents=True, exist_ok=True)
        # v0.8.0-alpha.1: generate one SceneWorld per case and derive all
        # variants from it.  This guarantees raw/target/background/clutter/air
        # variants are homologous and differ only by component masking, not by
        # re-randomised geology or clutter.
        world = generate_scene_world(cfg, i, case_id=case_id)
        written = write_scene_world_case(cfg, world, cdir, dirs["datasets"], vs)
        label_json = Path(written["label_json"])
        iface_csv = Path(written["interface_csv"])
        mask_npy = Path(written["mask_npy"])
        model_preview_png = Path(written["model_preview_png"])
        scene_world_json = Path(written["scene_world_json"])
        metadata_summary_json = Path(written["metadata_summary_json"])
        interface_gt_npy = Path(written["interface_gt_npy"])
        layer_gt_npy = Path(written["layer_gt_npy"])
        layer_gt_x_axis_m_npy = Path(written.get("layer_gt_x_axis_m_npy", ""))
        layer_gt_y_axis_m_npy = Path(written.get("layer_gt_y_axis_m_npy", ""))
        raw_bscan_npy = Path(written.get("raw_bscan_npy", ""))
        target_bscan_npy = Path(written.get("target_only_bscan_npy", ""))
        background_bscan_npy = Path(written.get("background_only_bscan_npy", ""))
        clutter_bscan_npy = Path(written.get("clutter_only_bscan_npy", ""))
        air_bscan_npy = Path(written.get("air_only_bscan_npy", ""))
        clutter_gt_bscan_npy = cdir / "outputs" / "clutter_gt_bscan.npy"
        time_axis_ns_npy = Path(written.get("time_axis_ns_npy", ""))
        distance_axis_m_npy = Path(written.get("distance_axis_m_npy", ""))
        tx_x_axis_m_npy = Path(written.get("tx_x_axis_m_npy", ""))
        rx_x_axis_m_npy = Path(written.get("rx_x_axis_m_npy", ""))
        midpoint_x_axis_m_npy = Path(written.get("midpoint_x_axis_m_npy", ""))
        interface_mask_bscan_npy = Path(written.get("interface_mask_bscan_npy", ""))
        layer_mask_bscan_npy = Path(written.get("layer_mask_bscan_npy", ""))
        bscan_placeholder_status_json = Path(written.get("bscan_placeholder_status_json", ""))
        bscan_qc_report_json = Path(written.get("bscan_qc_report_json", ""))
        depths = np.asarray(world.bedrock_interface_depth_m, dtype=float)
        ground = np.asarray(world.ground_y, dtype=float)
        for variant in vs:
            p = cdir / f"{variant}.in"
            preview_png = Path(written["variant_previews"].get(variant, model_preview_png))
            rows.append(CaseRecord(
                case_id=case_id,
                variant=variant,
                input_file=_rel_to_workspace(p, workspace),
                split=_split(i, n, cfg),
                n_traces=int(cfg.geometry.trace_count),
                trace_step_m=float(cfg.geometry.trace_step_m),
                model_length_m=float(world.model_length_actual_m),
                dx_m=float(cfg.geometry.dx_m),
                time_window_ns=float(cfg.radar.time_window_ns),
                center_frequency_mhz=float(cfg.radar.center_frequency_mhz),
                flight_height_m=float(cfg.radar.nominal_flight_height_m),
                interface_depth_mean_m=float(np.mean(depths)),
                interface_depth_min_m=float(np.min(depths)),
                interface_depth_max_m=float(np.max(depths)),
                slope_deg=float(np.degrees(np.arctan2(float(ground[-1] - ground[0]), max(float(world.ground_x[-1] - world.ground_x[0]), 1.0)))),
                clutter_level=_family_clutter_level(world.family, getattr(cfg.geology, "clutter_level", "medium")),
                label_json=_rel_to_workspace(label_json, workspace),
                interface_csv=_rel_to_workspace(iface_csv, workspace),
                mask_npy=_rel_to_workspace(mask_npy, workspace),
                family=world.family,
                random_seed=world.random_seed,
                model_length_config_m=float(world.model_length_config_m),
                model_length_actual_m=float(world.model_length_actual_m),
                domain_x_m=float(world.domain_x_m),
                scan_start_x_m=float(world.trajectory.scan_start_x_m),
                scan_end_x_m=float(world.trajectory.scan_end_x_m),
                flight_height_mode=world.trajectory.mode,
                scene_world_json=_rel_to_workspace(scene_world_json, workspace),
                metadata_summary_json=_rel_to_workspace(metadata_summary_json, workspace),
                interface_gt_npy=_rel_to_workspace(interface_gt_npy, workspace),
                layer_gt_npy=_rel_to_workspace(layer_gt_npy, workspace),
                model_preview_png=_rel_to_workspace(model_preview_png, workspace),
                variant_preview_png=_rel_to_workspace(preview_png, workspace),
                bscan_npy=_rel_to_workspace(Path(written.get(f"{variant}_bscan_npy", preview_png)), workspace),
                raw_bscan_npy=_rel_to_workspace(raw_bscan_npy, workspace),
                target_bscan_npy=_rel_to_workspace(target_bscan_npy, workspace),
                background_bscan_npy=_rel_to_workspace(background_bscan_npy, workspace),
                clutter_bscan_npy=_rel_to_workspace(clutter_bscan_npy, workspace),
                air_bscan_npy=_rel_to_workspace(air_bscan_npy, workspace),
                clutter_gt_bscan_npy=_rel_to_workspace(clutter_gt_bscan_npy, workspace),
                time_axis_ns_npy=_rel_to_workspace(time_axis_ns_npy, workspace),
                distance_axis_m_npy=_rel_to_workspace(distance_axis_m_npy, workspace),
                distance_axis_role="midpoint_x",
                tx_x_axis_m_npy=_rel_to_workspace(tx_x_axis_m_npy, workspace),
                rx_x_axis_m_npy=_rel_to_workspace(rx_x_axis_m_npy, workspace),
                midpoint_x_axis_m_npy=_rel_to_workspace(midpoint_x_axis_m_npy, workspace),
                layer_gt_x_axis_m_npy=_rel_to_workspace(layer_gt_x_axis_m_npy, workspace),
                layer_gt_y_axis_m_npy=_rel_to_workspace(layer_gt_y_axis_m_npy, workspace),
                interface_mask_bscan_npy=_rel_to_workspace(interface_mask_bscan_npy, workspace),
                layer_mask_bscan_npy=_rel_to_workspace(layer_mask_bscan_npy, workspace),
                bscan_placeholder_status_json=_rel_to_workspace(bscan_placeholder_status_json, workspace),
                bscan_qc_report_json=_rel_to_workspace(bscan_qc_report_json, workspace),
                bscan_status="not_run",
                bscan_error="",
                is_ml_pair_valid="true",
            ))
    manifest = dirs["datasets"] / f"{Path(workspace).name}_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        wr.writeheader(); [wr.writerow(asdict(r)) for r in rows]
    dataset_grade, not_for_training = _dataset_grade_for_workspace(workspace)
    summary = {
        "workspace": ".",
        "manifest": _rel_to_workspace(manifest, workspace),
        "manifest_paths": "relative_to_workspace",
        "case_count": n,
        "variants": vs,
        "records": len(rows),
        "trace_count_per_input": cfg.geometry.trace_count,
        "scene_world_schema": "uavgpr_simlab.scene_world.v1alpha1",
        "homologous_variants": True,
        "samples": int(cfg.radar.frequency_points),
        "time_window_ns": float(cfg.radar.time_window_ns),
        "center_frequency_mhz": float(cfg.radar.center_frequency_mhz),
        "dataset_grade": dataset_grade,
        "ready_to_run": True,
        "not_ready_to_train": bool(not_for_training),
        "training_ready": False,
        "bscan_file_contract": {
            "canonical_variant_files": {
                "raw": "outputs/raw_bscan.npy",
                "target_only": "outputs/target_only_bscan.npy",
                "background_only": "outputs/background_only_bscan.npy",
                "clutter_only": "outputs/clutter_only_bscan.npy",
                "air_only": "outputs/air_only_bscan.npy",
                "clutter_gt": "outputs/clutter_gt_bscan.npy",
            },
            "distance_axis_role": "midpoint_x",
        },
        "bscan_run_status": {"status": "not_run", "success_cases": 0, "failed_cases": 0},
    }
    (dirs["reports"] / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_smoke_readme(workspace, manifest, summary)
    return dirs["models"], manifest


def write_manifest_commands_bat(
    manifest_csv: str | Path,
    out_bat: str | Path,
    conda_env: str = "gprMax",
    gpu: bool = True,
    gpu_ids: Sequence[int] | None = None,
    geometry_only: bool = False,
    variants: Sequence[str] | None = None,
    max_tasks: int = 0,
    safe_runner: bool = True,
    workspace: str | Path | None = None,
    skip_completed: bool = True,
    force: bool = False,
    postprocess: bool = False,
    python_executable: str = "python",
) -> Path:
    """Write a Windows BAT for running manifest tasks.

    v0.4 defaults to the registry-backed safe runner. It records successful jobs
    under workspace/jobs/done and skips matching completed jobs on rerun.
    """
    from .runner import GprMaxRunOptions, build_gprmax_command, command_to_string

    out = Path(out_bat); out.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(variants or [])
    ws = Path(workspace) if workspace else Path(manifest_csv).resolve().parent.parent
    lines = [
        "@echo off",
        "setlocal EnableExtensions",
        'pushd "%~dp0\\.."',
        "echo UavGPR-SimLab gprMax batch started",
        'set "WORKSPACE=%CD%"',
        'set "PROJECT_ROOT=%WORKSPACE%\\..\\.."',
        'call "%PROJECT_ROOT%\\scripts\\windows_runtime_bootstrap.bat"',
        'if errorlevel 1 exit /b 1',
        ""
    ]
    count = 0
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if wanted and row.get("variant") not in wanted:
                continue
            if safe_runner:
                input_path = Path(row["input_file"])
                if not input_path.is_absolute():
                    input_path = (ws / input_path).resolve()
                cmd = [
                    "%PY_RUN%",
                    "-m", "uavgpr_simlab.cli", "run-one",
                    "--input-file", str(input_path),
                    "--workspace", str(ws),
                    "--case-id", row.get("case_id", ""),
                    "--variant", row.get("variant", "raw"),
                    "--n-traces", str(int(float(row.get("n_traces", 1) or 1))),
                    "--conda-env", conda_env,
                    "--gpu-ids", ",".join(str(x) for x in list(gpu_ids or [0])),
                ]
                if not gpu:
                    cmd.append("--no-gpu")
                if geometry_only:
                    cmd.append("--geometry-only")
                if postprocess:
                    cmd.append("--postprocess")
                    cmd += ["--postprocess-out-dir", str(ws / "outputs" / "gprmax_qc")]
                if not skip_completed:
                    cmd.append("--no-skip-completed")
                if force:
                    cmd.append("--force")
                command = command_to_string(cmd)
            else:
                input_path = Path(row["input_file"])
                if not input_path.is_absolute():
                    input_path = (ws / input_path).resolve()
                opts = GprMaxRunOptions(str(input_path), int(row.get("n_traces", 1) or 1), conda_env=conda_env, use_conda_run=True, use_gpu=gpu, gpu_ids=list(gpu_ids or [0]), geometry_only=geometry_only, geometry_fixed=True)
                command = command_to_string(build_gprmax_command(opts))
            lines += [f"echo Running {row.get('case_id')} {row.get('variant')}", command, "if errorlevel 1 goto :error", ""]
            count += 1
            if max_tasks and count >= max_tasks:
                break
    lines += ["echo Done.", "goto :eof", ":error", "echo gprMax failed. Check the log above.", "exit /b 1", ""]
    out.write_text("\r\n".join(lines), encoding="utf-8")
    return out
