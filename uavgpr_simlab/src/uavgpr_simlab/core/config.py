from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

import yaml

T = TypeVar("T")


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return repo_root() / "configs"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class RuntimeConfig:
    project_root: str = "workspace/default_project"
    conda_env_gprmax: str = ""
    gprmax_source_dir: str = ""
    python_executable: str = "python"
    use_conda_run: bool = False
    gpu_enabled: bool = False
    gpu_ids: str = "0"
    mpi_tasks: int = 0
    omp_threads: int = 1
    geometry_only_first: bool = True
    auto_merge_outputs: bool = True
    geometry_fixed: bool = True
    write_processed: bool = False


@dataclass
class RadarConfig:
    radar_name: str = "CDUT-UavGPR UG10"
    frequency_start_mhz: float = 20.0
    frequency_stop_mhz: float = 170.0
    center_frequency_mhz: float = 100.0
    intermediate_bandwidth_khz: float = 10.0
    frequency_step_mhz: float = 1.0
    frequency_points: int = 501
    time_window_ns: float = 700.0
    tx_gain_dbm: float = 36.0
    trace_rate_hz: float = 10.0
    flight_speed_mps: float = 1.0
    nominal_flight_height_m: float = 8.0
    flight_height_min_m: float = 2.0
    flight_height_max_m: float = 20.0
    tx_rx_offset_m: float = 1.4
    antenna_length_m: float = 1.4
    antenna_model: str = "wire_dipole_equivalent"
    simulation_mode: str = "both"


@dataclass
class GeometryConfig:
    dimension: str = "2D_patch"
    model_length_m: float = 48.0
    subsurface_depth_m: float = 24.0
    y_thickness_m: float = 0.05
    dx_m: float = 0.08
    dy_m: float = 0.05
    dz_m: float = 0.08
    trace_step_m: float = 0.20
    trace_count: int = 180
    air_margin_m: float = 1.0
    pml_cells: int = 10
    geometry_column_width_m: float = 0.40


@dataclass
class GeologyConfig:
    scenario_family: str = "landslide_bedrock_interface"
    interface_depth_min_m: float = 3.7
    interface_depth_max_m: float = 16.3
    interface_roughness_m: float = 1.5
    slope_deg_min: float = 0.0
    slope_deg_max: float = 25.0
    terrace_probability: float = 0.35
    water_table_probability: float = 0.30
    blocky_gravel_probability: float = 0.55
    paddy_water_probability: float = 0.35
    overhead_wire_probability: float = 0.35
    tree_probability: float = 0.45
    building_probability: float = 0.12
    anomaly_probability: float = 0.25
    clutter_level: str = "medium"
    random_seed: int = 20250301


@dataclass
class DatasetConfig:
    cases: int = 10
    variants: List[str] = field(default_factory=lambda: ["raw", "target_only", "clutter_only", "background_only", "air_only"])
    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15
    export_npz: bool = True
    export_hdf5: bool = True
    export_png: bool = True
    keep_gprmax_out: bool = True
    paper_mode: bool = True


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    geology: GeologyConfig = field(default_factory=GeologyConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    domain_randomization: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _dataclass_from_dict(cls: Type[T], data: Dict[str, Any] | None) -> T:
    obj = cls()  # type: ignore[call-arg]
    if not data:
        return obj
    valid = {f.name for f in fields(obj)} if is_dataclass(obj) else set()
    for k, v in data.items():
        if k in valid:
            setattr(obj, k, v)
    return obj


def app_config_from_dict(data: Dict[str, Any]) -> AppConfig:
    return AppConfig(
        runtime=_dataclass_from_dict(RuntimeConfig, data.get("runtime", {})),
        radar=_dataclass_from_dict(RadarConfig, data.get("radar", {})),
        geometry=_dataclass_from_dict(GeometryConfig, data.get("geometry", {})),
        geology=_dataclass_from_dict(GeologyConfig, data.get("geology", {})),
        dataset=_dataclass_from_dict(DatasetConfig, data.get("dataset", {})),
        domain_randomization=data.get("domain_randomization", {}),
    )


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        return AppConfig()
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = load_yaml(p)
    return app_config_from_dict(data)


def save_config(cfg: AppConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict()
    if p.suffix.lower() == ".json":
        save_json(data, p)
    else:
        save_yaml(data, p)


def ensure_project_dirs(project_root: str | Path) -> Dict[str, Path]:
    root = Path(project_root)
    dirs = {
        "root": root,
        "models": root / "models",
        "datasets": root / "datasets",
        "real_data": root / "real_data",
        "outputs": root / "outputs",
        "reports": root / "reports",
        "logs": root / "logs",
        "configs": root / "configs",
        "exports": root / "exports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

# -----------------------------------------------------------------------------
# Lightweight local environment persistence used by the GUI launchers.
# The file is intentionally plain KEY=VALUE so it can also be edited by hand.
# -----------------------------------------------------------------------------
def simlab_env_path() -> Path:
    return repo_root() / ".simlab_env"


def read_simlab_env(path: str | Path | None = None) -> Dict[str, str]:
    p = Path(path) if path else simlab_env_path()
    out: Dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out




def _default_runtime_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    import os
    env_root = os.environ.get("UAVGPR_RUNTIME_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    if os.name == "nt":
        candidates.extend([Path("D:/UavGPR_Runtime"), Path("E:/UavGPR_Runtime")])
    candidates.append(repo_root() / "UavGPR_Runtime")
    out: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def read_persistent_runtime_env() -> Dict[str, str]:
    """Read persistent RuntimeRoot settings shared by multiple release folders."""

    merged: Dict[str, str] = {}
    for root in _default_runtime_root_candidates():
        env_file = root / "uavgpr_runtime.env"
        if env_file.exists():
            merged.update(read_simlab_env(env_file))
    return merged


def read_combined_simlab_env() -> Dict[str, str]:
    """Merge persistent RuntimeRoot settings with project-local .simlab_env.

    Project-local settings win when both files define the same key.
    """

    merged = read_persistent_runtime_env()
    merged.update(read_simlab_env())
    return merged

def write_simlab_env(values: Dict[str, str], path: str | Path | None = None) -> Path:
    p = Path(path) if path else simlab_env_path()
    old = read_simlab_env(p)
    old.update({str(k): str(v) for k, v in values.items()})
    lines = ["# UavGPR-SimLab local settings. Safe to edit."]
    for k in sorted(old):
        lines.append(f"{k}={old[k]}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
