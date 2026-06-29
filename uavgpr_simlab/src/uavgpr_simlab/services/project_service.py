from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uavgpr_simlab.cli import _cfg_from_plan
from uavgpr_simlab.core.config import AppConfig, load_yaml
from uavgpr_simlab.core.scenario import generate_cases
from uavgpr_simlab.services.easy_batch_service import read_manifest_rows


@dataclass(frozen=True)
class ModelGenerationResult:
    """Generated model batch information for the easy GUI."""

    cfg: AppConfig
    model_root: Path
    models: list[Any]
    manifest: Path
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class ModelPlanPreset:
    """A selectable run-plan preset for one-click model generation."""

    label: str
    path: Path
    plan_name: str
    scene_count: int | None
    trace_count: int | None
    dx_m: float | None
    domain_depth_m: float | None
    source: str = "configs"


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_plan_label(path: Path, data: dict[str, Any]) -> ModelPlanPreset:
    plan_name = str(data.get("plan_name") or path.stem)
    scene = data.get("scene") if isinstance(data.get("scene"), dict) else {}
    scene_count = _as_int(data.get("scene_count"))
    trace_count = _as_int(scene.get("trace_count"))
    dx_m = _as_float(scene.get("dx_m"))
    domain_depth_m = _as_float(scene.get("domain_depth_m"))

    bits = [plan_name]
    if scene_count is not None:
        bits.append(f"默认 {scene_count} 个场景")
    if trace_count is not None:
        bits.append(f"{trace_count} 道")
    if dx_m is not None:
        bits.append(f"dx={dx_m:g}m")
    if domain_depth_m is not None:
        bits.append(f"深度 {domain_depth_m:g}m")
    bits.append(path.name)

    return ModelPlanPreset(
        label=" | ".join(bits),
        path=path,
        plan_name=plan_name,
        scene_count=scene_count,
        trace_count=trace_count,
        dx_m=dx_m,
        domain_depth_m=domain_depth_m,
    )


def discover_model_plan_presets(root_dir: str | Path) -> list[ModelPlanPreset]:
    """Discover run-plan YAML files that can be selected before model generation.

    Only ``configs/run_plan*.yaml`` and ``configs/run_plan*.yml`` are treated as
    model-generation presets. Other YAML files under ``configs/`` may describe
    materials, ML training or automation pipelines and should not be offered as
    one-click model-generation plans.
    """

    root = Path(root_dir).expanduser()
    configs = root / "configs"
    candidates = sorted({*configs.glob("run_plan*.yaml"), *configs.glob("run_plan*.yml")})
    presets: list[ModelPlanPreset] = []
    for path in candidates:
        try:
            data = load_yaml(path)
            if not isinstance(data, dict):
                continue
            if "scene" not in data or "plan_name" not in data:
                continue
            presets.append(_format_plan_label(path, data))
        except Exception:
            # A broken preset should not prevent the GUI from starting. The user
            # can still select the file manually and see the detailed error in
            # the plan preview panel.
            continue
    return presets


def preview_plan_yaml(plan_path: str | Path) -> dict[str, Any]:
    """Load a run-plan YAML for display in the project page."""

    return load_yaml(Path(plan_path).expanduser())


def find_latest_manifest(workspace: str | Path) -> Path | None:
    """Find the newest generated manifest under a workspace.

    The easy GUI uses this when the manifest field is empty and the user clicks
    "加载模型图库".  Generated manifests live under::

        <workspace>/datasets/<workspace-name>_manifest.csv

    The fallback glob keeps compatibility with manually renamed manifests.
    """

    ws = Path(workspace).expanduser()
    datasets = ws / "datasets"
    if datasets.exists():
        preferred = datasets / f"{ws.name}_manifest.csv"
        if preferred.exists():
            return preferred
        candidates = sorted(
            datasets.glob("*manifest*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    # Some run plans create a named project directory below the selected
    # workspace, for example <workspace>/3060_quick_smoke/datasets/*.csv.
    nested = sorted(
        ws.glob("*/datasets/*manifest*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return nested[0] if nested else None


def generate_model_batch(plan_path: str | Path, workspace: str | Path, case_count: int) -> ModelGenerationResult:
    """Generate models from the selected plan and return GUI-ready metadata."""

    cfg = _cfg_from_plan(Path(plan_path).expanduser(), Path(workspace).expanduser())
    cfg.dataset.cases = int(case_count)
    model_root, manifest = generate_cases(cfg, cfg.runtime.project_root, cfg.dataset.cases)
    manifest_path = Path(manifest)
    rows = read_manifest_rows(manifest_path)

    # core.scenario.generate_cases() returns (models_directory, manifest_path), not
    # an iterable of model objects.  Count unique case ids from the manifest so
    # the GUI reports the actual number of generated models rather than the
    # number of input variants.
    seen: set[str] = set()
    models: list[dict[str, str]] = []
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id and case_id not in seen:
            seen.add(case_id)
            models.append(row)

    return ModelGenerationResult(
        cfg=cfg,
        model_root=Path(model_root),
        models=models,
        manifest=manifest_path,
        rows=rows,
    )
