"""Portable repository runtime configuration.

Machine-specific executable locations are intentionally kept outside Git. This
module resolves one local JSON profile plus explicit environment overrides while
leaving all project-relative data paths rooted at the checked-out repository.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "environment"
DEFAULT_PROFILE = RUNTIME_DIR / "project_runtime.local.json"


class RuntimeConfigError(RuntimeError):
    """Raised when a required machine-local runtime value is absent or invalid."""


@dataclass(frozen=True)
class RuntimeProfile:
    profile_path: Path | None
    profile_name: str
    project_python: Path
    gprmax_python: Path | None
    gprmax_source: Path | None
    gprmax_vcvars: Path | None
    gpu_index: int
    cuda_visible_devices: str | None
    output_root: Path
    scratch_root: Path
    solver_run_root: Path


def _optional_path(value: object, *, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path)


def _read_profile(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeConfigError(f"invalid runtime profile {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"runtime profile {path} must be a JSON object")
    if value.get("schema_version", 1) != 1:
        raise RuntimeConfigError(f"unsupported runtime profile schema: {value.get('schema_version')!r}")
    return value


def _profile_path(profile_path: Path | None) -> Path | None:
    configured = os.environ.get("PGDA_RUNTIME_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    if profile_path is not None:
        return profile_path.expanduser().resolve()
    return DEFAULT_PROFILE if DEFAULT_PROFILE.is_file() else None


def load_runtime(profile_path: Path | None = None) -> RuntimeProfile:
    """Load a local profile and explicit environment overrides."""
    path = _profile_path(profile_path)
    payload = _read_profile(path)

    def value(key: str, env_key: str, default: object = None) -> object:
        return os.environ.get(env_key, payload.get(key, default))

    project_python = _optional_path(value("project_python", "PGDA_PROJECT_PYTHON", sys.executable), root=ROOT)
    assert project_python is not None
    gpu_value = value("gpu_index", "PGDA_GPU_INDEX", 0)
    try:
        gpu_index = int(gpu_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"gpu_index must be an integer, got {gpu_value!r}") from exc
    return RuntimeProfile(
        profile_path=path,
        profile_name=str(payload.get("profile_name") or "unconfigured-machine"),
        project_python=project_python,
        gprmax_python=_optional_path(value("gprmax_python", "PGDA_GPRMAX_PYTHON"), root=ROOT),
        gprmax_source=_optional_path(value("gprmax_source", "PGDA_GPRMAX_ROOT"), root=ROOT),
        gprmax_vcvars=_optional_path(value("gprmax_vcvars", "PGDA_GPRMAX_VCVARS"), root=ROOT),
        gpu_index=gpu_index,
        cuda_visible_devices=(str(value("cuda_visible_devices", "CUDA_VISIBLE_DEVICES", "")).strip() or None),
        output_root=_optional_path(value("output_root", "PGDA_OUTPUT_ROOT", "outputs"), root=ROOT) or ROOT / "outputs",
        scratch_root=_optional_path(value("scratch_root", "PGDA_SCRATCH_ROOT", "workspace"), root=ROOT) or ROOT / "workspace",
        solver_run_root=_optional_path(value("solver_run_root", "PGDA_SOLVER_RUN_ROOT", "data/PGDA_SYNTH_DATASET_V2/01_solver_runs"), root=ROOT)
        or ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "01_solver_runs",
    )


def require_gprmax(profile: RuntimeProfile) -> tuple[Path, Path]:
    """Return verified gprMax interpreter/source locations."""
    if profile.gprmax_python is None or profile.gprmax_source is None:
        raise RuntimeConfigError(
            "gprMax is not configured. Fill environment/project_runtime.local.json "
            "or set PGDA_GPRMAX_PYTHON and PGDA_GPRMAX_ROOT."
        )
    if not profile.gprmax_python.is_file():
        raise RuntimeConfigError(f"gprMax Python does not exist: {profile.gprmax_python}")
    if not (profile.gprmax_source / "gprMax").is_dir():
        raise RuntimeConfigError(f"gprMax source tree does not contain gprMax/: {profile.gprmax_source}")
    return profile.gprmax_python, profile.gprmax_source


def profile_summary(profile: RuntimeProfile) -> dict[str, object]:
    return {
        "profile_path": str(profile.profile_path) if profile.profile_path else None,
        "profile_name": profile.profile_name,
        "project_root": str(ROOT),
        "project_python": str(profile.project_python),
        "gprmax_python": str(profile.gprmax_python) if profile.gprmax_python else None,
        "gprmax_source": str(profile.gprmax_source) if profile.gprmax_source else None,
        "gpu_index": profile.gpu_index,
        "output_root": str(profile.output_root),
        "scratch_root": str(profile.scratch_root),
        "solver_run_root": str(profile.solver_run_root),
    }
