from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

STANDARD_SCENEWORLD_VARIANTS = ["raw", "target_only", "background_only", "clutter_only", "air_only"]
STANDARD_SCENEWORLD_VARIANT_TEXT = ",".join(STANDARD_SCENEWORLD_VARIANTS)


@dataclass(frozen=True)
class SimulationRunProfile:
    """A reusable batch-run profile, not a dedicated GUI button.

    The GUI should expose profiles in a combo box and run them through the same
    batch runner.  This avoids adding one button per dataset type.
    """

    key: str
    title: str
    workspace_name: str = ""
    variants: tuple[str, ...] = tuple(STANDARD_SCENEWORLD_VARIANTS)
    max_cases: int = 0
    max_tasks_hint: int = 0
    one_case_per_family: bool = False
    allow_resample: bool = False
    timeout_sec: int = 3600
    strict_qc: bool = True
    qc_mode: str = "strict_native"
    description: str = ""


def default_simulation_run_profiles() -> list[SimulationRunProfile]:
    return [
        SimulationRunProfile(
            key="custom",
            title="自定义：使用当前 manifest 和参数",
            workspace_name="",
            max_cases=0,
            max_tasks_hint=60,
            one_case_per_family=False,
            allow_resample=False,
            timeout_sec=3600,
            strict_qc=True,
            description="不改当前清单；适合后续 pilot/formal 或手工选择数据集。",
        ),
        SimulationRunProfile(
            key="ultra_tiny_check",
            title="最小链路验证：ultra tiny（1 case × 5 variants）",
            workspace_name="yingshan_sceneworld_ultra_tiny_v080a14",
            max_cases=1,
            max_tasks_hint=5,
            one_case_per_family=False,
            allow_resample=True,
            timeout_sec=600,
            strict_qc=False,
            qc_mode="chain_resample",
            description="只验证 .in → gprMax → .out → .npy → QC → clutter_gt；允许显式重采样，不作为训练数据。",
        ),
        SimulationRunProfile(
            key="smoke_25run",
            title="五类场景 smoke：25-run（5 cases × 5 variants）",
            workspace_name="yingshan_sceneworld_smoke_v080a14",
            max_cases=0,
            max_tasks_hint=25,
            one_case_per_family=True,
            allow_resample=True,
            timeout_sec=3600,
            strict_qc=False,
            qc_mode="chain_resample",
            description="正式 pilot 前验证五类营山场景链路；gprMax 原生输出会显式保存并重采样到 manifest 目标网格，报告中标记 resampled，不作为训练数据。",
        ),
        SimulationRunProfile(
            key="pilot_4090",
            title="4090 pilot：使用当前选择的数据集（严格 QC）",
            workspace_name="",
            max_cases=0,
            max_tasks_hint=0,
            one_case_per_family=False,
            allow_resample=False,
            timeout_sec=7200,
            strict_qc=True,
            qc_mode="strict_native",
            description="预留给 4090 正式 pilot；不绑定内置小骨架，必须由用户选择 manifest。正式数据默认严格 QC，不做自动重采样。",
        ),
    ]


def profile_by_key(key: str) -> SimulationRunProfile:
    for p in default_simulation_run_profiles():
        if p.key == key:
            return p
    return default_simulation_run_profiles()[0]


def find_dataset_manifest(root_dir: str | Path, workspace_name: str) -> Path | None:
    if not workspace_name:
        return None
    workspace = Path(root_dir) / "workspace" / workspace_name
    manifests = sorted((workspace / "datasets").glob("*_manifest.csv"))
    return manifests[-1] if manifests else None


def manifest_is_sceneworld(manifest: str | Path) -> bool:
    try:
        with Path(manifest).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
    except Exception:
        return False
    required_any = {"scene_world_json", "bscan_qc_report_json", "raw_bscan_npy", "target_bscan_npy"}
    return bool(required_any & fields)


def normalize_variants(variants: Iterable[str] | str | None) -> list[str]:
    if variants is None:
        return list(STANDARD_SCENEWORLD_VARIANTS)
    if isinstance(variants, str):
        parts = variants.replace(";", ",").split(",")
    else:
        parts = list(variants)
    out: list[str] = []
    for item in parts:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out or list(STANDARD_SCENEWORLD_VARIANTS)
