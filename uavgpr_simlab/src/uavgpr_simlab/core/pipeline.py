from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

from .config import AppConfig, ensure_project_dirs, load_config, load_yaml, save_config
from .environment import run_environment_checks, save_report
from .hpc import write_local_bash_script, write_slurm_array_script
from .real_data import convert_real_csv
from .reporting import build_auto_report
from .scenario import generate_cases, write_manifest_commands_bat
from .softmask import generate_borehole_soft_mask


def _cfg_from_plan_inline(plan_path: str | Path, workspace: str | Path) -> AppConfig:
    # Kept local to avoid importing CLI from core and creating circular dependencies.
    from uavgpr_simlab.cli import _cfg_from_plan

    return _cfg_from_plan(plan_path, workspace)


def _as_bool(data: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(data.get(key, default))


def _project_out_path(value: Any, workspace: Path, default: Path) -> Path:
    """Resolve output paths against the active workspace.

    If a template contains paths like ``workspace/old_name/logs/x.sh`` and
    the user changes only the top-level ``workspace`` value, this function
    rewrites the path under the active workspace. Input paths such as
    ``sample_data/...`` are intentionally not passed here.
    """
    if value is None or str(value).strip() == "":
        return default
    p = Path(str(value))
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "workspace":
        # Rewrite template paths to the active workspace.
        # "workspace/old_project/logs/x.sh" -> active/logs/x.sh
        # "workspace/logs/x.sh"             -> active/logs/x.sh
        known_roots = {"models", "datasets", "real_data", "outputs", "reports", "logs", "configs", "exports", "paper", "jobs", "previews"}
        if len(parts) >= 3 and parts[1] not in known_roots:
            rest = parts[2:]
        else:
            rest = parts[1:]
        return workspace.joinpath(*rest) if rest else workspace
    return p


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def run_automation_pipeline(config_path: str | Path, dry_run: bool = False) -> dict[str, Any]:
    """Run a reproducible automation pipeline described by YAML.

    The default mode is safe: it generates inputs, command scripts, soft masks and
    reports, but does not execute gprMax unless ``run_solver: true`` is specified.
    This avoids accidental long HPC jobs from a paper-preparation wrapper.
    """

    cfg_file = Path(config_path)
    plan = load_yaml(cfg_file)
    workspace = Path(plan.get("workspace", "workspace/pipeline_run"))
    ensure_project_dirs(workspace)
    result: dict[str, Any] = {"config": str(cfg_file.resolve()), "workspace": str(workspace.resolve()), "dry_run": bool(dry_run), "steps": []}

    if _as_bool(plan.get("environment", {}), "enabled", False):
        env_cfg = plan.get("environment", {})
        out = workspace / "reports" / "environment_report.json"
        if dry_run:
            result["steps"].append({"step": "environment", "planned_out": str(out)})
        else:
            rep = run_environment_checks(env_cfg.get("conda_env", "gprMax"), not env_cfg.get("no_conda_run", False), env_cfg.get("gprmax_root", ""))
            save_report(rep, out)
            result["steps"].append({"step": "environment", "report": str(out.resolve()), "ok": rep.ok})

    manifest_path: Path | None = None
    gen_cfg = plan.get("generate", {})
    if _as_bool(gen_cfg, "enabled", True):
        if gen_cfg.get("plan"):
            cfg = _cfg_from_plan_inline(gen_cfg["plan"], workspace)
        elif gen_cfg.get("config"):
            cfg = load_config(gen_cfg["config"])
            cfg.runtime.project_root = str(workspace)
        else:
            cfg = AppConfig()
            cfg.runtime.project_root = str(workspace)
        if gen_cfg.get("count") is not None:
            cfg.dataset.cases = int(gen_cfg["count"])
        cfg_save = workspace / "configs" / "pipeline_generated_config.yaml"
        if dry_run:
            result["steps"].append({"step": "generate", "planned_cases": cfg.dataset.cases, "workspace": str(workspace)})
        else:
            _, manifest = generate_cases(cfg, workspace, cases=cfg.dataset.cases)
            save_config(cfg, cfg_save)
            manifest_path = manifest
            result["steps"].append({"step": "generate", "manifest": str(manifest.resolve()), "config": str(cfg_save.resolve())})

    commands_cfg = plan.get("commands", {})
    if _as_bool(commands_cfg, "enabled", True):
        manifest_for_commands = Path(commands_cfg.get("manifest", manifest_path or workspace / "datasets" / f"{workspace.name}_manifest.csv"))
        bat_out = _project_out_path(commands_cfg.get("out_bat", None), workspace, workspace / "logs" / "run_all_gprmax.bat")
        if dry_run:
            result["steps"].append({"step": "commands", "planned_manifest": str(manifest_for_commands), "planned_bat": str(bat_out)})
        else:
            variants = [x.strip() for x in str(commands_cfg.get("variants", "raw")).replace(";", ",").split(",") if x.strip()]
            gpu_ids = [int(x) for x in str(commands_cfg.get("gpu_ids", "0")).replace(";", ",").split(",") if x.strip()]
            bat = write_manifest_commands_bat(
                manifest_for_commands,
                bat_out,
                conda_env=commands_cfg.get("conda_env", "gprMax"),
                gpu=not commands_cfg.get("no_gpu", False),
                gpu_ids=gpu_ids,
                geometry_only=commands_cfg.get("geometry_only", False),
                variants=variants,
                max_tasks=int(commands_cfg.get("max_tasks", 0) or 0),
            )
            result["steps"].append({"step": "commands", "bat": str(bat.resolve())})

    hpc_cfg = plan.get("hpc", {})
    if _as_bool(hpc_cfg, "enabled", False):
        manifest_for_hpc = Path(hpc_cfg.get("manifest", manifest_path or workspace / "datasets" / f"{workspace.name}_manifest.csv"))
        mode = str(hpc_cfg.get("mode", "slurm")).lower()
        if dry_run:
            result["steps"].append({"step": "hpc", "mode": mode, "planned_manifest": str(manifest_for_hpc)})
        else:
            if mode == "local":
                rep = write_local_bash_script(
                    manifest_for_hpc,
                    _project_out_path(hpc_cfg.get("out_sh", None), workspace, workspace / "logs" / "run_gprmax_local.sh"),
                    variants=hpc_cfg.get("variants", "raw"),
                    conda_env=hpc_cfg.get("conda_env", "gprMax"),
                    gpu_ids=hpc_cfg.get("gpu_ids", "0"),
                    no_gpu=bool(hpc_cfg.get("no_gpu", False)),
                    geometry_only=bool(hpc_cfg.get("geometry_only", False)),
                    max_tasks=int(hpc_cfg.get("max_tasks", 0) or 0),
                    safe_runner=bool(hpc_cfg.get("safe_runner", True)),
                    workspace=str(workspace.resolve()),
                    gprmax_root=hpc_cfg.get("gprmax_root", ""),
                    skip_completed=bool(hpc_cfg.get("skip_completed", True)),
                    force=bool(hpc_cfg.get("force", False)),
                    postprocess=bool(hpc_cfg.get("postprocess", False)),
                    postprocess_out_dir=hpc_cfg.get("postprocess_out_dir", str(workspace / "outputs" / "gprmax_qc")),
                )
            else:
                rep = write_slurm_array_script(
                    manifest_for_hpc,
                    _project_out_path(hpc_cfg.get("out_sh", None), workspace, workspace / "logs" / "run_gprmax_slurm_array.sh"),
                    variants=hpc_cfg.get("variants", "raw"),
                    conda_env=hpc_cfg.get("conda_env", "gprMax"),
                    gprmax_root=hpc_cfg.get("gprmax_root", ""),
                    partition=hpc_cfg.get("partition", "gpu"),
                    time_limit=hpc_cfg.get("time_limit", "12:00:00"),
                    gpus_per_task=int(hpc_cfg.get("gpus_per_task", 1) or 1),
                    no_gpu=bool(hpc_cfg.get("no_gpu", False)),
                    cpus_per_task=int(hpc_cfg.get("cpus_per_task", 4) or 4),
                    mem=hpc_cfg.get("mem", "24G"),
                    gpu_ids=hpc_cfg.get("gpu_ids", "0"),
                    geometry_only=bool(hpc_cfg.get("geometry_only", False)),
                    array_parallelism=int(hpc_cfg.get("array_parallelism", 4) or 4),
                    max_tasks=int(hpc_cfg.get("max_tasks", 0) or 0),
                    postprocess=bool(hpc_cfg.get("postprocess", False)),
                    postprocess_out_dir=hpc_cfg.get("postprocess_out_dir", str(workspace / "outputs" / "gprmax_qc")),
                    safe_runner=bool(hpc_cfg.get("safe_runner", True)),
                    workspace=str(workspace.resolve()),
                    skip_completed=bool(hpc_cfg.get("skip_completed", True)),
                    force=bool(hpc_cfg.get("force", False)),
                )
            result["steps"].append({"step": "hpc", **rep})

    real_cfg = plan.get("real_csv", {})
    if _as_bool(real_cfg, "enabled", False):
        out_dir = _project_out_path(real_cfg.get("out_dir", None), workspace, workspace / "real_data" / "qc")
        if dry_run:
            result["steps"].append({"step": "real_csv", "planned_csv": real_cfg.get("csv", ""), "planned_out": str(out_dir)})
        else:
            rep = convert_real_csv(real_cfg["csv"], out_dir, max_traces=real_cfg.get("max_traces", None), make_baselines=not real_cfg.get("no_baselines", False))
            result["steps"].append({"step": "real_csv", "report": str((out_dir / "qc_report.json").resolve()), "bscan_shape": rep.get("bscan_shape")})

    soft_cfg = plan.get("soft_mask", {})
    if _as_bool(soft_cfg, "enabled", False):
        out_dir = _project_out_path(soft_cfg.get("out_dir", None), workspace, workspace / "real_data" / "soft_masks")
        bscan_npz = soft_cfg.get("bscan_npz")
        if not bscan_npz and real_cfg.get("enabled", False):
            bscan_npz = str(_project_out_path(real_cfg.get("out_dir", None), workspace, workspace / "real_data" / "qc") / "real_uavgpr_bscan_preview.npz")
        if dry_run:
            result["steps"].append({"step": "soft_mask", "planned_bscan_npz": bscan_npz, "planned_boreholes": soft_cfg.get("boreholes", ""), "planned_out": str(out_dir)})
        else:
            rep = generate_borehole_soft_mask(
                bscan_npz,
                soft_cfg["boreholes"],
                out_dir,
                velocity_m_per_ns=float(soft_cfg.get("velocity_m_per_ns", 0.10)),
                trace_interval_m=_optional_float(soft_cfg.get("trace_interval_m", None)),
                line_id=soft_cfg.get("line_id", None),
                trace_sigma=float(soft_cfg.get("trace_sigma", 4.0)),
                time_sigma_ns=_optional_float(soft_cfg.get("time_sigma_ns", None)),
                two_way_time=bool(soft_cfg.get("two_way_time", True)),
            )
            result["steps"].append({"step": "soft_mask", "report": str((out_dir / "soft_mask_report.json").resolve()), "total_picks_used": rep.get("total_picks_used")})

    report_cfg = plan.get("report", {})
    if _as_bool(report_cfg, "enabled", True):
        out_dir = _project_out_path(report_cfg.get("out_dir", None), workspace, workspace / "reports")
        if dry_run:
            result["steps"].append({"step": "report", "planned_out": str(out_dir)})
        else:
            rep = build_auto_report(workspace, out_dir=out_dir, title=report_cfg.get("title", "UavGPR-SimLab 自动实验报告"))
            result["steps"].append({"step": "report", "auto_report_md": rep.get("auto_report_md"), "auto_report_json": rep.get("auto_report_json")})

    out_path = workspace / "reports" / "pipeline_run_report.json"
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["pipeline_run_report"] = str(out_path.resolve())
    return result
