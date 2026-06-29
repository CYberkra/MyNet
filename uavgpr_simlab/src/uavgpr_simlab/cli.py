from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .core.config import AppConfig, load_config, load_yaml, save_config
from .core.environment import run_environment_checks, save_report
from .core.hpc import write_local_bash_script, write_slurm_array_script
from .core.job_registry import run_registered_task, write_job_plan
from .core.runner import GprMaxTask
from .core.pipeline import run_automation_pipeline
from .core.real_data import convert_real_csv, summarize_real_csv
from .core.reporting import build_auto_report
from .core.scenario import generate_cases, write_manifest_commands_bat
from .core.softmask import generate_borehole_soft_mask
from .core.postprocess import export_gprmax_bscan_for_input
from .core.history import scan_simulation_history, export_history_csv, delete_history_record, delete_history_bulk, _read_json
from .core.dataset_contract import validate_dataset_skeleton
from .core.run_dashboard import summarize_dataset_run_dashboard
from .core.workspace_relocator import relocate_workspace_paths
from .core.visual_history import build_history_preview
from .services.sceneworld_bscan_service import run_sceneworld_bscan_from_manifest, check_sceneworld_case_package


def _cfg_from_plan(plan_path: str | Path, workspace: str | Path) -> AppConfig:
    cfg = AppConfig()
    plan = load_yaml(plan_path)
    scene: Dict[str, Any] = plan.get("scene", {})
    run: Dict[str, Any] = plan.get("run", {})
    plan_name = plan.get("plan_name", Path(plan_path).stem)
    cfg.runtime.project_root = str(Path(workspace) / plan_name)
    cfg.runtime.gpu_enabled = bool(run.get("use_gpu", True))
    gids = run.get("gpu_ids", [0])
    cfg.runtime.gpu_ids = ",".join(str(x) for x in gids) if isinstance(gids, list) else str(gids)
    cfg.runtime.geometry_only_first = bool(run.get("geometry_only_first", True))
    cfg.runtime.mpi_tasks = int(run.get("mpi_tasks", 0) or 0)
    cfg.runtime.omp_threads = int(run.get("openmp_threads", 0) or 0)
    cfg.runtime.geometry_fixed = bool(run.get("geometry_fixed", True))
    cfg.dataset.cases = int(plan.get("scene_count", 10))
    components = plan.get("components", ["raw", "target", "clutter", "background", "air"])
    variant_map = {"target": "target_only", "clutter": "clutter_only", "background": "background_only", "air": "air_only"}
    cfg.dataset.variants = [variant_map.get(str(x), str(x)) for x in components]
    cfg.geology.random_seed = int(plan.get("random_seed", cfg.geology.random_seed))
    cfg.geometry.dimension = str(scene.get("dimension", cfg.geometry.dimension))
    cfg.geometry.model_length_m = float(scene.get("scan_length_m", cfg.geometry.model_length_m))
    cfg.geometry.trace_count = int(scene.get("trace_count", cfg.geometry.trace_count))
    cfg.geometry.trace_step_m = float(scene.get("trace_interval_m", cfg.geometry.trace_step_m))
    cfg.geometry.subsurface_depth_m = float(scene.get("domain_depth_m", cfg.geometry.subsurface_depth_m))
    cfg.geometry.air_margin_m = float(scene.get("air_margin_m", cfg.geometry.air_margin_m))
    cfg.geometry.dx_m = float(scene.get("dx_m", cfg.geometry.dx_m))
    cfg.geometry.dy_m = float(scene.get("dx_m", cfg.geometry.dy_m))
    cfg.geometry.dz_m = float(scene.get("dx_m", cfg.geometry.dz_m))
    cfg.geometry.geometry_column_width_m = float(scene.get("column_width_m", cfg.geometry.geometry_column_width_m))
    cfg.radar.time_window_ns = float(scene.get("time_window_ns", cfg.radar.time_window_ns))
    if "samples" in scene:
        cfg.radar.frequency_points = int(scene.get("samples", cfg.radar.frequency_points))
    if "frequency_points" in scene:
        cfg.radar.frequency_points = int(scene.get("frequency_points", cfg.radar.frequency_points))
    cfg.radar.center_frequency_mhz = float(scene.get("center_frequency_hz", cfg.radar.center_frequency_mhz * 1e6)) / 1e6
    cfg.radar.nominal_flight_height_m = float(scene.get("flight_height_m", cfg.radar.nominal_flight_height_m))
    cfg.radar.tx_rx_offset_m = float(scene.get("tx_rx_offset_m", cfg.radar.tx_rx_offset_m))
    if "antenna_model" in scene:
        cfg.radar.antenna_model = str(scene["antenna_model"])
    idepth = scene.get("interface_depth_m")
    if isinstance(idepth, list) and len(idepth) >= 2:
        cfg.geology.interface_depth_min_m = float(idepth[0])
        cfg.geology.interface_depth_max_m = float(idepth[1])
    slope = scene.get("slope_deg")
    if isinstance(slope, list) and len(slope) >= 2:
        cfg.geology.slope_deg_min = float(slope[0])
        cfg.geology.slope_deg_max = float(slope[1])
    rough = scene.get("interface_roughness_m")
    if isinstance(rough, list) and len(rough) >= 2:
        cfg.geology.interface_roughness_m = float(rough[1])
    if "clutter_level" in scene:
        cfg.geology.clutter_level = str(scene.get("clutter_level"))
    if "family" in scene:
        cfg.geology.scenario_family = str(scene.get("family"))
    if "scene_family" in scene:
        cfg.geology.scenario_family = str(scene.get("scene_family"))
    # Domain randomization config
    cfg.domain_randomization = plan.get("domain_randomization", {})
    return cfg


def _split_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def cmd_generate(args: argparse.Namespace) -> int:
    if getattr(args, "plan", None):
        cfg = _cfg_from_plan(args.plan, args.workspace)
        if args.count is not None:
            cfg.dataset.cases = int(args.count)
    else:
        cfg = load_config(args.config) if Path(args.config).exists() else AppConfig()
        cfg.runtime.project_root = args.workspace
        if args.count is not None:
            cfg.dataset.cases = int(args.count)
    models, manifest = generate_cases(cfg, cfg.runtime.project_root, cases=cfg.dataset.cases)
    # Store a portable config inside the generated dataset. The workspace itself
    # is represented by "." so the folder can be moved between machines without
    # keeping the generation host path.
    portable_cfg = copy.deepcopy(cfg)
    portable_cfg.runtime.project_root = "."
    save_config(portable_cfg, Path(cfg.runtime.project_root) / "configs" / "generated_config.yaml")
    gpu_ids = [int(x) for x in cfg.runtime.gpu_ids.replace(';', ',').split(',') if x.strip()]
    bat = write_manifest_commands_bat(manifest, Path(cfg.runtime.project_root) / "logs" / "run_all_gprmax.bat", conda_env=cfg.runtime.conda_env_gprmax, gpu=cfg.runtime.gpu_enabled, gpu_ids=gpu_ids, variants=list(cfg.dataset.variants))
    # SceneWorld datasets need a portable 5-variant run script that delegates to
    # scripts/run_all_gprmax.py instead of embedding generation-host paths.
    if "sceneworld" in str(Path(cfg.runtime.project_root).name).lower():
        variant_text = ",".join(str(v) for v in cfg.dataset.variants)
        gpu_text = cfg.runtime.gpu_ids if cfg.runtime.gpu_enabled else ""
        # Do not force a conda environment here. Many Windows machines already
        # have a usable Python and a downloaded gprMax source tree; the shared
        # bootstrap will use current Python unless UAVGPR_CONDA_ENV is set.
        timeout_sec = 600 if "ultra_tiny" in str(Path(cfg.runtime.project_root).name).lower() else 3600
        project_name_lower = str(Path(cfg.runtime.project_root).name).lower()
        allow_resample_flag = " --allow-resample" if ("ultra_tiny" in project_name_lower or "smoke" in project_name_lower) else ""
        bat_lines = [
            '@echo off',
            'setlocal EnableExtensions',
            'pushd "%~dp0\\.."',
            'set "WORKSPACE=%CD%"',
            'set "PROJECT_ROOT=%WORKSPACE%\\..\\.."',
            'if not exist "%PROJECT_ROOT%\\scripts\\run_all_gprmax.py" (',
            '  echo [ERROR] Cannot find %PROJECT_ROOT%\\scripts\\run_all_gprmax.py',
            '  echo Put this dataset under ^<project_root^>\\workspace\\',
            '  pause',
            '  popd',
            '  exit /b 1',
            ')',
            'call "%PROJECT_ROOT%\\scripts\\windows_runtime_bootstrap.bat"',
            'if errorlevel 1 (',
            '  echo [ERROR] Runtime bootstrap failed.',
            '  pause',
            '  popd',
            '  exit /b 1',
            ')',
            'if "%GPRMAX_SOURCE_DIR%"=="" (',
            '  echo [ERROR] GPRMAX_SOURCE_DIR is empty.',
            '  echo Set it to your gprMax source root, for example:',
            '  echo   set "GPRMAX_SOURCE_DIR=E:\\gprMax\\gprMax-v.3.1.7"',
            '  set /p GPRMAX_SOURCE_DIR=Enter gprMax source dir: ',
            ')',
            'echo.',
            'echo [UavGPR-SimLab] SceneWorld all-variant gprMax run',
            'echo Workspace: %WORKSPACE%',
            'echo Project:   %PROJECT_ROOT%',
            'echo Python:    %PY_RUN%',
            'echo gprMax:    %GPRMAX_SOURCE_DIR%',
            'if "%UAVGPR_CONDA_ENV%"=="" (echo Conda env: [not used]) else (echo Conda env: %UAVGPR_CONDA_ENV%)',
            'echo.',
            f'  %PY_RUN% "%PROJECT_ROOT%\\scripts\\run_all_gprmax.py" --workspace "%WORKSPACE%" --gprmax-source-dir "%GPRMAX_SOURCE_DIR%" --python-executable "%PY_EXE%" --gpu-ids "{gpu_text}" --variants "{variant_text}" --omp-threads {max(1, int(cfg.runtime.omp_threads or 1))} --timeout {timeout_sec} --force{allow_resample_flag}',
            'set RC=%ERRORLEVEL%',
            'echo.',
            'if %RC%==0 (echo [OK] SceneWorld gprMax run completed.) else (echo [FAILED] SceneWorld gprMax run failed.)',
            'pause',
            'popd',
            'exit /b %RC%',
            '',
        ]
        bat.write_text("\r\n".join(bat_lines), encoding="utf-8")
    print(json.dumps({"models": str(models), "manifest": str(manifest), "config": str(Path(cfg.runtime.project_root) / "configs" / "generated_config.yaml"), "run_bat": str(bat)}, ensure_ascii=False, indent=2))
    return 0


def cmd_sceneworld_bscan(args: argparse.Namespace) -> int:
    rep = run_sceneworld_bscan_from_manifest(
        args.manifest,
        gprmax_root=args.gprmax_root,
        variants=_split_csv(args.variants) or None,
        one_case_per_family=args.one_case_per_family,
        max_cases=args.max_cases,
        python_executable=args.python_executable,
        omp_threads=args.omp_threads,
        timeout_sec=args.timeout,
        no_gpu=bool(getattr(args, "no_gpu", False) or not _split_csv(getattr(args, "gpu_ids", ""))),
        gpu_ids=[int(x) for x in _split_csv(getattr(args, "gpu_ids", ""))],
        force=args.force,
        allow_resample=getattr(args, "allow_resample", False),
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_check_sceneworld_case_package(args: argparse.Namespace) -> int:
    rep = check_sceneworld_case_package(args.workspace, manifest_csv=args.manifest)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_check_env(args: argparse.Namespace) -> int:
    rep = run_environment_checks(args.conda_env, not args.no_conda_run, args.gprmax_root)
    save_report(rep, args.out)
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_preview_csv(args: argparse.Namespace) -> int:
    max_traces = args.max_traces if args.max_traces and args.max_traces > 0 else None
    if args.convert or args.png:
        out_dir = Path(args.out) if args.out else Path("outputs") / Path(args.csv).stem
        rep = convert_real_csv(args.csv, out_dir, max_traces=max_traces, make_baselines=True)
    else:
        s = summarize_real_csv(args.csv, max_traces=max_traces)
        rep = {**s.__dict__, "meta": s.meta.__dict__}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_postprocess_out(args: argparse.Namespace) -> int:
    rep = export_gprmax_bscan_for_input(args.input, args.out, stem=args.stem, time_window_ns=args.time_window_ns)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_commands(args: argparse.Namespace) -> int:
    ids = [int(x) for x in args.gpu_ids.replace(";", ",").split(",") if x.strip()]
    variants = _split_csv(args.variants)
    out = write_manifest_commands_bat(args.manifest, args.out_bat, conda_env=args.conda_env, gpu=not args.no_gpu, gpu_ids=ids, geometry_only=args.geometry_only, variants=variants, max_tasks=args.max_tasks)
    print(json.dumps({"bat": str(out)}, ensure_ascii=False, indent=2))
    return 0


def cmd_soft_mask(args: argparse.Namespace) -> int:
    rep = generate_borehole_soft_mask(
        args.bscan_npz,
        args.boreholes,
        args.out,
        velocity_m_per_ns=args.velocity_m_per_ns,
        trace_interval_m=args.trace_interval_m,
        line_id=args.line_id,
        trace_sigma=args.trace_sigma,
        time_sigma_ns=args.time_sigma_ns,
        two_way_time=not args.one_way_time,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_auto_report(args: argparse.Namespace) -> int:
    rep = build_auto_report(args.workspace, out_dir=args.out, title=args.title)
    print(json.dumps({"auto_report_md": rep.get("auto_report_md"), "auto_report_json": rep.get("auto_report_json"), "issues": rep.get("issues", [])}, ensure_ascii=False, indent=2))
    return 0


def cmd_hpc_script(args: argparse.Namespace) -> int:
    if args.mode == "local":
        rep = write_local_bash_script(
            args.manifest,
            args.out_sh,
            variants=args.variants,
            conda_env=args.conda_env,
            gpu_ids=args.gpu_ids,
            no_gpu=args.no_gpu,
            geometry_only=args.geometry_only,
            write_processed=args.write_processed,
            geometry_fixed=not args.no_geometry_fixed,
            max_tasks=args.max_tasks,
            python_executable=args.python_executable,
            safe_runner=not args.no_safe_runner,
            workspace=args.workspace,
            gprmax_root=args.gprmax_root,
            skip_completed=not args.no_skip_completed,
            force=args.force,
            postprocess=args.postprocess,
            postprocess_out_dir=args.postprocess_out_dir,
        )
    else:
        rep = write_slurm_array_script(
            args.manifest,
            args.out_sh,
            variants=args.variants,
            conda_env=args.conda_env,
            gprmax_root=args.gprmax_root,
            partition=args.partition,
            time_limit=args.time_limit,
            gpus_per_task=args.gpus_per_task,
            no_gpu=args.no_gpu,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            gpu_ids=args.gpu_ids,
            geometry_only=args.geometry_only,
            write_processed=args.write_processed,
            geometry_fixed=not args.no_geometry_fixed,
            array_parallelism=args.array_parallelism,
            max_tasks=args.max_tasks,
            python_executable=args.python_executable,
            postprocess=args.postprocess,
            postprocess_out_dir=args.postprocess_out_dir,
            safe_runner=not args.no_safe_runner,
            workspace=args.workspace,
            skip_completed=not args.no_skip_completed,
            force=args.force,
        )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0



def cmd_plan_jobs(args: argparse.Namespace) -> int:
    variants = _split_csv(args.variants)
    out_csv = args.out_csv or str(Path(args.workspace) / "jobs" / "job_plan.csv")
    rep = write_job_plan(
        args.manifest,
        args.workspace,
        out_csv,
        variants=variants,
        max_tasks=args.max_tasks,
        skip_completed=not args.no_skip_completed,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_run_one(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    cfg.runtime.project_root = args.workspace
    cfg.runtime.conda_env_gprmax = args.conda_env
    cfg.runtime.gprmax_source_dir = args.gprmax_root
    cfg.runtime.python_executable = args.python_executable
    cfg.runtime.use_conda_run = not args.no_conda_run
    cfg.runtime.gpu_enabled = not args.no_gpu
    cfg.runtime.gpu_ids = args.gpu_ids
    cfg.runtime.geometry_fixed = not args.no_geometry_fixed
    cfg.runtime.write_processed = args.write_processed
    cfg.runtime.omp_threads = args.omp_threads
    cfg.runtime.mpi_tasks = args.mpi_tasks
    task = GprMaxTask(
        input_file=args.input_file,
        case_id=args.case_id,
        variant=args.variant,
        n_traces=args.n_traces,
        geometry_only=args.geometry_only,
        write_processed=args.write_processed,
        geometry_fixed=not args.no_geometry_fixed,
    )
    rep = run_registered_task(
        cfg,
        task,
        args.workspace,
        skip_completed=not args.no_skip_completed,
        force=args.force,
        postprocess=args.postprocess,
        postprocess_out_dir=args.postprocess_out_dir,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    if args.export_csv:
        rep = export_history_csv(args.workspace, args.export_csv)
    else:
        records = [r.to_dict() for r in scan_simulation_history(args.workspace)]
        if args.status:
            records = [r for r in records if r.get("status") == args.status]
        rep = {"workspace": str(Path(args.workspace).resolve()), "count": len(records), "records": records[:args.limit if args.limit > 0 else None]}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0



def cmd_history_preview(args: argparse.Namespace) -> int:
    records = scan_simulation_history(args.workspace)
    if args.status:
        records = [r for r in records if r.status == args.status]
    if args.limit and args.limit > 0:
        records = records[:args.limit]
    previews = []
    for rec in records:
        marker_data = _read_json(Path(rec.marker_file))
        previews.append(build_history_preview(rec, args.workspace, marker_data=marker_data, time_window_ns=args.time_window_ns, make_png=not args.no_png).to_dict())
    rep = {"workspace": str(Path(args.workspace).resolve()), "count": len(previews), "previews": previews}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0

def cmd_delete_history(args: argparse.Namespace) -> int:
    if args.failed:
        rep = delete_history_bulk(args.workspace, statuses=["failed"], delete_outputs=args.delete_outputs)
    else:
        rep = delete_history_record(args.workspace, job_id=args.job_id, marker_file=args.marker_file, delete_outputs=args.delete_outputs, delete_case_model=args.delete_case_model)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_check_dataset_skeleton(args: argparse.Namespace) -> int:
    variants = _split_csv(args.variants) or None
    rep = validate_dataset_skeleton(
        args.manifest,
        expected_variants=variants or (),
        require_relative_paths=not args.allow_absolute_paths,
        write_report=bool(args.write_report),
    )
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    return 0 if rep.ok else 2

def cmd_run_dashboard(args: argparse.Namespace) -> int:
    variants = _split_csv(args.variants) or None
    rep = summarize_dataset_run_dashboard(
        args.manifest,
        expected_variants=variants or (),
        write_report=bool(args.write_report),
    )
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_relocate_workspace(args: argparse.Namespace) -> int:
    old_roots = _split_csv(args.old_roots)
    rep = relocate_workspace_paths(
        args.manifest,
        old_root=args.old_root or None,
        old_roots=old_roots,
        new_root=args.new_root or None,
        to_relative=not bool(args.keep_absolute),
        dry_run=not bool(args.apply),
        write_report=True,
        make_backup=not bool(args.no_backup),
        validate_after=not bool(args.no_validate),
    )
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    return 0 if rep.ok else 2

def cmd_pipeline(args: argparse.Namespace) -> int:
    rep = run_automation_pipeline(args.config, dry_run=args.dry_run)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uavgpr-simlab", description="UavGPR-SimLab command helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate gprMax input files from default config or a run plan")
    g.add_argument("--plan", default=None)
    g.add_argument("--config", default="configs/default_app.yaml")
    g.add_argument("--workspace", default="workspace/demo_cases")
    g.add_argument("--count", type=int, default=None)
    g.set_defaults(func=cmd_generate)

    gb = sub.add_parser("generate-batch", help="Alias of generate for existing batch files")
    gb.add_argument("--plan", required=True)
    gb.add_argument("--workspace", default="workspace")
    gb.add_argument("--count", type=int, default=None)
    gb.set_defaults(func=cmd_generate)

    ce = sub.add_parser("check-env", help="Diagnose local gprMax/GPU/GUI environment")
    ce.add_argument("--conda-env", default="gprMax")
    ce.add_argument("--gprmax-root", default="")
    ce.add_argument("--no-conda-run", action="store_true")
    ce.add_argument("--out", default="workspace/reports/environment_report.json")
    ce.set_defaults(func=cmd_check_env)

    pc = sub.add_parser("preview-csv", help="Summarize/convert an uploaded UavGPR CSV")
    pc.add_argument("csv")
    pc.add_argument("--max-traces", type=int, default=300)
    pc.add_argument("--out", default="outputs/real_csv_qc")
    pc.add_argument("--convert", action="store_true")
    pc.add_argument("--snr", action="store_true")
    pc.add_argument("--png", default=None)
    pc.set_defaults(func=cmd_preview_csv)

    po = sub.add_parser("postprocess-out", help="Merge readable gprMax .out files for one input and export QC/ML products")
    po.add_argument("input")
    po.add_argument("--out", default="outputs/gprmax_qc")
    po.add_argument("--stem", default=None)
    po.add_argument("--time-window-ns", type=float, default=700.0)
    po.set_defaults(func=cmd_postprocess_out)

    cm = sub.add_parser("commands", help="Write a gprMax batch BAT from manifest")
    cm.add_argument("--manifest", required=True)
    cm.add_argument("--out-bat", default="workspace/run_all_gprmax.bat")
    cm.add_argument("--conda-env", default="gprMax")
    cm.add_argument("--gpu-ids", default="0")
    cm.add_argument("--no-gpu", action="store_true")
    cm.add_argument("--geometry-only", action="store_true")
    cm.add_argument("--variants", default="raw")
    cm.add_argument("--max-tasks", type=int, default=0)
    cm.set_defaults(func=cmd_commands)

    sm = sub.add_parser("soft-mask", help="Generate borehole/interface weak-supervision soft mask for real B-scan NPZ")
    sm.add_argument("--bscan-npz", required=True)
    sm.add_argument("--boreholes", required=True, help="CSV/JSON with depth_m plus trace_index or distance_m/x_m")
    sm.add_argument("--out", default="workspace/soft_masks")
    sm.add_argument("--velocity-m-per-ns", type=float, default=0.10)
    sm.add_argument("--trace-interval-m", type=float, default=None)
    sm.add_argument("--line-id", default=None)
    sm.add_argument("--trace-sigma", type=float, default=4.0)
    sm.add_argument("--time-sigma-ns", type=float, default=None)
    sm.add_argument("--one-way-time", action="store_true", help="Use one-way depth/velocity instead of two-way GPR time")
    sm.set_defaults(func=cmd_soft_mask)

    ar = sub.add_parser("auto-report", help="Collect manifest/QC/soft-mask artifacts into Markdown and CSV paper tables")
    ar.add_argument("--workspace", default="workspace")
    ar.add_argument("--out", default=None)
    ar.add_argument("--title", default="UavGPR-SimLab 自动实验报告")
    ar.set_defaults(func=cmd_auto_report)

    hs = sub.add_parser("hpc-script", help="Write SLURM array or local bash scripts from manifest")
    hs.add_argument("--manifest", required=True)
    hs.add_argument("--out-sh", default="workspace/logs/run_gprmax_slurm_array.sh")
    hs.add_argument("--mode", choices=["slurm", "local"], default="slurm")
    hs.add_argument("--variants", default="raw")
    hs.add_argument("--conda-env", default="gprMax")
    hs.add_argument("--gprmax-root", default="")
    hs.add_argument("--partition", default="gpu")
    hs.add_argument("--time-limit", default="12:00:00")
    hs.add_argument("--gpus-per-task", type=int, default=1)
    hs.add_argument("--cpus-per-task", type=int, default=4)
    hs.add_argument("--mem", default="24G")
    hs.add_argument("--gpu-ids", default="0")
    hs.add_argument("--no-gpu", action="store_true", help="Do not pass -gpu to gprMax/run-one and do not request GPU resources")
    hs.add_argument("--geometry-only", action="store_true")
    hs.add_argument("--write-processed", action="store_true")
    hs.add_argument("--no-geometry-fixed", action="store_true")
    hs.add_argument("--array-parallelism", type=int, default=4)
    hs.add_argument("--max-tasks", type=int, default=0)
    hs.add_argument("--python-executable", default="python")
    hs.add_argument("--postprocess", action="store_true")
    hs.add_argument("--postprocess-out-dir", default="outputs/gprmax_qc")
    hs.add_argument("--workspace", default="workspace")
    hs.add_argument("--no-safe-runner", action="store_true", help="Generate direct gprMax commands without registry/resume protection")
    hs.add_argument("--no-skip-completed", action="store_true")
    hs.add_argument("--force", action="store_true", help="Rerun even when a matching success marker exists")
    hs.set_defaults(func=cmd_hpc_script)


    pj = sub.add_parser("plan-jobs", help="Build a resumable job plan and mark already-completed simulations")
    pj.add_argument("--manifest", required=True)
    pj.add_argument("--workspace", required=True)
    pj.add_argument("--variants", default="raw")
    pj.add_argument("--out-csv", default=None)
    pj.add_argument("--max-tasks", type=int, default=0)
    pj.add_argument("--no-skip-completed", action="store_true")
    pj.set_defaults(func=cmd_plan_jobs)

    ro = sub.add_parser("run-one", help="Run one gprMax task through the resumable registry wrapper")
    ro.add_argument("--input-file", required=True)
    ro.add_argument("--workspace", required=True)
    ro.add_argument("--case-id", default="")
    ro.add_argument("--variant", default="raw")
    ro.add_argument("--n-traces", type=int, default=1)
    ro.add_argument("--conda-env", default="gprMax")
    ro.add_argument("--gprmax-root", default="")
    ro.add_argument("--python-executable", default=sys.executable)
    ro.add_argument("--no-conda-run", action="store_true")
    ro.add_argument("--gpu-ids", default="0")
    ro.add_argument("--no-gpu", action="store_true")
    ro.add_argument("--geometry-only", action="store_true")
    ro.add_argument("--write-processed", action="store_true")
    ro.add_argument("--no-geometry-fixed", action="store_true")
    ro.add_argument("--omp-threads", type=int, default=4)
    ro.add_argument("--mpi-tasks", type=int, default=0)
    ro.add_argument("--postprocess", action="store_true")
    ro.add_argument("--postprocess-out-dir", default=None)
    ro.add_argument("--no-skip-completed", action="store_true")
    ro.add_argument("--force", action="store_true")
    ro.set_defaults(func=cmd_run_one)


    hi = sub.add_parser("history", help="List/export simulation history from the workspace registry")
    hi.add_argument("--workspace", default="workspace")
    hi.add_argument("--status", choices=["running", "done", "failed", "stale_running"], default=None)
    hi.add_argument("--limit", type=int, default=50)
    hi.add_argument("--export-csv", default=None)
    hi.set_defaults(func=cmd_history)


    hp = sub.add_parser("history-preview", help="Generate/read model and B-scan preview metadata for history records")
    hp.add_argument("--workspace", default="workspace")
    hp.add_argument("--status", choices=["running", "done", "failed", "stale_running"], default=None)
    hp.add_argument("--limit", type=int, default=50)
    hp.add_argument("--time-window-ns", type=float, default=700.0)
    hp.add_argument("--no-png", action="store_true", help="Only report preview metadata; do not render thumbnail PNGs")
    hp.set_defaults(func=cmd_history_preview)

    dh = sub.add_parser("delete-history", help="Delete one simulation history marker and optionally its outputs")
    dh.add_argument("--workspace", required=True)
    dh.add_argument("--job-id", default=None)
    dh.add_argument("--marker-file", default=None)
    dh.add_argument("--delete-outputs", action="store_true")
    dh.add_argument("--delete-case-model", action="store_true")
    dh.add_argument("--failed", action="store_true", help="Delete all failed markers in the workspace")
    dh.set_defaults(func=cmd_delete_history)


    sb = sub.add_parser("run-sceneworld-bscan", help="Run gprMax for SceneWorld manifest rows and replace B-scan placeholders")
    sb.add_argument("--manifest", required=True)
    sb.add_argument("--gprmax-root", required=True)
    sb.add_argument("--variants", default="raw,target_only,background_only,clutter_only,air_only")
    sb.add_argument("--one-case-per-family", action="store_true")
    sb.add_argument("--max-cases", type=int, default=0)
    sb.add_argument("--python-executable", default="python")
    sb.add_argument("--omp-threads", type=int, default=1)
    sb.add_argument("--timeout", type=int, default=3600)
    sb.add_argument("--gpu-ids", default="", help="Comma-separated GPU ids. Empty or --no-gpu means CPU mode.")
    sb.add_argument("--no-gpu", action="store_true")
    sb.add_argument("--force", action="store_true")
    sb.add_argument("--allow-resample", action="store_true", help="Explicitly align/resample gprMax native B-scans to the manifest target grid for chain validation.")
    sb.set_defaults(func=cmd_sceneworld_bscan)

    sc = sub.add_parser("check-sceneworld-case-package", help="Check SceneWorld case folder completeness and B-scan QC status")
    sc.add_argument("--workspace", required=True)
    sc.add_argument("--manifest", default=None)
    sc.set_defaults(func=cmd_check_sceneworld_case_package)


    ds = sub.add_parser("check-dataset-skeleton", help="Validate an imported/generated dataset skeleton before batch simulation")
    ds.add_argument("--manifest", required=True)
    ds.add_argument("--variants", default="raw,target_only,background_only,clutter_only,air_only")
    ds.add_argument("--allow-absolute-paths", action="store_true", help="Do not warn about absolute manifest paths")
    ds.add_argument("--write-report", action="store_true", help="Write reports/dataset_contract_report.json next to the dataset")
    ds.set_defaults(func=cmd_check_dataset_skeleton)

    rd = sub.add_parser("run-dashboard", help="Summarize imported/running/pending/done/failed tasks for one dataset manifest")
    rd.add_argument("--manifest", required=True)
    rd.add_argument("--variants", default="raw,target_only,background_only,clutter_only,air_only")
    rd.add_argument("--write-report", action="store_true", help="Write reports/run_dashboard_report.json next to the dataset")
    rd.set_defaults(func=cmd_run_dashboard)


    rw = sub.add_parser("relocate-workspace", help="Inspect/fix absolute paths after moving a dataset workspace between machines")
    rw.add_argument("--manifest", required=True)
    rw.add_argument("--old-root", default="", help="Old workspace root to replace, e.g. E:\\old_project\\workspace\\dataset")
    rw.add_argument("--old-roots", default="", help="Comma-separated additional old roots")
    rw.add_argument("--new-root", default="", help="Current workspace root; defaults to manifest parent parent")
    rw.add_argument("--apply", action="store_true", help="Write changes. Default is dry-run only")
    rw.add_argument("--keep-absolute", action="store_true", help="Keep relocated values as absolute paths instead of workspace-relative paths")
    rw.add_argument("--no-backup", action="store_true", help="Do not create reports/relocation_backups before writing")
    rw.add_argument("--no-validate", action="store_true", help="Skip post-relocation dataset contract/dashboard checks")
    rw.set_defaults(func=cmd_relocate_workspace)

    pl = sub.add_parser("pipeline", help="Run YAML automation pipeline: generate -> scripts -> real QC -> soft mask -> report")
    pl.add_argument("--config", required=True)
    pl.add_argument("--dry-run", action="store_true")
    pl.set_defaults(func=cmd_pipeline)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
