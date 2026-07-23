#!/usr/bin/env python3
"""Run a native-256 source deck from a disposable, locally ignored work area.

The versioned ``01_native_256_release_pilot`` directory is a source-deck
registry, not a solver working directory.  This runner therefore copies a
case into ``runtime.solver_run_root`` before invoking gprMax, captures the
per-trace provenance before merging, and leaves all raw solver products out
of Git.  Subset runs are explicitly smoke tests and are never postprocessed
as 256-trace release candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.runtime import RuntimeConfigError, load_runtime, require_gprmax  # noqa: E402

STATIC_AUDIT = ROOT / ".claude" / "skills" / "gprmax-physics-audit" / "scripts" / "audit_gprmax_input.py"
SOLVER_ARTIFACT_PATTERNS = (
    "*.out",
    "*.h5",
    "*.hdf5",
    "*.vti",
    "run_logs",
    "preflight",
    "run_manifest.json",
    "run_state.json",
    "postprocess_validation.json",
)


def _command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _run(command: list[str], *, cwd: Path, env: dict[str, str], execute: bool) -> None:
    print(_command_text(command))
    if execute:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def _load_vcvars_environment(env: dict[str, str], vcvars: Path | None) -> dict[str, str]:
    """Return an environment initialised by an optional Windows MSVC script.

    ``nvcc`` delegates host compilation to ``cl.exe`` on Windows.  A compiler
    may be installed without being on the parent shell's PATH, so the local
    runtime profile can name ``vcvars64.bat`` without committing a drive path.
    """
    if vcvars is None:
        return env
    if os.name != "nt":
        raise RuntimeError("gprmax_vcvars is only supported on Windows")
    if not vcvars.is_file():
        raise FileNotFoundError(f"gprMax MSVC environment script does not exist: {vcvars}")
    temp_root = Path(env.get("TEMP") or env.get("TMP") or tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)
    descriptor, script_name = tempfile.mkstemp(prefix="pgda_vcvars_", suffix=".cmd", dir=temp_root)
    os.close(descriptor)
    script_path = Path(script_name)
    script_path.write_text(f'@echo off\ncall "{vcvars}" >nul\nset\n', encoding="ascii")
    try:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(script_path)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        script_path.unlink(missing_ok=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to initialise MSVC from {vcvars}: {detail}")
    initialized = env.copy()
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            initialized[key] = value
    if not shutil.which("cl.exe", path=initialized.get("PATH")):
        raise RuntimeError(f"MSVC initialisation from {vcvars} did not expose cl.exe")
    return initialized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_geometry_views(case_dir: Path) -> list[dict[str, object]]:
    """Hash transient geometry views, then delete them from the run cache."""
    records: list[dict[str, object]] = []
    for view in sorted(case_dir.glob("*.vti")):
        record: dict[str, object] = {
            "name": view.name,
            "bytes": view.stat().st_size,
            "sha256": _sha256(view),
        }
        view.unlink()
        record["deleted"] = not view.exists()
        records.append(record)
    return records


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _set_spatial_step(input_path: Path, step_m: float) -> None:
    text = input_path.read_text(encoding="utf-8")
    replacement = f"{step_m:.12g} 0 0"
    source_label = re.escape("#src_steps:")
    receiver_label = re.escape("#rx_steps:")
    updated, source_count = re.subn(
        rf"(?m)^({source_label}\s*)[^\r\n]+$", rf"\g<1>{replacement}", text
    )
    updated, receiver_count = re.subn(
        rf"(?m)^({receiver_label}\s*)[^\r\n]+$", rf"\g<1>{replacement}", updated
    )
    if source_count != 1 or receiver_count != 1:
        raise RuntimeError(f"expected one source and receiver step declaration in {input_path}")
    input_path.write_text(updated, encoding="utf-8")


def _set_trace_start(input_path: Path, offset_m: float) -> None:
    """Translate the source and receiver origins for a subset probe.

    gprMax applies ``#src_steps`` and ``#rx_steps`` after the origin declared
    by the corresponding directives.  A one-trace smoke test therefore has to
    move both origins explicitly when it is meant to sample a central (rather
    than edge) canonical trace.
    """
    if offset_m == 0:
        return
    directives = ("#hertzian_dipole:", "#rx:")
    updated_lines: list[str] = []
    changed: dict[str, int] = {directive: 0 for directive in directives}
    for line in input_path.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.lstrip()
        directive = next((item for item in directives if stripped.startswith(item)), None)
        if directive is None:
            updated_lines.append(line)
            continue
        prefix, payload = stripped.split(":", 1)
        values = payload.strip().split()
        coordinate_index = 1 if directive == "#hertzian_dipole:" else 0
        if len(values) <= coordinate_index + 2:
            raise RuntimeError(f"invalid {directive} declaration in {input_path}")
        try:
            values[coordinate_index] = f"{float(values[coordinate_index]) + offset_m:.12g}"
        except ValueError as exc:
            raise RuntimeError(f"invalid x coordinate in {directive} declaration in {input_path}") from exc
        newline = "\n" if line.endswith("\n") else ""
        indentation = line[: len(line) - len(stripped)]
        updated_lines.append(f"{indentation}{prefix}: {' '.join(values)}{newline}")
        changed[directive] += 1
    if any(count != 1 for count in changed.values()):
        raise RuntimeError(f"expected one source and receiver origin declaration in {input_path}")
    input_path.write_text("".join(updated_lines), encoding="utf-8")


def stage_case(
    source_case_dir: Path,
    run_case_dir: Path,
    *,
    requested_trace_count: int,
    geometry_only: bool,
    trace_stride: int = 1,
    trace_start: int = 0,
    include_air_reference: bool = True,
    full_scene_only: bool = False,
) -> Path:
    """Copy a source deck to a fresh solver work directory with provenance."""
    source_case_dir = source_case_dir.resolve()
    run_case_dir = run_case_dir.resolve()
    if source_case_dir == run_case_dir or _inside(run_case_dir, source_case_dir):
        raise ValueError("solver run directory must not be the source deck or a child of it")
    if run_case_dir.exists():
        raise FileExistsError(f"refusing to reuse an existing solver run directory: {run_case_dir}")
    manifest_path = source_case_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing scene manifest: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_trace_count = int(source_manifest["grid"]["trace_count"])
    if trace_stride < 1:
        raise ValueError("trace stride must be positive")
    if trace_start < 0:
        raise ValueError("trace start must be non-negative")
    selected_indices = [trace_start + index * trace_stride for index in range(requested_trace_count)]
    if selected_indices and selected_indices[-1] >= declared_trace_count:
        raise ValueError(
            f"distributed subset ends at trace index {selected_indices[-1]}, "
            f"outside declared range [0, {declared_trace_count - 1}]"
        )
    shutil.copytree(source_case_dir, run_case_dir, ignore=shutil.ignore_patterns(*SOLVER_ARTIFACT_PATTERNS))
    geometry_index = source_manifest.get("geometry", {}).get("index_file")
    if geometry_index:
        source_geometry = source_case_dir / str(geometry_index)
        staged_geometry = run_case_dir / str(geometry_index)
        if not source_geometry.is_file():
            raise FileNotFoundError(f"manifest geometry index does not exist: {source_geometry}")
        staged_geometry.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_geometry, staged_geometry)
    input_names = _input_names(
        source_manifest,
        include_air_reference=include_air_reference,
        full_scene_only=full_scene_only,
    )
    if trace_stride > 1:
        step_m = float(source_manifest["grid"]["trace_spacing_m"]) * trace_stride
        for input_name in input_names:
            _set_spatial_step(run_case_dir / input_name, step_m)
    if trace_start:
        offset_m = float(source_manifest["grid"]["trace_spacing_m"]) * trace_start
        for input_name in input_names:
            _set_trace_start(run_case_dir / input_name, offset_m)
    provenance = {
        "schema": "native_256_solver_run_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_case_dir": str(source_case_dir),
        "source_scene_manifest_sha256": _sha256(manifest_path),
        "requested_trace_count": requested_trace_count,
        "declared_trace_count": declared_trace_count,
        "trace_stride": trace_stride,
        "trace_start": trace_start,
        "input_groups": [Path(name).stem for name in input_names],
        "selected_trace_indices_zero_based": selected_indices,
        "mode": "geometry_only" if geometry_only else (
            "full" if requested_trace_count == declared_trace_count and trace_stride == 1
            else "distributed_smoke_subset" if trace_stride > 1
            else "smoke_subset"
        ),
        "source_deck_read_only": True,
        "causal_pair_complete": bool(
            not full_scene_only and bool(source_manifest.get("target_presence"))
        ),
    }
    (run_case_dir / "run_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return run_case_dir


def _input_names(
    manifest: dict[str, object],
    *,
    include_air_reference: bool = True,
    full_scene_only: bool = False,
) -> list[str]:
    names = ["full_scene.in"]
    if bool(manifest.get("target_presence")) and not full_scene_only:
        names.append("no_basal_contrast_control.in")
    if include_air_reference and not full_scene_only:
        names.append("air_reference.in")
    return names


def _geometry_input(input_name: str) -> str:
    """Use the intentionally small geometry-check wrappers when supplied."""
    mapping = {
        "full_scene.in": "geometry_check_full.in",
        "no_basal_contrast_control.in": "geometry_check_control.in",
    }
    return mapping.get(input_name, input_name)


def _write_state(case_dir: Path, **state: object) -> None:
    (case_dir / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _solver_trace_chunks(trace_count: int, max_per_call: int | None) -> list[tuple[int, int]]:
    """Return gprMax ``(restart_index, model_count)`` chunks.

    gprMax interprets ``-n`` as the number of models *after* ``-restart``.
    Keeping both values in one helper prevents a resumed subset from silently
    producing traces beyond its declared acquisition contract.
    """
    if max_per_call is None or max_per_call >= trace_count:
        return [(1, trace_count)]
    return [
        (start, min(max_per_call, trace_count - start + 1))
        for start in range(1, trace_count + 1, max_per_call)
    ]


def _solver_command(
    python: Path,
    input_name: str,
    *,
    restart_index: int,
    model_count: int,
    gpu: int,
) -> list[str]:
    command = [str(python), "-m", "gprMax", input_name, "-n", str(model_count)]
    if restart_index > 1:
        command.extend(["-restart", str(restart_index)])
    command.extend(["--geometry-fixed", "-gpu", str(gpu)])
    return command


def _resume_control_preflight(
    source_case_dir: Path,
    run_case_dir: Path,
    *,
    trace_count: int,
    trace_stride: int,
    trace_start: int,
) -> dict[str, object]:
    """Validate that a stopped causal subset can safely run its control only.

    A resume is deliberately narrower than a general restart: the completed
    full-scene acquisition must already be present and no control trace may
    exist.  This prevents a later command from silently mixing source decks,
    trace positions, or partially written control outputs.
    """
    source_manifest_path = source_case_dir / "scene_manifest.json"
    run_manifest_path = run_case_dir / "run_manifest.json"
    if not source_manifest_path.is_file() or not run_manifest_path.is_file():
        raise FileNotFoundError("control resume requires source and staged run manifests")
    if not (run_case_dir / "full_scene.in").is_file():
        raise FileNotFoundError("control resume requires the staged full_scene input")
    if not (run_case_dir / "no_basal_contrast_control.in").is_file():
        raise FileNotFoundError("control resume requires a staged no-basal input")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("source_scene_manifest_sha256") != _sha256(source_manifest_path):
        raise RuntimeError("control resume source manifest hash does not match the staged run")
    expected_indices = [trace_start + index * trace_stride for index in range(trace_count)]
    for key, expected in (
        ("requested_trace_count", trace_count),
        ("trace_stride", trace_stride),
        ("trace_start", trace_start),
        ("selected_trace_indices_zero_based", expected_indices),
    ):
        if run_manifest.get(key) != expected:
            raise RuntimeError(f"control resume acquisition contract mismatch for {key}")
    if trace_count > 1:
        full_output = run_case_dir / "full_scene_merged.out"
    else:
        full_output = run_case_dir / "full_scene.out"
    if not full_output.is_file():
        raise FileNotFoundError(f"control resume requires completed full-scene output: {full_output}")
    full_contract = run_case_dir / "run_logs" / "full_scene_trace_contract.json"
    if not full_contract.is_file():
        raise FileNotFoundError(f"control resume requires full-scene trace contract: {full_contract}")
    output_pattern = re.compile(r"^no_basal_contrast_control(?P<index>\d*)\.out$")
    completed_indices: list[int] = []
    for output in run_case_dir.glob("no_basal_contrast_control*.out"):
        match = output_pattern.match(output.name)
        if match is None:
            raise RuntimeError(f"control resume found an unrecognised control output: {output.name}")
        completed_indices.append(int(match.group("index") or "1"))
    completed_indices.sort()
    if completed_indices and completed_indices != list(range(1, len(completed_indices) + 1)):
        raise RuntimeError("control resume requires a contiguous control-trace prefix starting at index 1")
    if len(completed_indices) >= trace_count:
        raise RuntimeError("control resume found a complete control acquisition; capture or merge it instead")
    if not bool(source_manifest.get("target_presence")):
        raise RuntimeError("control resume is only valid for target-present source decks")
    return {
        "source_manifest": source_manifest,
        "completed_control_trace_count": len(completed_indices),
        "next_restart_index": len(completed_indices) + 1,
    }


def _remaining_solver_trace_chunks(
    *,
    next_restart_index: int,
    trace_count: int,
    max_per_call: int | None,
) -> list[tuple[int, int]]:
    """Return only the still-missing contiguous suffix of a trace acquisition."""
    if not 1 <= next_restart_index <= trace_count:
        raise ValueError("next restart index must identify an unfinished trace")
    remaining = trace_count - next_restart_index + 1
    size = remaining if max_per_call is None else min(max_per_call, remaining)
    if size < 1:
        raise ValueError("maximum traces per call must be positive")
    return [
        (start, min(size, trace_count - start + 1))
        for start in range(next_restart_index, trace_count + 1, size)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="Versioned pre-solver source deck.")
    parser.add_argument("--gprmax-python", help="Override gprmax_python from the local runtime profile.")
    parser.add_argument("--gprmax-root", type=Path, help="Override gprmax_source from the local runtime profile.")
    parser.add_argument("--gpu", type=int, help="Override gpu_index from the local runtime profile.")
    parser.add_argument("--cuda-bin", type=Path, help="Directory containing nvcc.exe; required for gprMax GPU kernel compilation.")
    parser.add_argument("--run-dir", type=Path, help="Fresh disposable solver work directory.")
    parser.add_argument("--run-id", help="Run identifier below runtime.solver_run_root (default: UTC timestamp).")
    parser.add_argument("--trace-count", type=int, help="Run 1, 32, 64, or the full declared trace count.")
    parser.add_argument(
        "--max-traces-per-solver-call",
        type=int,
        help=(
            "Bound one gprMax invocation to this many traces. Later chunks use "
            "the matching -restart and remaining-model count, so no extra traces are generated."
        ),
    )
    parser.add_argument("--trace-stride", type=int, default=1, help="Audit-only stride through canonical trace positions.")
    parser.add_argument(
        "--trace-start",
        type=int,
        default=0,
        help="Zero-based canonical trace index represented by the first subset trace.",
    )
    parser.add_argument(
        "--skip-air-reference",
        action="store_true",
        help="Run only the causal full/control pair. Air remains a separate diagnostic, not a control substitute.",
    )
    parser.add_argument(
        "--full-scene-only",
        action="store_true",
        help=(
            "Run only full_scene for an early morphology decision. This mode cannot prove "
            "causal attribution and is never release eligible."
        ),
    )
    parser.add_argument(
        "--resume-control",
        action="store_true",
        help=(
            "Run only a missing no-basal control in an existing interrupted causal run. "
            "The staged source hash, full-scene contract, and requested acquisition must match exactly."
        ),
    )
    parser.add_argument("--geometry-only", action="store_true", help="Run geometry checks and remove VTI views afterwards.")
    parser.add_argument("--execute", action="store_true", help="Stage the deck and execute the listed commands.")
    args = parser.parse_args()

    source_case_dir = args.case_dir.resolve()
    manifest_path = source_case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("formal_training_allowed") is not False:
        raise RuntimeError("native pilot runner only accepts blocked pre-promotion scenes")
    declared_trace_count = int(manifest["grid"]["trace_count"])
    if declared_trace_count != 256:
        raise RuntimeError(f"native pilot requires 256 traces, got {declared_trace_count}")
    trace_count = args.trace_count or declared_trace_count
    if not 1 <= trace_count <= declared_trace_count:
        raise ValueError(f"trace count must be in [1, {declared_trace_count}], got {trace_count}")
    if args.max_traces_per_solver_call is not None and args.max_traces_per_solver_call < 1:
        raise ValueError("max traces per solver call must be positive")
    if args.trace_start < 0 or args.trace_stride < 1 or args.trace_start + (trace_count - 1) * args.trace_stride >= declared_trace_count:
        raise ValueError("trace start, stride, and count exceed the declared acquisition span")
    if args.trace_stride > 1 and (args.geometry_only or trace_count == declared_trace_count):
        raise ValueError("trace stride is only valid for non-full spatial audit subsets")
    if args.resume_control and (args.geometry_only or args.full_scene_only):
        raise ValueError("control resume cannot be combined with geometry-only or full-scene-only modes")
    if args.resume_control and not args.run_dir and not args.run_id:
        raise ValueError("control resume requires --run-dir or --run-id")

    runtime = load_runtime()
    try:
        configured_python, configured_root = require_gprmax(runtime)
    except RuntimeConfigError as exc:
        if args.gprmax_python and args.gprmax_root:
            configured_python, configured_root = Path(args.gprmax_python), args.gprmax_root
        else:
            raise RuntimeError(str(exc)) from exc
    gprmax_python = Path(args.gprmax_python).resolve() if args.gprmax_python else configured_python
    root = args.gprmax_root.resolve() if args.gprmax_root else configured_root
    if not gprmax_python.is_file():
        raise FileNotFoundError(f"gprMax Python does not exist: {gprmax_python}")
    if not (root / "gprMax").is_dir():
        raise FileNotFoundError(f"not a gprMax source root: {root}")
    if not STATIC_AUDIT.is_file():
        raise FileNotFoundError(f"missing required static gprMax audit: {STATIC_AUDIT}")
    gpu = int(args.gpu) if args.gpu is not None else runtime.gpu_index
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    toolchain_tmp = runtime.scratch_root / "toolchain_tmp"
    toolchain_tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(toolchain_tmp)
    env["TMP"] = str(toolchain_tmp)
    pycuda_cache = toolchain_tmp / "pycuda_cache"
    pycuda_cache.mkdir(parents=True, exist_ok=True)
    # Keep generated CUDA cubins in the project runtime area, never in C:\\Users.
    env["PYCUDA_CACHE_DIR"] = str(pycuda_cache)
    if runtime.cuda_nvcc_flags:
        env["PYCUDA_DEFAULT_NVCC_FLAGS"] = " ".join(runtime.cuda_nvcc_flags)
        env["GPRMAX_NVCC_FLAGS"] = " ".join(runtime.cuda_nvcc_flags)
    if runtime.gprmax_force_cuda_arch:
        env["GPRMAX_FORCE_CUDA_ARCH"] = runtime.gprmax_force_cuda_arch
    if args.execute and not args.geometry_only:
        env = _load_vcvars_environment(env, runtime.gprmax_vcvars)
        cuda_bin = args.cuda_bin.resolve() if args.cuda_bin else runtime.cuda_bin
        nvcc = (cuda_bin / "nvcc.exe") if cuda_bin else shutil.which("nvcc", path=env.get("PATH"))
        if not nvcc or not Path(nvcc).is_file():
            raise RuntimeError(
                "gprMax GPU execution requires the CUDA Toolkit compiler nvcc.exe. "
                "Install a compatible CUDA Toolkit outside the system drive, then pass --cuda-bin <toolkit>\\bin "
                "or expose nvcc.exe on PATH."
            )
        nvcc_path = Path(nvcc).resolve()
        env["CUDA_PATH"] = str(nvcc_path.parents[1])
        env["GPRMAX_CUDA_INCLUDE"] = str(nvcc_path.parents[1] / "include")
        nvvm_bin = nvcc_path.parents[1] / "nvvm" / "bin"
        env["PATH"] = os.pathsep.join(
            part for part in (str(nvcc_path.parent), str(nvvm_bin), env.get("PATH", "")) if part
        )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_case_dir = args.run_dir.resolve() if args.run_dir else (runtime.solver_run_root / source_case_dir.name / run_id).resolve()
    if args.resume_control:
        if not args.execute:
            print("PLAN ONLY: a matching staged run will be checked before resuming the no-basal control.")
            print(f"source deck: {source_case_dir}")
            print(f"solver run:  {run_case_dir}")
            return 0
        resume = _resume_control_preflight(
            source_case_dir,
            run_case_dir,
            trace_count=trace_count,
            trace_stride=args.trace_stride,
            trace_start=args.trace_start,
        )
        log_dir = run_case_dir / "run_logs"
        preflight_dir = run_case_dir / "preflight"
        log_dir.mkdir(exist_ok=True)
        preflight_dir.mkdir(exist_ok=True)
        control_input = run_case_dir / "no_basal_contrast_control.in"
        # A full-scene-only subset stages only the full input, so regenerate the
        # control input from the immutable source deck before applying the same
        # acquisition transform.  Otherwise a distributed full scene can be
        # incorrectly paired with a contiguous control scan.
        shutil.copy2(source_case_dir / control_input.name, control_input)
        if args.trace_stride > 1:
            step_m = float(manifest["grid"]["trace_spacing_m"]) * args.trace_stride
            _set_spatial_step(control_input, step_m)
        if args.trace_start:
            offset_m = float(manifest["grid"]["trace_spacing_m"]) * args.trace_start
            _set_trace_start(control_input, offset_m)
        _run(
            [
                str(gprmax_python),
                str(STATIC_AUDIT),
                str(control_input),
                "--json",
                str(preflight_dir / "no_basal_contrast_control_static_audit.json"),
            ],
            cwd=run_case_dir,
            env=env,
            execute=True,
        )
        for restart_index, model_count in _remaining_solver_trace_chunks(
            next_restart_index=int(resume["next_restart_index"]),
            trace_count=trace_count,
            max_per_call=args.max_traces_per_solver_call,
        ):
            _run(
                _solver_command(
                    gprmax_python,
                    control_input.name,
                    restart_index=restart_index,
                    model_count=model_count,
                    gpu=gpu,
                ),
                cwd=run_case_dir,
                env=env,
                execute=True,
            )
        _run(
            [
                str(gprmax_python),
                str(ROOT / "scripts" / "capture_gprmax_trace_contract.py"),
                str(run_case_dir),
                "--prefix",
                "no_basal_contrast_control",
                "--expected",
                str(trace_count),
                "--output",
                str(log_dir / "no_basal_contrast_control_trace_contract.json"),
            ],
            cwd=run_case_dir,
            env=env,
            execute=True,
        )
        if trace_count > 1:
            _run(
                [str(gprmax_python), "-m", "tools.outputfiles_merge", "no_basal_contrast_control", "--remove-files"],
                cwd=run_case_dir,
                env=env,
                execute=True,
            )
        _write_state(
            run_case_dir,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            status="smoke_subset_causal_pair_resumed_complete",
            requested_trace_count=trace_count,
            declared_trace_count=declared_trace_count,
            release_eligible=False,
            causal_pair_complete=True,
            completed_control_trace_count_before_resume=int(resume["completed_control_trace_count"]),
            note="Only the missing no-basal control suffix was resumed after a matching full-scene subset completed.",
        )
        return 0
    include_air_reference = not args.skip_air_reference and not args.full_scene_only
    inputs = _input_names(
        manifest,
        include_air_reference=include_air_reference,
        full_scene_only=args.full_scene_only,
    )
    solver_chunks = _solver_trace_chunks(trace_count, args.max_traces_per_solver_call)
    if not args.execute:
        print("PLAN ONLY: source deck will remain untouched. Add --execute to stage and run.")
        print(f"source deck: {source_case_dir}")
        print(f"solver run:  {run_case_dir}")
        mode = "geometry_only" if args.geometry_only else (
            "full" if trace_count == declared_trace_count
            else "distributed_smoke_subset" if args.trace_stride > 1
            else "smoke_subset"
        )
        print(f"mode: {mode}")
        print(f"trace start: {args.trace_start}")
        print(f"trace stride: {args.trace_stride}")
        print(f"solver chunks: {solver_chunks}")
        for input_name in inputs:
            print(f"static audit: {input_name}")
            if args.geometry_only:
                _run([str(gprmax_python), "-m", "gprMax", _geometry_input(input_name), "--geometry-only"], cwd=run_case_dir, env=env, execute=False)
            else:
                for restart_index, model_count in solver_chunks:
                    _run(
                        _solver_command(
                            gprmax_python,
                            input_name,
                            restart_index=restart_index,
                            model_count=model_count,
                            gpu=gpu,
                        ),
                        cwd=run_case_dir,
                        env=env,
                        execute=False,
                    )
        return 0

    case_dir = stage_case(
        source_case_dir,
        run_case_dir,
        requested_trace_count=trace_count,
        geometry_only=args.geometry_only,
        trace_stride=args.trace_stride,
        trace_start=args.trace_start,
        include_air_reference=include_air_reference,
        full_scene_only=args.full_scene_only,
    )
    log_dir = case_dir / "run_logs"
    preflight_dir = case_dir / "preflight"
    log_dir.mkdir()
    preflight_dir.mkdir()
    geometry_view_cleanup: list[dict[str, object]] = []
    for input_name in inputs:
        input_path = case_dir / input_name
        if not input_path.is_file():
            raise FileNotFoundError(f"missing solver input in staged deck: {input_path}")
        _run([str(gprmax_python), str(STATIC_AUDIT), str(input_path), "--json", str(preflight_dir / f"{input_path.stem}_static_audit.json")], cwd=case_dir, env=env, execute=True)
        stem = input_path.stem
        if args.geometry_only:
            _run([str(gprmax_python), "-m", "gprMax", _geometry_input(input_name), "--geometry-only"], cwd=case_dir, env=env, execute=True)
            cleanup_records = _remove_geometry_views(case_dir)
            for record in cleanup_records:
                record["geometry_input"] = _geometry_input(input_name)
            geometry_view_cleanup.extend(cleanup_records)
            continue
        for restart_index, model_count in solver_chunks:
            _run(
                _solver_command(
                    gprmax_python,
                    input_name,
                    restart_index=restart_index,
                    model_count=model_count,
                    gpu=gpu,
                ),
                cwd=case_dir,
                env=env,
                execute=True,
            )
        _run([str(gprmax_python), str(ROOT / "scripts" / "capture_gprmax_trace_contract.py"), str(case_dir), "--prefix", stem, "--expected", str(trace_count), "--output", str(log_dir / f"{stem}_trace_contract.json")], cwd=case_dir, env=env, execute=True)
        if trace_count > 1:
            _run([str(gprmax_python), "-m", "tools.outputfiles_merge", stem, "--remove-files"], cwd=case_dir, env=env, execute=True)

    if args.geometry_only:
        cleanup_payload = {
            "schema": "gprmax_geometry_view_cleanup_v1",
            "policy": "transient_hash_then_delete",
            "solver_or_training_input": False,
            "generated_count": len(geometry_view_cleanup),
            "total_bytes_deleted": sum(int(item["bytes"]) for item in geometry_view_cleanup),
            "artifacts": geometry_view_cleanup,
        }
        (log_dir / "geometry_view_cleanup.json").write_text(
            json.dumps(cleanup_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_state(case_dir, completed_utc=datetime.now(timezone.utc).isoformat(), status="geometry_passed", declared_trace_count=declared_trace_count)
        return 0
    if trace_count != declared_trace_count:
        note = "Subset runs must not be postprocessed or promoted as canonical 256-trace data."
        if args.full_scene_only:
            note += " Full-scene-only morphology runs do not establish causal attribution."
        _write_state(
            case_dir,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            status="smoke_subset_complete",
            requested_trace_count=trace_count,
            declared_trace_count=declared_trace_count,
            release_eligible=False,
            causal_pair_complete=False if args.full_scene_only else bool(manifest.get("target_presence")),
            note=note,
        )
        return 0
    if args.full_scene_only:
        _write_state(
            case_dir,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            status="full_scene_morphology_complete",
            requested_trace_count=trace_count,
            declared_trace_count=declared_trace_count,
            release_eligible=False,
            causal_pair_complete=False,
            note="Full scene is complete for morphology review only; no causal control was run.",
        )
        return 0
    if args.skip_air_reference:
        _write_state(
            case_dir,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            status="full_causal_pair_complete",
            requested_trace_count=trace_count,
            declared_trace_count=declared_trace_count,
            release_eligible=False,
            note="Full/control causal pair is complete. Existing postprocessing requires air_reference and was intentionally skipped.",
        )
        return 0
    _run([str(gprmax_python), str(ROOT / "scripts" / "postprocess_physical_sim_v2.py"), str(case_dir)], cwd=case_dir, env=env, execute=True)
    _run([str(gprmax_python), str(ROOT / "scripts" / "validate_physical_sim_v2.py"), "--root", str(case_dir.parent)], cwd=ROOT, env=env, execute=True)
    _write_state(case_dir, completed_utc=datetime.now(timezone.utc).isoformat(), status="full_run_complete", requested_trace_count=trace_count, release_eligible=False, note="Promotion still requires an explicit independent audit and human decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
