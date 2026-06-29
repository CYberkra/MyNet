from __future__ import annotations

import csv
import shlex
from pathlib import Path
from typing import Any, Sequence


def _variants_set(variants: str | Sequence[str] | None) -> set[str]:
    if variants is None:
        return set()
    if isinstance(variants, str):
        return {x.strip() for x in variants.replace(";", ",").split(",") if x.strip()}
    return {str(x).strip() for x in variants if str(x).strip()}


def _repo_src() -> str:
    return str(Path(__file__).resolve().parents[3] / "src")


def manifest_to_task_list(manifest_csv: str | Path, out_tsv: str | Path, variants: str | Sequence[str] | None = None, max_tasks: int = 0) -> Path:
    """Create a portable TSV task list from a UavGPR manifest."""

    wanted = _variants_set(variants)
    out = Path(out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[tuple[str, str]] = set()
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as fin, out.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fields = ["input_file", "n_traces", "case_id", "variant", "split"]
        wr = csv.DictWriter(fout, fieldnames=fields, delimiter="\t")
        wr.writeheader()
        for row in reader:
            variant = row.get("variant", "")
            if wanted and variant not in wanted:
                continue
            inp = row.get("input_file", "")
            key = (str(Path(inp).resolve()) if inp else "", variant)
            if key in seen:
                continue
            seen.add(key)
            n_traces = row.get("n_traces") or row.get("trace_count") or "1"
            wr.writerow({
                "input_file": inp,
                "n_traces": n_traces,
                "case_id": row.get("case_id", ""),
                "variant": variant,
                "split": row.get("split", ""),
            })
            count += 1
            if max_tasks and count >= max_tasks:
                break
    return out


def count_tasks(task_tsv: str | Path) -> int:
    with Path(task_tsv).open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def _shell_bool_flag(name: str, enabled: bool) -> str:
    return name if enabled else ""


def _safe_runner_command(
    python_executable: str,
    workspace: str,
    conda_env: str,
    gprmax_root: str,
    gpu_ids: str,
    use_gpu: bool,
    geometry_only: bool,
    write_processed: bool,
    geometry_fixed: bool,
    postprocess: bool,
    skip_completed: bool,
    force: bool,
    postprocess_out_dir: str,
) -> str:
    flags = []
    if not use_gpu:
        flags.append("--no-gpu")
    if geometry_only:
        flags.append("--geometry-only")
    if write_processed:
        flags.append("--write-processed")
    if not geometry_fixed:
        flags.append("--no-geometry-fixed")
    if postprocess:
        flags.append("--postprocess")
    if not skip_completed:
        flags.append("--no-skip-completed")
    if force:
        flags.append("--force")
    parts = [
        shlex.quote(python_executable), "-m", "uavgpr_simlab.cli", "run-one",
        "--input-file", '"$INPUT_FILE"',
        "--workspace", shlex.quote(workspace),
        "--case-id", '"$CASE_ID"',
        "--variant", '"$VARIANT"',
        "--n-traces", '"$N_TRACES"',
        "--conda-env", shlex.quote(conda_env),
        "--gprmax-root", shlex.quote(gprmax_root),
        "--gpu-ids", shlex.quote(gpu_ids),
        "--postprocess-out-dir", shlex.quote(postprocess_out_dir),
    ] + flags
    return " ".join(str(x) for x in parts if str(x))


def write_slurm_array_script(
    manifest_csv: str | Path,
    out_sh: str | Path,
    variants: str | Sequence[str] | None = "raw",
    conda_env: str = "gprMax",
    gprmax_root: str = "",
    partition: str = "gpu",
    time_limit: str = "12:00:00",
    gpus_per_task: int = 1,
    no_gpu: bool = False,
    cpus_per_task: int = 4,
    mem: str = "24G",
    gpu_ids: str = "0",
    geometry_only: bool = False,
    write_processed: bool = False,
    geometry_fixed: bool = True,
    array_parallelism: int = 4,
    max_tasks: int = 0,
    python_executable: str = "python",
    postprocess: bool = False,
    postprocess_out_dir: str = "outputs/gprmax_qc",
    safe_runner: bool = True,
    workspace: str = "",
    skip_completed: bool = True,
    force: bool = False,
    simlab_src: str = "",
) -> dict[str, Any]:
    """Write a SLURM array script and companion task TSV.

    v0.4 defaults to a safe registered runner. The generated script will skip
    jobs that already have a matching success marker unless --force is used.
    """

    out = Path(out_sh)
    out.parent.mkdir(parents=True, exist_ok=True)
    task_tsv = out.with_suffix(".tasks.tsv")
    manifest_to_task_list(manifest_csv, task_tsv, variants=variants, max_tasks=max_tasks)
    n = count_tasks(task_tsv)
    array_spec = f"0-{max(n - 1, 0)}%{max(1, int(array_parallelism))}" if n else "0-0"
    gpu_tokens = [x.strip() for x in str(gpu_ids).replace(";", ",").split(",") if x.strip()]
    use_gpu = (not bool(no_gpu)) and int(gpus_per_task) > 0 and bool(gpu_tokens)
    gpu_arg = " ".join(shlex.quote(x) for x in gpu_tokens)
    gprmax_root_line = f"cd {shlex.quote(gprmax_root)}" if (gprmax_root and not safe_runner) else "# gprMax root is handled by run-one; edit if your cluster requires module paths"
    extra_flags = []
    if geometry_only:
        extra_flags.append("--geometry-only")
    if write_processed:
        extra_flags.append("--write-processed")
    if geometry_fixed:
        extra_flags.append("--geometry-fixed")
    extra = " ".join(extra_flags)
    src_path = simlab_src or _repo_src()
    workspace = workspace or str(Path(out).resolve().parents[1] if len(Path(out).resolve().parents) > 1 else Path.cwd())
    workspace = str(Path(workspace).resolve())
    if postprocess_out_dir and not Path(postprocess_out_dir).is_absolute():
        # If caller passes the default output folder, make it project-relative.
        if str(postprocess_out_dir).startswith("outputs"):
            postprocess_out_dir = str((Path(workspace) / postprocess_out_dir).resolve())
        else:
            postprocess_out_dir = str(Path(postprocess_out_dir).resolve())
    if safe_runner:
        cmd_line = _safe_runner_command(
            python_executable,
            workspace,
            conda_env,
            gprmax_root,
            gpu_ids,
            use_gpu,
            geometry_only,
            write_processed,
            geometry_fixed,
            postprocess,
            skip_completed,
            force,
            postprocess_out_dir,
        )
        run_block = f"{cmd_line}"
        py_path_line = f"export PYTHONPATH={shlex.quote(src_path)}:${{PYTHONPATH:-}}"
    else:
        run_block = f"CMD=({shlex.quote(python_executable)} -m gprMax \"$INPUT_FILE\" -n \"$N_TRACES\" {extra})\nif [[ {1 if use_gpu else 0} -gt 0 ]]; then\n  CMD+=(-gpu {gpu_arg})\nfi\necho \"[CMD] ${{CMD[*]}}\"\n\"${{CMD[@]}}\""
        py_path_line = "# safe runner disabled; PYTHONPATH not required"
    gpu_sbatch_line = f"#SBATCH --gres=gpu:{int(gpus_per_task)}" if use_gpu else "# GPU disabled for this job"
    script = f"""#!/usr/bin/env bash
#SBATCH --job-name=uavgpr_gprmax
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
{gpu_sbatch_line}
#SBATCH --cpus-per-task={int(cpus_per_task)}
#SBATCH --mem={mem}
#SBATCH --array={array_spec}
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -euo pipefail
PROJECT_WORKSPACE={shlex.quote(str(Path(workspace).resolve()))}
mkdir -p "$PROJECT_WORKSPACE/logs"
cd "$PROJECT_WORKSPACE"
TASK_FILE={shlex.quote(str(task_tsv.resolve()))}
TASK_LINE=$(awk -v IDX=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==IDX {{print}}' "$TASK_FILE")
if [[ -z "${{TASK_LINE}}" ]]; then
  echo "No task for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}"
  exit 0
fi
IFS=$'\t' read -r INPUT_FILE N_TRACES CASE_ID VARIANT SPLIT <<< "$TASK_LINE"
echo "[TASK] case=${{CASE_ID}} variant=${{VARIANT}} split=${{SPLIT}} input=${{INPUT_FILE}} n=${{N_TRACES}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {shlex.quote(conda_env)}
{py_path_line}
{gprmax_root_line}
{run_block}
"""
    out.write_text(script, encoding="utf-8")
    try:
        out.chmod(0o755)
    except Exception:
        pass
    return {"script": str(out.resolve()), "task_tsv": str(task_tsv.resolve()), "task_count": n, "array": array_spec, "safe_runner": bool(safe_runner), "skip_completed": bool(skip_completed)}


def write_local_bash_script(
    manifest_csv: str | Path,
    out_sh: str | Path,
    variants: str | Sequence[str] | None = "raw",
    conda_env: str = "gprMax",
    gpu_ids: str = "0",
    no_gpu: bool = False,
    geometry_only: bool = False,
    write_processed: bool = False,
    geometry_fixed: bool = True,
    max_tasks: int = 0,
    python_executable: str = "python",
    safe_runner: bool = True,
    workspace: str = "",
    gprmax_root: str = "",
    skip_completed: bool = True,
    force: bool = False,
    postprocess: bool = False,
    postprocess_out_dir: str = "outputs/gprmax_qc",
    simlab_src: str = "",
) -> dict[str, Any]:
    """Write a local Linux/macOS bash runner for sequential smoke tests."""

    out = Path(out_sh)
    out.parent.mkdir(parents=True, exist_ok=True)
    task_tsv = out.with_suffix(".tasks.tsv")
    manifest_to_task_list(manifest_csv, task_tsv, variants=variants, max_tasks=max_tasks)
    n = count_tasks(task_tsv)
    extra_flags = []
    if geometry_only:
        extra_flags.append("--geometry-only")
    if write_processed:
        extra_flags.append("--write-processed")
    if geometry_fixed:
        extra_flags.append("--geometry-fixed")
    extra = " ".join(extra_flags)
    gpu_tokens = [x.strip() for x in str(gpu_ids).replace(";", ",").split(",") if x.strip()]
    use_gpu = (not bool(no_gpu)) and bool(gpu_tokens)
    gpu_arg = " ".join(shlex.quote(x) for x in gpu_tokens)
    src_path = simlab_src or _repo_src()
    workspace = workspace or str(Path(out).resolve().parents[1] if len(Path(out).resolve().parents) > 1 else Path.cwd())
    workspace = str(Path(workspace).resolve())
    if postprocess_out_dir and not Path(postprocess_out_dir).is_absolute():
        if str(postprocess_out_dir).startswith("outputs"):
            postprocess_out_dir = str((Path(workspace) / postprocess_out_dir).resolve())
        else:
            postprocess_out_dir = str(Path(postprocess_out_dir).resolve())
    safe_cmd = _safe_runner_command(python_executable, workspace, conda_env, gprmax_root, gpu_ids, use_gpu, geometry_only, write_processed, geometry_fixed, postprocess, skip_completed, force, postprocess_out_dir)
    run_line = safe_cmd if safe_runner else f"{shlex.quote(python_executable)} -m gprMax \"$INPUT_FILE\" -n \"$N_TRACES\" {extra}" + (f" -gpu {gpu_arg}" if use_gpu else "")
    py_line = f"export PYTHONPATH={shlex.quote(src_path)}:${{PYTHONPATH:-}}" if safe_runner else "# safe runner disabled"
    script = f"""#!/usr/bin/env bash
set -euo pipefail
PROJECT_WORKSPACE={shlex.quote(str(Path(workspace).resolve()))}
mkdir -p "$PROJECT_WORKSPACE/logs"
cd "$PROJECT_WORKSPACE"
TASK_FILE={shlex.quote(str(task_tsv.resolve()))}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {shlex.quote(conda_env)}
{py_line}
mkdir -p logs
idx=0
while IFS=$'\t' read -r INPUT_FILE N_TRACES CASE_ID VARIANT SPLIT; do
  if [[ "$INPUT_FILE" == "input_file" ]]; then
    continue
  fi
  echo "[TASK $idx] case=${{CASE_ID}} variant=${{VARIANT}} input=${{INPUT_FILE}}"
  {run_line}
  idx=$((idx + 1))
done < "$TASK_FILE"
"""
    out.write_text(script, encoding="utf-8")
    try:
        out.chmod(0o755)
    except Exception:
        pass
    return {"script": str(out.resolve()), "task_tsv": str(task_tsv.resolve()), "task_count": n, "safe_runner": bool(safe_runner), "skip_completed": bool(skip_completed)}
