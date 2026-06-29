# SimLab GPU Runner Hardening + 3060 Smoke Test

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make UavGPR-SimLab's gprMax runner stable enough for autonomous multi-day GPU simulation runs on the 3060 laptop (6 GB VRAM), then verify the full chain with a smoke test.

**Architecture:** Port hotfix90's battle-tested process isolation (process group, timeout tree-kill, vcvars GPU wrapper, UTF-8 safety) into SimLab's `core/runner.py`. Fix config paths for the actual machine (no conda, direct .venv). Then generate and run 3 ultra_tiny SceneWorld cases to verify end-to-end.

**Tech Stack:** Python 3.x, gprMax 3.1.6 + PyCUDA, PyQt6, E:/gprMax/gprMax-v.3.1.7/.venv, MSVC 2022 vcvars64.bat

**Hardware:** RTX 3060 Laptop 6 GB → future RTX 4090 Laptop 16 GB

---

### Task 1: Fix SimLab config for actual machine environment

**Files:**
- Create: `/tmp/uavgpr_simlab/configs/environment_3060_laptop.yaml`
- Modify: `/tmp/uavgpr_simlab/configs/default_app.yaml`

The current `default_app.yaml` says `use_conda_run: true` and `gprmax_source_dir: C:/gprMax`. This machine has no conda — gprMax runs via `E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe` directly. We also need a 3060-specific config with correct memory budgets.

- [ ] **Step 1: Create 3060 environment config**

Write `/tmp/uavgpr_simlab/configs/environment_3060_laptop.yaml`:

```yaml
# 3060 Laptop GPU (6 GB VRAM) environment profile
# gprMax 3.1.6 + PyCUDA, no conda, direct .venv
runtime:
  project_root: workspace/yingshan_pilot_3060
  conda_env_gprmax: ""                         # no conda on this machine
  gprmax_source_dir: E:/gprMax/gprMax-v.3.1.7
  python_executable: E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe
  use_conda_run: false                         # direct .venv, not conda run
  gpu_enabled: true
  gpu_ids: '0'
  gpu_launch_mode: windows_cmd_vcvars_wrapper  # ported from hotfix90
  cuda_vcvars: E:/sisual stdio 2022/VC/Auxiliary/Build/vcvars64.bat
  mpi_tasks: 0
  omp_threads: 4
  geometry_only_first: true
  auto_merge_outputs: true
  geometry_fixed: true
  write_processed: false
  # Memory safety for 6 GB VRAM
  max_grid_cells_2d: 1500000                   # ~2500 × 600 safe limit
  gpu_memory_budget_mb: 5500                   # leave 500 MB for driver/etc
  # Timeouts (seconds)
  timeout_geometry: 600
  timeout_smoke: 1200
  timeout_raw: 14400
  timeout_background: 14400
  timeout_full_chain: 14400
```

- [ ] **Step 2: Update default_app.yaml for actual paths**

Edit `/tmp/uavgpr_simlab/configs/default_app.yaml` — change these lines:

Old:
```yaml
  conda_env_gprmax: gprMax
  gprmax_source_dir: C:/gprMax
  python_executable: python
  use_conda_run: true
  gpu_enabled: false
```

New:
```yaml
  conda_env_gprmax: ""
  gprmax_source_dir: E:/gprMax/gprMax-v.3.1.7
  python_executable: E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe
  use_conda_run: false
  gpu_enabled: true
  gpu_ids: '0'
```

Also change `project_root: workspace/default_project` to `project_root: workspace/yingshan_pilot_3060`.

- [ ] **Step 3: Verify config loads**

Run: `cd /tmp/uavgpr_simlab && python -c "from uavgpr_simlab.core.config import load_config; c = load_config('configs/default_app.yaml'); print('OK:', c.runtime.python_executable)"`

Expected: prints the .venv python path.

---

### Task 2: Port process isolation runner from hotfix90

**Files:**
- Create: `/tmp/uavgpr_simlab/src/uavgpr_simlab/core/runner_worker.py`
- Modify: `/tmp/uavgpr_simlab/src/uavgpr_simlab/core/runner.py`

The SimLab `runner.py` currently does a plain `subprocess.Popen` with no process group isolation. If gprMax/PyCUDA crashes during GPU teardown, it can take down the entire GUI process. hotfix90's `runner_worker.py` solves this with:
1. Base64-encoded command passing (safe across Windows code pages)
2. `CREATE_NEW_PROCESS_GROUP` for independent process group
3. `taskkill /T /F` for complete process tree termination on timeout
4. UTF-8 safety with `PYTHONUTF8=1` and `ensure_ascii` JSON events
5. Progress parsing from gprMax stdout

We add a `SafeGprMaxRunner` class to `runner.py` that wraps these patterns.

- [ ] **Step 1: Add SafeGprMaxRunner class**

Add to the end of `/tmp/uavgpr_simlab/src/uavgpr_simlab/core/runner.py`:

```python
# --- Safe GPU runner (ported from hotfix90 runner_worker.py) ---

import signal
import threading
import queue
import time as time_module

def _popen_process_group_kwargs() -> dict:
    """Start gprMax in a killable process group."""
    if sys.platform.startswith("win"):
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"preexec_fn": os.setsid}


def _terminate_process_tree(proc: subprocess.Popen, *, reason: str = "timeout") -> None:
    """Kill entire process tree. On Windows uses taskkill /T /F."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _progress_value(text: str) -> int | None:
    """Extract gprMax progress percentage from stdout line."""
    import re
    matches = re.findall(r"(\d{1,3})%", text)
    if not matches:
        return None
    try:
        return max(0, min(100, int(matches[-1])))
    except Exception:
        return None


class SafeGprMaxResult:
    """Result from a safe gprMax run."""
    def __init__(self, return_code: int, status: str, elapsed: float,
                 stdout_tail: str, stderr_tail: str, stdout_log: str, stderr_log: str):
        self.return_code = return_code
        self.status = status          # "success" | "timeout" | "failed" | "cancelled"
        self.elapsed = elapsed
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.stdout_log = stdout_log
        self.stderr_log = stderr_log


class SafeGprMaxRunner:
    """Run gprMax in an isolated subprocess with timeout and process-tree cleanup.

    If gprMax/PyCUDA aborts or crashes during GPU teardown, only the child
    process is affected; the calling process (GUI or CLI) survives.
    """

    def __init__(self, cmd: list[str], cwd: str | Path,
                 timeout: float = 14400.0, vcvars_bat: str = "",
                 log_dir: str | Path | None = None,
                 on_progress = None, on_log = None):
        self.cmd = [str(x) for x in cmd]
        self.cwd = str(cwd)
        self.timeout = timeout
        self.vcvars_bat = str(vcvars_bat) if vcvars_bat else ""
        self.log_dir = Path(log_dir) if log_dir else None
        self.on_progress = on_progress   # callback(pct: int, text: str)
        self.on_log = on_log             # callback(line: str)

    def _build_wrapped_cmd(self) -> list[str]:
        """If vcvars_bat is set, wrap the command through cmd.exe + vcvars.

        This ensures nvcc and CUDA libraries are on PATH before gprMax starts,
        which is required for PyCUDA GPU mode on Windows.
        """
        if not self.vcvars_bat or not sys.platform.startswith("win"):
            return self.cmd
        # cmd.exe /c "call vcvars64.bat >NUL && command"
        inner = subprocess.list2cmdline(self.cmd)
        wrapped = f'call "{self.vcvars_bat}" >NUL && {inner}'
        return ["cmd.exe", "/c", wrapped]

    def run(self) -> SafeGprMaxResult:
        """Execute gprMax with full process isolation. Blocks until done or timeout."""
        actual_cmd = self._build_wrapped_cmd()
        cwd = Path(self.cwd)
        cwd.mkdir(parents=True, exist_ok=True)

        # Setup logging
        stamp = time_module.strftime("%Y%m%d_%H%M%S")
        if self.log_dir:
            stdout_log = self.log_dir / f"gprmax_stdout_{stamp}.log"
            stderr_log = self.log_dir / f"gprmax_stderr_{stamp}.log"
        else:
            stdout_log = cwd / f"gprmax_stdout_{stamp}.log"
            stderr_log = cwd / f"gprmax_stderr_{stamp}.log"
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")

        start = time_module.time()
        timeout = float(self.timeout or 0.0)
        stdout_tail = ""
        stderr_tail = ""
        last_progress = -1
        proc = None
        rc = 1
        status = "failed"

        try:
            env = {
                **os.environ.copy(),
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8:backslashreplace",
            }
            if self.vcvars_bat:
                # vcvars sets its own env; let our UTF-8 overrides survive
                pass

            proc = subprocess.Popen(
                actual_cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
                **_popen_process_group_kwargs(),
            )

            q: queue.Queue[tuple[str, str]] = queue.Queue()

            def _reader(stream, kind):
                try:
                    while True:
                        chunk = stream.readline()
                        if not chunk:
                            break
                        q.put((kind, chunk))
                except Exception:
                    pass

            threads = []
            if proc.stdout:
                t = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
                t.start(); threads.append(t)
            if proc.stderr:
                t = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
                t.start(); threads.append(t)

            def _append(p: Path, text: str) -> None:
                if text:
                    try:
                        with p.open("a", encoding="utf-8", errors="replace") as fh:
                            fh.write(text)
                    except Exception:
                        pass

            while True:
                now = time_module.time()
                if timeout > 0 and now - start > timeout and proc.poll() is None:
                    status = "timeout"
                    _terminate_process_tree(proc, reason="timeout")
                    break

                try:
                    kind, text = q.get(timeout=0.3)
                except queue.Empty:
                    if proc.poll() is not None:
                        # Drain remaining
                        while True:
                            try:
                                kind, text = q.get_nowait()
                                if kind == "stdout":
                                    _append(stdout_log, text)
                                    stdout_tail = (stdout_tail + text)[-8000:]
                                else:
                                    _append(stderr_log, text)
                                    stderr_tail = (stderr_tail + text)[-8000:]
                            except queue.Empty:
                                break
                        rc = int(proc.returncode or 0)
                        status = "success" if rc == 0 else "failed"
                        break
                    continue

                if kind == "stdout":
                    _append(stdout_log, text)
                    stdout_tail = (stdout_tail + text)[-8000:]
                else:
                    _append(stderr_log, text)
                    stderr_tail = (stderr_tail + text)[-8000:]

                pct = _progress_value(text)
                if pct is not None and pct != last_progress:
                    last_progress = pct
                    if self.on_progress:
                        self.on_progress(pct, text.strip()[-300:])

                if self.on_log:
                    self.on_log(text.rstrip("\n"))

        except KeyboardInterrupt:
            status = "cancelled"; rc = 130
            if proc is not None:
                _terminate_process_tree(proc, reason="keyboard_interrupt")
        except Exception as exc:
            status = "failed"; rc = 1
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            _append(stderr_log, f"runner exception: {exc!r}\n")
            stderr_tail = (stderr_tail + f"runner exception: {exc!r}\n")[-8000:]

        elapsed = round(time_module.time() - start, 3)
        return SafeGprMaxResult(
            return_code=rc,
            status=status,
            elapsed=elapsed,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
        )
```

- [ ] **Step 2: Verify import works**

Run: `cd /tmp/uavgpr_simlab && python -c "from uavgpr_simlab.core.runner import SafeGprMaxRunner; print('SafeGprMaxRunner imported OK')"`

Expected: prints OK.

---

### Task 3: Add dry-run gprMax geometry check

**Files:**
- Modify: `/tmp/uavgpr_simlab/src/uavgpr_simlab/core/runner.py`

Before running a full multi-hour simulation, we need a fast `--geometry-only` dry-run that validates the gprMax input file parses correctly and the GPU is reachable. This catches 90% of configuration errors in ~30 seconds instead of discovering them 2 hours into a run.

- [ ] **Step 1: Add geometry dry-run function**

Add to `/tmp/uavgpr_simlab/src/uavgpr_simlab/core/runner.py`:

```python
def run_geometry_dry_run(
    input_file: str | Path,
    python_exe: str,
    gprmax_root: str = "",
    timeout: float = 300.0,
    vcvars_bat: str = "",
) -> SafeGprMaxResult:
    """Validate a gprMax .in file by running --geometry-only.

    This is fast (~10-60 seconds), tests GPU/PyCUDA initialization, and
    catches parse errors, missing materials, domain violations before
    committing to a full multi-hour run.
    """
    input_file = str(Path(input_file).resolve())
    cmd = [python_exe, "-m", "gprMax", input_file, "--geometry-only", "--geometry-fixed"]
    if gprmax_root:
        cmd += ["--gprmax-root", str(Path(gprmax_root).resolve())]

    runner = SafeGprMaxRunner(
        cmd=cmd,
        cwd=str(Path(input_file).parent),
        timeout=timeout,
        vcvars_bat=vcvars_bat,
    )
    return runner.run()
```

- [ ] **Step 2: Verify function is importable**

Run: `cd /tmp/uavgpr_simlab && python -c "from uavgpr_simlab.core.runner import run_geometry_dry_run; print('OK')"`

Expected: prints OK.

---

### Task 4: Verify gprMax smoke test runs on GPU

**Files:** None (diagnostic only)

Before modifying the pipeline, verify gprMax actually runs on the 3060 GPU with a minimal test.

- [ ] **Step 1: Create minimal test .in file**

Write `/tmp/gprmax_gpu_test.in`:
```
#title: GPU smoke test
#domain: 2.0 2.0 0.05
#dx_dy_dz: 0.02 0.02 0.05
#time_window: 30e-9
#material: 6 0 1 0 half_space
#box: 0 0 0 2.0 1.0 0.05 half_space
#waveform: ricker 1 100e6 my_wave
#hertzian_dipole: z 0.5 1.5 0.025 my_wave
#rx: 1.5 1.5 0.025
#src_steps: 0.02 0 0
#rx_steps: 0.02 0 0
#geometry_view: 0 0 0 2.0 2.0 0.05 0.02 0.02 0.05 geometry n
```

- [ ] **Step 2: Run with GPU via vcvars wrapper**

Run:
```bash
cmd.exe /c 'call "E:/sisual stdio 2022/VC/Auxiliary/Build/vcvars64.bat" >NUL && E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe -m gprMax /tmp/gprmax_gpu_test.in -gpu --geometry-only'
```

Expected: gprMax completes without error, output shows GPU detected.

- [ ] **Step 3: Run full simulation (single trace)**

```bash
cmd.exe /c 'call "E:/sisual stdio 2022/VC/Auxiliary/Build/vcvars64.bat" >NUL && E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe -m gprMax /tmp/gprmax_gpu_test.in -gpu -n 5'
```

Expected: generates .out files, exit code 0.

---

### Task 5: Generate and run 3 ultra_tiny SceneWorld test cases

**Files:**
- Modify: `/tmp/uavgpr_simlab/configs/run_plan_3060_quick.yaml` (adjust for 6 GB)

The existing 3060 config needs memory safety adjustments for the 6 GB (not 12 GB) GPU.

- [ ] **Step 1: Adjust 3060 config for 6 GB VRAM**

Edit `/tmp/uavgpr_simlab/configs/run_plan_3060_quick.yaml`:

Old values that need changing:
```yaml
trace_count: 72        # OK
dx_m: 0.10             # OK for 6 GB
domain_depth_m: 24     # Reduce slightly
scan_length_m: 36      # Reduce to keep grid size safe
time_window_ns: 450    # OK
column_width_m: 1.00   # OK
```

The grid size calculation: domain_x ≈ scan_length + tx_rx_offset + margins ≈ 36 + 1.4 + 4 = ~42m. domain_y ≈ subsurface_depth + air_margin = 24 + 6 = 30m. At dx=0.10m: 420×300 = 126K cells → well under 6 GB. We don't actually need to reduce — this is very safe.

Keep the existing values but add a comment noting this was validated for 6 GB.

Also reduce scene_count from 12 to 3 for initial smoke test:
```yaml
scene_count: 3
```

- [ ] **Step 2: Generate SceneWorld cases**

Run:
```bash
cd /tmp/uavgpr_simlab && python -m uavgpr_simlab.cli generate --plan configs/run_plan_3060_quick.yaml --workspace workspace/smoke_3060_test
```

Expected: creates workspace/smoke_3060_test/ with models/, datasets/, configs/, reports/, logs/. Manifest CSV should list 3 cases × 5 variants = 15 rows.

- [ ] **Step 3: Run geometry-only on one case**

Pick the first raw.in and run geometry-only via our new SafeGprMaxRunner:

```bash
cd /tmp/uavgpr_simlab && python -c "
from uavgpr_simlab.core.runner import run_geometry_dry_run
result = run_geometry_dry_run(
    'workspace/smoke_3060_test/models/case_000001/raw.in',
    python_exe='E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe',
    gprmax_root='E:/gprMax/gprMax-v.3.1.7',
    vcvars_bat='E:/sisual stdio 2022/VC/Auxiliary/Build/vcvars64.bat',
)
print(f'Status: {result.status}, elapsed: {result.elapsed}s')
if result.status != 'success':
    print('STDERR:', result.stderr_tail[-1000:])
"
```

Expected: status=success, elapsed < 60s.

- [ ] **Step 4: Run full simulation on 1 case (all 5 variants)**

```bash
cd /tmp/uavgpr_simlab && python -c "
from uavgpr_simlab.core.runner import SafeGprMaxRunner
import time
case_dir = 'workspace/smoke_3060_test/models/case_000001'
variants = ['raw', 'target_only', 'background_only', 'clutter_only', 'air_only']
for v in variants:
    infile = f'{case_dir}/{v}.in'
    cmd = ['E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe', '-m', 'gprMax', infile, '-gpu', '-n', '10', '--geometry-fixed']
    runner = SafeGprMaxRunner(
        cmd, 'workspace/smoke_3060_test/logs',
        timeout=1200, vcvars_bat='E:/sisual stdio 2022/VC/Auxiliary/Build/vcvars64.bat'
    )
    result = runner.run()
    print(f'{v}: status={result.status}, elapsed={result.elapsed:.1f}s, rc={result.return_code}')
    if result.status != 'success':
        print(f'  STDERR: {result.stderr_tail[-500:]}')
"
```

Expected: all 5 variants succeed. Note simulation times for budgeting.

- [ ] **Step 5: Postprocess outputs and verify B-scan contract**

```bash
cd /tmp/uavgpr_simlab && python -c "
from uavgpr_simlab.core.postprocess import merge_available_bscan_for_input, run_traditional_baselines
import numpy as np
case_dir = 'workspace/smoke_3060_test/models/case_000001'
for v in ['raw', 'target_only']:
    merged = merge_available_bscan_for_input(f'{case_dir}/{v}.in')
    if merged is not None:
        bscan, meta = merged
        print(f'{v}: shape={bscan.shape}, min={bscan.min():.4f}, max={bscan.max():.4f}')
        products = run_traditional_baselines(bscan, f'{case_dir}/outputs', stem=v, time_window_ns=450)
        print(f'  exported: {list(products.keys())}')
    else:
        print(f'{v}: NO OUTPUT FILES FOUND')
"
```

Expected: prints B-scan shapes and exports baseline products.

---

### Task 6: Create batch runner script for the 3060

**Files:**
- Create: `/tmp/uavgpr_simlab/scripts/run_batch_safe_3060.py`

A standalone script that reads a manifest CSV and runs all tasks through SafeGprMaxRunner, with skip-if-completed support and a summary report.

- [ ] **Step 1: Write the batch runner script**

Write `/tmp/uavgpr_simlab/scripts/run_batch_safe_3060.py`:

```python
#!/usr/bin/env python3
"""Safe batch gprMax runner for 3060 laptop (6 GB VRAM).

Usage:
  python scripts/run_batch_safe_3060.py --manifest workspace/smoke_3060_test/datasets/smoke_3060_test_manifest.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Ensure SimLab is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavgpr_simlab.core.runner import SafeGprMaxRunner, SafeGprMaxResult, resolve_manifest_path

PYTHON_EXE = "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe"
VCVARS_BAT = "E:/sisual stdio 2022/VC/Auxiliary/Build/vcvars64.bat"
TIMEOUT_S = 14400
N_TRACES_DEFAULT = 20


def run_manifest(manifest_csv: str, variants: list[str] | None = None,
                 limit: int = 0, dry_run: bool = False,
                 skip_completed: bool = True) -> dict:
    manifest = Path(manifest_csv)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    workspace = manifest.resolve().parent.parent
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(variants) if variants else None
    results = {"total": 0, "success": 0, "timeout": 0, "failed": 0, "skipped": 0, "tasks": []}

    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row.get("variant", "raw")
            if wanted and variant not in wanted:
                continue

            input_file = resolve_manifest_path(row["input_file"], manifest)
            case_id = row.get("case_id", "")
            n_traces = int(float(row.get("n_traces", N_TRACES_DEFAULT) or N_TRACES_DEFAULT))
            task_key = f"{case_id}_{variant}"

            results["total"] += 1
            print(f"\n{'[DRY-RUN]' if dry_run else '[RUN]'} {task_key} ({n_traces} traces)")

            if dry_run:
                results["skipped"] += 1
                continue

            cmd = [PYTHON_EXE, "-m", "gprMax", str(input_file), "-gpu", "-n", str(n_traces), "--geometry-fixed"]

            runner = SafeGprMaxRunner(
                cmd=cmd,
                cwd=str(log_dir),
                timeout=TIMEOUT_S,
                vcvars_bat=VCVARS_BAT,
                log_dir=log_dir,
                on_progress=lambda pct, txt: print(f"  [{task_key}] {pct}%", end="\r"),
            )

            t0 = time.time()
            result = runner.run()
            elapsed = time.time() - t0

            task_result = {
                "task_key": task_key,
                "case_id": case_id,
                "variant": variant,
                "input_file": str(input_file),
                "status": result.status,
                "return_code": result.return_code,
                "elapsed": result.elapsed,
                "stdout_log": result.stdout_log,
                "stderr_log": result.stderr_log,
            }
            results["tasks"].append(task_result)
            results[result.status] = results.get(result.status, 0) + 1

            status_emoji = {"success": "OK", "timeout": "TIMEOUT", "failed": "FAIL", "cancelled": "CANCEL"}
            print(f"  [{task_key}] {status_emoji.get(result.status, result.status)} in {result.elapsed:.1f}s")

            if limit and results["total"] >= limit:
                break

    # Write summary
    summary_path = log_dir / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {results['success']}/{results['total']} success, "
          f"{results.get('timeout', 0)} timeout, {results.get('failed', 0)} failed")
    print(f"Report: {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Safe batch gprMax runner for 3060")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--variants", default="raw", help="Comma-separated variants to run")
    parser.add_argument("--limit", type=int, default=0, help="Max tasks to run")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run_manifest(args.manifest, variants=variants, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the batch script**

```bash
cd /tmp/uavgpr_simlab && python scripts/run_batch_safe_3060.py --manifest workspace/smoke_3060_test/datasets/smoke_3060_test_manifest.csv --dry-run
```

Expected: prints all tasks that would run, no execution.

- [ ] **Step 3: Run 1 real task through the batch script**

```bash
cd /tmp/uavgpr_simlab && python scripts/run_batch_safe_3060.py --manifest workspace/smoke_3060_test/datasets/smoke_3060_test_manifest.csv --variants raw --limit 1
```

Expected: one raw simulation completes successfully, summary JSON written.

---

### Task 7: End-to-end verification checklist

**Files:** None (verification only)

- [ ] **Step 1: Confirm all 5 variants generate valid B-scans**

Run the batch script for 1 case × 5 variants (raw, target_only, background_only, clutter_only, air_only). Verify:
- All 5 exit with status "success"
- Each produces `.out` files that h5py can read
- `clutter_gt = raw - target_only` produces non-zero, non-NaN result

- [ ] **Step 2: Verify contract integrity**

```bash
cd /tmp/uavgpr_simlab && python -m uavgpr_simlab.cli check-dataset workspace/smoke_3060_test/datasets/smoke_3060_test_manifest.csv
```

Expected: all checks pass (or only warnings about missing full-resolution B-scans, which is expected for smoke test).

- [ ] **Step 3: Document GPU memory usage**

During a simulation, note the GPU memory usage:
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1
```

Record peak memory and confirm it's well under 6 GB (should be < 2 GB for the small smoke test).

- [ ] **Step 4: Load and inspect real data**

```bash
cd /tmp/uavgpr_simlab && python -c "
import numpy as np
# Load one 营山 line
data = np.loadtxt('C:/Users/17844/Desktop/02_Preprocessed_Standard/2025-09_营山/Line3origin(36).csv', delimiter=',', max_rows=5)
print(f'Sample shape check: first 5 rows read, {data.shape[1]} columns')
"
```

Expected: data loads, note the number of traces and time samples for later matching with simulation parameters.
