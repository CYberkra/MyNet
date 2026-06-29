from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

from .config import AppConfig


@dataclass
class GprMaxRunOptions:
    input_file: str
    n_traces: int = 1
    conda_env: str = "gprMax"
    gprmax_root: str = ""
    python_executable: str = "python"
    use_conda_run: bool = True
    use_gpu: bool = True
    gpu_ids: Optional[List[int]] = None
    geometry_only: bool = False
    write_processed: bool = False
    geometry_fixed: bool = True
    restart_from: int = 0
    mpi_tasks: int = 0
    openmp_threads: int = 0
    extra_args: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GprMaxTask:
    input_file: str
    case_id: str | int = ""
    variant: str = "raw"
    n_traces: int = 1
    geometry_only: bool = False
    write_processed: bool = False
    geometry_fixed: bool = True
    restart_from: int = 0
    extra_args: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _parse_gpu_ids(text: str | Sequence[int] | None) -> List[int]:
    if text is None:
        return []
    if isinstance(text, str):
        out = []
        for item in text.replace(";", ",").split(","):
            item = item.strip()
            if item:
                out.append(int(item))
        return out
    return [int(x) for x in text]


def _cmd_from_options(options: GprMaxRunOptions) -> List[str]:
    use_conda = bool(options.use_conda_run and str(options.conda_env).strip())
    if use_conda:
        cmd = ["conda", "run", "-n", options.conda_env, options.python_executable, "-m", "gprMax", options.input_file]
    else:
        cmd = [options.python_executable, "-m", "gprMax", options.input_file]
    if options.n_traces and int(options.n_traces) > 1:
        cmd += ["-n", str(int(options.n_traces))]
    if options.geometry_only:
        cmd += ["--geometry-only"]
    if options.write_processed:
        cmd += ["--write-processed"]
    if options.geometry_fixed:
        cmd += ["--geometry-fixed"]
    if options.restart_from and int(options.restart_from) > 0:
        cmd += ["-restart", str(int(options.restart_from))]
    if options.mpi_tasks and int(options.mpi_tasks) > 0:
        cmd += ["-mpi", str(int(options.mpi_tasks))]
    if options.use_gpu:
        cmd += ["-gpu"]
        if options.gpu_ids:
            cmd += [str(int(g)) for g in options.gpu_ids]
    if options.extra_args:
        cmd += list(options.extra_args)
    return cmd


def options_from_config_task(cfg: AppConfig, task: GprMaxTask) -> GprMaxRunOptions:
    return GprMaxRunOptions(
        input_file=str(Path(task.input_file).resolve()),
        n_traces=int(task.n_traces or cfg.geometry.trace_count),
        conda_env=cfg.runtime.conda_env_gprmax,
        gprmax_root=cfg.runtime.gprmax_source_dir,
        python_executable=cfg.runtime.python_executable,
        use_conda_run=cfg.runtime.use_conda_run,
        use_gpu=cfg.runtime.gpu_enabled,
        gpu_ids=_parse_gpu_ids(cfg.runtime.gpu_ids),
        geometry_only=task.geometry_only,
        write_processed=task.write_processed or cfg.runtime.write_processed,
        geometry_fixed=task.geometry_fixed and cfg.runtime.geometry_fixed,
        restart_from=int(task.restart_from or 0),
        mpi_tasks=int(cfg.runtime.mpi_tasks or 0),
        openmp_threads=int(cfg.runtime.omp_threads or 0),
        extra_args=task.extra_args,
    )


def build_gprmax_command(*args) -> List[str]:
    if len(args) == 1 and isinstance(args[0], GprMaxRunOptions):
        return _cmd_from_options(args[0])
    if len(args) == 2 and isinstance(args[0], AppConfig) and isinstance(args[1], GprMaxTask):
        return _cmd_from_options(options_from_config_task(args[0], args[1]))
    raise TypeError("build_gprmax_command expects GprMaxRunOptions or (AppConfig, GprMaxTask)")


def command_to_string(cmd: List[str]) -> str:
    """Return a Windows-safe command line string for generated BAT files and logs."""

    return subprocess.list2cmdline([str(c) for c in cmd])


def _env_for_options(options: GprMaxRunOptions) -> dict[str, str]:
    env = os.environ.copy()
    if options.openmp_threads and int(options.openmp_threads) > 0:
        env["OMP_NUM_THREADS"] = str(int(options.openmp_threads))
    if options.gprmax_root:
        env["PYTHONPATH"] = str(Path(options.gprmax_root).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    # Inject MSVC + CUDA + Windows SDK paths so that nvcc can find cl.exe
    # during gprMax GPU kernel compilation. SafeGprMaxRunner already does
    # this via its own code path; this ensures the CLI pipeline path
    # (run_task -> _env_for_options) also works.
    _extra_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64",
    ]
    for _p in _extra_paths:
        if os.path.isdir(_p):
            env["PATH"] = _p + ";" + env.get("PATH", "")
    _msvc_include = r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\include"
    _sdk_include = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt"
    _include_parts = []
    for _d in [_msvc_include, _sdk_include]:
        if os.path.isdir(_d):
            _include_parts.append(_d)
    if _include_parts:
        env["INCLUDE"] = ";".join(_include_parts) + ";" + env.get("INCLUDE", "")
    _msvc_lib = r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\lib\x64"
    _sdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"
    _ucrt_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64"
    _lib_parts = []
    for _d in [_msvc_lib, _sdk_lib, _ucrt_lib]:
        if os.path.isdir(_d):
            _lib_parts.append(_d)
    if _lib_parts:
        env["LIB"] = ";".join(_lib_parts) + ";" + env.get("LIB", "")
    return env


def stream_run(options: GprMaxRunOptions) -> Iterator[str]:
    cmd = build_gprmax_command(options)
    env = _env_for_options(options)
    cwd = options.gprmax_root or str(Path(options.input_file).parent)
    yield f"[CMD] {command_to_string(cmd)}\n"
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line
    rc = proc.wait()
    yield f"\n[EXIT] returncode={rc}\n"
    if rc != 0:
        raise RuntimeError(f"gprMax failed with return code {rc}")


def run_task(cfg: AppConfig, task: GprMaxTask, log_dir: str | Path, log: Callable[[str], None] | None = None) -> int:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_case = str(task.case_id).replace("/", "_").replace("\\", "_") or Path(task.input_file).stem
    log_path = log_dir / f"gprmax_{safe_case}_{task.variant}_{stamp}.log"
    options = options_from_config_task(cfg, task)
    cmd = build_gprmax_command(options)
    env = _env_for_options(options)
    cwd = options.gprmax_root or str(Path(options.input_file).parent)
    if log:
        log(f"[TASK] case={task.case_id} variant={task.variant}")
        log(f"[CMD] {command_to_string(cmd)}")
        log(f"[LOG] {log_path}")
    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
        lf.write(command_to_string(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            if log:
                log(line.rstrip("\n"))
        rc = proc.wait()
        lf.write(f"\n[EXIT] returncode={rc}\n")
    if rc != 0:
        raise RuntimeError(f"gprMax failed with return code {rc}; see {log_path}")
    return rc


def resolve_manifest_path(value: str | Path, manifest_csv: str | Path) -> Path:
    """Resolve a manifest-stored path.

    v0.8 manifests intentionally store relative paths for portability.  Runtime
    code resolves them against the workspace root inferred from
    <workspace>/datasets/<manifest>.csv.
    """
    p = Path(str(value))
    if p.is_absolute():
        return p
    base = Path(manifest_csv).expanduser().resolve().parent.parent
    return (base / p).resolve()


def tasks_from_manifest(manifest_csv: str | Path, variants: Optional[Sequence[str]] = None, limit: int = 0, trace_count: int = 1) -> List[GprMaxTask]:
    manifest_csv = Path(manifest_csv).expanduser()
    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest.csv 不存在：{manifest_csv}")
    if not manifest_csv.is_file():
        raise IsADirectoryError(f"manifest.csv 路径不是文件：{manifest_csv}")
    wanted = set(variants or [])
    tasks: List[GprMaxTask] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row.get("variant", "raw")
            if wanted and variant not in wanted:
                continue
            inp = row.get("input_file", "")
            if not inp:
                continue
            inp_path = resolve_manifest_path(inp, manifest_csv)
            n_text = row.get("n_traces") or row.get("trace_count") or row.get("trace_count_per_input") or trace_count
            try:
                n_val = int(float(n_text))
            except Exception:
                n_val = int(trace_count)
            tasks.append(GprMaxTask(input_file=str(inp_path), case_id=row.get("case_id", ""), variant=variant, n_traces=max(1, n_val), geometry_only=False))
            if limit and int(limit) > 0 and len(tasks) >= int(limit):
                break
    return tasks


def manifest_input_files(manifest_csv: str | Path) -> List[Dict[str, str]]:
    manifest_path = Path(manifest_csv).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv 不存在：{manifest_path}")
    if not manifest_path.is_file():
        raise IsADirectoryError(f"manifest.csv 路径不是文件：{manifest_path}")
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# --- Safe GPU runner (ported from hotfix90 runner_worker.py) ---

import signal
import threading
import queue as py_queue
import time as time_module


def _popen_process_group_kwargs():
    """Start gprMax in a killable process group."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"preexec_fn": os.setsid}


def _terminate_process_tree(proc, *, reason="timeout"):
    """Kill entire process tree. On Windows uses taskkill /T /F."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
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


def _progress_value(text):
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

    def __init__(self, return_code, status, elapsed,
                 stdout_tail, stderr_tail, stdout_log, stderr_log):
        self.return_code = return_code
        self.status = status
        self.elapsed = elapsed
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.stdout_log = stdout_log
        self.stderr_log = stderr_log

    def __repr__(self):
        return f"SafeGprMaxResult(status={self.status!r}, rc={self.return_code}, elapsed={self.elapsed:.1f}s)"


class SafeGprMaxRunner:
    """Run gprMax in an isolated subprocess with timeout and process-tree cleanup.

    If gprMax/PyCUDA aborts or crashes during GPU teardown, only the child
    process is affected; the calling process (GUI or CLI) survives.

    On Windows with vcvars_bat set, wraps the command through cmd.exe + vcvars64.bat
    so that nvcc and CUDA libraries are on PATH before gprMax starts.
    """

    def __init__(self, cmd, cwd,
                 timeout=14400.0, vcvars_bat="",
                 log_dir=None,
                 on_progress=None, on_log=None):
        self.cmd = [str(x) for x in cmd]
        self.cwd = str(cwd)
        self.timeout = float(timeout)
        self.vcvars_bat = str(vcvars_bat) if vcvars_bat else ""
        self.log_dir = Path(log_dir) if log_dir else None
        self.on_progress = on_progress
        self.on_log = on_log

    def _build_wrapped_cmd(self):
        """Return the command to execute. No vcvars wrapping needed --
        MSVC and CUDA paths are injected via environment in run()."""
        return self.cmd

    def run(self):
        """Execute gprMax with full process isolation. Blocks until done or timeout."""
        actual_cmd = self._build_wrapped_cmd()
        cwd = Path(self.cwd)
        cwd.mkdir(parents=True, exist_ok=True)

        stamp = time_module.strftime("%Y%m%d_%H%M%S")
        if self.log_dir:
            stdout_log = Path(self.log_dir) / f"gprmax_stdout_{stamp}.log"
            stderr_log = Path(self.log_dir) / f"gprmax_stderr_{stamp}.log"
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
            # Inject MSVC + CUDA paths directly instead of relying on
            # vcvars64.bat (which requires cmd.exe wrapper and has quoting
            # issues with paths containing spaces).
            _extra_paths = [
                # nvcc (CUDA compiler)
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
                # cl.exe (MSVC host compiler)
                r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64",
                # Windows SDK tools
                r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64",
            ]
            for _p in _extra_paths:
                if os.path.isdir(_p):
                    env["PATH"] = _p + ";" + env.get("PATH", "")
            # MSVC needs INCLUDE and LIB for the C++ standard library
            _msvc_include = r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\include"
            _sdk_include = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt"
            _msvc_lib = r"E:\sisual stdio 2022\VC\Tools\MSVC\14.39.33519\lib\x64"
            _sdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"
            _ucrt_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64"
            _include_parts = []
            for _d in [_msvc_include, _sdk_include]:
                if os.path.isdir(_d):
                    _include_parts.append(_d)
            if _include_parts:
                env["INCLUDE"] = ";".join(_include_parts) + ";" + env.get("INCLUDE", "")
            _lib_parts = []
            for _d in [_msvc_lib, _sdk_lib, _ucrt_lib]:
                if os.path.isdir(_d):
                    _lib_parts.append(_d)
            if _lib_parts:
                env["LIB"] = ";".join(_lib_parts) + ";" + env.get("LIB", "")

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

            q = py_queue.Queue()

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
                t.start()
                threads.append(t)
            if proc.stderr:
                t = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
                t.start()
                threads.append(t)

            def _append(p, text):
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
                except py_queue.Empty:
                    if proc.poll() is not None:
                        while True:
                            try:
                                kind, text = q.get_nowait()
                                if kind == "stdout":
                                    _append(stdout_log, text)
                                    stdout_tail = (stdout_tail + text)[-8000:]
                                else:
                                    _append(stderr_log, text)
                                    stderr_tail = (stderr_tail + text)[-8000:]
                            except py_queue.Empty:
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
            status = "cancelled"
            rc = 130
            if proc is not None:
                _terminate_process_tree(proc, reason="keyboard_interrupt")
        except Exception as exc:
            status = "failed"
            rc = 1
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


def run_geometry_dry_run(input_file, python_exe, gprmax_root="", timeout=300.0, vcvars_bat=""):
    """Validate a gprMax .in file by running --geometry-only.

    Fast (~10-60 seconds), tests GPU/PyCUDA init, catches parse errors,
    missing materials, and domain violations before full multi-hour runs.
    """
    input_file = str(Path(input_file).resolve())
    cmd = [python_exe, "-m", "gprMax", input_file, "--geometry-only", "--geometry-fixed"]
    # Note: gprmax_root sets PYTHONPATH via env, not a CLI flag.
    # The env is set by SafeGprMaxRunner internally.

    runner = SafeGprMaxRunner(
        cmd=cmd,
        cwd=str(Path(input_file).parent),
        timeout=timeout,
        vcvars_bat=vcvars_bat,
    )
    return runner.run()
