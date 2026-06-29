from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

try:
    import psutil
except Exception:
    psutil = None


@dataclass
class CheckItem:
    name: str
    ok: bool
    message: str


@dataclass
class EnvironmentReport:
    python: str
    platform: str
    cpu_count_logical: int
    cpu_count_physical: int
    memory_gb: float
    disk_free_gb: float
    gpu_summary: str
    items: List[CheckItem]

    @property
    def ok(self) -> bool:
        """True when all required environment checks passed.

        This property is intentionally conservative and is used by the
        automation pipeline to summarize readiness without re-parsing the
        item list in callers.
        """
        return all(item.ok for item in self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python": self.python,
            "platform": self.platform,
            "cpu_count_logical": self.cpu_count_logical,
            "cpu_count_physical": self.cpu_count_physical,
            "memory_gb": self.memory_gb,
            "disk_free_gb": self.disk_free_gb,
            "gpu_summary": self.gpu_summary,
            "ok": self.ok,
            "items": [asdict(x) for x in self.items],
        }


def _run(cmd: List[str], timeout: int = 15, env_extra: Dict[str, str] | None = None) -> tuple[bool, str]:
    """Run a short diagnostics command with a hard timeout."""

    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=max(1, int(timeout)),
            **kwargs,
        )
        text = (proc.stdout or "").strip()
        return proc.returncode == 0, text[-8000:]
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        tail = (text or "")[-4000:]
        return False, f"timeout after {timeout}s" + (f"\n{tail}" if tail else "")
    except Exception as e:
        return False, repr(e)

def _module_item(module_name: str, timeout: int = 5) -> CheckItem:
    # Probe availability without importing heavy GUI/scientific modules.  This
    # avoids headless hangs while still reporting whether the package is installed.
    try:
        import importlib.metadata as metadata
        import importlib.util

        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return CheckItem(f"Python module: {module_name}", False, "not found")
        dist = {"yaml": "PyYAML"}.get(module_name, module_name)
        try:
            version = metadata.version(dist)
        except Exception:
            version = spec.origin or "installed"
        return CheckItem(f"Python module: {module_name}", True, str(version))
    except Exception as exc:
        return CheckItem(f"Python module: {module_name}", False, repr(exc))

def gpu_summary() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not found"
    ok, out = _run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout=10)
    return out if ok else f"nvidia-smi failed: {out}"


def run_environment_checks(
    conda_env: str = "gprMax",
    use_conda_run: bool = True,
    gprmax_root: str = "",
    *,
    conda_env_prefix: str = "",
    conda_exe: str = "",
    module_timeout: int = 5,
    gprmax_timeout: int = 10,
) -> EnvironmentReport:
    items: List[CheckItem] = []
    items.append(CheckItem("Python executable", True, sys.executable))
    conda_cmd = conda_exe or shutil.which("conda") or ""
    conda_required = bool(use_conda_run or conda_env_prefix or conda_exe)
    if conda_cmd:
        items.append(CheckItem("Conda command", True, conda_cmd))
    else:
        msg = "not found" if conda_required else "not found; optional because conda run is disabled"
        items.append(CheckItem("Conda command", not conda_required, msg))
    items.append(CheckItem("Git command", shutil.which("git") is not None, shutil.which("git") or "not found"))
    items.append(CheckItem("NVIDIA SMI", shutil.which("nvidia-smi") is not None, gpu_summary()))
    items.append(CheckItem("CUDA nvcc", shutil.which("nvcc") is not None, shutil.which("nvcc") or "not found; install CUDA Toolkit for GPU/PyCUDA build"))
    if gprmax_root:
        p = Path(gprmax_root)
        items.append(CheckItem("gprMax source directory", p.exists(), str(p)))
    for mod in ["numpy", "pandas", "h5py", "yaml", "matplotlib", "PySide6"]:
        items.append(_module_item(mod, timeout=module_timeout))
    if use_conda_run and conda_cmd:
        if conda_env_prefix:
            base_cmd = [conda_cmd, "run", "-p", conda_env_prefix, "python"]
            label = conda_env_prefix
        else:
            base_cmd = [conda_cmd, "run", "-n", conda_env, "python"]
            label = conda_env
        ok, out = _run(base_cmd + ["-c", "import sys; print(sys.executable)"], timeout=gprmax_timeout)
        items.append(CheckItem(f"Conda env: {label}", ok, out))
        root_env: Dict[str, str] = {}
        if gprmax_root:
            root_env["PYTHONPATH"] = str(Path(gprmax_root).expanduser()) + os.pathsep + os.environ.get("PYTHONPATH", "")
        ok, out = _run(base_cmd + ["-c", "import gprMax; print('gprMax import ok')"], timeout=gprmax_timeout, env_extra=root_env or None)
        items.append(CheckItem("gprMax import in conda env", ok, out))
    elif use_conda_run and not conda_cmd:
        items.append(CheckItem("Conda env", False, "conda run is enabled, but conda was not found"))
    else:
        root_env: Dict[str, str] = {}
        if gprmax_root:
            root = Path(gprmax_root).expanduser()
            root_env["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
            ok, out = _run([sys.executable, "-c", "import sys, gprMax; print(sys.executable); print('gprMax import ok'); print(getattr(gprMax, '__file__', ''))"], timeout=gprmax_timeout, env_extra=root_env)
            items.append(CheckItem("gprMax import via source tree", ok, out))
        else:
            ok, out = _run([sys.executable, "-c", "import gprMax; print('gprMax import ok')"], timeout=gprmax_timeout)
            items.append(CheckItem("gprMax import in current Python", ok, out))
    if psutil:
        cpu_logical = psutil.cpu_count(logical=True) or 0
        cpu_physical = psutil.cpu_count(logical=False) or cpu_logical
        mem_gb = round(psutil.virtual_memory().total / 1024**3, 2)
        try:
            disk_gb = round(psutil.disk_usage(Path.cwd()).free / 1024**3, 2)
        except Exception:
            disk_gb = -1.0
    else:
        cpu_logical = os.cpu_count() or 0
        cpu_physical = cpu_logical
        mem_gb = -1.0
        disk_gb = -1.0
    return EnvironmentReport(sys.version.replace("\n", " "), platform.platform(), cpu_logical, cpu_physical, mem_gb, disk_gb, gpu_summary(), items)


def save_report(report: EnvironmentReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
