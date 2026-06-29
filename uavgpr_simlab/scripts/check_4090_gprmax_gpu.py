from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from uavgpr_simlab.services.environment_service import inspect_gprmax_source


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _split_gpu_ids(text: str | None) -> list[int]:
    ids: list[int] = []
    for part in str(text or "").replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        ids.append(int(item))
    return ids


def _python_cmd(conda_env: str = "", python_executable: str = "python") -> list[str]:
    env_name = str(conda_env or "").strip()
    if env_name:
        return ["conda", "run", "-n", env_name, "python"]
    exe = str(python_executable or "python").strip().strip('"') or "python"
    return [exe]


def _run_step(
    name: str,
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    required: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    step: dict[str, Any] = {
        "name": name,
        "cmd": [str(x) for x in cmd],
        "cwd": str(cwd) if cwd else "",
        "required": bool(required),
    }
    try:
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1, int(timeout)),
        )
        output = (proc.stdout or "")[-12000:]
        step.update({"ok": proc.returncode == 0, "returncode": proc.returncode, "seconds": round(time.time() - t0, 3), "output": output})
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        step.update({"ok": False, "returncode": None, "seconds": round(time.time() - t0, 3), "output": f"timeout after {timeout}s\n{out[-8000:]}"})
    except FileNotFoundError as exc:
        step.update({"ok": False, "returncode": None, "seconds": round(time.time() - t0, 3), "output": f"not found: {exc}"})
    except Exception as exc:
        step.update({"ok": False, "returncode": None, "seconds": round(time.time() - t0, 3), "output": repr(exc)})
    return step


def _write_tiny_input(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "#title: UavGPR-SimLab 4090 tiny GPU smoke",
                "#domain: 0.080 0.080 0.002",
                "#dx_dy_dz: 0.002 0.002 0.002",
                "#time_window: 1e-10",
                "#material: 6 0 1 0 halfspace",
                "#waveform: ricker 1 1e9 src",
                "#hertzian_dipole: z 0.040 0.040 0.001 src",
                "#rx: 0.044 0.040 0.001",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _check_hdf5(path: Path, py_cmd: Sequence[str], env: dict[str, str], timeout: int) -> dict[str, Any]:
    code = (
        "import h5py, sys; "
        "p=sys.argv[1]; "
        "f=h5py.File(p, 'r'); "
        "print('hdf5 keys=' + ','.join(f.keys())); "
        "f.close()"
    )
    return _run_step("read gprMax HDF5 output", list(py_cmd) + ["-c", code, str(path)], env=env, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Windows RTX 4090 + gprMax GPU readiness for UavGPR-SimLab.")
    ap.add_argument("--gprmax-root", required=True, help="gprMax source root containing gprMax/__main__.py")
    ap.add_argument("--conda-env", default="", help="Run checks through conda run -n <env> python")
    ap.add_argument("--python-executable", default="python", help="Python executable when --conda-env is not used")
    ap.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids; first id is used for smoke")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "logs" / "check_4090_gprmax_gpu_report.json"))
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--no-gpu", action="store_true", help="Skip PyCUDA and gprMax -gpu smoke. CPU/import checks still run.")
    ns = ap.parse_args()

    gprmax_root = Path(ns.gprmax_root).expanduser().resolve()
    out_path = Path(ns.out).expanduser().resolve()
    work_dir = PROJECT_ROOT / "workspace" / "check_4090_gprmax_gpu"
    tiny_input = work_dir / "tiny_gpu_Ascan_2D.in"
    cpu_out = tiny_input.with_name("tiny_gpu_Ascan_2D.out")
    gpu_ids = _split_gpu_ids(ns.gpu_ids)
    py_cmd = _python_cmd(ns.conda_env, ns.python_executable)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(gprmax_root) + os.pathsep + str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["OMP_NUM_THREADS"] = "1"

    report: dict[str, Any] = {
        "ok": False,
        "project_root": str(PROJECT_ROOT),
        "gprmax_root": str(gprmax_root),
        "python_cmd": py_cmd,
        "gpu_ids": gpu_ids,
        "no_gpu": bool(ns.no_gpu),
        "work_dir": str(work_dir),
        "steps": [],
        "notes": [],
    }

    try:
        report["gprmax_source"] = inspect_gprmax_source(gprmax_root).to_dict()
    except Exception as exc:
        report["gprmax_source"] = {"ok": False, "error": repr(exc)}

    if not (gprmax_root / "gprMax" / "__main__.py").exists():
        report["steps"].append({"name": "gprMax source root", "ok": False, "required": True, "output": "gprMax/__main__.py not found"})
    else:
        report["steps"].append({"name": "gprMax source root", "ok": True, "required": True, "output": str(gprmax_root)})

    report["steps"].append(_run_step("nvidia-smi", ["nvidia-smi"], env=env, timeout=30, required=not ns.no_gpu))
    report["steps"].append(_run_step("nvcc --version", ["nvcc", "--version"], env=env, timeout=30, required=not ns.no_gpu))
    report["steps"].append(_run_step("Python core imports", list(py_cmd) + ["-c", "import sys, numpy, h5py, Cython; print(sys.version); print('numpy', numpy.__version__); print('h5py', h5py.__version__); print('cython', Cython.__version__)"], env=env, timeout=60))
    report["steps"].append(_run_step("PySide6 import", list(py_cmd) + ["-c", "import PySide6; print('PySide6 import ok')"], env=env, timeout=60))
    report["steps"].append(_run_step("gprMax import/help", list(py_cmd) + ["-m", "gprMax", "--help"], cwd=gprmax_root, env=env, timeout=60))

    if not ns.no_gpu:
        pycuda_code = "import pycuda.driver as cuda; cuda.init(); print('device_count', cuda.Device.count()); print('device0', cuda.Device(0).name())"
        report["steps"].append(_run_step("PyCUDA CUDA driver check", list(py_cmd) + ["-c", pycuda_code], env=env, timeout=60))
    else:
        report["notes"].append("GPU smoke skipped by --no-gpu.")

    _write_tiny_input(tiny_input)
    for stale in work_dir.glob("tiny_gpu_Ascan_2D*.out"):
        try:
            stale.unlink()
        except Exception:
            pass
    cpu_step = _run_step("gprMax tiny CPU smoke", list(py_cmd) + ["-m", "gprMax", str(tiny_input), "-n", "1"], cwd=gprmax_root, env=env, timeout=int(ns.timeout))
    report["steps"].append(cpu_step)
    if cpu_out.exists():
        report["steps"].append(_check_hdf5(cpu_out, py_cmd, env, timeout=60))
    else:
        report["steps"].append({"name": "CPU output exists", "ok": False, "required": True, "output": str(cpu_out)})

    if not ns.no_gpu:
        for stale in work_dir.glob("tiny_gpu_Ascan_2D*.out"):
            try:
                stale.unlink()
            except Exception:
                pass
        gpu_id = gpu_ids[0] if gpu_ids else 0
        gpu_cmd = list(py_cmd) + ["-m", "gprMax", str(tiny_input), "-n", "1", "-gpu", str(gpu_id)]
        gpu_step = _run_step(f"gprMax tiny GPU smoke gpu={gpu_id}", gpu_cmd, cwd=gprmax_root, env=env, timeout=int(ns.timeout))
        report["steps"].append(gpu_step)
        if cpu_out.exists():
            report["steps"].append(_check_hdf5(cpu_out, py_cmd, env, timeout=60))
        else:
            report["steps"].append({"name": "GPU output exists", "ok": False, "required": True, "output": str(cpu_out)})

    required_steps = [s for s in report["steps"] if s.get("required", True)]
    report["ok"] = bool(required_steps) and all(bool(s.get("ok")) for s in required_steps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
