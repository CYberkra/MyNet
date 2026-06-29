from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py

from uavgpr_simlab.services.environment_service import inspect_gprmax_source


@dataclass(frozen=True)
class GprMaxSmokeStep:
    """Single step in a local-source gprMax CPU smoke test."""

    name: str
    ok: bool
    message: str
    returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GprMaxSourceSmokeReport:
    """Machine-readable result for a minimal CPU smoke test."""

    ok: bool
    gprmax_root: str
    work_dir: str
    python: str
    steps: list[GprMaxSmokeStep]
    output_file: str
    output_size: int
    hdf5_summary: dict[str, Any]
    report_path: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> GprMaxSmokeStep:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout.strip()
        tail = "\n".join(out.splitlines()[-24:])
        return GprMaxSmokeStep(
            name=" ".join(cmd[:3]),
            ok=proc.returncode == 0,
            message=tail,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        return GprMaxSmokeStep(
            name=" ".join(cmd[:3]),
            ok=False,
            message=f"timeout after {timeout}s: {exc}",
            returncode=None,
        )
    except Exception as exc:
        return GprMaxSmokeStep(name=" ".join(cmd[:3]), ok=False, message=repr(exc), returncode=None)


def write_tiny_cpu_input(path: Path) -> None:
    """Write a tiny 2D A-scan input file that should complete quickly on CPU."""

    path.write_text(
        "\n".join(
            [
                "#title: Tiny CPU smoke test for UavGPR-SimLab integration",
                "#domain: 0.080 0.080 0.002",
                "#dx_dy_dz: 0.002 0.002 0.002",
                "#time_window: 1e-10",
                "",
                "#waveform: ricker 1 1e9 src",
                "#hertzian_dipole: z 0.040 0.040 0 src",
                "#rx: 0.044 0.040 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_gprmax_source_smoke(
    gprmax_root: str | Path,
    work_dir: str | Path,
    *,
    build: bool = False,
    omp_threads: int = 1,
    timeout: int = 180,
) -> GprMaxSourceSmokeReport:
    """Run a minimal local-source gprMax CPU smoke test.

    This function uses the current Python interpreter and prepends *gprmax_root*
    to PYTHONPATH. It is intentionally CPU-only and does not change the normal
    batch runner, fingerprint, marker or B-scan post-processing semantics.
    """

    work = Path(work_dir).expanduser().resolve()
    root = Path(gprmax_root).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "gprmax_source_smoke_report.json"
    steps: list[GprMaxSmokeStep] = []

    source = inspect_gprmax_source(root)
    steps.append(GprMaxSmokeStep("inspect gprMax source", source.is_source_tree, source.message))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = str(max(1, int(omp_threads)))

    if build and source.is_source_tree:
        build_step = _run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=root,
            env=env,
            timeout=max(timeout, 300),
        )
        build_step = GprMaxSmokeStep("python setup.py build_ext --inplace", build_step.ok, build_step.message, build_step.returncode)
        steps.append(build_step)
        source = inspect_gprmax_source(root)
        steps.append(GprMaxSmokeStep("inspect compiled extensions", not source.missing_compiled_extensions, source.message))

    output_file = work / "tiny_Ascan_2D.out"
    hdf5_summary: dict[str, Any] = {}
    output_size = output_file.stat().st_size if output_file.exists() else 0

    if source.is_source_tree:
        help_step = _run([sys.executable, "-m", "gprMax", "--help"], cwd=root, env=env, timeout=timeout)
        steps.append(GprMaxSmokeStep("python -m gprMax --help", help_step.ok, help_step.message, help_step.returncode))

        input_file = work / "tiny_Ascan_2D.in"
        write_tiny_cpu_input(input_file)
        run_step = _run([sys.executable, "-m", "gprMax", str(input_file), "-n", "1"], cwd=root, env=env, timeout=timeout)
        steps.append(GprMaxSmokeStep("python -m gprMax tiny_Ascan_2D.in -n 1", run_step.ok, run_step.message, run_step.returncode))

        if output_file.exists():
            output_size = output_file.stat().st_size
            try:
                with h5py.File(output_file, "r") as h5:
                    hdf5_summary = {
                        "title": str(h5.attrs.get("Title", "")),
                        "iterations": int(h5.attrs.get("Iterations", 0)),
                        "groups": list(h5.keys()),
                        "rxs": list(h5.get("rxs", {}).keys()) if "rxs" in h5 else [],
                    }
                steps.append(GprMaxSmokeStep("inspect generated HDF5 .out", True, json.dumps(hdf5_summary, ensure_ascii=False)))
            except Exception as exc:
                steps.append(GprMaxSmokeStep("inspect generated HDF5 .out", False, repr(exc)))
        else:
            steps.append(GprMaxSmokeStep("inspect generated HDF5 .out", False, f"missing output file: {output_file}"))

    ok = all(step.ok for step in steps) and output_file.exists()
    report = GprMaxSourceSmokeReport(
        ok=ok,
        gprmax_root=str(root),
        work_dir=str(work),
        python=sys.version.replace("\n", " "),
        steps=steps,
        output_file=str(output_file),
        output_size=output_size,
        hdf5_summary=hdf5_summary,
        report_path=str(report_path),
    )
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def format_gprmax_source_smoke_report(report: GprMaxSourceSmokeReport) -> str:
    """Format a gprMax smoke report for the easy GUI settings log."""

    lines: list[str] = []
    lines.append("gprMax 最小 CPU 测试摘要")
    lines.append(f"- 总体状态：{'通过' if report.ok else '需要处理'}")
    lines.append(f"- gprMax 源码目录：{report.gprmax_root}")
    lines.append(f"- 工作目录：{report.work_dir}")
    lines.append(f"- 输出文件：{report.output_file}")
    lines.append(f"- 报告文件：{report.report_path}")
    if report.hdf5_summary:
        lines.append(f"- HDF5 摘要：{json.dumps(report.hdf5_summary, ensure_ascii=False)}")
    lines.append("")
    lines.append("步骤结果：")
    for step in report.steps:
        status = "通过" if step.ok else "失败"
        lines.append(f"- {step.name}：{status}")
        if step.message:
            tail = step.message.strip().splitlines()[-6:]
            for line in tail:
                lines.append(f"  {line}")
    lines.append("")
    lines.append("说明：此测试只验证当前 Python 对本地 gprMax 源码的最小 CPU 求解能力，不等同于 Windows/CUDA/GPU 批量验收。")
    return "\n".join(lines)
