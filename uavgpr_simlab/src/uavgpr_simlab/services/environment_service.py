from __future__ import annotations

import json
import re
import subprocess
import sys
import shutil
import time
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Sequence

from uavgpr_simlab.cli import _cfg_from_plan
from uavgpr_simlab.core.config import AppConfig, read_combined_simlab_env, read_simlab_env, write_simlab_env
from uavgpr_simlab.core.environment import EnvironmentReport, run_environment_checks
from uavgpr_simlab.services.easy_batch_service import workspace_from_manifest


_REQUIRED_GPRMAX_FILES = (
    "gprMax/__init__.py",
    "gprMax/__main__.py",
    "gprMax/gprMax.py",
    "setup.py",
)
_OPTIONAL_GPRMAX_FILES = (
    "conda_env.yml",
    "requirements.txt",
    "README.rst",
    "user_models/cylinder_Ascan_2D.in",
)
_COMPILED_GPRMAX_MODULES = (
    "gprMax/fields_updates_ext",
    "gprMax/fractals_generate_ext",
    "gprMax/geometry_outputs_ext",
    "gprMax/geometry_primitives_ext",
    "gprMax/snapshots_ext",
    "gprMax/yee_cell_build_ext",
    "gprMax/yee_cell_setget_rigid_ext",
    "gprMax/pml_updates/pml_updates_electric_HORIPML_ext",
    "gprMax/pml_updates/pml_updates_electric_MRIPML_ext",
    "gprMax/pml_updates/pml_updates_magnetic_HORIPML_ext",
    "gprMax/pml_updates/pml_updates_magnetic_MRIPML_ext",
)
_COMPILED_SUFFIXES = {".so", ".pyd"}


@dataclass(frozen=True)
class GprMaxSourceInfo:
    """Structural diagnostics for a local gprMax source tree."""

    path: str
    exists: bool
    is_source_tree: bool
    detected_version: str
    missing_required: list[str]
    present_optional: list[str]
    user_model_count: int
    compiled_extension_count: int
    missing_compiled_extensions: list[str]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




@dataclass(frozen=True)
class RuntimePreflightStep:
    """One subprocess or structural check before starting a real gprMax run."""

    name: str
    ok: bool
    required: bool
    command: list[str]
    output: str
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePreflightReport:
    """Runtime Python/gprMax/GPU readiness check used before batch execution."""

    ok: bool
    python_cmd: list[str]
    gprmax_root: str
    require_gpu: bool
    steps: list[RuntimePreflightStep]
    issues: list[str]
    suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "python_cmd": self.python_cmd,
            "gprmax_root": self.gprmax_root,
            "require_gpu": self.require_gpu,
            "steps": [x.to_dict() for x in self.steps],
            "issues": self.issues,
            "suggestions": self.suggestions,
        }

    def format_for_user(self) -> str:
        lines = ["运行前环境检查", f"- 结果：{'通过' if self.ok else '未通过'}", f"- Python 命令：{' '.join(self.python_cmd)}", f"- gprMax 源码目录：{self.gprmax_root}", f"- GPU 检查：{'开启' if self.require_gpu else '关闭'}"]
        if self.issues:
            lines.append("")
            lines.append("问题：")
            for item in self.issues:
                lines.append(f"- {item}")
        if self.suggestions:
            lines.append("")
            lines.append("处理建议：")
            for item in self.suggestions:
                lines.append(f"- {item}")
        lines.append("")
        lines.append("检查明细：")
        for step in self.steps:
            status = "OK" if step.ok else "FAIL"
            lines.append(f"- [{status}] {step.name}: {step.output[:500].strip() if step.output else ''}")
        return "\n".join(lines)


@dataclass(frozen=True)
class EasyEnvironmentSettings:
    """Settings shown on the easy GUI environment page."""

    runtime_root: str = ""
    gprmax_root: str = ""
    conda_env: str = ""
    conda_env_prefix: str = ""
    conda_exe: str = ""
    gpu_ids: str = "0"
    omp_threads: int = 1
    use_gpu: bool = False
    use_conda_run: bool = False


@dataclass(frozen=True)
class EasyEnvironmentReport:
    """Environment check result with extra gprMax-source diagnostics."""

    environment: EnvironmentReport
    gprmax_source: GprMaxSourceInfo
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "environment": self.environment.to_dict(),
            "gprmax_source": self.gprmax_source.to_dict(),
            "report_path": str(self.report_path),
        }

    @property
    def ok(self) -> bool:
        return bool(self.environment.ok and self.gprmax_source.is_source_tree)


def _bool_from_env(value: str | bool | None, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_from_env(value: str | int | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)


def load_easy_environment_settings() -> EasyEnvironmentSettings:
    """Load local GUI environment settings from .simlab_env plus runtime env.

    v0.8.0-alpha.7 defaults to source-tree mode: current Python + PYTHONPATH.
    A conda environment is used only when the user explicitly sets one.
    """

    saved = read_combined_simlab_env()

    def pick(key: str, default: str = "") -> str:
        return saved.get(key) or os.environ.get(key, default) or default

    runtime_root = pick("UAVGPR_RUNTIME_ROOT")
    gprmax_root = pick("UAVGPR_GPRMAX_ROOT") or os.environ.get("GPRMAX_SOURCE_DIR", "")
    conda_env = pick("UAVGPR_CONDA_ENV")
    conda_env_prefix = pick("UAVGPR_CONDA_ENV_PREFIX")
    conda_exe = pick("UAVGPR_CONDA_EXE")
    conda_available = bool(conda_exe.strip() and Path(conda_exe.strip()).expanduser().exists()) or shutil.which("conda") is not None
    prefix_available = bool(conda_env_prefix.strip() and (Path(conda_env_prefix.strip()).expanduser() / ("python.exe" if os.name == "nt" else "bin/python")).exists())
    use_conda_default = bool(prefix_available or (conda_env.strip() and conda_available))
    return EasyEnvironmentSettings(
        runtime_root=runtime_root,
        gprmax_root=gprmax_root,
        conda_env=conda_env,
        conda_env_prefix=conda_env_prefix,
        conda_exe=conda_exe,
        gpu_ids=pick("UAVGPR_GPU_IDS", "0") or "0",
        omp_threads=max(0, _int_from_env(pick("UAVGPR_OMP_THREADS", "1"), 1)),
        use_gpu=_bool_from_env(pick("UAVGPR_USE_GPU", "0"), False),
        use_conda_run=_bool_from_env(pick("UAVGPR_USE_CONDA_RUN", ""), use_conda_default),
    )


def save_easy_environment_settings(settings: EasyEnvironmentSettings) -> Path:
    """Persist the easy GUI environment settings without requiring a config rewrite."""

    values = {
        "UAVGPR_RUNTIME_ROOT": settings.runtime_root.strip(),
        "UAVGPR_GPRMAX_ROOT": settings.gprmax_root.strip(),
        "GPRMAX_SOURCE_DIR": settings.gprmax_root.strip(),
        "UAVGPR_CONDA_ENV": settings.conda_env.strip(),
        "UAVGPR_CONDA_ENV_PREFIX": settings.conda_env_prefix.strip(),
        "UAVGPR_CONDA_EXE": settings.conda_exe.strip(),
        "UAVGPR_GPU_IDS": settings.gpu_ids.strip() or "0",
        "UAVGPR_OMP_THREADS": str(max(0, int(settings.omp_threads))),
        "UAVGPR_USE_GPU": "1" if settings.use_gpu else "0",
        "UAVGPR_USE_CONDA_RUN": "1" if settings.use_conda_run else "0",
    }
    project_path = write_simlab_env(values)
    runtime_root = settings.runtime_root.strip()
    if runtime_root:
        try:
            runtime_env = Path(runtime_root).expanduser() / "uavgpr_runtime.env"
            runtime_env.parent.mkdir(parents=True, exist_ok=True)
            write_simlab_env(values, runtime_env)
        except Exception:
            pass
    return project_path


def _detect_gprmax_version(root: Path) -> str:
    version_file = root / "gprMax" / "_version.py"
    if not version_file.exists():
        return ""
    text = version_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else ""


def _compiled_gprmax_extension_present(root: Path, module_rel: str) -> bool:
    base = root / module_rel
    parent = base.parent
    stem = base.name
    if not parent.exists():
        return False
    for candidate in parent.glob(f"{stem}.*"):
        if candidate.suffix.lower() in _COMPILED_SUFFIXES:
            return True
    return False


def _missing_compiled_gprmax_extensions(root: Path) -> list[str]:
    return [rel for rel in _COMPILED_GPRMAX_MODULES if not _compiled_gprmax_extension_present(root, rel)]


def inspect_gprmax_source(root: str | Path | None) -> GprMaxSourceInfo:
    """Check whether *root* looks like a usable gprMax source directory.

    This is intentionally structural and side-effect free: it does not import,
    build or modify gprMax. The real solver still has to be verified in the
    target conda/CUDA environment.
    """

    raw = "" if root is None else str(root).strip()
    path = Path(raw).expanduser() if raw else Path("")
    if not raw:
        return GprMaxSourceInfo(
            path="",
            exists=False,
            is_source_tree=False,
            detected_version="",
            missing_required=list(_REQUIRED_GPRMAX_FILES),
            present_optional=[],
            user_model_count=0,
            compiled_extension_count=0,
            missing_compiled_extensions=list(_COMPILED_GPRMAX_MODULES),
            message="未填写 gprMax 源码目录。请指向包含 gprMax/__main__.py 和 setup.py 的根目录。",
        )

    exists = path.exists()
    missing = [rel for rel in _REQUIRED_GPRMAX_FILES if not (path / rel).exists()]
    present_optional = [rel for rel in _OPTIONAL_GPRMAX_FILES if (path / rel).exists()]
    user_models = list((path / "user_models").glob("*.in")) if (path / "user_models").exists() else []
    version = _detect_gprmax_version(path) if exists else ""
    missing_compiled = _missing_compiled_gprmax_extensions(path) if exists else list(_COMPILED_GPRMAX_MODULES)
    compiled_count = len(_COMPILED_GPRMAX_MODULES) - len(missing_compiled)
    is_tree = exists and not missing
    if not exists:
        msg = f"目录不存在：{path}"
    elif missing:
        msg = "目录存在，但不像 gprMax 源码根目录；缺少：" + ", ".join(missing)
    else:
        suffix = f"，检测到版本 {version}" if version else ""
        compiled_suffix = f"；已编译扩展 {compiled_count}/{len(_COMPILED_GPRMAX_MODULES)}"
        if missing_compiled:
            compiled_suffix += "，需要在目标环境执行 build_ext 或安装 gprMax 后再真实运行"
        msg = f"gprMax 源码目录结构有效{suffix}{compiled_suffix}。"
    return GprMaxSourceInfo(
        path=str(path),
        exists=exists,
        is_source_tree=is_tree,
        detected_version=version,
        missing_required=missing,
        present_optional=present_optional,
        user_model_count=len(user_models),
        compiled_extension_count=compiled_count,
        missing_compiled_extensions=missing_compiled,
        message=msg,
    )


def run_easy_environment_diagnostics(
    settings: EasyEnvironmentSettings,
    *,
    report_dir: str | Path,
) -> EasyEnvironmentReport:
    """Run environment checks and persist a GUI-friendly diagnostics report."""

    report = run_environment_checks(
        settings.conda_env.strip(),
        settings.use_conda_run,
        settings.gprmax_root.strip(),
        conda_env_prefix=settings.conda_env_prefix.strip(),
        conda_exe=settings.conda_exe.strip(),
    )
    source = inspect_gprmax_source(settings.gprmax_root)
    out = Path(report_dir).expanduser() / "environment_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = EasyEnvironmentReport(environment=report, gprmax_source=source, report_path=out)
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _status_label(ok: bool) -> str:
    return "通过" if ok else "需要处理"


def _guide_for_failed_item(name: str, message: str, settings: EasyEnvironmentSettings) -> str:
    """Return concise Chinese remediation guidance for common environment failures."""

    lower_name = name.lower()
    lower_msg = message.lower()
    if "conda command" in lower_name:
        return "未找到 conda。可安装 Miniconda/Anaconda，或在设置页暂时关闭“使用 conda run”。"
    if lower_name.startswith("conda env"):
        target = settings.conda_env_prefix or settings.conda_env
        return (
            f"未能进入 conda 环境 `{target}`。请确认环境名或环境路径是否正确；"
            "如未创建，可运行 setup_gprmax_4090_windows.bat 统一配置。"
        )
    if "gprmax import" in lower_name:
        if settings.use_conda_run:
            target = settings.conda_env_prefix or settings.conda_env
            flag = "-p" if settings.conda_env_prefix else "-n"
            return (
                f"当前设置会通过 `conda run {flag} {target}` 调用 gprMax。"
                "请在该环境中安装 gprMax，或确认源码目录已以可导入方式安装。"
            )
        return "当前 Python 无法导入 gprMax。请安装 gprMax，或改用已安装 gprMax 的 conda 环境。"
    if "nvidia smi" in lower_name:
        return "未找到 nvidia-smi。CPU 模式仍可运行；GPU 模式需要 NVIDIA 驱动正常安装。"
    if "cuda nvcc" in lower_name:
        return "未找到 nvcc。CPU 模式可忽略；GPU/PyCUDA 构建需要 CUDA Toolkit。"
    if "python module: pyside6" in lower_name:
        return "缺少 PySide6。请安装 requirements_gui.txt，或运行项目提供的 GUI 依赖安装脚本。"
    if lower_name.startswith("python module"):
        module = name.split(":", 1)[-1].strip()
        return f"缺少 Python 模块 `{module}`。请检查 requirements_gui.txt / requirements_gpu_extra.txt。"
    if "gprmax source directory" in lower_name:
        return "请把 gprMax 源码目录指向包含 gprMax/__main__.py、gprMax/gprMax.py 和 setup.py 的根目录。"
    if "not found" in lower_msg:
        return "系统 PATH 中未找到对应命令。请确认软件已安装，并重新打开终端或 GUI。"
    return "请查看上方原始输出，并优先确认路径、环境名和当前 Python 解释器是否符合预期。"


def format_easy_environment_report(report: EasyEnvironmentReport, settings: EasyEnvironmentSettings) -> str:
    """Format diagnostics for the easy GUI settings page.

    The returned text is intended for users. The machine-readable JSON report is
    still written separately by run_easy_environment_diagnostics().
    """

    lines: list[str] = []
    lines.append("诊断摘要")
    lines.append(f"- 总体状态：{_status_label(report.ok)}")
    lines.append(f"- Python：{report.environment.python.split()[0] if report.environment.python else 'unknown'}")
    lines.append(f"- 平台：{report.environment.platform}")
    lines.append(f"- gprMax 源码目录：{report.gprmax_source.message}")
    if report.gprmax_source.detected_version:
        lines.append(f"- gprMax 源码版本：{report.gprmax_source.detected_version}")
    if report.gprmax_source.exists:
        total_ext = len(_COMPILED_GPRMAX_MODULES)
        lines.append(f"- gprMax 编译扩展：{report.gprmax_source.compiled_extension_count}/{total_ext}")
    lines.append(f"- 报告文件：{report.report_path}")

    failed = [item for item in report.environment.items if not item.ok]
    if report.gprmax_source.missing_required:
        lines.append("")
        lines.append("gprMax 源码目录缺失关键文件：")
        for rel in report.gprmax_source.missing_required:
            lines.append(f"- {rel}")
    elif report.gprmax_source.missing_compiled_extensions:
        lines.append("")
        lines.append("gprMax 源码目录尚未检测到完整编译扩展。")
        lines.append("- CPU/GPU 真实求解前，需要在目标 Python/conda 环境中完成 gprMax 构建或安装。")
        lines.append("- 可运行：python setup.py build_ext --inplace")

    if failed:
        lines.append("")
        lines.append("需要处理的检查项：")
        for item in failed:
            lines.append(f"- {item.name}：{item.message}")
            lines.append(f"  建议：{_guide_for_failed_item(item.name, item.message, settings)}")
    else:
        lines.append("")
        lines.append("所有基础环境检查项已通过。")

    lines.append("")
    lines.append("目标机 smoke test 建议顺序：")
    lines.append("1. 在设置页保存 gprMax 源码目录、conda 环境名、GPU/OpenMP 配置。")
    lines.append("2. 先运行 geometry-only 或极小规模 CPU 任务，确认 .in / marker / history 链路。")
    lines.append("3. 再运行 1 个 raw full 任务，确认真实 .out 和 B-scan 可生成。")
    lines.append("4. 最后再启用 GPU 和批量任务。")
    return "\n".join(lines)


def build_runtime_config_for_easy(
    *,
    plan_path: str | Path,
    workspace: str | Path,
    current_manifest: str | Path | None,
    settings: EasyEnvironmentSettings,
) -> AppConfig:
    """Build the AppConfig used by the live batch worker from easy GUI state."""

    cfg = _cfg_from_plan(Path(plan_path).expanduser(), Path(workspace).expanduser())
    manifest_path = Path(current_manifest).expanduser() if current_manifest else None
    if manifest_path and manifest_path.exists():
        cfg.runtime.project_root = str(workspace_from_manifest(manifest_path))
    cfg.runtime.conda_env_gprmax = settings.conda_env.strip()
    cfg.runtime.gprmax_source_dir = settings.gprmax_root.strip() or os.environ.get("GPRMAX_SOURCE_DIR", "")
    cfg.runtime.gpu_enabled = bool(settings.use_gpu)
    cfg.runtime.gpu_ids = settings.gpu_ids.strip() or "0"
    cfg.runtime.omp_threads = max(0, int(settings.omp_threads))
    cfg.runtime.use_conda_run = bool(settings.use_conda_run)
    return cfg


def python_command_from_easy_settings(settings: EasyEnvironmentSettings) -> list[str]:
    """Return the Python command that batch jobs will use for gprMax.

    v0.8.0-alpha.21 supports both a named conda environment and a fixed
    centralized environment prefix under UAVGPR_RUNTIME_ROOT. The prefix path is
    preferred because it does not depend on conda being available on PATH.
    """

    if settings.use_conda_run:
        conda = settings.conda_exe.strip() or os.environ.get("UAVGPR_CONDA_EXE") or "conda"
        if settings.conda_env_prefix.strip():
            return [conda, "run", "-p", settings.conda_env_prefix.strip(), "python"]
        if settings.conda_env.strip():
            return [conda, "run", "-n", settings.conda_env.strip(), "python"]
    if settings.conda_env_prefix.strip():
        candidate = Path(settings.conda_env_prefix.strip()) / ("python.exe" if os.name == "nt" else "bin/python")
        if candidate.exists():
            return [str(candidate)]
    return [os.environ.get("UAVGPR_PYTHON_EXE") or os.environ.get("PYTHON") or sys.executable]


def _run_preflight_step(
    name: str,
    cmd: Sequence[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    timeout: int = 30,
    required: bool = True,
) -> RuntimePreflightStep:
    t0 = time.time()
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
        out = (proc.stdout or "")[-4000:]
        return RuntimePreflightStep(name=name, ok=proc.returncode == 0, required=required, command=[str(x) for x in cmd], output=out, seconds=round(time.time() - t0, 3))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return RuntimePreflightStep(name=name, ok=False, required=required, command=[str(x) for x in cmd], output=f"timeout after {timeout}s\n{out[-2000:]}", seconds=round(time.time() - t0, 3))
    except FileNotFoundError as exc:
        return RuntimePreflightStep(name=name, ok=False, required=required, command=[str(x) for x in cmd], output=f"command not found: {exc}", seconds=round(time.time() - t0, 3))
    except Exception as exc:
        return RuntimePreflightStep(name=name, ok=False, required=required, command=[str(x) for x in cmd], output=repr(exc), seconds=round(time.time() - t0, 3))


def check_gprmax_runtime_preflight(
    settings: EasyEnvironmentSettings,
    *,
    require_gpu: bool | None = None,
    timeout: int = 30,
) -> RuntimePreflightReport:
    """Check the exact Python command before launching a long gprMax batch job.

    The failure shown by Windows users is usually: GUI runs with a system
    python.exe, GPU is enabled, but PyCUDA is installed only in the gprMax conda
    environment. This preflight fails fast before spending 25 runs on the same
    configuration error.
    """

    require_gpu = bool(settings.use_gpu if require_gpu is None else require_gpu)
    source = inspect_gprmax_source(settings.gprmax_root)
    py_cmd = python_command_from_easy_settings(settings)
    env = os.environ.copy()
    if settings.gprmax_root.strip():
        env["PYTHONPATH"] = settings.gprmax_root.strip() + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")
    env["OMP_NUM_THREADS"] = str(max(1, int(settings.omp_threads or 1)))

    steps: list[RuntimePreflightStep] = []
    issues: list[str] = []
    suggestions: list[str] = []

    if not source.is_source_tree:
        steps.append(RuntimePreflightStep("gprMax source tree", False, True, [], source.message, 0.0))
        issues.append(source.message)
    else:
        steps.append(RuntimePreflightStep("gprMax source tree", True, True, [], source.message, 0.0))

    if settings.use_conda_run and not (settings.conda_env.strip() or settings.conda_env_prefix.strip()):
        steps.append(RuntimePreflightStep("conda environment name", False, True, [], "已勾选 conda run，但没有填写 conda 环境名或环境路径。", 0.0))
        issues.append("已勾选 conda run，但没有填写 conda 环境名或环境路径。")
        suggestions.append("在设置页填写 gprMax conda 环境名，或使用 setup_gprmax_4090_windows.bat 生成 UAVGPR_CONDA_ENV_PREFIX；也可关闭 conda run 并确保当前 Python 已安装 gprMax/PyCUDA。")

    steps.append(_run_preflight_step("selected Python", py_cmd + ["-c", "import sys; print(sys.executable); print(sys.version)"], env=env, timeout=timeout))
    steps.append(_run_preflight_step("gprMax import/help", py_cmd + ["-m", "gprMax", "--help"], env=env, cwd=Path(settings.gprmax_root).expanduser() if settings.gprmax_root.strip() else None, timeout=timeout))

    if require_gpu:
        pycuda_code = (
            "import pycuda.driver as drv; "
            "drv.init(); "
            "n=drv.Device.count(); "
            "print('pycuda_device_count', n); "
            "print('device0', drv.Device(0).name() if n else 'none'); "
            "raise SystemExit(0 if n > 0 else 3)"
        )
        steps.append(_run_preflight_step("PyCUDA GPU driver", py_cmd + ["-c", pycuda_code], env=env, timeout=timeout))

    for step in steps:
        if step.required and not step.ok and step.name not in {"gprMax source tree", "conda environment name"}:
            issues.append(f"{step.name} 未通过：{step.output.strip()[:500]}")

    if require_gpu and any((not step.ok and step.name == "PyCUDA GPU driver") for step in steps):
        suggestions.append("当前批量任务已启用 GPU，但所选 Python 不能导入 PyCUDA 或不能访问 CUDA 设备。不要继续批量运行。")
        if (settings.conda_env.strip() or settings.conda_env_prefix.strip()) and not settings.use_conda_run:
            target = settings.conda_env_prefix.strip() or settings.conda_env.strip()
            suggestions.append(f"设置页已填写 conda 环境 `{target}`，请勾选“使用 conda run 调用 gprMax”，然后保存设置并重试。")
        suggestions.append("推荐运行 setup_gprmax_4090_windows.bat 完成 4090/gprMax 环境，并用 scripts\\Verify_4090_GPRMAX_GPU.bat 验证。")
        suggestions.append("不建议把 PyCUDA 临时装进系统 Python；4090/gprMax/GPU 链路应固定在 gprMax conda 环境中。")
    elif any((not step.ok and step.name == "gprMax import/help") for step in steps):
        suggestions.append("当前 Python 不能运行 `python -m gprMax --help`。请确认 gprMax 源码目录、conda 环境和 PYTHONPATH。")

    ok = bool(steps) and all(step.ok for step in steps if step.required)
    return RuntimePreflightReport(ok=ok, python_cmd=py_cmd, gprmax_root=settings.gprmax_root.strip(), require_gpu=require_gpu, steps=steps, issues=issues, suggestions=suggestions)
