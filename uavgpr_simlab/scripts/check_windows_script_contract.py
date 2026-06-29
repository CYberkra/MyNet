from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT_FILES = [
    "setup_uavgpr_gpu_runtime_windows.bat",
    "setup_local_3060_gpu_runtime.bat",
    "setup_laptop_4090_gpu_runtime.bat",
    "setup_gprmax_4090_windows.bat",
    "scripts/OneClick_Install_GPRMAX_Windows.bat",
    "scripts/Verify_Current_GPU_Runtime.bat",
    "scripts/Verify_4090_GPRMAX_GPU.bat",
    "scripts/Generate_4090_Formal_Dataset.bat",
    "scripts/Generate_4090_Validation_Dataset.bat",
    "scripts/Generate_3060_Quick_Dataset.bat",
    "scripts/Preview_Example_CSV.bat",
    "scripts/Run_Full_Pipeline_Example.bat",
    "scripts/Setup_GUI_Only.bat",
    "scripts/Configure_Local_CPU_GprMax.bat",
    "install_gui_deps_into_gprmax_env.bat",
    "run_gui.bat",
]
BOOTSTRAP_KEYS = [
    "UAVGPR_RUNTIME_ROOT",
    "UAVGPR_MINICONDA_DIR",
    "UAVGPR_CONDA_EXE",
    "UAVGPR_CONDA_ENV_PREFIX",
    "UAVGPR_GPRMAX_ROOT",
    "GPRMAX_SOURCE_DIR",
    "UAVGPR_CONDA_ENV",
    "UAVGPR_PYTHON_EXE",
    "UAVGPR_USE_CONDA_RUN",
    "UAVGPR_GPU_IDS",
    "UAVGPR_USE_GPU",
    "UAVGPR_OMP_THREADS",
    "UAVGPR_MACHINE_PROFILE",
    "UAVGPR_GPU_RUNTIME_ENV",
    "UAVGPR_RUN_SCALE",
]
PS_ENV_KEYS = [
    "UAVGPR_RUNTIME_ROOT",
    "UAVGPR_MINICONDA_DIR",
    "UAVGPR_CONDA_EXE",
    "UAVGPR_CONDA_ENV_PREFIX",
    "UAVGPR_GPRMAX_ROOT",
    "GPRMAX_SOURCE_DIR",
    "UAVGPR_CONDA_ENV",
    "UAVGPR_USE_CONDA_RUN",
    "UAVGPR_GPU_IDS",
    "UAVGPR_USE_GPU",
    "UAVGPR_GPU_ENABLED",
    "UAVGPR_OMP_THREADS",
    "UAVGPR_MACHINE_PROFILE",
    "UAVGPR_GPU_RUNTIME_ENV",
    "UAVGPR_RUN_SCALE",
]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def check_bat_basics() -> Check:
    failed: list[str] = []
    details: list[str] = []
    for rel in BAT_FILES:
        path = ROOT / rel
        if not path.exists():
            failed.append(f"missing {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()
        checks = {
            "@echo off": "@echo off" in lower,
            "setlocal": "setlocal" in lower,
            "exit /b": "exit /b" in lower,
            "quoted_cd": ('cd /d "%~dp0' in lower) or ('pushd "%~dp0' in lower),
        }
        if rel.startswith("scripts/Generate_4090") or rel.endswith("Verify_Current_GPU_Runtime.bat") or rel == "run_gui.bat":
            checks["bootstrap"] = "windows_runtime_bootstrap.bat" in lower
            checks["bootstrap_errorlevel"] = "if errorlevel 1" in lower
        if rel.endswith("Verify_4090_GPRMAX_GPU.bat"):
            checks["legacy_verify_delegates"] = "verify_current_gpu_runtime.bat" in lower
        if rel == "setup_uavgpr_gpu_runtime_windows.bat":
            checks["explicit_powershell_path"] = "windowspowershell" in lower and "ps_exe" in lower
        if rel in {"scripts/Preview_Example_CSV.bat", "scripts/Run_Full_Pipeline_Example.bat", "scripts/Setup_GUI_Only.bat"}:
            checks["uses_bootstrap"] = "windows_runtime_bootstrap.bat" in lower
            checks["uses_py_run"] = "%py_run%" in lower
            checks["no_bare_python_module_run"] = "python -m uavgpr_simlab" not in lower and "python -m pip" not in lower
        bad = [k for k, ok in checks.items() if not ok]
        details.append(f"{rel}:" + ("ok" if not bad else ",".join(bad)))
        if bad:
            failed.append(f"{rel} missing {bad}")
    return Check("bat_basics", not failed, "; ".join(details))


def check_powershell_contract() -> Check:
    texts = {rel: _read(rel) for rel in ["scripts/install_gprmax_windows.ps1", "scripts/setup_gprmax_4090_windows.ps1"]}
    failed: list[str] = []
    details: list[str] = []
    for rel, text in texts.items():
        has_steps = all(f"[{i}/14]" in text for i in range(1, 15))
        has_env = all(key in text for key in PS_ENV_KEYS)
        has_external_source = all(tok in text for tok in ["Resolve-DefaultGprMaxDir", "This release no longer bundles gprMax", "-GprMaxDir", "AllowCloneGprMax"])
        has_optional_zip = "GprMaxZip" in text and "Expand-Archive" in text
        has_conda_env = "environment_gprmax_4090_windows.yml" in text and "conda" in text.lower() and "CondaEnvPrefix" in text
        has_verify = "check_4090_gprmax_gpu.py" in text and "MachineProfile" in text and "uavgpr_gprmax_py310_gpu" in text
        has_safe_path = all(tok in text for tok in ["Test-PathQuiet", "Join-PathSafe", "Test-GprMaxSourceDir", "Ensure-DirectoryChecked", "Test-PathRootAvailable"])
        has_conda_fallback = all(tok in text for tok in ["Find-ExternalCondaExe", "NoExternalCondaFallback", "controller only"])
        has_native_array_invocation = "& $exe @argv" in text and "Start-Process -FilePath" not in text
        has_core_path_refresh = all(tok in text for tok in ["System32", "WindowsPowerShell", "OpenSSH", "Refresh-Path", "Test-CoreWindowsTools", "chcp.com", "where.exe"])
        has_vsdev_env = all(tok in text for tok in ["Initialize-MSVCBuildEnvironment", "Import-VSDeveloperEnvironment", "VsDevCmd.bat", "vcvars64.bat"])
        has_sdk_header_guard = all(tok in text for tok in ["Test-HeaderInIncludePath", "Find-WindowsSdkHeader", "io.h", "windows.h", "Windows SDK/UCRT"])
        has_distutils_sdk = "DISTUTILS_USE_SDK" in text and "MSSdk" in text
        has_cuda_header_guard = all(tok in text for tok in ["Test-CudaBuildEnvironment", "cuda.h", "CUDA_PATH", "CUDA_HOME"])
        has_env_probe = "import sys, numpy, Cython, setuptools" in text
        has_compiled_extension_guard = "Compiled gprMax .pyd count" in text and "*.pyd" in text
        has_vs_installer_sdk_components = "Microsoft.VisualStudio.Component.Windows11SDK" in text and "Microsoft.VisualStudio.Component.Windows10SDK" in text
        ok = has_steps and has_env and has_external_source and has_optional_zip and has_conda_env and has_verify and has_safe_path and has_conda_fallback and has_native_array_invocation and has_core_path_refresh and has_vsdev_env and has_sdk_header_guard and has_distutils_sdk and has_cuda_header_guard and has_env_probe and has_compiled_extension_guard and has_vs_installer_sdk_components
        details.append(f"{rel}: steps={has_steps} env={has_env} external_gprmax={has_external_source} optional_zip={has_optional_zip} conda={has_conda_env} verify={has_verify} safe_path={has_safe_path} conda_fallback={has_conda_fallback} native_array={has_native_array_invocation} path_refresh={has_core_path_refresh} vsdev_env={has_vsdev_env} sdk_headers={has_sdk_header_guard} distutils_sdk={has_distutils_sdk} cuda_headers={has_cuda_header_guard} env_probe={has_env_probe} compiled_ext_guard={has_compiled_extension_guard} vs_sdk_components={has_vs_installer_sdk_components}")
        if not ok:
            failed.append(rel)
    # The two setup scripts should remain synchronized.
    normalized = [re.sub(r"\s+", " ", t.strip()) for t in texts.values()]
    if len(set(normalized)) != 1:
        failed.append("install/setup PowerShell scripts are not synchronized")
    return Check("powershell_contract", not failed, "; ".join(details) + ("; failed=" + ", ".join(failed) if failed else ""))


def check_bootstrap_contract() -> Check:
    text = _read("scripts/windows_runtime_bootstrap.bat")
    missing_keys = [key for key in BOOTSTRAP_KEYS if key not in text]
    required_tokens = ["PYTHONPATH", "GPRMAX_SOURCE_DIR", "PY_RUN", "PY_EXE", "UAVGPR_RUNTIME_ROOT", "UAVGPR_CONDA_ENV_PREFIX", "where python", "exit /b 0", "USE_PREFIX_PYTHON", "PYTHON_SELECTED"]
    missing_tokens = [tok for tok in required_tokens if tok.lower() not in text.lower()]
    fragile_tokens = [tok for tok in ['if "%PY_RUN%"==""', 'set "PY_RUN=%UAVGPR_CONDA_ENV_PREFIX%\\python.exe"'] if tok.lower() in text.lower()]
    ok = not missing_keys and not missing_tokens and not fragile_tokens
    return Check("bootstrap_contract", ok, f"missing_keys={missing_keys}; missing_tokens={missing_tokens}; fragile_tokens={fragile_tokens}")



def check_launcher_contract() -> Check:
    text = _read("run_gui.bat")
    failed: list[str] = []
    if "windows_runtime_bootstrap.bat" not in text:
        failed.append("run_gui missing bootstrap")
    if "%PY_RUN% -m uavgpr_simlab.app" not in text:
        failed.append("run_gui does not launch through PY_RUN")
    if 'if "%UAVGPR_CONDA_ENV%"==""' in text:
        failed.append("run_gui still branches on UAVGPR_CONDA_ENV instead of PY_RUN")
    if '%PY_RUN% -c "import sys; print(sys.executable)"' not in text:
        failed.append("run_gui missing selected Python probe")
    return Check("launcher_contract", not failed, "; ".join(failed) if failed else "run_gui uses bootstrap-selected PY_RUN only")

def check_4090_bats() -> Check:
    failed: list[str] = []
    details: list[str] = []
    verify = _read("scripts/Verify_Current_GPU_Runtime.bat")
    if "check_current_gpu_runtime_report.json" not in verify or "--gpu-ids" not in verify or "--gprmax-root" not in verify:
        failed.append("Verify_Current_GPU_Runtime.bat does not pass fixed report/gpu/root arguments")
    legacy_verify = _read("scripts/Verify_4090_GPRMAX_GPU.bat")
    if "Verify_Current_GPU_Runtime.bat" not in legacy_verify:
        failed.append("Verify_4090_GPRMAX_GPU.bat is not mapped to generic verifier")
    for rel, plan in [
        ("scripts/Generate_4090_Formal_Dataset.bat", "run_plan_4090_formal.yaml"),
        ("scripts/Generate_4090_Validation_Dataset.bat", "run_plan_4090_validation_hifi.yaml"),
        ("scripts/Generate_3060_Quick_Dataset.bat", "run_plan_3060_quick.yaml"),
    ]:
        text = _read(rel)
        ok = "windows_runtime_bootstrap.bat" in text and plan in text and "%PY_RUN%" in text and "uavgpr_simlab.cli generate" in text
        details.append(f"{rel}:{'ok' if ok else 'bad'}")
        if not ok:
            failed.append(rel)
    return Check("4090_bats", not failed, "; ".join(details) + ("; failed=" + ", ".join(failed) if failed else ""))


def check_generated_run_all_contract() -> Check:
    paths = sorted((ROOT / "workspace").glob("yingshan_sceneworld_*_v080a14/logs/run_all_gprmax.bat"))
    failed: list[str] = []
    details: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        ok = "windows_runtime_bootstrap.bat" in text and "scripts\\run_all_gprmax.py" in text and "raw,target_only,background_only,clutter_only,air_only" in text
        rel = str(path.relative_to(ROOT))
        details.append(f"{rel}:{'ok' if ok else 'bad'}")
        if not ok:
            failed.append(rel)
    return Check("generated_run_all_contract", not failed, "; ".join(details) if details else "no generated run_all_gprmax.bat skeletons found")



def check_python_source_contract() -> Check:
    failed: list[str] = []
    details: list[str] = []
    runner = _read("src/uavgpr_simlab/core/runner.py")
    scenario = _read("src/uavgpr_simlab/core/scenario.py")
    if "subprocess.list2cmdline" not in runner:
        failed.append("runner.command_to_string does not use subprocess.list2cmdline")
    if 'pushd "%~dp0\\\\.."' not in scenario:
        failed.append("write_manifest_commands_bat does not quote pushd")
    if "windows_runtime_bootstrap.bat" not in scenario or '"%PY_RUN%"' not in scenario:
        failed.append("write_manifest_commands_bat does not route generated BAT through runtime bootstrap/PY_RUN")
    details.append("runner_list2cmdline=" + str("subprocess.list2cmdline" in runner))
    details.append("scenario_bootstrap=" + str("windows_runtime_bootstrap.bat" in scenario))
    details.append("scenario_py_run=" + str('"%PY_RUN%"' in scenario))
    return Check("python_source_contract", not failed, "; ".join(details) + ("; failed=" + ", ".join(failed) if failed else ""))

def main() -> int:
    checks = [
        check_bat_basics(),
        check_powershell_contract(),
        check_bootstrap_contract(),
        check_launcher_contract(),
        check_4090_bats(),
        check_generated_run_all_contract(),
        check_python_source_contract(),
    ]
    payload = {"ok": all(c.ok for c in checks), "checks": [c.to_dict() for c in checks]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
