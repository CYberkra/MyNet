from __future__ import annotations

import ast
import csv
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - release env should include PyYAML
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VARIANTS = ["raw", "target_only", "background_only", "clutter_only", "air_only"]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from uavgpr_simlab.core.dataset_contract import validate_dataset_skeleton
from uavgpr_simlab.core.run_dashboard import summarize_dataset_run_dashboard
from uavgpr_simlab.core.workspace_relocator import relocate_workspace_paths
REQUIRED_FILES = [
    "pyproject.toml",
    "src/uavgpr_simlab/__init__.py",
    "src/uavgpr_simlab/app.py",
    "src/uavgpr_simlab/gui/easy_window.py",
    "scripts/check_architecture_guard.py",
    "scripts/check_easy_ui_contract.py",
    "scripts/check_windows_script_contract.py",
    "scripts/check_4090_gprmax_gpu.py",
    "scripts/check_dataset_skeleton.py",
    "src/uavgpr_simlab/core/dataset_contract.py",
    "src/uavgpr_simlab/core/run_dashboard.py",
    "src/uavgpr_simlab/core/workspace_relocator.py",
    "src/uavgpr_simlab/gui/controllers/batch_actions.py",
    "src/uavgpr_simlab/gui/controllers/batch_recent_preview.py",
    "scripts/check_workspace_relocation.py",
    "scripts/install_gprmax_windows.ps1",
    "scripts/setup_gprmax_4090_windows.ps1",
    "scripts/windows_runtime_bootstrap.bat",
    "scripts/Verify_Current_GPU_Runtime.bat",
    "scripts/Verify_4090_GPRMAX_GPU.bat",
    "scripts/Generate_4090_Formal_Dataset.bat",
    "scripts/Generate_4090_Validation_Dataset.bat",
    "setup_uavgpr_gpu_runtime_windows.bat",
    "setup_local_3060_gpu_runtime.bat",
    "setup_laptop_4090_gpu_runtime.bat",
    "setup_gprmax_4090_windows.bat",
    "run_gui.bat",
    "configs/environment_gprmax_4090_windows.yml",
    "configs/run_plan_4090_formal.yaml",
    "configs/run_plan_4090_validation_hifi.yaml",
]
DOC_VERSION_FILES = ["README.md", "CURRENT_STATE.md", "DEV_HANDOFF.md", "CHANGELOG.md"]
SIMLAB_ENV_KEYS = [
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


def _read(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def _project_version() -> str:
    data = tomllib.loads(_read("pyproject.toml"))
    return str(data["project"]["version"])


def _init_value(name: str) -> str:
    tree = ast.parse(_read("src/uavgpr_simlab/__init__.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return str(ast.literal_eval(node.value))
    return ""


def _init_version() -> str:
    return _init_value("__version__")


def check_version_sync() -> Check:
    project_v = _project_version()
    init_v = _init_version()
    display_v = _init_value("__display_version__")
    alpha = project_v.replace("a", "-alpha.") if "a" in project_v else project_v
    ok = project_v == init_v and bool(project_v) and bool(display_v) and (display_v.endswith(alpha.split("-alpha.")[-1]) or "alpha" in display_v)
    return Check("version_sync", ok, f"pyproject={project_v}; __init__={init_v}; display={display_v}")


def check_doc_versions() -> Check:
    version = _project_version()
    alpha = version.replace("a", "-alpha.") if "a" in version else version
    missing: list[str] = []
    details: list[str] = []
    for rel in DOC_VERSION_FILES:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        ok = version in text or alpha in text
        details.append(f"{rel}={'ok' if ok else 'missing'}")
        if not ok:
            missing.append(rel)
    return Check("doc_versions", not missing, "; ".join(details))


def check_required_files() -> Check:
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).exists()]
    return Check("required_files", not missing, "missing=" + ", ".join(missing) if missing else f"{len(REQUIRED_FILES)} required files present")


def check_no_bundled_gprmax_zip() -> Check:
    bundled = sorted(p.name for p in ROOT.glob("gprMax*.zip"))
    return Check(
        "no_bundled_gprmax_zip",
        not bundled,
        "no gprMax zip bundled; persistent external solver expected" if not bundled else "bundled=" + ", ".join(bundled),
    )


def _run(cmd: list[str], timeout: int = 45) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return proc.returncode == 0, (proc.stdout or "")[-4000:]
    except Exception as exc:
        return False, repr(exc)


def check_entry_help() -> Check:
    checks = [
        ([sys.executable, "-m", "uavgpr_simlab.app", "--help"], "app"),
        ([sys.executable, "-m", "uavgpr_simlab.cli", "--help"], "cli"),
    ]
    failed: list[str] = []
    details: list[str] = []
    for cmd, name in checks:
        ok, out = _run(cmd)
        details.append(f"{name}={'ok' if ok else 'failed'}")
        if not ok:
            failed.append(f"{name}: {out[:300].replace(chr(10), ' ')}")
    return Check("entry_help", not failed, "; ".join(details) if not failed else " | ".join(failed))


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not available")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_4090_run_plans() -> Check:
    failed: list[str] = []
    details: list[str] = []
    for rel in ["configs/run_plan_4090_formal.yaml", "configs/run_plan_4090_validation_hifi.yaml"]:
        path = ROOT / rel
        try:
            data = _load_yaml(path)
        except Exception as exc:
            failed.append(f"{rel}: yaml error {exc}")
            continue
        variants = [str(x) for x in data.get("components", [])]
        run = data.get("run", {}) if isinstance(data.get("run"), dict) else {}
        scene = data.get("scene", {}) if isinstance(data.get("scene"), dict) else {}
        ok = (
            variants == EXPECTED_VARIANTS
            and bool(run.get("use_gpu"))
            and str(scene.get("dimension", "")).upper() == "2D"
            and int(scene.get("samples", 0)) == 501
            and int(scene.get("time_window_ns", 0)) == 700
            and int(data.get("scene_count", 0)) > 0
        )
        details.append(f"{Path(rel).name}: variants={variants}, gpu={run.get('use_gpu')}, cases={data.get('scene_count')}")
        if not ok:
            failed.append(rel)
    return Check("4090_run_plans", not failed, "; ".join(details) + ("; failed=" + ", ".join(failed) if failed else ""))


def check_simlab_env_contract() -> Check:
    writers = (_read("scripts/install_gprmax_windows.ps1") + "\n" + _read("scripts/setup_gprmax_4090_windows.ps1"))
    bootstrap = _read("scripts/windows_runtime_bootstrap.bat")
    missing_written = [k for k in SIMLAB_ENV_KEYS if k not in writers]
    missing_loaded = [k for k in SIMLAB_ENV_KEYS if k not in bootstrap]
    ok = not missing_written and not missing_loaded
    return Check(
        "simlab_env_contract",
        ok,
        f"written_missing={missing_written}; bootstrap_missing={missing_loaded}",
    )


def check_workspace_skeletons() -> Check:
    workspace = ROOT / "workspace"
    allowed = {"yingshan_sceneworld_ultra_tiny_v080a14", "yingshan_sceneworld_smoke_v080a14"}
    if not workspace.exists():
        return Check("workspace_skeletons", True, "workspace not present in package")
    actual = {p.name for p in workspace.iterdir() if p.is_dir()}
    unexpected = sorted(actual - allowed)
    details: list[str] = []
    for name in sorted(actual & allowed):
        manifest = next((workspace / name / "datasets").glob("*manifest*.csv"), None)
        run_bat = workspace / name / "logs" / "run_all_gprmax.bat"
        details.append(f"{name}: manifest={bool(manifest)} run_bat={run_bat.exists()}")
        if not manifest or not run_bat.exists():
            unexpected.append(f"{name}:missing_contract_files")
        else:
            try:
                contract = validate_dataset_skeleton(manifest, expected_variants=EXPECTED_VARIANTS, write_report=False)
                dashboard = summarize_dataset_run_dashboard(manifest, expected_variants=EXPECTED_VARIANTS, write_report=False)
                details.append(f"{name}: contract_ok={contract.ok} cases={contract.case_count} rows={contract.row_count} warnings={contract.warning_count}; dashboard_total={dashboard.total} pending={dashboard.pending}")
                if not contract.ok:
                    unexpected.append(f"{name}:dataset_contract_errors={contract.error_count}")
                if dashboard.total != contract.row_count:
                    unexpected.append(f"{name}:dashboard_row_mismatch={dashboard.total}!={contract.row_count}")
            except Exception as exc:
                unexpected.append(f"{name}:contract_error={exc}")
    return Check("workspace_skeletons", not unexpected, "; ".join(details) + ("; unexpected=" + ", ".join(unexpected) if unexpected else ""))


def check_docs_root_clean() -> Check:
    docs = ROOT / "docs"
    offenders: list[str] = []
    stable_exceptions = {"YINGSHAN_REAL_DATA_AUDIT.md"}
    versioned_doc_patterns = ("_v080a", "_v0_", "CURRENT_ARCHITECTURE_v", "MULTI_MACHINE_GPU_RUNTIME_v", "RUNTIME_ROOT_4090_SETUP_v")
    historical_doc_patterns = ("AUDIT", "V0_", "OPTIMIZATION", "REPORT")
    for p in docs.glob("*.md"):
        if p.name not in stable_exceptions and (any(tag in p.name for tag in historical_doc_patterns) or any(tag in p.name for tag in versioned_doc_patterns)):
            offenders.append(p.name)
    for p in docs.glob("*.png"):
        if "UavGPR-SimLab_v" in p.name or "GUI" in p.name:
            offenders.append(p.name)
    detail = "offenders=" + ", ".join(sorted(offenders)) if offenders else "root docs ok; versioned docs archived under docs/history"
    return Check("docs_root_clean", not offenders, detail)



def check_workspace_relocator() -> Check:
    manifest = ROOT / "workspace" / "yingshan_sceneworld_smoke_v080a14" / "datasets" / "yingshan_sceneworld_smoke_v080a14_manifest.csv"
    if not manifest.exists():
        return Check("workspace_relocator", False, f"missing sample manifest: {manifest}")
    try:
        rep = relocate_workspace_paths(manifest, dry_run=True, write_report=False, validate_after=True)
        ok = rep.ok and rep.change_count == 0 and rep.dataset_contract_ok
        return Check("workspace_relocator", ok, f"dry_run ok={rep.ok}; changes={rep.change_count}; abs={rep.absolute_path_count}; contract={rep.dataset_contract_ok}")
    except Exception as exc:
        return Check("workspace_relocator", False, repr(exc))

def main() -> int:
    checks = [
        check_version_sync(),
        check_doc_versions(),
        check_required_files(),
        check_no_bundled_gprmax_zip(),
        check_entry_help(),
        check_4090_run_plans(),
        check_simlab_env_contract(),
        check_workspace_skeletons(),
        check_workspace_relocator(),
        check_docs_root_clean(),
    ]
    payload = {"ok": all(c.ok for c in checks), "checks": [c.to_dict() for c in checks]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
