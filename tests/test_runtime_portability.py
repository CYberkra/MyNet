from __future__ import annotations

import json
from pathlib import Path

from pgdacsnet.runtime import load_runtime, require_gprmax
from scripts.clean_disposable_sim_artifacts import find_disposable


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_profile_resolves_relative_roots(tmp_path: Path) -> None:
    fake_python = tmp_path / "python.exe"
    fake_python.write_bytes(b"")
    gprmax_root = tmp_path / "gprMax-source"
    (gprmax_root / "gprMax").mkdir(parents=True)
    profile_path = tmp_path / "runtime.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_python": str(fake_python),
                "gprmax_python": str(fake_python),
                "gprmax_source": str(gprmax_root),
                "output_root": "outputs-test",
                "scratch_root": "scratch-test",
                "solver_run_root": "solver-test",
            }
        ),
        encoding="utf-8",
    )
    profile = load_runtime(profile_path)
    assert require_gprmax(profile) == (fake_python, gprmax_root)
    assert profile.output_root == ROOT / "outputs-test"
    assert profile.scratch_root == ROOT / "scratch-test"


def test_cleanup_only_targets_raw_solver_suffixes(tmp_path: Path) -> None:
    raw = tmp_path / "nested" / "run.out"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    (tmp_path / "labels.npy").write_bytes(b"label")
    assert find_disposable([tmp_path]) == [raw]


def test_active_runtime_entrypoints_have_no_machine_drive_paths() -> None:
    paths = [
        ROOT / ".claude" / "settings.json",
        ROOT / ".claude" / "hooks" / "venv_python_guard.py",
        ROOT / "scripts" / "run_native_256_release_pilot.py",
        ROOT / "data" / "simulation_contract_v2" / "RECOMMENDED_NATIVE_256_V1.md",
        ROOT / ".claude" / "skills" / "gprmax-physics-audit" / "SKILL.md",
    ]
    for path in paths:
        assert not __import__("re").search(r"[A-Za-z]:[\\/]", path.read_text(encoding="utf-8")), path
