from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("handoff_record", ROOT / "scripts" / "handoff_record.py")
assert SPEC and SPEC.loader
handoff_record = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff_record)


def _valid_record() -> dict:
    readme_sha = hashlib.sha256((ROOT / "README.md").read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "task_id": "20260715-governance-handoff-test",
        "title": "Validate the handoff contract",
        "status": "complete",
        "work_type": "governance",
        "created_at": "2026-07-15T00:00:00+00:00",
        "updated_at": "2026-07-15T00:00:00+00:00",
        "implementation": {
            "branch": "agent/handoff-test-20260715",
            "implementation_head": "a" * 40,
            "base_branch": "master",
            "base_commit": "b" * 40,
            "remote": "https://github.com/CYberkra/MyNet.git",
            "worktree_dirty": False,
            "changed_files": ["docs/HANDOFF_STANDARD.md"],
        },
        "scope": {
            "objective": "Validate portable handoff records.",
            "in_scope": ["Handoff schema validation"],
            "out_of_scope": ["Training"],
        },
        "changes": ["Added a versioned handoff contract."],
        "artifacts": [{"role": "entrypoint", "path": "README.md", "sha256": readme_sha}],
        "validation": [
            {
                "command": "python -m pytest -q tests/test_handoff_record.py",
                "status": "passed",
                "result": "All handoff tests passed.",
                "timestamp": "2026-07-15T00:00:00+00:00",
            }
        ],
        "decisions": [],
        "risks": ["None known."],
        "blockers": [],
        "next_actions": [
            {
                "priority": "P1",
                "action": "Use the contract for milestone handoffs.",
                "owner": "next agent",
                "done_when": "A final record validates.",
            }
        ],
        "resume": {
            "required_capabilities": [],
            "entry_document": "docs/HANDOFF_STANDARD.md",
            "command": "python scripts/handoff_record.py --help",
        },
    }


def test_final_record_passes_and_verifies_artifacts() -> None:
    errors = handoff_record.validate_record(_valid_record(), final=True, verify_artifacts=True)
    assert errors == []


def test_final_record_rejects_absolute_machine_path() -> None:
    record = _valid_record()
    record["resume"]["command"] = r"F:\codex\envs\python.exe train.py"
    errors = handoff_record.validate_record(record, final=True, verify_artifacts=False)
    assert any("absolute path" in error for error in errors)


def test_complete_record_rejects_dirty_or_blocked_state() -> None:
    record = _valid_record()
    record["implementation"]["worktree_dirty"] = True
    record["blockers"] = [{"issue": "missing data", "unblock_when": "data arrives"}]
    errors = handoff_record.validate_record(record, final=True, verify_artifacts=False)
    assert any("clean implementation worktree" in error for error in errors)
    assert any("cannot contain active blockers" in error for error in errors)


def test_blocked_record_requires_unblock_information() -> None:
    record = _valid_record()
    record["status"] = "blocked"
    record["blockers"] = []
    errors = handoff_record.validate_record(record, final=True, verify_artifacts=False)
    assert any("requires at least one blocker" in error for error in errors)


def test_porcelain_parser_preserves_leading_dot() -> None:
    paths = handoff_record._porcelain_paths(" M .gitignore\n?? docs/new.md")
    assert paths == [".gitignore", "docs/new.md"]


def test_absolute_path_is_rejected_inside_a_command() -> None:
    record = _valid_record()
    record["resume"]["command"] = r"python train.py --data F:\private\data"
    errors = handoff_record.validate_record(record, final=True, verify_artifacts=False)
    assert any("absolute path" in error for error in errors)
