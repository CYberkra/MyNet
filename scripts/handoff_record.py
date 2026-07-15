"""Create and validate portable PGDA-CSNet task handoff records."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
TASK_ID_RE = re.compile(r"^[0-9]{8}-[a-z0-9]+-[a-z0-9-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
UNIX_ABSOLUTE_RE = re.compile(r"(?:^|\s)/(?:home|Users|mnt)/")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WORK_TYPES = {
    "code",
    "model",
    "data",
    "simulation",
    "training",
    "evaluation",
    "governance",
    "documentation",
    "mixed",
}
VALID_RESULTS = {"passed", "expected_fail"}
PLACEHOLDERS = ("TODO", "<fill", "<replace", "TBD")


def _git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.rstrip()


def _portable_remote(value: str) -> str:
    if "://" not in value:
        return value
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _changed_files(base_branch: str) -> list[str]:
    candidates = []
    for revision in (f"origin/{base_branch}", base_branch):
        if _git("rev-parse", "--verify", revision, check=False):
            candidates = _git("diff", "--name-only", f"{revision}...HEAD").splitlines()
            break
    candidates.extend(_porcelain_paths(_git("status", "--porcelain")))
    return sorted({path.replace("\\", "/") for path in candidates if path})


def _porcelain_paths(value: str) -> list[str]:
    return [line[3:] for line in value.splitlines() if len(line) > 3]


def build_draft(task_id: str, title: str, work_type: str, objective: str, base_branch: str) -> dict[str, Any]:
    now = _timestamp()
    branch = _git("branch", "--show-current") or "detached"
    head = _git("rev-parse", "HEAD")
    base_ref = f"origin/{base_branch}"
    base_commit = _git("merge-base", "HEAD", base_ref, check=False) or _git("rev-parse", "HEAD")
    remote = _portable_remote(_git("remote", "get-url", "origin", check=False))
    dirty = bool(_git("status", "--porcelain"))
    return {
        "schema_version": 1,
        "task_id": task_id,
        "title": title,
        "status": "in_progress",
        "work_type": work_type,
        "created_at": now,
        "updated_at": now,
        "implementation": {
            "branch": branch,
            "implementation_head": head,
            "base_branch": base_branch,
            "base_commit": base_commit,
            "remote": remote,
            "worktree_dirty": dirty,
            "changed_files": _changed_files(base_branch),
        },
        "scope": {
            "objective": objective,
            "in_scope": ["TODO: list durable work included in this task"],
            "out_of_scope": ["TODO: list explicit exclusions"],
        },
        "changes": ["TODO: summarize durable changes"],
        "artifacts": [],
        "validation": [
            {
                "command": "TODO: exact command",
                "status": "not_run",
                "result": "TODO: concise observed result",
                "timestamp": now,
            }
        ],
        "decisions": [],
        "risks": ["TODO: residual risk or 'none known'"],
        "blockers": [],
        "next_actions": [
            {
                "priority": "P0",
                "action": "TODO: next concrete action",
                "owner": "TODO: role or person",
                "done_when": "TODO: observable completion condition",
            }
        ],
        "resume": {
            "required_capabilities": [],
            "entry_document": "docs/current_state.md",
            "command": "TODO: exact first command or 'none'",
        },
    }


def _walk_strings(value: Any, location: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.append((location, value))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_strings(item, f"{location}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_walk_strings(item, f"{location}.{key}"))
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: dict[str, Any], *, final: bool, verify_artifacts: bool) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "task_id",
        "title",
        "status",
        "work_type",
        "implementation",
        "scope",
        "changes",
        "artifacts",
        "validation",
        "decisions",
        "risks",
        "blockers",
        "next_actions",
        "resume",
    }
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing top-level fields: {missing}")
        return errors
    if record["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if not TASK_ID_RE.fullmatch(str(record["task_id"])):
        errors.append("task_id must match YYYYMMDD-<area>-<short-purpose>")
    if record["work_type"] not in WORK_TYPES:
        errors.append(f"unknown work_type: {record['work_type']!r}")
    if record["status"] not in {"in_progress", "complete", "blocked"}:
        errors.append(f"unknown status: {record['status']!r}")

    implementation = record.get("implementation") or {}
    if not SHA_RE.fullmatch(str(implementation.get("implementation_head", ""))):
        errors.append("implementation.implementation_head must be a full Git SHA")
    if final and implementation.get("worktree_dirty") is not False:
        errors.append("final handoff must reference a clean implementation worktree")
    changed_files = implementation.get("changed_files") or []
    if final and not changed_files:
        errors.append("final handoff must list changed_files")

    for location, value in _walk_strings(record):
        if (
            WINDOWS_ABSOLUTE_RE.search(value)
            or UNIX_ABSOLUTE_RE.search(value)
            or value.startswith("\\\\")
        ):
            errors.append(f"machine-specific absolute path at {location}: {value!r}")
        if final and any(marker.lower() in value.lower() for marker in PLACEHOLDERS):
            errors.append(f"unresolved placeholder at {location}: {value!r}")

    validations = record.get("validation") or []
    if final and not validations:
        errors.append("final handoff must include validation results")
    for index, item in enumerate(validations):
        if not isinstance(item, dict):
            errors.append(f"validation[{index}] must be an object")
            continue
        if final and item.get("status") not in VALID_RESULTS:
            errors.append(f"validation[{index}].status must be passed or expected_fail")
        for field in ("command", "result", "timestamp"):
            if final and not str(item.get(field, "")).strip():
                errors.append(f"validation[{index}] lacks {field}")
        if item.get("status") == "expected_fail" and not str(item.get("result", "")).strip():
            errors.append(f"validation[{index}] expected_fail requires a result/reason")

    blockers = record.get("blockers") or []
    if record["status"] == "blocked" and not blockers:
        errors.append("blocked handoff requires at least one blocker")
    if record["status"] == "complete" and blockers:
        errors.append("complete handoff cannot contain active blockers")
    for index, blocker in enumerate(blockers):
        if not isinstance(blocker, dict):
            errors.append(f"blockers[{index}] must be an object")
            continue
        for field in ("issue", "unblock_when"):
            if not str(blocker.get(field, "")).strip():
                errors.append(f"blockers[{index}] lacks {field}")
    if final and record["status"] == "in_progress":
        errors.append("final handoff status must be complete or blocked")
    if final and not (record.get("next_actions") or []):
        errors.append("final handoff must include at least one next action")
    for index, action in enumerate(record.get("next_actions") or []):
        if not isinstance(action, dict):
            errors.append(f"next_actions[{index}] must be an object")
            continue
        for field in ("priority", "action", "owner", "done_when"):
            if not str(action.get(field, "")).strip():
                errors.append(f"next_actions[{index}] lacks {field}")

    scope = record.get("scope") or {}
    if final:
        if not str(scope.get("objective", "")).strip():
            errors.append("scope lacks objective")
        for field in ("in_scope", "out_of_scope"):
            if not isinstance(scope.get(field), list) or not scope[field]:
                errors.append(f"scope.{field} must be a non-empty list")
        if not (record.get("changes") or []):
            errors.append("final handoff must summarize changes")
        if not (record.get("artifacts") or []):
            errors.append("final handoff must identify at least one durable artifact")

    resume = record.get("resume") or {}
    if final:
        if not str(resume.get("entry_document", "")).strip():
            errors.append("resume lacks entry_document")
        if not str(resume.get("command", "")).strip():
            errors.append("resume lacks command")
        if not isinstance(resume.get("required_capabilities"), list):
            errors.append("resume.required_capabilities must be a list")

    for index, artifact in enumerate(record.get("artifacts") or []):
        if not isinstance(artifact, dict):
            errors.append(f"artifacts[{index}] must be an object")
            continue
        rel = str(artifact.get("path", ""))
        if not rel:
            errors.append(f"artifacts[{index}] lacks path")
            continue
        if not str(artifact.get("role", "")).strip():
            errors.append(f"artifacts[{index}] lacks role")
        path = ROOT / rel
        expected = str(artifact.get("sha256", ""))
        if final and not SHA256_RE.fullmatch(expected):
            errors.append(f"artifacts[{index}].sha256 must be a full SHA256")
        if verify_artifacts:
            if not path.is_file():
                errors.append(f"artifact missing: {rel}")
            elif expected and _sha256(path) != expected:
                errors.append(f"artifact SHA256 mismatch: {rel}")
    return errors


def _default_output(task_id: str) -> Path:
    return ROOT / "reports" / "handoffs" / "_drafts" / task_id / "HANDOFF.json"


def command_create(args: argparse.Namespace) -> int:
    if not TASK_ID_RE.fullmatch(args.task_id):
        raise SystemExit("task_id must match YYYYMMDD-<area>-<short-purpose>")
    if args.work_type not in WORK_TYPES:
        raise SystemExit(f"unsupported work_type: {args.work_type}")
    output = Path(args.output) if args.output else _default_output(args.task_id)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            build_draft(args.task_id, args.title, args.work_type, args.objective, args.base_branch),
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output)
    return 0


def command_validate(args: argparse.Namespace) -> int:
    path = Path(args.record)
    if not path.is_absolute():
        path = ROOT / path
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(json.dumps({"ok": False, "errors": [f"cannot read record: {exc}"]}, indent=2))
        return 1
    errors = validate_record(record, final=args.final, verify_artifacts=args.verify_artifacts)
    print(json.dumps({"ok": not errors, "record": str(path), "errors": errors}, indent=2))
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a local handoff draft")
    create.add_argument("--task-id", required=True)
    create.add_argument("--title", required=True)
    create.add_argument("--work-type", required=True, choices=sorted(WORK_TYPES))
    create.add_argument("--objective", required=True)
    create.add_argument("--base-branch", default="master")
    create.add_argument("--output")
    create.set_defaults(func=command_create)

    validate = subparsers.add_parser("validate", help="validate a handoff record")
    validate.add_argument("record")
    validate.add_argument("--final", action="store_true", help="enforce final handoff gates")
    validate.add_argument("--verify-artifacts", action="store_true")
    validate.set_defaults(func=command_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
