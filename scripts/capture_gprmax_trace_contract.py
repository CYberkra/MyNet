#!/usr/bin/env python3
"""Capture per-trace gprMax metadata before the merge tool removes files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import h5py


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialise(value: object) -> object:
    return value.tolist() if hasattr(value, "tolist") else value


def trace_filename(prefix: str, index: int, expected: int) -> str:
    """Return the gprMax filename for a trace in a single or multi-run job."""
    return f"{prefix}.out" if expected == 1 else f"{prefix}{index}.out"


def trace_index_for_path(path: Path, prefix: str, expected: int) -> int | None:
    if expected == 1:
        return 1 if path.name == f"{prefix}.out" else None
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.out", path.name)
    return int(match.group(1)) if match else None


def read_contract(path: Path, index: int) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        root = {key: serialise(value) for key, value in handle.attrs.items()}
        source_positions = []
        if "srcs" in handle:
            for name in sorted(handle["srcs"], key=lambda item: int(re.sub(r"\D", "", item) or 0)):
                group = handle[f"srcs/{name}"]
                source_positions.append(serialise(group.attrs.get("Position")))
        receiver_positions = []
        receiver_shapes: dict[str, dict[str, list[int]]] = {}
        if "rxs" in handle:
            for name in sorted(handle["rxs"], key=lambda item: int(re.sub(r"\D", "", item) or 0)):
                group = handle[f"rxs/{name}"]
                receiver_positions.append(serialise(group.attrs.get("Position")))
                receiver_shapes[name] = {
                    component: list(group[component].shape) for component in sorted(group.keys())
                }
        return {
            "trace_index": index,
            "filename": path.name,
            "sha256": sha256(path),
            "root_attributes": root,
            "source_positions_m": source_positions,
            "receiver_positions_m": receiver_positions,
            "receiver_shapes": receiver_shapes,
        }


def write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def load_existing_report(
    path: Path,
    *,
    case_dir: Path,
    prefix: str,
    expected: int,
) -> tuple[dict[int, dict[str, object]], list[str]]:
    if not path.is_file():
        return {}, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, []
    if (
        Path(str(payload.get("case_dir", ""))).resolve() != case_dir
        or payload.get("prefix") != prefix
        or int(payload.get("expected_trace_count", -1)) != expected
    ):
        return {}, []
    rows = payload.get("traces", [])
    captured: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or "trace_index" not in row:
            continue
        index = int(row["trace_index"])
        expected_name = trace_filename(prefix, index, expected)
        trace_path = case_dir / expected_name
        if row.get("filename") != expected_name or not trace_path.is_file():
            continue
        try:
            hash_matches = sha256(trace_path) == row.get("sha256")
        except OSError:
            hash_matches = False
        if hash_matches:
            captured[index] = row
    failures = [str(item) for item in payload.get("failures_tail", [])]
    return captured, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    output = args.output.resolve()
    captured, failures = load_existing_report(
        output,
        case_dir=case_dir,
        prefix=args.prefix,
        expected=args.expected,
    )
    deadline = time.monotonic() + args.timeout_seconds
    while len(captured) < args.expected and time.monotonic() < deadline:
        for path in case_dir.glob(f"{args.prefix}*.out"):
            index = trace_index_for_path(path, args.prefix, args.expected)
            if index is None:
                continue
            if index in captured:
                continue
            try:
                captured[index] = read_contract(path, index)
            except (OSError, KeyError, RuntimeError) as exc:
                failures.append(f"{path.name}: {exc}")
        payload = {
            "case_dir": str(case_dir),
            "prefix": args.prefix,
            "expected_trace_count": args.expected,
            "captured_trace_count": len(captured),
            "complete": len(captured) == args.expected,
            "failures_tail": failures[-20:],
            "traces": [captured[index] for index in sorted(captured)],
        }
        write_report(output, payload)
        if len(captured) < args.expected:
            time.sleep(args.poll_seconds)

    return 0 if len(captured) == args.expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
