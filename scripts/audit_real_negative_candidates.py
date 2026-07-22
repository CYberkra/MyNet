#!/usr/bin/env python3
"""Audit whether a measured release contains trustworthy true-negative traces.

This tool is deliberately conservative. An ignored, weak, or unlabeled trace
is not a true negative. A trace is reported as a confirmed true negative only
when the release itself explicitly assigns ``status_code == 0`` and it carries
neither target-label mass nor an ignore mask.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "data" / "contracts" / "dataset_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive contiguous index ranges where a one-dimensional mask is true."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("run mask must be one dimensional")
    starts = np.flatnonzero(values & np.r_[True, ~values[:-1]])
    ends = np.flatnonzero(values & np.r_[~values[1:], True])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _column_mask(values: np.ndarray, trace_count: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1 and values.shape == (trace_count,):
        return values.astype(bool)
    if values.ndim == 2 and values.shape[1] == trace_count:
        return values.astype(bool).any(axis=0)
    raise ValueError(f"{name} shape {values.shape} is not trace-aligned")


def _line_summary(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual_sha256 = _sha256(path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"{path.name}: SHA256 does not match dataset contract")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"status_code", "label_weight", "soft_mask_train"}
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"{path.name}: missing required arrays {sorted(missing)}")
        status = np.asarray(arrays["status_code"], dtype=np.int16)
        weight = np.asarray(arrays["label_weight"], dtype=np.float32)
        if status.ndim != 1 or weight.shape != status.shape:
            raise ValueError(f"{path.name}: status_code and label_weight must be trace vectors")
        if not np.isin(status, (0, 1, 2)).all():
            raise ValueError(f"{path.name}: status_code contains values outside {{0, 1, 2}}")
        label_present = _column_mask(arrays["soft_mask_train"] > 0, status.size, "soft_mask_train")
        ignored = (
            _column_mask(arrays["ignore_mask"], status.size, "ignore_mask")
            if "ignore_mask" in arrays.files
            else np.zeros(status.size, dtype=bool)
        )

    explicit_negative = status == 0
    confirmed_negative = explicit_negative & ~label_present & ~ignored & (weight <= 0)
    invalid_negative = explicit_negative & ~confirmed_negative
    ambiguous_or_ignore = (~explicit_negative) & (ignored | (weight <= 0) | ~label_present)

    line_id = path.stem
    summary = {
        "line_id": line_id,
        "path": str(path),
        "sha256": actual_sha256,
        "trace_count": int(status.size),
        "status_code_counts": {str(value): int((status == value).sum()) for value in (0, 1, 2)},
        "confirmed_true_negative_trace_count": int(confirmed_negative.sum()),
        "invalid_status_zero_trace_count": int(invalid_negative.sum()),
        "ambiguous_or_ignore_trace_count": int(ambiguous_or_ignore.sum()),
        "decision": (
            "contains_confirmed_true_negative_traces"
            if confirmed_negative.any()
            else "no_confirmed_true_negative_traces"
        ),
    }
    rows: list[dict[str, Any]] = []
    categories = (
        ("confirmed_true_negative", confirmed_negative, "eligible_for_human_window_review"),
        ("invalid_status_zero", invalid_negative, "repair_release_before_any_use"),
        ("ambiguous_or_ignore", ambiguous_or_ignore, "never_promote_to_negative_without_new_evidence"),
    )
    for category, mask, action in categories:
        for start, end in _runs(mask):
            rows.append(
                {
                    "line_id": line_id,
                    "category": category,
                    "trace_start": start,
                    "trace_end": end,
                    "trace_count": end - start + 1,
                    "required_action": action,
                }
            )
    return summary, rows


def _expected_hashes(contract_root: Path) -> dict[str, str]:
    path = contract_root / "real_lines.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["line_id"]: row.get("sha256", "") for row in csv.DictReader(handle)}


def audit(data_root: Path, contract_root: Path | None = DEFAULT_CONTRACT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines_dir = data_root / "lines"
    if not lines_dir.is_dir():
        raise FileNotFoundError(f"missing measured line directory: {lines_dir}")
    expected = _expected_hashes(contract_root) if contract_root else {}
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in sorted(lines_dir.glob("*.npz")):
        summary, line_rows = _line_summary(path, expected.get(path.stem))
        summaries.append(summary)
        rows.extend(line_rows)
    if not summaries:
        raise FileNotFoundError(f"no line NPZ files found below {lines_dir}")
    confirmed = sum(int(item["confirmed_true_negative_trace_count"]) for item in summaries)
    invalid = sum(int(item["invalid_status_zero_trace_count"]) for item in summaries)
    result = {
        "schema_version": "real_negative_candidate_audit_v1",
        "data_root": str(data_root),
        "contract_root": str(contract_root) if contract_root else None,
        "line_count": len(summaries),
        "confirmed_true_negative_trace_count": confirmed,
        "invalid_status_zero_trace_count": invalid,
        "formal_negative_supervision_ready": confirmed > 0 and invalid == 0,
        "decision": (
            "human_review_of_explicit_negative_traces_required"
            if confirmed
            else "blocked_no_confirmed_real_true_negative_traces"
        ),
        "required_next_action": (
            "review explicit status_code=0 traces and promote only approved windows"
            if confirmed
            else "acquire or manually label new no-interface windows; ignore and weak labels are not negative evidence"
        ),
        "lines": summaries,
    }
    return result, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["line_id", "category", "trace_start", "trace_end", "trace_count", "required_action"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result, rows = audit(args.data_root, args.contract_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "real_negative_candidate_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "real_negative_candidate_intervals.csv", rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
