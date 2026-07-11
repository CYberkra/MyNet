"""Validate the immutable policy and array integrity of the legacy V1 catalog."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "simulation_governance_v1_20260711"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate(catalog: Path) -> dict[str, object]:
    policy_path = catalog / "catalog_policy.json"
    registry_path = catalog / "manifests" / "legacy_simulation_registry.csv"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    with registry_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(policy["unique_case_count"]):
        raise RuntimeError("registry count does not match catalog policy")
    seen: set[str] = set()
    weak_cases = 0
    ignored_traces = 0
    for row in rows:
        case_id = row["case_id"]
        if case_id in seen:
            raise RuntimeError(f"duplicate canonical case_id: {case_id}")
        seen.add(case_id)
        if row["formal_training_allowed"].lower() != "false" or row["line9_conditioned"].lower() != "true":
            raise RuntimeError(f"{case_id}: legacy catalog formal isolation was modified")
        if float(row["mask_center_error_p90_ns"]) > 1.5:
            raise RuntimeError(f"{case_id}: override is not visible-phase centered")
        raw = _resolve(row["raw_path"])
        override = _resolve(row["visible_phase_override_path"])
        status_path = _resolve(row["status_path"])
        weight_path = _resolve(row["label_weight_path"])
        ignore_path = _resolve(row["ignore_mask_path"])
        if _sha256(raw) != row["raw_sha256"]:
            raise RuntimeError(f"{case_id}: raw provenance hash mismatch")
        if _sha256(override) != row["visible_phase_override_sha256"]:
            raise RuntimeError(f"{case_id}: override hash mismatch")
        mask = np.load(override, allow_pickle=False)
        status = np.load(status_path, allow_pickle=False)
        weight = np.load(weight_path, allow_pickle=False)
        ignore = np.load(ignore_path, allow_pickle=False)
        if mask.ndim != 2 or mask.shape != ignore.shape or status.shape != weight.shape or status.size != mask.shape[1]:
            raise RuntimeError(f"{case_id}: label/supervision shape contract failed")
        if not all(np.isfinite(array).all() for array in (mask, weight, ignore)):
            raise RuntimeError(f"{case_id}: non-finite governed label artifact")
        if np.any(status == 0):
            raise RuntimeError(f"{case_id}: legacy positive case was incorrectly converted to a negative")
        if np.any((ignore.mean(axis=0) > 0.5) & (weight > 0)):
            raise RuntimeError(f"{case_id}: ignored trace retains positive label weight")
        weak_cases += int(np.all(status == 2))
        ignored_traces += int((ignore.mean(axis=0) > 0.5).sum())
    return {
        "catalog": str(catalog),
        "unique_case_count": len(rows),
        "formal_training_allowed_count": 0,
        "weak_case_count": weak_cases,
        "ignored_trace_count": ignored_traces,
        "result": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    args = parser.parse_args()
    catalog = Path(args.catalog)
    if not catalog.is_absolute():
        catalog = ROOT / catalog
    print(json.dumps(validate(catalog), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
