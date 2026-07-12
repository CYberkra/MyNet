#!/usr/bin/env python3
"""Validate the portable MACRO05 ten-family pre-solver package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_macro05_nightly_batch import (
    DEFAULT_OUTPUT,
    FAMILIES,
    GridSpec,
    equivalence_geometry,
    family_properties,
    reference_arrival,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_case(case_dir: Path, spec: GridSpec, expected_id: str) -> dict[str, object]:
    errors: list[str] = []
    manifest_path = case_dir / "scene_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"case_id": expected_id, "ok": False, "errors": [f"manifest: {exc}"]}

    if manifest.get("case_id") != expected_id:
        errors.append(f"case_id mismatch: {manifest.get('case_id')!r}")
    if manifest.get("line9_conditioned") is not False:
        errors.append("line9_conditioned must be false")
    if manifest.get("formal_training_allowed") is not False:
        errors.append("formal_training_allowed must remain false")
    if manifest.get("reference_line") is not None:
        errors.append("reference_line must be null")
    if manifest.get("strict_pair", {}).get("changed_material_indices") != list(range(10, 30)):
        errors.append("strict pair must change material indices 10..29 only")
    if manifest.get("strict_pair", {}).get("only_transition_and_bedrock_changed") is not True:
        errors.append("strict pair material contract is not asserted")

    domain = manifest.get("domain_contract", {})
    if float(domain.get("physical_left_guard_m", -1)) < 80.0 - 1e-6:
        errors.append("left physical guard is below 80 m")
    if float(domain.get("physical_right_guard_m", -1)) < 80.0 - 1e-6:
        errors.append("right physical guard is below 80 m")
    if float(domain.get("earliest_free_space_side_roundtrip_ns", -1)) <= 500.0:
        errors.append("side-boundary return can enter the protected 0-500 ns window")
    if float(domain.get("latest_geometric_reference_ns", 1e9)) >= 500.0:
        errors.append("geometric reference extends beyond protected target window")

    geometry_path = case_dir / "geology_indices.h5"
    try:
        with h5py.File(geometry_path, "r") as handle:
            data = handle["data"]
            if data.shape != (spec.nx, spec.ny, 1):
                errors.append(f"geometry shape {data.shape} != {(spec.nx, spec.ny, 1)}")
            if data.dtype != np.dtype("int16"):
                errors.append(f"geometry dtype {data.dtype} is not int16")
            if tuple(float(v) for v in handle.attrs["dx_dy_dz"]) != (spec.dl_m,) * 3:
                errors.append("HDF5 dx_dy_dz mismatch")
    except Exception as exc:
        errors.append(f"HDF5: {exc}")

    reference_path = case_dir / "labels" / "reference_arrival_time_ns.npy"
    try:
        reference = np.load(reference_path, allow_pickle=False)
        if reference.shape != (spec.trace_count,):
            errors.append(f"reference shape {reference.shape} != {(spec.trace_count,)}")
        if not np.isfinite(reference).all():
            errors.append("reference contains NaN/Inf")
        if float(np.max(reference)) >= 500.0:
            errors.append("reference arrival reaches protected-window boundary")
    except Exception as exc:
        errors.append(f"reference label: {exc}")

    checksum_path = case_dir / "FILE_SHA256.csv"
    try:
        with checksum_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            errors.append("empty FILE_SHA256.csv")
        for row in rows:
            path = case_dir / row["relative_path"]
            if not path.is_file():
                errors.append(f"missing checksummed file: {row['relative_path']}")
                continue
            if int(row["size_bytes"]) != path.stat().st_size:
                errors.append(f"size mismatch: {row['relative_path']}")
            if row["sha256"] != sha256(path):
                errors.append(f"hash mismatch: {row['relative_path']}")
    except Exception as exc:
        errors.append(f"checksums: {exc}")

    return {
        "case_id": expected_id,
        "ok": not errors,
        "errors": errors,
        "reference_min_ns": float(manifest.get("geometry", {}).get("reference_arrival_min_ns", float("nan"))),
        "reference_max_ns": float(manifest.get("geometry", {}).get("reference_arrival_max_ns", float("nan"))),
        "left_guard_m": float(domain.get("physical_left_guard_m", float("nan"))),
        "right_guard_m": float(domain.get("physical_right_guard_m", float("nan"))),
        "boundary_roundtrip_ns": float(domain.get("earliest_free_space_side_roundtrip_ns", float("nan"))),
    }


def validate_domain_equivalence(root: Path, spec: GridSpec) -> list[str]:
    errors: list[str] = []
    case_dir = root / FAMILIES[0].case_id
    expected_data, profile, properties, _lenses, _full, _control, _thresholds = equivalence_geometry(spec)
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        actual_data = handle["data"][:]
    if not np.array_equal(actual_data, expected_data):
        errors.append("F01 HDF5 is not the exact cropped/shifted MACRO04 geometry")
    expected_labels = reference_arrival(spec, profile, properties)
    for name, expected in expected_labels.items():
        actual = np.load(case_dir / "labels" / f"{name}.npy", allow_pickle=False)
        if not np.array_equal(actual, expected):
            errors.append(f"F01 label mismatch: {name}")
    return errors


def write_batch_checksums(root: Path) -> None:
    excluded = {"BATCH_SHA256.csv", "preflight_report.json", "preflight_summary.csv"}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append((path.relative_to(root).as_posix(), sha256(path), path.stat().st_size))
    with (root / "BATCH_SHA256.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "sha256", "size_bytes"))
        writer.writerows(rows)


def validate_batch_checksums(root: Path) -> list[str]:
    errors: list[str] = []
    path = root / "BATCH_SHA256.csv"
    if not path.is_file():
        return ["BATCH_SHA256.csv is missing"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        artifact = root / row["relative_path"]
        if not artifact.is_file():
            errors.append(f"batch checksum references missing file: {row['relative_path']}")
        elif int(row["size_bytes"]) != artifact.stat().st_size:
            errors.append(f"batch checksum size mismatch: {row['relative_path']}")
        elif row["sha256"].lower() != sha256(artifact).lower():
            errors.append(f"batch checksum hash mismatch: {row['relative_path']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    spec = GridSpec()
    expected_ids = [family.case_id for family in FAMILIES]
    errors: list[str] = []

    try:
        batch = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
        if batch.get("case_order") != expected_ids:
            errors.append("batch case_order does not match the generator contract")
        if batch.get("case_count") != 10:
            errors.append("batch must contain exactly ten cases")
        if batch.get("line9_conditioned_case_count") != 0:
            errors.append("batch contains Line9-conditioned cases")
        if batch.get("formal_training_allowed") is not False:
            errors.append("batch must remain blocked from formal training")
    except Exception as exc:
        errors.append(f"batch manifest: {exc}")

    order = (root / "nightly_case_order.txt").read_text(encoding="ascii").splitlines()
    if order != expected_ids:
        errors.append("nightly_case_order.txt mismatch")
    runner_text = (root / "RUN_NIGHTLY_GPU.cmd").read_text(encoding="ascii")
    if 'if not "%~1"==""' not in runner_text or "pair_complete.marker" not in runner_text:
        errors.append("runner lacks direct-case or resume contract")

    cases = [validate_case(root / case_id, spec, case_id) for case_id in expected_ids]
    for case in cases:
        errors.extend(f"{case['case_id']}: {message}" for message in case["errors"])
    errors.extend(validate_domain_equivalence(root, spec))

    if args.write_checksums and not errors:
        write_batch_checksums(root)
    errors.extend(validate_batch_checksums(root))

    report = {
        "batch_id": "MACRO05_NIGHTLY_10FAMILY_20260712",
        "root": str(root),
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "case_count": len(cases),
        "case_results": cases,
        "domain_decision": {
            "domain_x_m": spec.domain_x_m,
            "scan_span_m": spec.scan_span_m,
            "physical_side_guard_m": spec.physical_guard_m,
            "protected_window_end_ns": 500.0,
            "earliest_free_space_side_roundtrip_ns": 2e9 * spec.physical_guard_m / 299_792_458.0,
        },
    }
    (root / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (root / "preflight_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in cases[0] if key != "errors"])
        writer.writeheader()
        for case in cases:
            writer.writerow({key: value for key, value in case.items() if key != "errors"})
    print(json.dumps({"ok": not errors, "case_count": len(cases), "error_count": len(errors)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
