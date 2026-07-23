#!/usr/bin/env python3
"""Validate the evidence gate for finite-antenna 3D gprMax studies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "data" / "contracts" / "simulation_v2" / "hardware_measurement_contract_v1.json"
READY_STATUS = "ready_for_3d_local_preflight"


def _value(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(payload: dict[str, Any], root: Path, require_ready: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("schema_version") != "uav_gpr_hardware_measurement_contract_v1":
        errors.append("unexpected schema_version")
    if payload.get("formal_training_allowed") is not False:
        errors.append("hardware measurement contract must not directly allow formal training")
    if payload.get("three_dimensional_preflight", {}).get("scope") != "bounded_local_window_only":
        errors.append("first 3D study must be bounded_local_window_only")
    if payload.get("three_dimensional_preflight", {}).get("max_native_traces", 0) > 16:
        errors.append("first 3D study may not exceed 16 native traces")

    status = payload.get("status")
    if status not in {"blocked_pending_measurement", READY_STATUS, "retired"}:
        errors.append(f"unknown status={status!r}")
    ready = status == READY_STATUS
    if require_ready and not ready:
        errors.append("contract is not ready_for_3d_local_preflight")
    if not ready:
        warnings.append("finite-antenna 3D solve remains blocked pending measured hardware evidence")
        return errors, warnings

    for field in payload.get("release_gate", {}).get("required_fields", []):
        value = _value(payload, str(field))
        if value is None or value == "":
            errors.append(f"missing required field: {field}")
    for field in (
        "antenna_geometry.physical_length_m",
        "antenna_geometry.tx_rx_separation_m",
        "source_measurement.sample_interval_s",
    ):
        try:
            if float(_value(payload, field)) <= 0:
                errors.append(f"{field} must be positive")
        except (TypeError, ValueError):
            errors.append(f"{field} must be numeric")

    source = payload.get("source_measurement", {})
    raw_path = Path(str(source.get("direct_or_air_reference_raw_path", "")))
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    expected_hash = str(source.get("direct_or_air_reference_sha256", "")).lower()
    if raw_path.is_file():
        if expected_hash and _sha256(raw_path).lower() != expected_hash:
            errors.append("direct_or_air_reference_raw_path SHA256 does not match")
    else:
        errors.append(f"direct_or_air_reference_raw_path is missing: {raw_path}")
    manifest_path = Path(str(source.get("preprocessing_manifest_path", "")))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        errors.append(f"preprocessing_manifest_path is missing: {manifest_path}")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.contract.read_text(encoding="utf-8"))
    errors, warnings = validate_contract(payload, ROOT, args.require_ready)
    result = {
        "contract": str(args.contract),
        "status": payload.get("status"),
        "ok": not errors,
        "finite_antenna_3d_run_allowed": not errors and payload.get("status") == READY_STATUS,
        "errors": errors,
        "warnings": warnings,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
