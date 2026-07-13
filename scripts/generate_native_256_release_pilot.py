#!/usr/bin/env python3
"""Generate and statically validate the recommended native 501x256 pilot."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import canonical_json_sha256, sha256_file  # noqa: E402
from scripts.generate_physical_sim_v2 import generate_case  # noqa: E402
from scripts.validate_physical_sim_v2 import validate_case_safe  # noqa: E402

CONTRACT_DIR = ROOT / "data" / "simulation_contract_v2"
DEFAULT_CASES = CONTRACT_DIR / "recommended_native_256_cases_v1.json"
DEFAULT_MATERIALS = CONTRACT_DIR / "recommended_native_256_materials_v1.json"
DEFAULT_STANDARD = CONTRACT_DIR / "recommended_native_256_v1.json"
DEFAULT_OUT = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "01_native_256_release_pilot"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _assert_case_matches_standard(case: dict[str, Any], standard: dict[str, Any]) -> None:
    output = standard["canonical_output"]
    fdtd = standard["fdtd"]
    acquisition = standard["acquisition"]
    grid = case.get("grid", {})
    source = case.get("source", {})
    checks = {
        "trace_count": (grid.get("trace_count"), output["trace_count"]),
        "trace_spacing_m": (grid.get("trace_spacing_m"), output["trace_spacing_m"]),
        "dl_m": (grid.get("dl_m"), fdtd["dl_m"]),
        "solver_time_window_ns": (grid.get("solver_time_window_ns"), fdtd["solver_time_window_ns"]),
        "pml_cells": (grid.get("pml_cells"), fdtd["pml_cells"]),
        "guard_cells": (grid.get("guard_cells"), fdtd["guard_cells"]),
        "center_frequency_hz": (source.get("center_frequency_hz"), acquisition["center_frequency_hz"]),
        "tx_rx_offset_m": (source.get("tx_rx_offset_m"), acquisition["tx_rx_offset_m"]),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{case['case_id']}: {name}={actual!r}, expected {expected!r}")
    for name in ("scan_left_margin_m", "scan_right_margin_m"):
        if float(case.get(name, 0.0)) < float(fdtd["minimum_side_guard_from_inner_pml_m"]):
            raise ValueError(f"{case['case_id']}: {name} is below long-window boundary guard")
    if case.get("target_presence") not in (True, False):
        raise ValueError(f"{case['case_id']}: target_presence must be boolean")


def build(
    *,
    cases_path: Path,
    materials_path: Path,
    standard_path: Path,
    out_root: Path,
    selected: set[str],
    overwrite: bool,
) -> dict[str, Any]:
    standard = _load(standard_path)
    cases_payload = _load(cases_path)
    materials = _load(materials_path)
    if cases_payload.get("standard_id") != standard.get("standard_id"):
        raise ValueError("case catalog is not bound to the recommended standard")
    if standard.get("formal_training_allowed") is not False:
        raise ValueError("recommended pre-solver standard must remain blocked from formal training")
    cases = list(cases_payload.get("cases", []))
    if selected:
        cases = [case for case in cases if case.get("case_id") in selected]
        missing = selected - {str(case["case_id"]) for case in cases}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    if not cases:
        raise ValueError("no native pilot cases selected")
    ids = [str(case["case_id"]) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("native pilot catalog contains duplicate case IDs")
    for case in cases:
        _assert_case_matches_standard(case, standard)

    if out_root.exists() and overwrite:
        for case_id in ids:
            shutil.rmtree(out_root / case_id, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    manifests = [generate_case(case, materials, out_root, overwrite=overwrite) for case in cases]
    validations = [validate_case_safe(out_root / case_id) for case_id in ids]
    report = {
        "standard_id": standard["standard_id"],
        "formal_training_allowed": False,
        "pre_solver_only": True,
        "standard_sha256": sha256_file(standard_path),
        "case_catalog_sha256": sha256_file(cases_path),
        "materials_sha256": sha256_file(materials_path),
        "case_spec_sha256": canonical_json_sha256({"cases": cases}),
        "case_count": len(ids),
        "positive_case_count": sum(bool(case["target_presence"]) for case in cases),
        "negative_case_count": sum(not bool(case["target_presence"]) for case in cases),
        "case_ids": ids,
        "static_validation_ok": all(item["ok"] for item in validations),
        "validations": validations,
        "promotion_blockers": [
            "No solver outputs exist yet.",
            "No signed visible-phase labels exist yet.",
            "No runtime gprMax and pre-merge trace evidence exists yet.",
            "No human release decisions exist yet."
        ],
        "generated_manifest_hashes": {
            item["case_id"]: sha256_file(out_root / item["case_id"] / "scene_manifest.json")
            for item in manifests
        },
    }
    (out_root / "native_256_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS)
    parser.add_argument("--standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build(
        cases_path=args.cases.resolve(),
        materials_path=args.materials.resolve(),
        standard_path=args.standard.resolve(),
        out_root=args.out_root.resolve(),
        selected=set(args.case_id),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["static_validation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
