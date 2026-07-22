"""Audit the promotion boundary for independent V2 simulation families.

This is intentionally narrower than ``validate_physical_sim_v2.py``. That
validator covers the original flat control schema, while the independent
families use a nested family/case layout and have their own provenance gates.
Neither audit may promote a family; human release approval remains required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "data" / "simulations" / "v2" / "00_controls"
REPORTS = ROOT / "reports"

FAMILIES = (
    {
        "id": "F01",
        "directory": "IV2_F01_GENTLE_APERIODIC_COVER_BEDROCK",
        "decision": "reports/independent_v2_family01_20260715/pilot_decision.json",
        "expected_disposition": "continue_to_native256",
    },
    {
        "id": "F02",
        "directory": "IV2_F02_FORMAL06C_MECHANISM_TRANSFER_DEVELOPMENT",
        "decision": "reports/independent_v2_family02_20260715/pilot_decision.json",
        "expected_disposition": "development_only_line9_conditioned",
    },
    {
        "id": "F03",
        "directory": "IV2_F03_INSTRUMENT_BAND_WEAK_INTERFACE_PILOT",
        "decision": "reports/independent_v2_family03_20260716/pilot_decision.json",
        "expected_disposition": "hold_not_preferred_morphology",
    },
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_family(spec: dict[str, str], root: Path = ROOT) -> dict[str, Any]:
    family_dir = root / "data" / "simulations" / "v2" / "00_controls" / spec["directory"]
    errors: list[str] = []
    warnings: list[str] = []
    if not family_dir.is_dir():
        return {"family": spec["id"], "disposition": "missing", "errors": [str(family_dir)]}

    family = load_json(family_dir / "family_manifest.json")
    positive_dir = family_dir / family["positive_case_id"]
    negative_dir = family_dir / family["true_negative_case_id"]
    positive = load_json(positive_dir / "scene_manifest.json")
    negative = load_json(negative_dir / "scene_manifest.json")

    for payload, label in ((family, "family"), (positive, "positive"), (negative, "negative")):
        if payload.get("formal_training_allowed") is not False:
            errors.append(f"{label} must remain formal_training_allowed=false before release")

    if positive.get("target_presence") is not True:
        errors.append("positive case does not declare target_presence=true")
    if negative.get("target_presence") is not False:
        errors.append("negative case does not declare target_presence=false")
    if not family.get("split_group_indivisible"):
        errors.append("positive and negative are not declared as one split group")
    if not family.get("exact_negative_equivalence"):
        errors.append("family does not declare exact target-absent equivalence")
    if positive.get("strict_pair", {}).get("positive_control_equals_family_negative_full") is not True:
        errors.append("positive control is not declared equal to negative full state")

    positive_geometry = positive_dir / positive["geometry"]["index_file"]
    negative_geometry = negative_dir / negative["geometry"]["index_file"]
    for path, expected, label in (
        (positive_geometry, family["shared_geometry_sha256"], "positive geometry"),
        (negative_geometry, family["shared_geometry_sha256"], "negative geometry"),
        (positive_dir / "materials_no_basal.txt", family["positive_control_materials_sha256"], "positive control materials"),
        (negative_dir / "materials_full.txt", family["negative_full_materials_sha256"], "negative full materials"),
    ):
        if not path.is_file():
            errors.append(f"missing {label}: {path.name}")
        elif sha256(path) != expected:
            errors.append(f"hash mismatch: {label}")

    decision = load_json(root / spec["decision"])
    line9_conditioned = bool(family.get("line9_conditioned"))
    if line9_conditioned != bool(decision.get("line9_conditioned")):
        errors.append("family and decision disagree on Line9 conditioning")

    if spec["id"] == "F02":
        disposition = "development_only_line9_conditioned"
    elif spec["id"] == "F03":
        disposition = "hold_not_preferred_morphology"
        warnings.append("Independent provenance passes, but the recorded visual review is not preferred.")
    else:
        disposition = "continue_to_native256"
        warnings.append("Pilot evidence exists, but native-256 paired solver evidence and human release remain missing.")

    if errors:
        disposition = "blocked_integrity_mismatch"

    return {
        "family": spec["id"],
        "scene_family_id": family["scene_family_id"],
        "line9_conditioned": line9_conditioned,
        "recorded_decision": decision["decision"],
        "disposition": disposition,
        "formal_training_allowed": False,
        "errors": errors,
        "warnings": warnings,
    }


def audit(root: Path = ROOT) -> dict[str, Any]:
    families = [audit_family(spec, root) for spec in FAMILIES]
    return {
        "schema": "independent_v2_promotion_audit_v1",
        "formal_training_allowed": False,
        "promotion_decision": "blocked_pending_new_independent_native256_release",
        "families": families,
        "next_required_run": "F01 positive full/control/air and matched negative full at native 256 traces",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if all(not family["errors"] for family in result["families"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
