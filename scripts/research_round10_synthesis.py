#!/usr/bin/env python3
"""Freeze Round 10: the auditable two-track simulator architecture.

Round 10 is an evidence synthesis, not a new data promotion.  It consumes the
previous nine rounds and writes the only allowed successor architecture:
development realism may inspect Line9-conditioned cases, whereas formal data
must be generated from independent/fold-only parameters and remain blocked
until the Dataset V2 gates pass.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.research_root.resolve()
    ledger_path = root / "research_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    completed = {int(item["round"]) for item in ledger["rounds"]}
    missing = set(range(1, 10)) - completed
    if missing:
        raise RuntimeError(f"cannot synthesize Round 10; missing rounds: {sorted(missing)}")

    round10 = {
        "round": 10,
        "factor": "minimum_causal_physics_to_measurement_combination",
        "decision": "two_track_architecture_selected_no_training_promotion",
        "formal_training_allowed": False,
        "development_track": {
            "purpose": "qualitative morphology calibration only",
            "permitted_reference": "FORMAL06C and IV2_F02 may be inspected with Line9 explicitly labelled as development-only",
            "forbidden_claim": "unseen Line9 generalisation or formal training eligibility",
        },
        "formal_track": {
            "physics": [
                "independent or fold-only basal geometry and cover property fields",
                "same-geometry full/no-basal causal control",
                "per-trace physical source/receiver height after metadata semantics validation",
                "weak P2-like finite lamina factor only outside the protected basal corridor",
                "no full-width periodic stratigraphy, point-target chains, or post-solve event injection",
            ],
            "measurement": [
                "bounded zero-phase effective response fit only on Line3, Line7, LineL1 and validated on Line6",
                "no measured phase transfer, copied patches, post-solve trace timing warp, or low-rank band injection",
                "explicit source-band ablation; amplitude-only proxy must never be described as SFCW",
            ],
            "labels_and_contract": [
                "retain geometry and visible-phase labels separately",
                "derive visible phase only after signed full/control review",
                "record source, control, material, and processing hashes",
                "keep Line9 test-only and LineX1 review-only",
            ],
        },
        "release_gates": [
            "independent non-Line9 scene family approved",
            "full/no-basal pair complete for every positive family",
            "true measured negative windows available",
            "V15 split and label semantics validation pass",
            "measurement-response fit excludes Line9",
            "human visual audit confirms no constructed layering or isolated hyperbola chain",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    prior = [item for item in ledger["rounds"] if int(item.get("round", -1)) != 10]
    prior.append(round10)
    prior.sort(key=lambda item: int(item["round"]))
    ledger["rounds"] = prior
    ledger["round10_status"] = "complete_architecture_synthesis"
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "round10_synthesis.json").write_text(
        json.dumps(round10, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(round10, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
