#!/usr/bin/env python3
"""Register already-solved physical FDTD ablations as research Rounds 06-09.

The project already contains audited solver runs for these factors.  This
script does not re-run them.  It checks that the cited run and decision
artifacts exist, then writes a compact immutable evidence index and updates
the ten-round research ledger.  Re-running rejected factors would spend GPU
time without creating new scientific evidence.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


ROUNDS: tuple[dict[str, object], ...] = (
    {
        "round": 6,
        "factor": "continuous_interface_roughness_and_lateral_coherence",
        "evidence_runs": (
            "data/simulations/v2/01_solver_runs/FORMAL07A_CONTINUOUS_STRATIGRAPHY_DEVELOPMENT/formal07a_blind8_span_full_20260715",
            "data/simulations/v2/01_solver_runs/FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE/formal09c_p2_native64_full_20260716",
        ),
        "decision_reports": (
            "reports/formal07a_continuous_stratigraphy_20260715/FORMAL06C_VS_FORMAL07A_DECISION.md",
            "reports/formal09c_p2_sparse_irregular_laminae_20260716/VISUAL_AUDIT_DECISION.md",
        ),
        "decision": "continuous_stack_rejected_sparse_lamina_optional_lower_bound",
        "conclusion": "Regular full-width stratigraphy reads as constructed layering; sparse finite laminae preserve the basal packet but do not close the real-data texture gap.",
        "formal_training_allowed": False,
    },
    {
        "round": 7,
        "factor": "volumetric_multiscale_cover_background",
        "evidence_runs": (
            "data/simulations/v2/01_solver_runs/FORMAL07B_WEAK_APERIODIC_BACKGROUND_DEVELOPMENT/formal07b_distributed32_full_20260715",
            "data/simulations/v2/01_solver_runs/FORMAL08A_LINE9_REALISM_BACKGROUND_DEVELOPMENT/formal08a_distributed32_full_20260716",
            "data/simulations/v2/01_solver_runs/FORMAL08B_LINE9_REALISM_DEEP_BACKGROUND_DEVELOPMENT/formal08b_distributed32_full_20260716",
        ),
        "decision_reports": (
            "reports/formal08a_line9_realism_background_20260716/RUNTIME_VISUAL_DECISION.md",
            "reports/formal08b_line9_realism_deep_background_20260716/RUNTIME_VISUAL_DECISION.md",
        ),
        "decision": "rejected_single_factor_cover_texture_insufficient_or_competing",
        "conclusion": "Weak texture is visually too small to justify promotion; stronger transition-following texture changes target conditioning instead of producing measured-like clutter.",
        "formal_training_allowed": False,
    },
    {
        "round": 8,
        "factor": "finite_facies_and_lamina_topology",
        "evidence_runs": (
            "data/simulations/v2/01_solver_runs/FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE/formal09c_p2_smoke1_pair_20260716",
            "data/simulations/v2/01_solver_runs/FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE/formal09c_p2_native64_full_20260716",
        ),
        "decision_reports": (
            "reports/formal09c_p1_dense_physical_finite_laminae_20260716/VISUAL_AUDIT_DECISION.md",
            "reports/formal09c_p2_sparse_irregular_laminae_20260716/VISUAL_AUDIT_DECISION.md",
        ),
        "decision": "dense_facies_rejected_sparse_factor_optional_only",
        "conclusion": "Finite topology is safer than full-width layering, but a physical facies factor must remain weak, causal, and outside the protected basal corridor.",
        "formal_training_allowed": False,
    },
    {
        "round": 9,
        "factor": "source_band_and_instrument_proxy",
        "evidence_runs": (
            "data/simulations/v2/01_solver_runs/IV2_F01_GENTLE_APERIODIC_POS/distributed32_stride8_full_20260715",
            "data/simulations/v2/01_solver_runs/IV2_F02_FORMAL06C_MECHANISM_POS/f02_distributed32_stride8_full_20260715",
            "data/simulations/v2/01_solver_runs/IV2_F03_INSTRUMENT_BAND_POS/f03_distributed32_stride8_full_20260716",
        ),
        "decision_reports": (
            "reports/formal08a_line9_realism_background_20260716/RUNTIME_VISUAL_DECISION.md",
        ),
        "decision": "retain_source_ablation_tracks_no_sfcw_claim_from_amplitude_proxy",
        "conclusion": "F01 is the independent 55 MHz Ricker baseline; F02 retains a development-only zero-mean mechanism; F03 is an amplitude-only instrument-band proxy and must never be called SFCW.",
        "formal_training_allowed": False,
    },
)


def _run_record(relative: str) -> dict[str, object]:
    run_dir = ROOT / relative
    manifest = run_dir / "run_manifest.json"
    outputs = sorted(run_dir.glob("*.out"))
    if not manifest.is_file():
        raise FileNotFoundError(f"missing audited run manifest: {manifest}")
    if not outputs:
        raise FileNotFoundError(f"missing solver output in audited run: {run_dir}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "run_dir": relative,
        "run_mode": payload.get("mode"),
        "requested_trace_count": payload.get("requested_trace_count"),
        "causal_pair_complete": payload.get("causal_pair_complete"),
        "solver_outputs": [item.name for item in outputs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.research_root.resolve()
    ledger_path = root / "research_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    results: list[dict[str, object]] = []
    for entry in ROUNDS:
        reports = [str(item) for item in entry["decision_reports"]]
        for report in reports:
            if not (ROOT / report).is_file():
                raise FileNotFoundError(f"missing cited visual decision: {ROOT / report}")
        record = dict(entry)
        record["audited_runs"] = [_run_record(str(item)) for item in entry["evidence_runs"]]
        record["registered_utc"] = datetime.now(timezone.utc).isoformat()
        results.append(record)

    evidence_path = root / "round06_09_physical_evidence.json"
    evidence_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prior = [item for item in ledger["rounds"] if int(item.get("round", -1)) not in {6, 7, 8, 9}]
    prior.extend(results)
    prior.sort(key=lambda item: int(item["round"]))
    ledger["rounds"] = prior
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
