#!/usr/bin/env python3
"""Build the single release-gate catalog for current independent V2 simulations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOLVED_ROOT = ROOT / "outputs" / "simulation_v2_controls" / "official_audited_20260711"
DEFAULT_MACRO04 = (
    ROOT
    / "data"
    / "PGDA_SYNTH_DATASET_V2"
    / "00_controls"
    / "MACRO04_DEEPER_GENTLE_DROPOUT_DIAGNOSTIC"
)
DEFAULT_MACRO05 = ROOT.parent / "PGDA_gprMax_MACRO05_NIGHTLY_20260712"
DEFAULT_OUT = ROOT / "reports" / "SIMULATION_V2_RELEASE_AUDIT_20260713"
DEFAULT_GOVERNANCE = ROOT / "data" / "simulation_governance_v2_20260713"


DECISIONS = {
    "CTRL01_FLAT_SHALLOW_LOWLOSS_POS": (
        "diagnostic_reference_only",
        "Flat low-loss reference is visually idealised and dominated by endpoint artifacts.",
    ),
    "CTRL02_FLAT_DEEP_MODERATE_POS": (
        "diagnostic_reference_only",
        "Flat deep reference is useful for timing regression, not morphology training.",
    ),
    "CTRL03_SMOOTH_INTERFACE_POS": (
        "development_positive_candidate",
        "Clean matched contrast, but the interface is too smooth and old per-trace provenance is incomplete.",
    ),
    "CTRL04_MATCHED_BACKGROUND_NEG": (
        "development_true_negative_candidate",
        "The scene is explicitly basal-free and its zero mask is validated; old metadata remains untrusted.",
    ),
    "CTRL05_GENTLE_TERRAIN_WEAK_LAYER_POS": (
        "development_positive_candidate",
        "Clean weak-layer response, but morphology and acquisition remain idealised.",
    ),
    "CTRL06_LATERAL_VARIATION_POS": (
        "artifact_diagnostic_only",
        "A strong crossing artifact intersects the target and can confound positive supervision.",
    ),
    "MACRO01_GENTLE_LONG_LINE_DIAGNOSTIC": (
        "development_positive_candidate",
        "Long-line response is clean but overly smooth and only 128 traces wide.",
    ),
    "MACRO02_MULTISCALE_LONG_LINE_DIAGNOSTIC": (
        "reprocess_required",
        "The solved pair exists, but the formal postprocess lifecycle record is missing.",
    ),
    "MACRO03_CORRELATED_VOXEL_LONG_LINE_DIAGNOSTIC": (
        "repair_and_relabel_required",
        "Missing height metadata, incomplete full-run provenance, and 31.9 ns phase-path residual.",
    ),
    "MACRO04_DEEPER_GENTLE_DROPOUT_DIAGNOSTIC": (
        "best_pair_pilot_candidate",
        "Strict hashes and per-trace provenance pass; still 128 traces, no air run, and manual review is pending.",
    ),
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path("..", path.resolve().relative_to(ROOT.parent.resolve())).as_posix()


def _analytic_pair_topology_ok(case_dir: Path) -> bool:
    full_path = case_dir / "full_scene_geometry.inc"
    control_path = case_dir / "no_basal_contrast_geometry.inc"
    if not full_path.is_file() or not control_path.is_file():
        return False
    full = [line.split() for line in full_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    control = [line.split() for line in control_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(full) != len(control) or not full:
        return False
    return all(len(left) >= 8 and len(right) >= 8 and left[:7] == right[:7] for left, right in zip(full, control))


def _solved_row(case_dir: Path) -> dict[str, Any]:
    manifest = _load_json(case_dir / "scene_manifest.json") or {}
    post = _load_json(case_dir / "postprocess_validation.json") or {}
    pair = _load_json(case_dir / "pair_audit" / "pair_audit_validation.json") or {}
    case_id = str(manifest.get("case_id", case_dir.name))
    decision, reason = DECISIONS[case_id]
    trace_count = int(manifest.get("grid", {}).get("trace_count", 0) or 0)
    target_presence = bool(manifest.get("target_presence"))
    strict_pair = manifest.get("strict_pair", {})
    static_pair_topology_ok = bool(
        pair.get("artifact_hash_contract_ok") is True
        and strict_pair.get("only_transition_and_bedrock_changed") is True
    ) or _analytic_pair_topology_ok(case_dir)
    full_done = (case_dir / "full_scene_merged.out").is_file()
    control_done = (case_dir / "no_basal_contrast_control_merged.out").is_file()
    air_done = (case_dir / "air_reference_merged.out").is_file()
    if full_done and control_done and air_done:
        solver_state = "full_control_air_complete"
    elif full_done and control_done:
        solver_state = "full_control_complete"
    elif full_done and air_done and not target_presence:
        solver_state = "negative_full_air_complete"
    else:
        solver_state = "incomplete"
    return {
        "case_id": case_id,
        "family": manifest.get("family", ""),
        "case_path": _relative(case_dir),
        "solver_state": solver_state,
        "target_semantics": "positive" if target_presence else "confirmed_negative_design",
        "trace_count": trace_count,
        "canonical_shape": f"501x{trace_count}" if trace_count else "",
        "line9_conditioned": str(manifest.get("line9_conditioned") is True).lower(),
        "postprocess_validated": str(post.get("postprocess_validated") is True).lower(),
        "metadata_trusted": str(post.get("metadata_trusted") is True).lower(),
        "static_pair_topology_ok": str(static_pair_topology_ok).lower(),
        "pair_audit_ok": str(pair.get("ok") is True).lower(),
        "pair_contract_ok": str(pair.get("pair_contract_ok") is True).lower(),
        "contrast_target_to_background_rms": pair.get("contrast_target_to_background_rms", ""),
        "max_phase_path_residual_ns": pair.get("max_abs_trace_residual_ns", ""),
        "release_tier": decision,
        "negative_export_candidate": str(decision == "development_true_negative_candidate").lower(),
        "formal_training_allowed": "false",
        "decision_basis": reason,
    }


def _macro05_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = _load_json(case_dir / "scene_manifest.json")
        if not manifest:
            continue
        trace_count = int(manifest.get("grid", {}).get("trace_count", 0) or 0)
        solved = list(case_dir.glob("*_merged.out"))
        rows.append(
            {
                "case_id": manifest.get("case_id", case_dir.name),
                "family": manifest.get("family", ""),
                "case_path": _relative(case_dir),
                "solver_state": "unexpected_partial_output" if solved else "preflight_only_not_run",
                "target_semantics": "positive_design_pending_solver",
                "trace_count": trace_count,
                "canonical_shape": f"501x{trace_count}" if trace_count else "",
                "line9_conditioned": str(manifest.get("line9_conditioned") is True).lower(),
                "postprocess_validated": "false",
                "metadata_trusted": "false",
                "static_pair_topology_ok": "false",
                "pair_audit_ok": "false",
                "pair_contract_ok": "false",
                "contrast_target_to_background_rms": "",
                "max_phase_path_residual_ns": "",
                "release_tier": "solver_run_required",
                "negative_export_candidate": "false",
                "formal_training_allowed": "false",
                "decision_basis": "Static design exists, but no completed solver pair or visual result is present.",
            }
        )
    return rows


def build(
    solved_root: Path,
    macro04: Path,
    macro05: Path,
    out_dir: Path,
    governance_dir: Path,
) -> dict[str, Any]:
    rows = [
        _solved_row(case_dir)
        for case_dir in sorted(path for path in solved_root.iterdir() if path.is_dir())
        if case_dir.name in DECISIONS
    ]
    if macro04.is_dir():
        rows.append(_solved_row(macro04))
    rows.extend(_macro05_rows(macro05))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = governance_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for path in (
        manifests_dir / "simulation_release_gate.csv",
        out_dir / "simulation_release_gate.csv",
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "audit_id": "PGDA_SIMULATION_V2_RELEASE_AUDIT_20260713",
        "case_count": len(rows),
        "solved_case_count": sum("complete" in row["solver_state"] for row in rows),
        "formal_training_allowed_count": 0,
        "development_true_negative_candidate_count": sum(
            row["release_tier"] == "development_true_negative_candidate" for row in rows
        ),
        "best_pair_pilot_candidate_count": sum(
            row["release_tier"] == "best_pair_pilot_candidate" for row in rows
        ),
        "solver_run_required_count": sum(row["release_tier"] == "solver_run_required" for row in rows),
        "formal_blockers": [
            "No case has explicit human promotion with trusted per-trace provenance.",
            "The only confirmed-negative design is one idealised scene family.",
            "MACRO04 is 128 traces wide and cannot enter the canonical 501x256 exporter.",
            "MACRO05 families have not been solved.",
        ],
    }
    payload = json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2) + "\n"
    (manifests_dir / "simulation_release_gate.json").write_text(payload, encoding="utf-8")
    (out_dir / "simulation_release_gate.json").write_text(payload, encoding="utf-8")
    (governance_dir / "catalog_policy.json").write_text(
        json.dumps(
            {
                **summary,
                "authoritative_manifest": "manifests/simulation_release_gate.csv",
                "raw_evidence_mutated": False,
                "directory_name_grants_training_permission": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (governance_dir / "README.md").write_text(
        "# Simulation V2 Governance\n\n"
        "`manifests/simulation_release_gate.csv` is the authoritative release-gate catalog for current independent V2 simulations. Raw solver outputs stay in their original case directories. Every case remains `formal_training_allowed=false` until an explicit promotion updates the contract.\n",
        encoding="utf-8",
    )

    lines = [
        "# Simulation V2 Release Audit",
        "",
        "All current independent simulations remain blocked from formal paper training. The audit separates useful development evidence from formal eligibility; it never treats directory names or successful solver completion as promotion.",
        "",
        "| Case | State | Decision | Key reason |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['solver_state']} | {row['release_tier']} | {row['decision_basis']} |"
        )
    lines.extend(
        [
            "",
            "## Release Decision",
            "",
            "- CTRL04 is a valid development true-negative candidate, not a formal negative release.",
            "- MACRO04 is the strongest completed positive pair and the reference design for the next 256-trace pilot.",
            "- CTRL01/02 remain timing and boundary-regression controls only.",
            "- CTRL06 and MACRO03 must not provide positive curve supervision in their current state.",
            "- MACRO05 requires solver execution before any morphology or label decision.",
        ]
    )
    (out_dir / "SIMULATION_V2_RELEASE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solved-root", default=str(DEFAULT_SOLVED_ROOT))
    parser.add_argument("--macro04", default=str(DEFAULT_MACRO04))
    parser.add_argument("--macro05", default=str(DEFAULT_MACRO05))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--governance-dir", default=str(DEFAULT_GOVERNANCE))
    args = parser.parse_args()
    summary = build(
        Path(args.solved_root),
        Path(args.macro04),
        Path(args.macro05),
        Path(args.out_dir),
        Path(args.governance_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
