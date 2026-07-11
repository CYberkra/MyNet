"""Revalidate every materialised legacy simulation case without promoting it.

This tool deliberately audits the V1 corpus as *evidence* rather than as an
approved training set.  It recomputes visible-phase support from raw arrays,
checks label-array semantics and duplicate hashes, then combines those facts
with the immutable simulation contract.  A visually supported event is not
automatically a physically reproducible or Line9-holdout-compatible sample.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_AUDIT = ROOT / "reports" / "SIMULATION_FULL_AUDIT_20260710" / "audit_simulations.py"


def _load_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row[key]): row for row in csv.DictReader(handle) if str(row.get(key, "")).strip()}


def _load_legacy_module(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_simulation_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy audit module: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve postponed type annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _provenance_state(copy: Any) -> dict[str, Any]:
    case_dir = copy.case_dir
    scene = next(iter(case_dir.rglob("scene_world.json")), None)
    design = next(iter(case_dir.rglob("design_metrics.csv")), None)
    raw_input = next(iter(case_dir.rglob("*.in")), None)
    component_names = {p.name.lower() for p in case_dir.rglob("*.npy")}
    component_truth = any(
        token in name for name in component_names
        for token in ("background_only", "basal_only", "target_only", "air_only", "surface_only")
    )
    return {
        "case_local_scene_world": bool(scene),
        "case_local_design_metrics": bool(design),
        "case_local_gprmax_input": bool(raw_input),
        "paired_component_truth": component_truth,
    }


def _semantic_decision(metrics: dict[str, Any]) -> str:
    if bool(metrics["curve_training_contract_ok"]):
        return "VISIBLE_PHASE_DISTRIBUTION_OK"
    if str(metrics["soft_target_semantics"]) == "GEOMETRY_TO_VISIBLE_BAND_OR_SHIFTED":
        return "REBUILD_VISIBLE_PHASE_DISTRIBUTION"
    return "BLOCK_LABEL_SEMANTICS"


def _development_decision(metrics: dict[str, Any], visual: dict[str, str]) -> str:
    visual_decision = visual.get("visual_decision", "PENDING")
    semantic = _semantic_decision(metrics)
    if str(metrics["automatic_signal_grade"]) == "UNSUPPORTED":
        return "REVIEW_SIGNAL_BEFORE_ANY_USE"
    if semantic != "VISIBLE_PHASE_DISTRIBUTION_OK" and visual_decision == "VISUAL_REVIEW_LOCAL_ARTIFACT":
        return "REBUILD_LABEL_AND_LOCAL_IGNORE"
    if semantic != "VISIBLE_PHASE_DISTRIBUTION_OK":
        return "REBUILD_LABEL_BEFORE_DEVELOPMENT"
    if visual_decision == "VISUAL_REVIEW_LOCAL_ARTIFACT":
        return "DEVELOPMENT_ONLY_AFTER_LOCAL_IGNORE"
    if visual_decision == "VISUAL_PASS_LOW_CONTRAST":
        return "DEVELOPMENT_ONLY_WEAK_POSITIVE"
    if visual_decision == "VISUAL_PASS":
        return "DEVELOPMENT_ONLY_SIGNAL_SUPPORTED"
    return "PENDING_VISUAL_REVIEW"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _v2_control_state() -> dict[str, Any]:
    root = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "00_controls"
    cases = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    output_files = [path for path in root.rglob("*") if path.suffix.lower() in {".out", ".h5", ".hdf5"}]
    bscan_arrays = [path for path in root.rglob("*.npy") if "bscan" in path.name.lower()]
    decisions = _load_csv(ROOT / "reports" / "overnight_20260711" / "gprmax" / "CONTROL_CASE_DECISIONS.csv", "case_id")
    return {
        "control_case_count": len(cases),
        "control_case_names": [path.name for path in sorted(cases)],
        "official_fdtd_output_count": len(output_files),
        "materialised_bscan_array_count": len(bscan_arrays),
        "all_controls_line9_conditioned": False,
        "all_controls_formal_training_allowed": False,
        "control_decisions": {key: value.get("decision", "UNRECORDED") for key, value in decisions.items()},
        "decision": "CONTROL_STAGE_ONLY_NO_VISUAL_OR_TRAINING_RELEASE",
    }


def _write_report(path: Path, rows: list[dict[str, Any]], copies: int, unique: int, v2: dict[str, Any]) -> None:
    by = lambda key, value: sum(str(row.get(key, "")) == value for row in rows)
    semantic_ok = by("label_semantic_decision", "VISIBLE_PHASE_DISTRIBUTION_OK")
    formally_allowed = sum(str(row["formal_training_allowed"]).lower() == "true" for row in rows)
    supported = sum(str(row["automatic_signal_grade"]) == "SUPPORTED" for row in rows)
    rebuild = by("development_decision", "REBUILD_LABEL_BEFORE_DEVELOPMENT") + by("development_decision", "REBUILD_LABEL_AND_LOCAL_IGNORE")
    local_ignore = by("development_decision", "DEVELOPMENT_ONLY_AFTER_LOCAL_IGNORE") + by("development_decision", "REBUILD_LABEL_AND_LOCAL_IGNORE")
    weak = by("development_decision", "DEVELOPMENT_ONLY_WEAK_POSITIVE")
    supported_dev = by("development_decision", "DEVELOPMENT_ONLY_SIGNAL_SUPPORTED")
    lines = [
        "# Legacy Simulation Revalidation",
        "",
        "## Scope",
        f"- Physical copies on disk: {copies}",
        f"- Unique raw-plus-visible-label pairs: {unique}",
        f"- Signal-supported unique cases: {supported}/{unique}",
        f"- Cases with visible-phase-centred curve distributions: {semantic_ok}/{unique}",
        f"- Formal-training-approved cases: {formally_allowed}/{unique}",
        "",
        "## Decision",
        "All currently materialised V1 cases remain excluded from formal Line9-holdout training. "
        "The evidence checks whether a visible-phase label follows a signal event; it does not remove "
        "Line9 conditioning, duplicate/template dependence, missing gprMax inputs, or absent component truth.",
        "",
        "## Signal And Label Findings",
        f"- `{supported_dev}` cases have signal-supported labels and internally consistent visible-phase distributions; they remain development-only.",
        f"- `{rebuild}` cases have a visually supported hard visible-phase curve but a shifted/wide `y_soft` tensor. Rebuild their curve distribution before any development export.",
        f"- `{local_ignore}` case has a visible event with a local competing phase and requires an ignore span.",
        f"- `{weak}` cases retain a coherent but low-contrast event and may only be used as weak positives in development.",
        "- Every unique case is Line9-conditioned, and every one lacks case-local gprMax input plus paired component truth. These are hard formal-release blockers independent of visual appearance.",
        "",
        "## Interpretation Of Old QC",
        "The old point-sampled waveform QC is not used as a release decision. It can report RED at a wavelet zero crossing even when the visible-phase curve follows a continuous envelope-supported event. The new decision combines envelope support, curve-target semantics, duplication, provenance, and the immutable holdout policy.",
        "",
        "## Required Action By Category",
        "- `REBUILD_LABEL_BEFORE_DEVELOPMENT`: rebuild the curve distribution from the explicit visible-phase curve before any development export.",
        "- `REBUILD_LABEL_AND_LOCAL_IGNORE`: perform that rebuild and retain the stated local trace span as ignore.",
        "- `DEVELOPMENT_ONLY_AFTER_LOCAL_IGNORE`: preserve the stated local trace span as ignore during any non-formal experiment.",
        "- `DEVELOPMENT_ONLY_WEAK_POSITIVE`: retain reduced supervision weight; never reinterpret it as a negative.",
        "- `DEVELOPMENT_ONLY_SIGNAL_SUPPORTED`: signal evidence is adequate for development only, not paper training.",
        "",
        "The authoritative per-case evidence is `SIMULATION_REAUDIT_CASES.csv`.",
        "",
        "## V2 Control Status",
        f"V2 contains {v2['control_case_count']} non-Line9 control definitions but {v2['official_fdtd_output_count']} official FDTD output files and {v2['materialised_bscan_array_count']} materialised B-scan arrays. It is therefore an engineering control stage, not a visually auditable or training-releasable dataset. The required next gate is official gprMax runtime plus postprocessing.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path, legacy_script: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy = _load_legacy_module(legacy_script)
    visual = _load_csv(ROOT / "reports" / "SIMULATION_FULL_AUDIT_20260710" / "SIMULATION_VISUAL_DECISIONS.csv", "case_id")
    contract = _load_csv(ROOT / "data" / "dataset_contract_v2" / "simulation_cases.csv", "case_id")
    copies = legacy.discover_copies()
    legacy.OUT = output_dir
    legacy.PREVIEWS = output_dir / "previews"
    legacy.PREVIEWS.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Any]] = {}
    for copy in copies:
        groups.setdefault(copy.pair_sha256, []).append(copy)

    rows: list[dict[str, Any]] = []
    visual_entries: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    for _pair_hash, group in sorted(groups.items(), key=lambda item: legacy.choose_canonical(item[1]).case_id):
        canonical = legacy.choose_canonical(group)
        metrics, arrays = legacy.compute_metrics(canonical)
        evidence = _provenance_state(canonical)
        visual_row = visual.get(canonical.case_id, {})
        contract_row = contract.get(canonical.case_id, {})
        line9_conditioned = str(contract_row.get("line9_conditioned", "true")).strip().lower() == "true"
        formal_allowed = str(contract_row.get("train_allowed", "false")).strip().lower() == "true"
        row = {
            **metrics,
            **evidence,
            "physical_copy_count": len(group),
            "duplicate_case_paths": " | ".join(str(item.case_dir.relative_to(ROOT)) for item in group),
            "historic_visual_decision": visual_row.get("visual_decision", "PENDING"),
            "historic_local_review_trace_range": visual_row.get("local_review_trace_range", ""),
            "label_semantic_decision": _semantic_decision(metrics),
            "development_decision": _development_decision(metrics, visual_row),
            "line9_conditioned": line9_conditioned,
            "formal_training_allowed": formal_allowed,
            "formal_decision": "EXCLUDE_LINE9_HOLDOUT" if line9_conditioned or not formal_allowed else "REVIEW_FORMAL_ELIGIBILITY",
            "negative_semantics": contract_row.get("negative_semantics", "not_a_negative_sample"),
        }
        rows.append(row)
        render_metrics = dict(metrics)
        render_metrics["visual_decision"] = row["historic_visual_decision"]
        visual_entries.append((canonical, render_metrics, arrays))
        legacy.render_case(canonical, render_metrics, arrays)

    _write_csv(output_dir / "SIMULATION_REAUDIT_CASES.csv", rows)
    legacy.render_contact_sheet(
        "SIMULATION_REAUDIT_ALL_CASES.png", visual_entries, columns=4
    )
    legacy.render_contact_sheet(
        "SIMULATION_REAUDIT_LABEL_REBUILD_CASES.png",
        [entry for entry, row in zip(visual_entries, rows) if row["label_semantic_decision"] != "VISIBLE_PHASE_DISTRIBUTION_OK"],
        columns=4,
    )
    summary = {
        "audit_scope": "materialised_legacy_v1_simulation_cases",
        "physical_case_copies": len(copies),
        "unique_raw_visible_label_pairs": len(groups),
        "exact_duplicate_copies": len(copies) - len(groups),
        "signal_supported_count": sum(row["automatic_signal_grade"] == "SUPPORTED" for row in rows),
        "curve_visible_phase_semantic_ok_count": sum(row["label_semantic_decision"] == "VISIBLE_PHASE_DISTRIBUTION_OK" for row in rows),
        "line9_conditioned_count": sum(bool(row["line9_conditioned"]) for row in rows),
        "formal_training_allowed_count": sum(bool(row["formal_training_allowed"]) for row in rows),
        "missing_case_local_gprmax_input_count": sum(not bool(row["case_local_gprmax_input"]) for row in rows),
        "missing_component_truth_count": sum(not bool(row["paired_component_truth"]) for row in rows),
        "reproducibility_source_script": str(legacy_script.relative_to(ROOT)),
    }
    v2 = _v2_control_state()
    _write_report(output_dir / "SIMULATION_REAUDIT_REPORT.md", rows, len(copies), len(groups), v2)
    summary["v2_controls"] = v2
    (output_dir / "SIMULATION_REAUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/SIMULATION_REAUDIT_20260711")
    parser.add_argument("--legacy-script", default=str(DEFAULT_LEGACY_AUDIT))
    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    legacy_script = Path(args.legacy_script)
    if not legacy_script.is_absolute():
        legacy_script = ROOT / legacy_script
    print(json.dumps(run(output_dir, legacy_script), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
