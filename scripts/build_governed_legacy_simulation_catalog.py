"""Organise legacy V1 simulations without mutating or promoting raw evidence.

The V1 corpus is Line9-conditioned, so this builder never creates a formal
training manifest.  It creates one canonical catalog over unique raw/label
pairs, explicit visible-phase label overrides, and trace-level supervision
artifacts for controlled development or regression use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "reports" / "SIMULATION_REAUDIT_20260711" / "SIMULATION_REAUDIT_CASES.csv"
DEFAULT_OUT = ROOT / "data" / "simulation_governance_v1_20260711"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _manifest_path(path: Path) -> str:
    """Prefer repository-relative paths, while allowing isolated test output."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _case_paths(row: dict[str, str]) -> tuple[Path, Path, Path, Path]:
    case_dir = ROOT / Path(row["canonical_case_path"])
    raw_candidates = sorted(set(case_dir.rglob("bscan.npy")) | set(case_dir.rglob("raw_bscan.npy")))
    curve_candidates = sorted(case_dir.rglob("target_visible_phase_time_ns.npy"))
    if len(raw_candidates) != 1 or len(curve_candidates) != 1:
        raise RuntimeError(
            f"{row['case_id']}: expected one raw and one visible curve, got "
            f"raw={len(raw_candidates)}, curve={len(curve_candidates)}"
        )
    raw_path = raw_candidates[0]
    visible_curve = curve_candidates[0]
    label_dir = visible_curve.parent
    visible_mask = label_dir / "interface_mask_visible_phase_bscan.npy"
    if not visible_mask.is_file():
        visible_mask = label_dir / "interface_mask_bscan.npy"
    for path in (raw_path, visible_mask, visible_curve):
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}: required source artifact is missing: {path}")
    return case_dir, raw_path, visible_mask, visible_curve


def _ignore_span(case_id: str, row: dict[str, str]) -> tuple[int, int] | None:
    raw = str(row.get("historic_local_review_trace_range", "")).strip()
    if not raw:
        return None
    try:
        start, end = (int(value) for value in raw.split("-", 1))
    except Exception as exc:
        raise ValueError(f"{case_id}: invalid local review span {raw!r}") from exc
    return start, end


def _catalog_role(row: dict[str, str]) -> str:
    # Batch 1 shares near-identical label/template structure.  Preserve it for
    # regression and diagnostics, but do not let it dominate gradient updates.
    if row["canonical_source_group"] == "batch3":
        return "development_train_candidate"
    return "diagnostic_only_template_or_artifact"


def _state_arrays(case_id: str, row: dict[str, str], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traces = mask.shape[1]
    status = np.ones(traces, dtype=np.int16)
    weight = np.ones(traces, dtype=np.float32)
    ignore = np.zeros(mask.shape, dtype=np.float32)
    if case_id in {"B003_SHALLOW_DISTRACTOR_011", "B003_SHALLOW_DISTRACTOR_012"}:
        status[:] = 2
        weight[:] = 0.35
    span = _ignore_span(case_id, row)
    if span is not None:
        start, end = max(0, span[0]), min(traces - 1, span[1])
        ignore[:, start : end + 1] = 1.0
        status[start : end + 1] = 2
        weight[start : end + 1] = 0.0
    return status, weight, ignore


def _mask_center_error_ns(mask: np.ndarray, curve_ns: np.ndarray, time_ns: np.ndarray) -> tuple[float, float]:
    if mask.ndim != 2 or curve_ns.ndim != 1 or mask.shape[1] != curve_ns.size:
        raise ValueError(f"mask/curve shape mismatch: {mask.shape} vs {curve_ns.shape}")
    mass = mask.sum(axis=0)
    center = (mask * time_ns[:, None]).sum(axis=0) / np.maximum(mass, 1e-12)
    error = np.abs(center - curve_ns)
    return float(np.median(error)), float(np.percentile(error, 90.0))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(audit_path: Path, out_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Catalog already exists: {out_dir}; use --overwrite to rebuild it")
        shutil.rmtree(out_dir)
    labels_dir = out_dir / "label_overrides"
    states_dir = out_dir / "supervision"
    manifests_dir = out_dir / "manifests"
    labels_dir.mkdir(parents=True)
    states_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    with audit_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise RuntimeError(f"No cases in re-audit table: {audit_path}")

    registry: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for row in source_rows:
        case_id = row["case_id"]
        case_dir, raw_path, visible_mask_path, visible_curve_path = _case_paths(row)
        mask = np.load(visible_mask_path, allow_pickle=False).astype(np.float32)
        curve = np.load(visible_curve_path, allow_pickle=False).astype(np.float64)
        time_path = visible_mask_path.parent / "time_501_ns.npy"
        if time_path.is_file():
            time_ns = np.load(time_path, allow_pickle=False).astype(np.float64)
        else:
            time_ns = np.linspace(0.0, 700.0, mask.shape[0], dtype=np.float64)
        if mask.shape != (time_ns.size, curve.size):
            raise ValueError(f"{case_id}: visible mask shape {mask.shape} is inconsistent with time/curve arrays")
        if not np.isfinite(mask).all() or not np.isfinite(curve).all():
            raise ValueError(f"{case_id}: visible label source has NaN/Inf")
        p50, p90 = _mask_center_error_ns(mask, curve, time_ns)
        if p90 > 1.5:
            raise RuntimeError(f"{case_id}: visible-phase override centre P90={p90:.3f} ns exceeds 1.5 ns")

        status, weight, ignore = _state_arrays(case_id, row, mask)
        override = labels_dir / f"{case_id}_visible_phase_mask_501x128.npy"
        status_path = states_dir / f"{case_id}_status_code.npy"
        weight_path = states_dir / f"{case_id}_label_weight.npy"
        ignore_path = states_dir / f"{case_id}_ignore_mask.npy"
        np.save(override, mask)
        np.save(status_path, status)
        np.save(weight_path, weight)
        np.save(ignore_path, ignore)

        role = _catalog_role(row)
        decision = row["development_decision"]
        registry.append({
            "case_id": case_id,
            "catalog_role": role,
            "development_export_allowed": str(role == "development_train_candidate").lower(),
            "formal_training_allowed": "false",
            "line9_conditioned": "true",
            "raw_path": str(raw_path.relative_to(ROOT)),
            "raw_sha256": _sha256(raw_path),
            "source_visible_mask_path": str(visible_mask_path.relative_to(ROOT)),
            "source_visible_mask_sha256": _sha256(visible_mask_path),
            "source_visible_curve_path": str(visible_curve_path.relative_to(ROOT)),
            "visible_phase_override_path": _manifest_path(override),
            "visible_phase_override_sha256": _sha256(override),
            "status_path": _manifest_path(status_path),
            "label_weight_path": _manifest_path(weight_path),
            "ignore_mask_path": _manifest_path(ignore_path),
            "label_semantics": "visible_phase_distribution_from_explicit_visible_mask",
            "mask_center_error_p50_ns": f"{p50:.6f}",
            "mask_center_error_p90_ns": f"{p90:.6f}",
            "development_decision": decision,
            "local_ignore_trace_range": row.get("historic_local_review_trace_range", ""),
            "weak_positive": str(case_id in {"B003_SHALLOW_DISTRACTOR_011", "B003_SHALLOW_DISTRACTOR_012"}).lower(),
            "source_case_path": str(case_dir.relative_to(ROOT)),
            "source_group": row["canonical_source_group"],
            "negative_semantics": "not_a_negative_sample",
        })
        for path_text in str(row["duplicate_case_paths"]).split(" | "):
            aliases.append({
                "physical_case_path": path_text,
                "canonical_case_id": case_id,
                "is_canonical": str(Path(path_text) == case_dir.relative_to(ROOT)).lower(),
                "raw_sha256": _sha256(raw_path),
                "visible_label_sha256": _sha256(visible_curve_path),
            })

    _write_csv(manifests_dir / "legacy_simulation_registry.csv", registry)
    _write_csv(manifests_dir / "physical_copy_aliases.csv", aliases)
    summary = {
        "catalog_id": "PGDA_LEGACY_SIMULATION_GOVERNANCE_V1_20260711",
        "source_audit": str(audit_path.relative_to(ROOT)),
        "unique_case_count": len(registry),
        "physical_copy_count": len(aliases),
        "development_train_candidate_count": sum(row["catalog_role"] == "development_train_candidate" for row in registry),
        "diagnostic_only_count": sum(row["catalog_role"] != "development_train_candidate" for row in registry),
        "formal_training_allowed_count": 0,
        "line9_conditioned_count": len(registry),
        "confirmed_negative_case_count": 0,
        "policy": {
            "raw_evidence_mutated": False,
            "legacy_y_soft_used": False,
            "all_curve_labels": "explicit visible-phase mask overrides",
            "formal_line9_holdout": "forbidden",
            "batch1_sampling": "diagnostic only because the family is template-heavy",
        },
    }
    (out_dir / "catalog_policy.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Governed Legacy Simulation Catalog\n\n"
        "This catalog references, but never rewrites, the legacy V1 raw cases. It standardises all label references to explicit visible-phase masks and records trace-level weak/ignore state.\n\n"
        "It is development-only. All cases are Line9-conditioned and `formal_training_allowed=false`; no downstream paper or strict Line9 holdout run may consume this catalog.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = Path(args.audit); audit = audit if audit.is_absolute() else ROOT / audit
    out = Path(args.out_dir); out = out if out.is_absolute() else ROOT / out
    print(json.dumps(build(audit, out, overwrite=args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
