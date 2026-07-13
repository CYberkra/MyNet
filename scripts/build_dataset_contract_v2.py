"""Build dataset_contract_v2 from repository evidence without promoting data.

The builder is deliberately conservative. Automatic QC grades and directory
names never make a sample trainable. Missing case bodies remain visible in the
human audit manifest, and every Line9-conditioned simulation is quarantined
from formal Line9 holdout training.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data" / "dataset_contract_v2"
RECOMMENDATIONS = ROOT / "reports" / "AUDIT_20260710" / "SIM_CASE_RECOMMENDATIONS.csv"
SYNTH_ROOT = ROOT / "data" / "PGDA_SYNTH_DATASET_V1"
ACCEPTED_ROOT = SYNTH_ROOT / "05_accepted_dataset"
BATCH3_ROOT = (
    SYNTH_ROOT
    / "03_runs"
    / "BATCH_003_SHALLOW_GENERALIZATION_24CASES_V3_AUDITED_20260704"
)
CANONICAL_REAL_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"
V15_FINAL_ROOT = ROOT / "data_yingshan_v15_final_20260710"
V2_RELEASE_GATE = (
    ROOT
    / "data"
    / "simulation_governance_v2_20260713"
    / "manifests"
    / "simulation_release_gate.csv"
)
REAL_ROOT = V15_FINAL_ROOT if (V15_FINAL_ROOT / "manifests" / "v15_final_manifest.json").is_file() else CANONICAL_REAL_ROOT
AUDIT_COMMIT = "799f229_source_candidate_plus_v15_final_release"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_contract_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_recommendations() -> dict[str, dict[str, str]]:
    if not RECOMMENDATIONS.is_file():
        return {}
    with RECOMMENDATIONS.open(encoding="utf-8-sig", newline="") as f:
        return {row["case_id"]: row for row in csv.DictReader(f)}


def accepted_case_dirs() -> list[Path]:
    if not ACCEPTED_ROOT.is_dir():
        return []
    return sorted({path.parent.parent for path in ACCEPTED_ROOT.rglob("input/raw_bscan.npy")})


def batch3_case_dirs() -> list[Path]:
    if not BATCH3_ROOT.is_dir():
        return []
    return sorted(path for path in BATCH3_ROOT.iterdir() if path.is_dir())


def first_file(*candidates: Path) -> Path | None:
    return next((path for path in candidates if path.is_file()), None)


def raw_file(case_dir: Path) -> Path | None:
    return first_file(case_dir / "input" / "raw_bscan.npy", case_dir / "raw" / "bscan.npy")


def label_file(case_dir: Path) -> Path | None:
    return first_file(
        case_dir / "label" / "target_visible_phase_time_ns.npy",
        case_dir / "labels" / "target_visible_phase_time_ns.npy",
        case_dir / "label" / "y_soft_501x128.npy",
        case_dir / "labels" / "y_soft_501x128.npy",
    )


def scene_world_file(case_dir: Path) -> Path | None:
    return first_file(
        case_dir / "metadata" / "scene_world.json",
        case_dir / "geometry" / "scene_world.json",
        case_dir / "scene_world.json",
    )


def design_metrics_file(case_dir: Path) -> Path | None:
    return first_file(
        case_dir / "metadata" / "design_metrics.csv",
        case_dir / "tables" / "design_metrics.csv",
        case_dir / "design_metrics.csv",
    )


def is_line9_conditioned(case_dir: Path, accepted: bool) -> bool:
    if accepted:
        return True
    evidence_names = {
        "line9_label_time_v14_resampled_ns.npy",
        "line9_label_depth_resampled_m.npy",
        "line9_reference_distance_m.npy",
    }
    for labels_dir in (case_dir / "label", case_dir / "labels"):
        if labels_dir.is_dir() and any((labels_dir / name).is_file() for name in evidence_names):
            return True
    for metadata in (scene_world_file(case_dir), design_metrics_file(case_dir)):
        if metadata is not None:
            try:
                if "line9" in metadata.read_text(encoding="utf-8", errors="ignore").lower():
                    return True
            except OSError:
                pass
    return case_dir.name.upper().startswith("LINE9_")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



def build_real_contract_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[str],
    int,
]:
    """Register canonical real artifacts built from the original YingShan CSV archive."""
    line_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    blockers: list[str] = []
    confirmed_negative_windows = 0
    index_path = REAL_ROOT / "window_index.csv"
    lines_dir = REAL_ROOT / "lines"
    windows_dir = REAL_ROOT / "windows"
    if not index_path.is_file() or not lines_dir.is_dir() or not windows_dir.is_dir():
        blockers.append("real line NPZ files and canonical window index are missing")
        return line_rows, window_rows, split_rows, blockers, confirmed_negative_windows

    with index_path.open(encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if not index_rows:
        blockers.append("real window index is empty")
        return line_rows, window_rows, split_rows, blockers, confirmed_negative_windows

    required_line_keys = {
        "raw_amplitude", "raw_full_normalized", "soft_mask_train", "status_code", "label_weight",
        "time_ns", "dt_ns", "longitude", "latitude", "ground_elevation_m", "flight_height_agl_m",
        "antenna_elevation_m", "gnss_cumulative_distance_m", "source_zip_sha256",
        "source_csv_member", "source_csv_sha256", "canonical_source",
        "profile_chainage_m", "acquisition_bearing_deg", "acquisition_compass",
        "engineering_profile", "profile_left", "profile_right", "profile_display_flip",
        "profile_orientation_confidence", "orientation_contract",
    }
    height_review_lines: list[str] = []
    profile_source = CANONICAL_REAL_ROOT / "source" / "ying_shan_profiles_and_boreholes.zip"
    profile_source_sha = sha256(profile_source) if profile_source.is_file() else ""
    if not profile_source.is_file():
        blockers.append("survey profile/borehole source archive is missing")
    for line_path in sorted(lines_dir.glob("*.npz")):
        with np.load(line_path, allow_pickle=False) as data:
            missing = sorted(required_line_keys - set(data.files))
            if missing:
                blockers.append(f"real line archive lacks canonical original-CSV metadata: {rel(line_path)} missing={missing}")
                continue
            raw = np.asarray(data["raw_full_normalized"])
            if raw.ndim != 2:
                blockers.append(f"real line archive has invalid shape: {rel(line_path)}={raw.shape}")
                continue
            distance = np.asarray(data["gnss_cumulative_distance_m"], dtype=np.float64)
            height = np.asarray(data["flight_height_agl_m"], dtype=np.float64)
            declared = np.asarray(data["declared_trace_distance_m"], dtype=np.float64)
            if distance.shape != (raw.shape[1],) or height.shape != (raw.shape[1],):
                blockers.append(f"real line spatial vectors do not match trace count: {rel(line_path)}")
                continue
            outside = int(np.sum((height < 2.0) | (height > 20.0)))
            if outside:
                height_review_lines.append(line_path.stem)
            source_member = str(np.asarray(data["source_csv_member"]).item())
            source_csv_sha = str(np.asarray(data["source_csv_sha256"]).item())
            source_zip_sha = str(np.asarray(data["source_zip_sha256"]).item())
            line_rows.append({
                "line_id": line_path.stem,
                "line_npz_path": rel(line_path),
                "source_path": rel(CANONICAL_REAL_ROOT / "source" / "ying_shan_measurement_lines_original.zip"),
                "source_csv_member": source_member,
                "source_csv_sha256": source_csv_sha,
                "source_zip_sha256": source_zip_sha,
                "profile_source_path": rel(profile_source) if profile_source.is_file() else "",
                "profile_source_sha256": profile_source_sha,
                "dt_ns": float(data["dt_ns"]),
                "trace_count": int(raw.shape[1]),
                "gnss_distance_m": float(distance[-1]),
                "declared_distance_m": float(declared[-1]),
                "flight_height_min_m": float(height.min()),
                "flight_height_median_m": float(np.median(height)),
                "flight_height_max_m": float(height.max()),
                "flight_height_outside_2_20_count": outside,
                "acquisition_bearing_deg": float(np.asarray(data["acquisition_bearing_deg"]).item()),
                "acquisition_compass": str(np.asarray(data["acquisition_compass"]).item()),
                "engineering_profile": str(np.asarray(data["engineering_profile"]).item()),
                "profile_left": str(np.asarray(data["profile_left"]).item()),
                "profile_right": str(np.asarray(data["profile_right"]).item()),
                "profile_display_flip": str(bool(np.asarray(data["profile_display_flip"]).item())).lower(),
                "profile_orientation_confidence": str(np.asarray(data["profile_orientation_confidence"]).item()),
                "sha256": sha256(line_path),
                "approved": "false",
            })

    for row in index_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        line_id = str(row.get("line", "")).strip()
        if not sample_id or not line_id:
            blockers.append("real window index contains blank sample_id/line")
            continue
        window_path = windows_dir / f"{sample_id}.npz"
        if not window_path.is_file():
            blockers.append(f"real window index references missing file: {rel(window_path)}")
            continue
        with np.load(window_path, allow_pickle=False) as data:
            status = np.asarray(data["status_code"], dtype=np.int16)
            ignored = (np.asarray(data["ignore_mask"], dtype=np.float32).mean(axis=0) > 0.5) if "ignore_mask" in data.files else np.zeros(status.shape, dtype=bool)
        has_negative = bool(np.any((status == 0) & ~ignored))
        confirmed_negative_windows += int(has_negative)
        split = str(row.get("split", "unassigned") or "unassigned")
        review_only = line_id in {"LineX1", "X1"}
        height_valid = str(row.get("antenna_height_agl_valid", "")).strip().lower() == "true"
        window_rows.append({
            "sample_id": sample_id,
            "line_id": line_id,
            "window_npz_path": rel(window_path),
            "trace_start": int(row["start"]),
            "trace_end": int(row["end"]),
            "status_semantics": (
                "contains_confirmed_negative" if has_negative
                else "positive_or_weak_positive_with_explicit_ignore" if bool(ignored.any())
                else "positive_or_weak_positive_only"
            ),
            "height_source": str(row.get("height_source", "original_csv_column_5_flight_height")),
            "antenna_height_agl_m": str(row.get("antenna_height_agl_m", "")),
            "antenna_height_agl_valid": str(height_valid).lower(),
            "height_quality": str(row.get("height_quality", "unknown")),
            "source_csv_member": str(row.get("source_csv_member", "")),
            "source_csv_sha256": str(row.get("source_csv_sha256", "")),
            "gnss_span_m": str(row.get("gnss_span_m", "")),
            "sha256": sha256(window_path),
            "split_group": line_id,
        })
        split_rows.append({
            "sample_id": sample_id,
            "group_id": line_id,
            "source_type": "real_original_csv_with_v15_final_labels" if REAL_ROOT == V15_FINAL_ROOT else "real_original_csv_with_audited_window_labels",
            "split": split,
            "holdout_compatible": "false" if review_only else "true",
            "exclusion_reason": "X1 is review-only" if review_only else "formal split remains config-controlled",
        })
    if not confirmed_negative_windows:
        blockers.append("no confirmed true-negative real windows")
    return line_rows, window_rows, split_rows, blockers, confirmed_negative_windows

def main() -> None:
    CONTRACT.mkdir(parents=True, exist_ok=True)
    recommendations = load_recommendations()
    case_dirs = accepted_case_dirs() + batch3_case_dirs()

    sim_fields = [
        "case_id", "source_group", "case_path", "scene_family_id", "raw_path", "label_path",
        "raw_sha256", "label_sha256", "scene_world_path", "scene_world_sha256",
        "design_metrics_path", "design_metrics_sha256", "metadata_trusted",
        "label_origin", "label_semantics", "reference_line", "line9_conditioned",
        "automatic_qc_grade", "human_decision", "train_allowed", "line9_holdout_allowed",
        "negative_semantics", "exclusion_reason",
        "contract_id", "target_presence", "trace_count", "trace_spacing_m",
        "physical_span_m", "gprmax_version", "postprocess_validated",
        "release_tier", "static_pair_topology_ok",
    ]
    audit_fields = ["case_id", "label", "auditor", "date", "method", "note", "source_sha256"]
    accepted_fields = [
        "case_id", "family", "accepted_path", "raw_sha256", "label_sha256",
        "qc_grade", "human_decision", "label_origin", "line9_conditioned",
        "formal_training_allowed", "metadata_trusted", "exclusion_reason",
    ]

    sim_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    accepted_rows: list[dict[str, object]] = []

    for case_dir in case_dirs:
        case_id = case_dir.name
        recommendation = recommendations.get(case_id, {})
        accepted = ACCEPTED_ROOT in case_dir.parents
        raw = raw_file(case_dir)
        label = label_file(case_dir)
        scene = scene_world_file(case_dir)
        metrics = design_metrics_file(case_dir)
        conditioned = is_line9_conditioned(case_dir, accepted)
        auto_grade = recommendation.get("automatic_qc_grade", "UNLISTED")
        audit_recommendation = recommendation.get("audit_recommendation", "manual_review_required")
        note = recommendation.get("note", "No authoritative case-wise human decision is recorded")
        reasons = [note]
        if conditioned:
            reasons.append("Line9-conditioned data are quarantined from formal Line9 holdout training")
        if scene is None or metrics is None:
            reasons.append("case-local scene/design provenance is incomplete")
        if accepted:
            reasons.append("historical accepted metadata are marked untrusted pending provenance rebuild")
        exclusion_reason = "; ".join(reason for reason in reasons if reason)
        raw_hash = sha256(raw) if raw else ""
        label_hash = sha256(label) if label else ""

        row: dict[str, object] = {
            "case_id": case_id,
            "source_group": "accepted_quarantine" if accepted else "batch3_audit",
            "case_path": rel(case_dir),
            "scene_family_id": case_id.rsplit("_", 1)[0],
            "raw_path": rel(raw),
            "label_path": rel(label),
            "raw_sha256": raw_hash,
            "label_sha256": label_hash,
            "scene_world_path": rel(scene),
            "scene_world_sha256": sha256(scene) if scene else "",
            "design_metrics_path": rel(metrics),
            "design_metrics_sha256": sha256(metrics) if metrics else "",
            "metadata_trusted": "false",
            "label_origin": "Line9_V14_conditioned" if conditioned else "case_local_unverified",
            "label_semantics": "visible_phase" if label and "visible_phase" in label.name else "unverified",
            "reference_line": "Line9" if conditioned else "",
            "line9_conditioned": str(conditioned).lower(),
            "automatic_qc_grade": auto_grade,
            "human_decision": audit_recommendation.upper(),
            "train_allowed": "false",
            "line9_holdout_allowed": "false" if conditioned else "pending",
            "negative_semantics": "not_a_negative_sample",
            "exclusion_reason": exclusion_reason,
        }
        sim_rows.append(row)
        audit_rows.append({
            "case_id": case_id,
            "label": recommendation.get("current_human_label", "unreviewed"),
            "auditor": "audit_session_20260710",
            "date": "2026-07-10",
            "method": "repository_audit_plus_visual_preview",
            "note": exclusion_reason,
            "source_sha256": raw_hash,
        })
        if accepted:
            try:
                family = case_dir.parent.relative_to(ACCEPTED_ROOT).as_posix()
            except ValueError:
                family = case_dir.parent.name
            accepted_rows.append({
                "case_id": case_id,
                "family": family,
                "accepted_path": rel(case_dir),
                "raw_sha256": raw_hash,
                "label_sha256": label_hash,
                "qc_grade": auto_grade,
                "human_decision": audit_recommendation,
                "label_origin": "Line9_V14_conditioned" if conditioned else "case_local_unverified",
                "line9_conditioned": str(conditioned).lower(),
                "formal_training_allowed": "false",
                "metadata_trusted": "false",
                "exclusion_reason": exclusion_reason,
            })

    # Audit decisions for missing V4/Pilot case bodies remain explicit. Missing
    # evidence cannot silently vanish from the governance record.
    existing = {str(row["case_id"]) for row in audit_rows}
    for case_id, recommendation in sorted(recommendations.items()):
        if case_id in existing:
            continue
        note = (
            recommendation.get("note", "")
            + "; case body is absent from the audited repository and cannot be promoted or trained"
        ).strip("; ")
        audit_rows.append({
            "case_id": case_id,
            "label": recommendation.get("current_human_label", "unreviewed"),
            "auditor": "audit_session_20260710",
            "date": "2026-07-10",
            "method": "repository_audit_missing_evidence_record",
            "note": note,
            "source_sha256": "",
        })

    if V2_RELEASE_GATE.is_file():
        with V2_RELEASE_GATE.open(encoding="utf-8-sig", newline="") as handle:
            v2_rows = list(csv.DictReader(handle))
        for release in v2_rows:
            case_id = release["case_id"]
            case_dir = resolve_contract_path(release["case_path"])
            manifest_path = case_dir / "scene_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {}
            )
            labels_dir = case_dir / "labels"
            pair_dir = case_dir / "pair_audit"
            trace_count = int(release.get("trace_count", "0") or 0)
            target_presence = release.get("target_semantics") != "confirmed_negative_design"
            raw = first_file(
                labels_dir / f"full_scene_501x{trace_count}.npy",
                pair_dir / f"full_501x{trace_count}.npy",
            )
            if target_presence:
                label = first_file(
                    labels_dir / "target_mask_visible_phase_501x256.npy",
                    pair_dir / "visible_phase_time_ns.npy",
                    labels_dir / "visible_phase_time_ns.npy",
                )
                label_semantics = "visible_phase_full_minus_no_basal"
                negative_semantics = "not_a_negative_sample"
            else:
                label = first_file(labels_dir / "target_mask_confirmed_negative_501x256.npy")
                label_semantics = "confirmed_negative_zero_target_mask"
                negative_semantics = "true_negative_candidate_not_formally_promoted"
            evidence = first_file(
                pair_dir / "pair_audit_validation.json",
                case_dir / "postprocess_validation.json",
            )
            grid = manifest.get("grid", {})
            span = grid.get("trace_midpoint_span_m", grid.get("trace_span_m", ""))
            source_hash = sha256(raw) if raw else ""
            decision_reason = release.get("decision_basis", "")
            sim_rows.append({
                "case_id": case_id,
                "source_group": "independent_simulation_v2_release_gate",
                "case_path": release["case_path"],
                "scene_family_id": release.get("family", case_id),
                "raw_path": rel(raw),
                "label_path": rel(label),
                "raw_sha256": source_hash,
                "label_sha256": sha256(label) if label else "",
                "scene_world_path": rel(manifest_path) if manifest_path.is_file() else "",
                "scene_world_sha256": sha256(manifest_path) if manifest_path.is_file() else "",
                "design_metrics_path": rel(evidence),
                "design_metrics_sha256": sha256(evidence) if evidence else "",
                "metadata_trusted": release.get("metadata_trusted", "false"),
                "label_origin": (
                    "explicit_target_absence_design"
                    if not target_presence
                    else "strict_or_topology_matched_full_minus_no_basal"
                ),
                "label_semantics": label_semantics,
                "reference_line": "",
                "line9_conditioned": release.get("line9_conditioned", "false"),
                "automatic_qc_grade": "V2_RELEASE_AUDIT",
                "human_decision": release.get("release_tier", "unreviewed").upper(),
                "train_allowed": "false",
                "line9_holdout_allowed": "pending_explicit_promotion",
                "negative_semantics": negative_semantics,
                "exclusion_reason": decision_reason,
                "contract_id": manifest.get("contract_id", "PGDA_SIMULATION_CONTRACT_V2"),
                "target_presence": str(target_presence).lower(),
                "trace_count": trace_count,
                "trace_spacing_m": grid.get("trace_spacing_m", ""),
                "physical_span_m": span,
                "gprmax_version": (
                    manifest.get("gprmax", {}).get("reviewed_version", "")
                    or "3.1.7" if "complete" in release.get("solver_state", "") else ""
                ),
                "postprocess_validated": release.get("postprocess_validated", "false"),
                "release_tier": release.get("release_tier", ""),
                "static_pair_topology_ok": release.get("static_pair_topology_ok", "false"),
            })
            audit_rows.append({
                "case_id": case_id,
                "label": release.get("release_tier", "unreviewed"),
                "auditor": "codex_physics_and_visual_audit_20260713",
                "date": "2026-07-13",
                "method": "gprMax_contract_pair_metrics_and_matched_scale_visual_review",
                "note": decision_reason,
                "source_sha256": source_hash,
            })

    sim_rows.sort(key=lambda row: str(row["case_id"]))
    audit_rows.sort(key=lambda row: str(row["case_id"]))
    accepted_rows.sort(key=lambda row: str(row["case_id"]))
    write_csv(CONTRACT / "simulation_cases.csv", sim_fields, sim_rows)
    write_csv(CONTRACT / "human_audit_manifest.csv", audit_fields, audit_rows)
    write_csv(ACCEPTED_ROOT / "accepted_manifest.csv", accepted_fields, accepted_rows)

    real_line_rows, real_window_rows, real_split_rows, real_blockers, confirmed_negative_windows = build_real_contract_rows()
    write_csv(
        CONTRACT / "real_lines.csv",
        [
            "line_id", "line_npz_path", "source_path", "source_csv_member", "source_csv_sha256",
            "source_zip_sha256", "profile_source_path", "profile_source_sha256",
            "dt_ns", "trace_count", "gnss_distance_m", "declared_distance_m",
            "flight_height_min_m", "flight_height_median_m", "flight_height_max_m",
            "flight_height_outside_2_20_count", "acquisition_bearing_deg", "acquisition_compass",
            "engineering_profile", "profile_left", "profile_right", "profile_display_flip",
            "profile_orientation_confidence", "sha256", "approved",
        ],
        real_line_rows,
    )
    write_csv(
        CONTRACT / "real_windows.csv",
        [
            "sample_id", "line_id", "window_npz_path", "trace_start", "trace_end",
            "status_semantics", "height_source", "antenna_height_agl_m", "antenna_height_agl_valid",
            "height_quality", "source_csv_member", "source_csv_sha256", "gnss_span_m", "sha256", "split_group",
        ],
        real_window_rows,
    )
    write_csv(
        CONTRACT / "split_manifest.csv",
        ["sample_id", "group_id", "source_type", "split", "holdout_compatible", "exclusion_reason"],
        real_split_rows,
    )

    crossing_audit_path = ROOT / "reports" / "yingshan_direction_profile_audit" / "line_intersections.csv"
    v15_manifest_path = V15_FINAL_ROOT / "manifests" / "v15_final_manifest.json"
    v15_released = v15_manifest_path.is_file()
    crossing_blockers: list[str] = []
    if not v15_released and crossing_audit_path.is_file():
        with crossing_audit_path.open(encoding="utf-8", newline="") as handle:
            crossing_rows = list(csv.DictReader(handle))
        risky = [row["crossing"] for row in crossing_rows if row.get("crossing_qc_grade", "").startswith(("critical", "high_risk"))]
        if risky:
            crossing_blockers.append("crossing-label consistency review required: " + ", ".join(risky))

    manifest = {
        "schema_version": "dataset_contract_v2",
        "generated_at": "2026-07-10",
        "audit_commit": AUDIT_COMMIT,
        "formal_training_allowed": False,
        "label_semantics": (
            "visible_phase_v15_final_with_weak_crossing_relabels_and_explicit_ignore"
            if v15_released else "visible_phase_pending_global_revalidation"
        ),
        "required_files": [
            "real_lines.csv", "real_windows.csv", "simulation_cases.csv",
            "human_audit_manifest.csv", "split_manifest.csv", "label_semantics.json",
        ],
        "blockers": real_blockers + crossing_blockers + [
            "Line9-conditioned simulations are quarantined and no formal independent simulation is approved",
            "formal line-level train/validation split remains unassigned",
            "Batch 3 requires case-wise geometry review",
        ],
        "observations": [
            "Line7 contains measured flight-height values above the planned 20 m operating range; "
            "the values remain valid physical metadata and are retained with an explicit QC flag.",
            "Canonical arrays use CSV acquisition order; engineering-profile flipping is display-only and governed by trace_direction_registry.csv.",
            *(
                ["YingShan V15 final labels resolve crossing supervision through two accepted weak relabels and two explicit local exclusions."]
                if v15_released else []
            ),
        ],
        "current_real_label_dataset": (
            {
                "path": rel(V15_FINAL_ROOT),
                "version": "YINGSHAN_V15_FINAL_20260710",
                "release_status": "final_label_release_not_formal_training_release",
                "line9_policy": "Line9 preserved as primary crossing anchor and test-only line.",
                "x1_policy": "X1 remains review-only/excluded.",
            }
            if v15_released else None
        ),
        "counts": {
            "simulation_cases_registered": len(sim_rows),
            "human_audit_records": len(audit_rows),
            "accepted_quarantine_cases": len(accepted_rows),
            "line9_conditioned_cases": sum(row["line9_conditioned"] == "true" for row in sim_rows),
            "training_allowed_cases": sum(row["train_allowed"] == "true" for row in sim_rows),
            "real_lines_registered": len(real_line_rows),
            "real_windows_registered": len(real_window_rows),
            "confirmed_true_negative_windows": confirmed_negative_windows,
        },
    }
    write_json(CONTRACT / "dataset_manifest.json", manifest)
    write_json(
        CONTRACT / "label_semantics.json",
        {
            "schema_version": "1.0",
            "primary_curve_target": "visible_phase_time_ns",
            "definitions": {
                "geometry_time_ns": "Two-way geometric propagation-time estimate; not interchangeable with the visible waveform phase.",
                "visible_phase_time_ns": "Consistently selected visible target phase or envelope feature in the B-scan.",
                "label_center_ns": "Approved training curve after one visible-phase rule is applied to every source.",
                "label_lower_ns": "Lower uncertainty/tolerance boundary.",
                "label_upper_ns": "Upper uncertainty/tolerance boundary.",
            },
            "forbidden_mixture": "Geometry-time and visible-phase labels must not be mixed without an explicit, auditable conversion.",
            "v4_status": "superseded_by_v15_final_for_current_real_label_release" if v15_released else "blocked_pending_visible_phase_relabel",
            "v15_status": "released_final_labels_not_formal_training_release" if v15_released else "not_released",
            "v15_version": "YINGSHAN_V15_FINAL_20260710" if v15_released else "",
            "v15_policy": (
                {
                    "primary_anchor": "Line9 at cross-line conflicts",
                    "accepted_weak_relabels": ["Line3-Line9: Line3", "LineL1-LineX1: LineX1"],
                    "explicit_ignores": ["Line6-Line9: Line6", "Line9-LineX1: LineX1"],
                    "unresolved_ambiguity_is_never_promoted_to_strong": True,
                }
                if v15_released else {}
            ),
        },
    )
    readme = """# dataset_contract_v2

本目录是实测与仿真训练数据的唯一治理层。目录名、自动 QC 等级或历史 `accepted` 状态均不构成训练许可。

## 实测数据

6 条 canonical 全测线由原始营山 CSV ZIP 生成。当前标签发布为 `YINGSHAN_V15_FINAL_20260710`：Line9 保持原标签并继续作为 test-only 主锚点；Line3 与 X1 各有一处弱标签重标；Line6 与 X1 各有一处歧义区明确排除监督。V14 标签完整保留用于回滚。

## 测线方向与剖面显示合同

- Canonical 数组、窗口索引、训练和指标始终保持原始 CSV 采集顺序。
- `profile_display_flip` 只能用于可视化/导出。
- 地图轨迹使用 `gnss_cumulative_distance_m`；工程剖面对照使用 `profile_chainage_m`。
- Line3、Line6、Line9、LineX1 的剖面显示相对采集顺序翻转；Line7、LineL1 不翻转。

## 当前冻结原因

- 没有确认的实测真负窗口；
- 没有获批的非 Line9-conditioned 正式仿真；
- 正式 train/validation 测线级划分尚未锁定；
- Batch 3 仍待逐 case 审核。

V15 最终标签发布完成不等于正式训练放行。只有 `dataset_manifest.json` 明确设置 `formal_training_allowed=true` 且 `python scripts/validate_project_contracts.py --require-formal-ready` 通过后才允许启动正式训练。
"""
    (CONTRACT / "README.md").write_text(readme, encoding="utf-8")

    write_json(
        ACCEPTED_ROOT / "dataset_policy.json",
        {
            "dataset_id": "PGDA_SYNTH_DATASET_V1_accepted_quarantine",
            "training_allowed": False,
            "line9_conditioned": True,
            "formal_line9_holdout_allowed": False,
            "reason": "Historical accepted cases are Line9-conditioned and have untrusted duplicated provenance metadata.",
            "authoritative_manifest": "accepted_manifest.csv",
        },
    )
    write_json(
        BATCH3_ROOT / "dataset_policy.json",
        {
            "dataset_id": "BATCH_003_SHALLOW_GENERALIZATION_AUDIT_QUARANTINE",
            "training_allowed": False,
            "line9_conditioned": True,
            "reason": "Case-wise geometry review is incomplete; automatic QC grades do not authorize training.",
            "authoritative_manifest": rel(CONTRACT / "simulation_cases.csv"),
        },
    )
    existing_real_policy = {}
    real_policy_path = REAL_ROOT / "dataset_policy.json"
    if real_policy_path.is_file():
        existing_real_policy = json.loads(real_policy_path.read_text(encoding="utf-8"))
    existing_real_policy.update({
        "training_allowed": False,
        "confirmed_true_negative_traces": 0,
        "missing_required_artifacts": [] if real_line_rows and real_window_rows else ["window_index.csv", "lines/*.npz"],
    })
    write_json(real_policy_path, existing_real_policy)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
