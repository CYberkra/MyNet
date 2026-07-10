"""Validate configuration and dataset-governance contracts without training."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAL = {"lolo_eval", "holdout_eval", "baseline_eval", "paper_eval", "paper_train"}
KNOWN_RUN_TYPES = FORMAL | {"pretrain_sim_only", "development", "debug", "final_train", "exploratory"}
REVIEW_ONLY = {"x1", "linex1"}


def norm_line(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "").replace("-", "")



def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_configs() -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = {"config_count": 0, "disabled_config_count": 0, "formal_config_count": 0}
    for path in sorted((ROOT / "configs").glob("*.json")):
        if path.name.startswith("paper_splits_"):
            continue
        counts["config_count"] += 1
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        run_type = str(cfg.get("run_type", "")).lower()
        if not run_type:
            errors.append(f"{path}: missing run_type")
        elif run_type not in KNOWN_RUN_TYPES:
            errors.append(f"{path}: unknown run_type={run_type!r}")
        disabled = cfg.get("enabled") is False
        if disabled:
            counts["disabled_config_count"] += 1
            if not str(cfg.get("training_block_reason", "")).strip():
                errors.append(f"{path}: disabled config lacks training_block_reason")
        active_formal = run_type in FORMAL and not disabled
        if active_formal:
            counts["formal_config_count"] += 1
        split_policy = None
        split_policy_path = None
        split_value = str(cfg.get("paper_split_file", "")).strip()
        if active_formal and not split_value:
            errors.append(f"{path}: formal config lacks paper_split_file")
        if split_value:
            split_policy_path = Path(split_value)
            if not split_policy_path.is_absolute():
                split_policy_path = ROOT / split_policy_path
            if not split_policy_path.is_file():
                errors.append(f"{path}: paper_split_file missing: {split_policy_path}")
            else:
                try:
                    split_policy = json.loads(split_policy_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"{path}: invalid paper_split_file: {exc}")

        train = set(cfg.get("train_lines") or [])
        val = set(cfg.get("val_lines") or [])
        test = set(cfg.get("test_lines") or [])
        review_only = set(REVIEW_ONLY)
        if split_policy:
            review_only.update(norm_line(v) for v in split_policy.get("review_only_lines", []))
        misuse = [line for line in train | val | test if norm_line(line) in review_only]
        if misuse:
            errors.append(f"{path}: review-only line used in train/val/test: {sorted(misuse)}")
        if val & test:
            errors.append(f"{path}: validation/test overlap: {sorted(val & test)}")
        if active_formal and not val:
            errors.append(f"{path}: formal config has no validation split")
        if active_formal and train & test:
            errors.append(f"{path}: formal config has train/test line overlap: {sorted(train & test)}")
        if active_formal and train & val:
            errors.append(f"{path}: formal config has train/validation line overlap: {sorted(train & val)}")
        if split_policy and active_formal and len(test) == 1:
            holdout = next(iter(test))
            expected = (split_policy.get("inner_validation_by_holdout") or {}).get(holdout)
            if expected and val != {str(expected)}:
                errors.append(
                    f"{path}: holdout {holdout} requires validation {expected} by {split_policy_path}"
                )
        if float(cfg.get("sim_batch_ratio", 0.0) or 0.0) > 0 and not cfg.get("sim_data_root"):
            errors.append(f"{path}: sim_batch_ratio>0 without sim_data_root")

        loss = cfg.get("loss") or {}
        arrival = max(
            float(loss.get("arrival_prior_weight", 0.0) or 0.0),
            float(loss.get("arrival_time_prior_weight", 0.0) or 0.0),
        )
        if arrival > 0 and cfg.get("arrival_prior_missing_height_policy") not in {"error", "skip", "default"}:
            errors.append(f"{path}: arrival prior lacks explicit missing-height policy")
        if cfg.get("arrival_prior_missing_height_policy") == "default":
            try:
                if float(cfg.get("default_altitude_m", 0.0)) <= 0:
                    errors.append(f"{path}: default height policy requires positive default_altitude_m")
            except Exception:
                errors.append(f"{path}: invalid default_altitude_m")
        if not disabled:
            warnings.append(f"{path}: enabled config must be checked against resolved dataset manifests")
    return errors, warnings, counts


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_dataset_contract(require_formal_ready: bool) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    contract = ROOT / "data" / "dataset_contract_v2"
    manifest_path = contract / "dataset_manifest.json"
    if not manifest_path.is_file():
        return [f"missing {manifest_path}"], warnings, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid {manifest_path}: {exc}"], warnings, {}
    if manifest.get("schema_version") != "dataset_contract_v2":
        errors.append(f"unexpected dataset schema: {manifest.get('schema_version')!r}")
    for name in manifest.get("required_files", []):
        if not (contract / name).is_file():
            errors.append(f"dataset contract missing required file: {name}")
    if manifest.get("formal_training_allowed") is not True:
        message = "dataset_contract_v2 explicitly blocks formal training"
        (errors if require_formal_ready else warnings).append(message)

    sim_path = contract / "simulation_cases.csv"
    human_path = contract / "human_audit_manifest.csv"
    sim_rows = _read_csv(sim_path) if sim_path.is_file() else []
    human_rows = _read_csv(human_path) if human_path.is_file() else []
    human_ids = [row.get("case_id", "") for row in human_rows]
    duplicate_human = sorted({case_id for case_id in human_ids if human_ids.count(case_id) > 1})
    if duplicate_human:
        errors.append(f"duplicate human audit rows: {duplicate_human}")
    for row in sim_rows:
        case_id = row.get("case_id", "")
        if row.get("line9_conditioned") == "true" and row.get("line9_holdout_allowed") == "true":
            errors.append(f"{case_id}: Line9-conditioned case incorrectly allowed in Line9 holdout")
        raw_text = row.get("raw_path", "").strip()
        label_text = row.get("label_path", "").strip()
        if raw_text and not (ROOT / raw_text).is_file():
            errors.append(f"{case_id}: registered raw path is missing: {raw_text}")
        if label_text and not (ROOT / label_text).is_file():
            errors.append(f"{case_id}: registered label path is missing: {label_text}")
        matching_human = [item for item in human_rows if item.get("case_id") == case_id]
        if matching_human and matching_human[0].get("source_sha256") and row.get("raw_sha256"):
            if matching_human[0]["source_sha256"].lower() != row["raw_sha256"].lower():
                errors.append(f"{case_id}: human audit source hash differs from simulation manifest raw hash")
        if row.get("train_allowed") == "true":
            if not row.get("human_decision"):
                errors.append(f"{case_id}: training allowed without human decision")
            if case_id not in human_ids:
                errors.append(f"{case_id}: training allowed without a human audit record")
            if row.get("metadata_trusted") != "true":
                errors.append(f"{case_id}: training allowed with untrusted provenance metadata")
            raw_path = ROOT / row.get("raw_path", "")
            label_path = ROOT / row.get("label_path", "")
            if not raw_path.is_file() or not label_path.is_file():
                errors.append(f"{case_id}: training allowed with missing raw/label source")
        if row.get("negative_semantics") == "true_negative" and row.get("human_decision", "").lower() not in {
            "approved_true_negative", "promote_true_negative"
        }:
            errors.append(f"{case_id}: true-negative semantics lack explicit human approval")

    accepted_path = ROOT / "data" / "PGDA_SYNTH_DATASET_V1" / "05_accepted_dataset" / "accepted_manifest.csv"
    accepted_rows = _read_csv(accepted_path) if accepted_path.is_file() else []
    real_lines = _read_csv(contract / "real_lines.csv") if (contract / "real_lines.csv").is_file() else []
    real_windows = _read_csv(contract / "real_windows.csv") if (contract / "real_windows.csv").is_file() else []
    orientation_registry = ROOT / "data_corrected_v1_4_terrain_direction" / "trace_direction_registry.csv"
    orientation_contract = ROOT / "data_corrected_v1_4_terrain_direction" / "orientation_contract.json"
    if not orientation_registry.is_file():
        errors.append("canonical real dataset is missing trace_direction_registry.csv")
    if not orientation_contract.is_file():
        errors.append("canonical real dataset is missing orientation_contract.json")
    # Canonical real-data provenance must resolve to original CSV sources and
    # agree with the hashes embedded in each line archive.
    source_hash_cache: dict[Path, str] = {}
    for row in real_lines:
        line_id = row.get("line_id", "")
        line_path = ROOT / row.get("line_npz_path", "")
        source_path = ROOT / row.get("source_path", "")
        profile_source_path = ROOT / row.get("profile_source_path", "") if row.get("profile_source_path") else None
        if not line_path.is_file():
            errors.append(f"{line_id}: registered line NPZ is missing: {line_path}")
            continue
        if row.get("sha256") and _sha256(line_path).lower() != row["sha256"].lower():
            errors.append(f"{line_id}: line NPZ hash differs from real_lines.csv")
        if not source_path.is_file():
            errors.append(f"{line_id}: original source archive is missing: {source_path}")
        else:
            if source_path not in source_hash_cache:
                source_hash_cache[source_path] = _sha256(source_path)
            if row.get("source_zip_sha256") and source_hash_cache[source_path].lower() != row["source_zip_sha256"].lower():
                errors.append(f"{line_id}: original source archive hash mismatch")
        if profile_source_path is None or not profile_source_path.is_file():
            errors.append(f"{line_id}: survey profile/borehole source archive is missing")
        elif row.get("profile_source_sha256") and _sha256(profile_source_path).lower() != row["profile_source_sha256"].lower():
            errors.append(f"{line_id}: survey profile/borehole source archive hash mismatch")
        try:
            with np.load(line_path, allow_pickle=False) as data:
                required = {
                    "raw_amplitude", "raw_full_normalized", "longitude", "latitude",
                    "ground_elevation_m", "flight_height_agl_m", "antenna_elevation_m",
                    "gnss_cumulative_distance_m", "source_csv_member", "source_csv_sha256",
                    "source_zip_sha256", "canonical_source", "profile_chainage_m",
                    "acquisition_bearing_deg", "profile_display_flip", "orientation_contract",
                }
                missing = sorted(required - set(data.files))
                if missing:
                    errors.append(f"{line_id}: canonical line archive missing {missing}")
                else:
                    if str(np.asarray(data["canonical_source"]).item()) != "original_yingshan_csv":
                        errors.append(f"{line_id}: canonical_source is not original_yingshan_csv")
                    if row.get("source_csv_sha256") and str(np.asarray(data["source_csv_sha256"]).item()).lower() != row["source_csv_sha256"].lower():
                        errors.append(f"{line_id}: embedded source CSV hash mismatch")
                    height = np.asarray(data["flight_height_agl_m"], dtype=np.float64)
                    ground = np.asarray(data["ground_elevation_m"], dtype=np.float64)
                    antenna = np.asarray(data["antenna_elevation_m"], dtype=np.float64)
                    if not np.allclose(antenna, ground + height, atol=2e-4, rtol=0.0):
                        errors.append(f"{line_id}: antenna elevation is inconsistent with ground + flight height")
                    profile_distance = np.asarray(data["profile_chainage_m"], dtype=np.float64)
                    if profile_distance.shape != height.shape or np.any(np.diff(profile_distance) < -1e-8):
                        errors.append(f"{line_id}: invalid profile_chainage_m")
                    if str(np.asarray(data["orientation_contract"]).item()) != "canonical arrays remain acquisition order; profile flip is display-only":
                        errors.append(f"{line_id}: invalid orientation contract")
        except Exception as exc:
            errors.append(f"{line_id}: failed to validate canonical line archive: {exc}")

    for row in real_windows:
        sample_id = row.get("sample_id", "")
        if row.get("height_source") != "original_csv_column_5_flight_height":
            errors.append(f"{sample_id}: window height source is not original CSV column 5")
        try:
            if float(row.get("antenna_height_agl_m", "nan")) <= 0:
                errors.append(f"{sample_id}: invalid window AGL height")
        except ValueError:
            errors.append(f"{sample_id}: invalid window AGL height")
        if row.get("source_csv_sha256", "") == "":
            errors.append(f"{sample_id}: missing original CSV hash")

    accepted_ids = [row.get("case_id", "") for row in accepted_rows]
    if len(accepted_ids) != len(set(accepted_ids)):
        errors.append("accepted_manifest.csv contains duplicate case_id rows")
    if any(row.get("formal_training_allowed") == "true" for row in accepted_rows):
        errors.append("historical accepted quarantine unexpectedly grants formal training")
    counts = {
        "registered_simulation_cases": len(sim_rows),
        "human_audit_records": len(human_rows),
        "accepted_quarantine_cases": len(accepted_rows),
        "registered_real_lines": len(real_lines),
        "registered_real_windows": len(real_windows),
        "train_allowed_simulation_cases": sum(row.get("train_allowed") == "true" for row in sim_rows),
        "line9_conditioned_cases": sum(row.get("line9_conditioned") == "true" for row in sim_rows),
    }
    expected = manifest.get("counts", {})
    if expected.get("simulation_cases_registered") not in (None, counts["registered_simulation_cases"]):
        errors.append("dataset_manifest simulation case count does not match simulation_cases.csv")
    if expected.get("human_audit_records") not in (None, counts["human_audit_records"]):
        errors.append("dataset_manifest human audit count does not match human_audit_manifest.csv")
    return errors, warnings, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-formal-ready", action="store_true")
    args = parser.parse_args()
    errors1, warnings1, counts1 = validate_configs()
    errors2, warnings2, counts2 = validate_dataset_contract(args.require_formal_ready)
    report = {
        "ok": not (errors1 or errors2),
        "errors": errors1 + errors2,
        "warnings": warnings1 + warnings2,
        "counts": {**counts1, **counts2},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
