#!/usr/bin/env python3
"""Validate the real-data contract used by PGDA-CSNet training/evaluation.

This command reports structural completeness separately from formal-paper
readiness. It never fabricates a missing line archive or window index.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.experiment_contract import (  # noqa: E402
    ContractError,
    REQUIRED_WINDOW_INDEX_COLUMNS,
    inspect_window_dataset,
    load_dataset_usage_policy,
    resolve_window_npz,
)

FORBIDDEN_INPUT_TOKENS = {
    "bg501", "bg501_agc9", "processed_view", "response_teacher",
    "full_background", "target_only", "qa_view", "agc9", "bg501_agc",
}
REQUIRED_WINDOW_KEYS = {"x_raw", "y_mask", "status_code", "label_weight"}
REQUIRED_LINE_KEYS = {
    "raw_amplitude", "raw_full_normalized", "soft_mask_train", "status_code", "label_weight",
    "time_ns", "dt_ns", "trace_interval_m", "longitude", "latitude", "ground_elevation_m",
    "flight_height_agl_m", "antenna_elevation_m", "gnss_cumulative_distance_m",
    "source_zip_sha256", "source_csv_member", "source_csv_sha256", "canonical_source",
    "profile_chainage_m", "acquisition_bearing_deg", "acquisition_compass",
    "engineering_profile", "profile_left", "profile_right", "profile_display_flip",
    "profile_orientation_confidence", "orientation_contract",
}
HEIGHT_COLUMNS = (
    "antenna_height_agl_m", "flight_height_agl_m", "altitude",
    "flight_height_m", "antenna_height_m",
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _forbidden_keys(keys: set[str]) -> list[str]:
    return sorted(key for key in keys if any(token in key.lower() for token in FORBIDDEN_INPUT_TOKENS))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(data_root: Path, *, require_formal_ready: bool = False) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, object] = {"data_root": str(data_root)}

    policy = load_dataset_usage_policy(data_root) or {}
    facts["dataset_policy"] = policy
    training_allowed = policy.get("training_allowed", policy.get("train_allowed", True))
    if training_allowed is False:
        message = f"dataset policy blocks training: {policy.get('reason', policy.get('blocking_reasons', 'unspecified'))}"
        (errors if require_formal_ready else warnings).append(message)

    lines_dir = data_root / "lines"
    if not lines_dir.is_dir():
        errors.append(f"missing full-line archive directory: {lines_dir}")

    try:
        summary = inspect_window_dataset(data_root, require_windows=True)
    except ContractError as exc:
        errors.append(str(exc))
        return {"ok": False, "formal_ready": False, "errors": errors, "warnings": warnings, "facts": facts}

    rows = _read_rows(summary.index_path)
    facts.update({
        "window_index": str(summary.index_path),
        "window_count": summary.row_count,
        "lines": list(summary.lines),
        "index_columns": list(summary.fieldnames),
        "height_columns": [name for name in HEIGHT_COLUMNS if name in summary.fieldnames],
    })

    status_counts: Counter[int] = Counter()
    indexed_ids = set()
    referenced_lines = set()
    for row in rows:
        sample_id = row["sample_id"]
        indexed_ids.add(sample_id)
        referenced_lines.add(row["line"])
        path = resolve_window_npz(data_root, row)
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
                missing = REQUIRED_WINDOW_KEYS - keys
                if missing:
                    errors.append(f"{path}: missing window keys {sorted(missing)}")
                    continue
                forbidden = _forbidden_keys(keys)
                if forbidden:
                    errors.append(f"{path}: forbidden processed-input keys {forbidden}")
                arrays = [np.asarray(data[name]) for name in REQUIRED_WINDOW_KEYS]
                if any(not np.isfinite(array).all() for array in arrays):
                    errors.append(f"{path}: required array contains NaN/Inf")
                values, counts = np.unique(np.asarray(data["status_code"]).astype(np.int64), return_counts=True)
                status_counts.update({int(value): int(count) for value, count in zip(values, counts)})
        except Exception as exc:
            errors.append(f"{path}: unreadable NPZ: {exc}")

    unindexed = sorted(path.name for path in (data_root / "windows").glob("*.npz") if path.stem not in indexed_ids)
    if unindexed:
        errors.append(f"unindexed window files: {unindexed[:20]}")

    if lines_dir.is_dir():
        for line_name in sorted(referenced_lines):
            path = lines_dir / f"{line_name}.npz"
            if not path.is_file():
                errors.append(f"missing full-line NPZ: {path}")
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    keys = set(data.files)
                    missing = REQUIRED_LINE_KEYS - keys
                    if missing:
                        errors.append(f"{path}: missing line keys {sorted(missing)}")
                    forbidden = _forbidden_keys(keys)
                    if forbidden:
                        errors.append(f"{path}: forbidden processed-input keys {forbidden}")
                    if not missing:
                        raw = np.asarray(data["raw_full_normalized"])
                        traces = int(raw.shape[1]) if raw.ndim == 2 else -1
                        vector_keys = (
                            "longitude", "latitude", "ground_elevation_m", "flight_height_agl_m",
                            "antenna_elevation_m", "gnss_cumulative_distance_m", "profile_chainage_m",
                        )
                        for key in vector_keys:
                            array = np.asarray(data[key])
                            if array.shape != (traces,):
                                errors.append(f"{path}: {key} shape {array.shape} does not match traces={traces}")
                            elif not np.isfinite(array).all():
                                errors.append(f"{path}: {key} contains NaN/Inf")
                        if traces > 0:
                            ground = np.asarray(data["ground_elevation_m"], dtype=np.float64)
                            height = np.asarray(data["flight_height_agl_m"], dtype=np.float64)
                            antenna = np.asarray(data["antenna_elevation_m"], dtype=np.float64)
                            distance = np.asarray(data["gnss_cumulative_distance_m"], dtype=np.float64)
                            if np.any(height <= 0):
                                errors.append(f"{path}: flight_height_agl_m contains non-positive values")
                            if not np.allclose(antenna, ground + height, atol=2e-4, rtol=0.0):
                                errors.append(f"{path}: antenna_elevation_m != ground_elevation_m + flight_height_agl_m")
                            if distance.size and (abs(float(distance[0])) > 1e-6 or np.any(np.diff(distance) < -1e-8)):
                                errors.append(f"{path}: gnss_cumulative_distance_m is not monotonic from zero")
                            profile_distance = np.asarray(data["profile_chainage_m"], dtype=np.float64)
                            if profile_distance.shape != (traces,) or abs(float(profile_distance[0])) > 1e-6 or np.any(np.diff(profile_distance) < -1e-8):
                                errors.append(f"{path}: profile_chainage_m is not a trace-aligned monotonic axis from zero")
                            if str(np.asarray(data["orientation_contract"]).item()) != "canonical arrays remain acquisition order; profile flip is display-only":
                                errors.append(f"{path}: invalid orientation contract")
            except Exception as exc:
                errors.append(f"{path}: unreadable line NPZ: {exc}")

    registry_path = data_root / "trace_direction_registry.csv"
    orientation_path = data_root / "orientation_contract.json"
    if not registry_path.is_file():
        errors.append(f"missing trace direction registry: {registry_path}")
    if not orientation_path.is_file():
        errors.append(f"missing orientation contract: {orientation_path}")
    facts["orientation_registry"] = str(registry_path)
    facts["orientation_contract"] = str(orientation_path)

    facts["status_code_counts"] = {str(key): value for key, value in sorted(status_counts.items())}
    confirmed_negative_count = int(status_counts.get(0, 0))
    facts["confirmed_negative_trace_count"] = confirmed_negative_count
    if confirmed_negative_count == 0:
        message = "no confirmed status_code=0 traces; presence/no-target rejection supervision is not trainable"
        (errors if require_formal_ready else warnings).append(message)

    # Dataset-level split metadata is optional. Config/manifests own the actual
    # split; when a split column exists, only enforce immutable review/test locks.
    if rows and "split" in rows[0]:
        if any(row["line"] == "Line9" and row.get("split") != "test" for row in rows):
            errors.append("Line9 rows must remain dataset-level test rows")
        if any(row["line"] in {"LineX1", "X1"} and row.get("split") != "exclude" for row in rows):
            errors.append("X1 rows must remain excluded review rows")

    formal_ready = not errors and training_allowed is not False and confirmed_negative_count > 0
    if require_formal_ready and not formal_ready and not errors:
        errors.append("dataset is not formal-training-ready")
    return {
        "ok": not errors,
        "formal_ready": formal_ready,
        "errors": errors,
        "warnings": warnings,
        "facts": facts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/measured/yingshan_v15")
    parser.add_argument("--require-formal-ready", action="store_true")
    args = parser.parse_args()
    report = validate(_resolve(args.data_root), require_formal_ready=args.require_formal_ready)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
