"""Export explicitly approved simulation cases to the canonical training schema.

Only `train_allowed=true` rows from dataset_contract_v2 are eligible. Directory
names and automatic QC grades never grant training permission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.label_semantics import derive_supervision_state

DEFAULT_MANIFEST = ROOT / "data" / "dataset_contract_v2" / "simulation_cases.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _case_mask(case_dir: Path, row: dict[str, str]) -> Path:
    """Resolve the one manifest-approved training label for a simulation case.

    Historical directories contain several geometry and soft-label artifacts.
    File-order selection previously preferred legacy ``y_soft`` labels, which
    are not uniformly visible-phase aligned.  A training-eligible case must
    now name one label artifact in the contract.
    """
    label_path = str(row.get("label_path", "")).strip()
    if not label_path:
        raise RuntimeError(
            f"{row.get('case_id', case_dir.name)}: label_path is required for training export; "
            "automatic legacy label selection is forbidden"
        )
    path = _resolve(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest-approved label is missing for {row.get('case_id', case_dir.name)}: {path}")
    if str(row.get("contract_id", "")).strip() == "PGDA_SIMULATION_CONTRACT_V2":
        declared_presence = str(row.get("target_presence", "")).strip().lower()
        label_name = (
            "target_mask_confirmed_negative_501x256.npy"
            if declared_presence in {"false", "0", "no"}
            else "target_mask_visible_phase_501x256.npy"
        )
        expected = case_dir / "labels" / label_name
        if path.resolve() != expected.resolve():
            raise RuntimeError(
                f"{row.get('case_id', case_dir.name)}: Simulation V2 must export the "
                f"postprocessed {label_name} label"
            )
    return path


def _resample_time(raw: np.ndarray, target_h: int) -> np.ndarray:
    if raw.ndim != 2:
        raise ValueError(f"raw B-scan must be 2D, got {raw.shape}")
    if raw.shape[0] == target_h:
        return raw.astype(np.float32, copy=False)
    src = np.linspace(0.0, 1.0, raw.shape[0], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, target_h, dtype=np.float64)
    out = np.empty((target_h, raw.shape[1]), np.float32)
    for trace in range(raw.shape[1]):
        out[:, trace] = np.interp(dst, src, raw[:, trace]).astype(np.float32)
    return out




def _validate_v2_spatial_contract(row: dict[str, str], raw: np.ndarray, mask: np.ndarray) -> None:
    """Reject image-resized or incompletely validated Simulation V2 cases."""
    if str(row.get("contract_id", "")).strip() != "PGDA_SIMULATION_CONTRACT_V2":
        return
    if str(row.get("postprocess_validated", "")).strip().lower() != "true":
        raise RuntimeError(f"{row.get('case_id')}: Simulation V2 postprocess_validated must be true")
    if str(row.get("metadata_trusted", "")).strip().lower() != "true":
        raise RuntimeError(f"{row.get('case_id')}: Simulation V2 metadata_trusted must be true")
    if str(row.get("line9_conditioned", "")).strip().lower() != "false":
        raise RuntimeError(f"{row.get('case_id')}: Simulation V2 must be non-Line9-conditioned")
    if raw.shape != (501, 256) or mask.shape != (501, 256):
        raise RuntimeError(
            f"{row.get('case_id')}: Simulation V2 must already be canonical 501x256; "
            f"got raw={raw.shape}, mask={mask.shape}. Width resize/padding is forbidden."
        )
    trace_count = int(row.get("trace_count", "0") or 0)
    spacing = float(row.get("trace_spacing_m", "nan") or "nan")
    span = float(row.get("physical_span_m", "nan") or "nan")
    if trace_count != 256:
        raise RuntimeError(f"{row.get('case_id')}: trace_count must be 256")
    if not np.isclose(spacing, 0.09, atol=1e-9, rtol=0):
        raise RuntimeError(f"{row.get('case_id')}: trace_spacing_m must be 0.09")
    if not np.isclose(span, 22.95, atol=1e-6, rtol=0):
        raise RuntimeError(f"{row.get('case_id')}: physical_span_m must be 22.95")
    if not str(row.get("gprmax_version", "")).strip():
        raise RuntimeError(f"{row.get('case_id')}: gprmax_version is required")

def load_eligible_rows(manifest: Path, formal_test_line: str = "") -> list[dict[str, str]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Simulation contract manifest not found: {manifest}")
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    eligible = [row for row in rows if str(row.get("train_allowed", "")).lower() == "true"]
    if formal_test_line.strip().lower() == "line9":
        leaked = [
            row.get("case_id", "")
            for row in eligible
            if str(row.get("line9_conditioned", "")).lower() == "true"
        ]
        if leaked:
            raise RuntimeError(
                f"Line9-conditioned cases are forbidden in formal Line9 holdout export: {leaked}"
            )
    return eligible


def export(
    manifest: Path,
    out_root: Path,
    formal_test_line: str = "",
    overwrite: bool = False,
) -> dict[str, object]:
    eligible = load_eligible_rows(manifest, formal_test_line)
    if not eligible:
        raise RuntimeError(
            "No simulation cases have train_allowed=true; human approval and data contract remain incomplete."
        )
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {out_root}")
        shutil.rmtree(out_root)
    windows = out_root / "windows"
    windows.mkdir(parents=True)

    index_fields = [
        "sample_id", "line", "start", "end", "present", "weak", "no_pick",
        "source_case_id", "source_raw_sha256", "source_label_sha256",
        "antenna_height_agl_m", "label_semantics", "target_presence",
        "scene_family_id", "contract_id", "trace_count", "trace_spacing_m",
        "physical_span_m", "gprmax_version", "postprocess_validated",
    ]
    index_rows: list[dict[str, object]] = []
    for row in eligible:
        case_id = row["case_id"]
        case_dir = _resolve(row["case_path"])
        raw_path = _resolve(row["raw_path"])
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw source for {case_id}: {raw_path}")
        expected_raw_hash = str(row.get("raw_sha256", "")).strip().lower()
        actual_raw_hash = _sha256(raw_path)
        if not expected_raw_hash or expected_raw_hash != actual_raw_hash:
            raise RuntimeError(
                f"Raw provenance hash mismatch for {case_id}: manifest={expected_raw_hash or '<missing>'}, actual={actual_raw_hash}"
            )
        mask_path = _case_mask(case_dir, row)
        expected_label_hash = str(row.get("label_sha256", "")).strip().lower()
        actual_label_hash = _sha256(mask_path)
        if expected_label_hash and expected_label_hash != actual_label_hash:
            raise RuntimeError(
                f"Label provenance hash mismatch for {case_id}: manifest={expected_label_hash}, actual={actual_label_hash}"
            )

        raw = np.load(raw_path, allow_pickle=False).astype(np.float32)
        mask = np.load(mask_path, allow_pickle=False).astype(np.float32)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D for {case_id}: {mask.shape}")
        raw = _resample_time(raw, mask.shape[0])
        if raw.shape[1] != mask.shape[1]:
            raise ValueError(f"Raw/mask trace mismatch for {case_id}: {raw.shape} vs {mask.shape}")
        if not np.isfinite(raw).all() or not np.isfinite(mask).all():
            raise ValueError(f"NaN/Inf in {case_id}")
        _validate_v2_spatial_contract(row, raw, mask)
        mask = np.clip(mask, 0.0, 1.0)
        trace_mass = mask.sum(axis=0)
        status = np.where(trace_mass > 1e-6, 1, 0).astype(np.int16)
        label_weight = np.where(status == 1, 1.0, 0.0).astype(np.float32)
        ignore_mask = np.zeros_like(mask, dtype=np.float32)

        declared_presence = str(row.get("target_presence", "")).strip().lower()
        if declared_presence in {"false", "0", "no"} and np.any(status != 0):
            raise RuntimeError(f"{case_id}: manifest declares target_presence=false but label mask is non-zero")
        if declared_presence in {"true", "1", "yes"} and not np.any(status == 1):
            raise RuntimeError(f"{case_id}: manifest declares target_presence=true but label mask is empty")

        supervision_state = derive_supervision_state(status, label_weight, ignore_mask)
        if declared_presence in {"false", "0", "no"} and not np.all(supervision_state == 0):
            raise RuntimeError(f"{case_id}: background-only case failed confirmed-negative semantics")

        np.savez_compressed(
            windows / f"{case_id}.npz",
            x_raw=raw,
            y_mask=mask,
            status_code=status,
            label_weight=label_weight,
            ignore_mask=ignore_mask,
            supervision_state=supervision_state,
        )
        index_rows.append({
            "sample_id": case_id,
            "line": case_id,
            "start": 0,
            "end": raw.shape[1] - 1,
            "present": int((status == 1).sum()),
            "weak": 0,
            "no_pick": int((status == 0).sum()),
            "source_case_id": case_id,
            "source_raw_sha256": actual_raw_hash,
            "source_label_sha256": actual_label_hash,
            "antenna_height_agl_m": row.get("antenna_height_agl_m", ""),
            "label_semantics": row.get("label_semantics", "visible_phase"),
            "target_presence": (
                declared_presence if declared_presence else ("true" if np.any(status == 1) else "false")
            ),
            "scene_family_id": row.get("scene_family_id", row.get("family", case_id)),
            "contract_id": row.get("contract_id", ""),
            "trace_count": raw.shape[1],
            "trace_spacing_m": row.get("trace_spacing_m", ""),
            "physical_span_m": row.get("physical_span_m", ""),
            "gprmax_version": row.get("gprmax_version", ""),
            "postprocess_validated": row.get("postprocess_validated", ""),
        })

    with (out_root / "window_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=index_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(index_rows)
    policy = {
        "dataset_id": out_root.name,
        "training_allowed": True,
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "formal_test_line": formal_test_line or None,
        "line9_conditioned": any(
            str(row.get("line9_conditioned", "")).lower() == "true" for row in eligible
        ),
        "exported_cases": len(eligible),
        "confirmed_negative_cases": sum(
            str(row.get("target_presence", "")).strip().lower() in {"false", "0", "no"}
            for row in eligible
        ),
        "label_semantics": "visible_phase",
        "supervision_state_runtime": "derived from status_code + label_weight + ignore_mask",
    }
    (out_root / "dataset_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--formal-test-line", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = export(Path(args.manifest), Path(args.out_root), args.formal_test_line, args.overwrite)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
