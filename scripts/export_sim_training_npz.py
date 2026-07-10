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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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


def _case_mask(case_dir: Path) -> Path:
    candidates = [
        case_dir / "label" / "y_soft_501x128.npy",
        case_dir / "labels" / "y_soft_501x128.npy",
        case_dir / "label" / "interface_mask_visible_phase_bscan.npy",
        case_dir / "labels" / "interface_mask_visible_phase_bscan.npy",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No visible-phase training mask in {case_dir}")


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
        "antenna_height_agl_m", "label_semantics",
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
        mask_path = _case_mask(case_dir)
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
        mask = np.clip(mask, 0.0, 1.0)
        trace_mass = mask.sum(axis=0)
        status = np.where(trace_mass > 1e-6, 1, 0).astype(np.int16)
        label_weight = np.where(status == 1, 1.0, 0.0).astype(np.float32)
        np.savez_compressed(
            windows / f"{case_id}.npz",
            x_raw=raw,
            y_mask=mask,
            status_code=status,
            label_weight=label_weight,
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
            "antenna_height_agl_m": "",
            "label_semantics": row.get("label_semantics", "visible_phase"),
        })

    with (out_root / "window_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=index_fields)
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
        "label_semantics": "visible_phase",
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
