#!/usr/bin/env python3
"""Copy a solver-validated native case into the compact Git-sync release area."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "02_released_canonical"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _required_trace_contracts(target_presence: bool) -> tuple[str, ...]:
    stems = ["full_scene", "air_reference"]
    if target_presence:
        stems.insert(1, "no_basal_contrast_control")
    return tuple(f"{stem}_trace_contract.json" for stem in stems)


def promote(case_dir: Path, out_root: Path, *, replace: bool = False) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    manifest_path = case_dir / "scene_manifest.json"
    postprocess_path = case_dir / "postprocess_validation.json"
    manifest = _load_json(manifest_path)
    postprocess = _load_json(postprocess_path)
    case_id = str(manifest["case_id"])
    target_presence = bool(manifest["target_presence"])
    if manifest.get("formal_training_allowed") is not False:
        raise ValueError("only blocked pre-promotion cases may enter canonical release sync")
    if postprocess.get("ok") is not True or postprocess.get("postprocess_validated") is not True:
        raise ValueError("postprocess validation must be successful before release sync")
    if tuple(postprocess.get("output_shape_canonical", ())) != (501, 256):
        raise ValueError("canonical release must have exactly shape 501x256")

    labels = case_dir / "labels"
    if not labels.is_dir():
        raise FileNotFoundError(labels)
    required_labels = ["full_scene_501x256.npy", "air_reference_501x256.npy"]
    if target_presence:
        required_labels.extend(
            [
                "no_basal_contrast_501x256.npy",
                "contrast_response_501x256.npy",
                "target_mask_visible_phase_501x256.npy",
                "visible_phase_support_ratio.npy",
            ]
        )
    else:
        required_labels.append("target_mask_confirmed_negative_501x256.npy")
    missing = [name for name in required_labels if not (labels / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical labels: {missing}")
    contracts = [case_dir / "run_logs" / name for name in _required_trace_contracts(target_presence)]
    missing_contracts = [str(path) for path in contracts if not path.is_file()]
    if missing_contracts:
        raise FileNotFoundError(f"missing pre-merge trace contracts: {missing_contracts}")

    destination = out_root.resolve() / case_id
    if destination.exists():
        if not replace:
            raise FileExistsError(f"release already exists: {destination}")
        shutil.rmtree(destination)
    (destination / "labels").mkdir(parents=True)
    (destination / "trace_contracts").mkdir()
    shutil.copy2(manifest_path, destination / "scene_manifest.json")
    shutil.copy2(postprocess_path, destination / "postprocess_validation.json")
    checksum_path = case_dir / "FILE_SHA256.csv"
    if checksum_path.is_file():
        shutil.copy2(checksum_path, destination / "source_FILE_SHA256.csv")
    for source in sorted(labels.glob("*.npy")):
        shutil.copy2(source, destination / "labels" / source.name)
    for source in contracts:
        shutil.copy2(source, destination / "trace_contracts" / source.name)

    files = sorted(path for path in destination.rglob("*") if path.is_file())
    release = {
        "release_id": f"native_256_canonical::{case_id}",
        "case_id": case_id,
        "release_state": "solver_validated_not_formal",
        "formal_training_allowed": False,
        "target_presence": target_presence,
        "source_case_dir": str(case_dir),
        "source_manifest_sha256": sha256(manifest_path),
        "source_postprocess_sha256": sha256(postprocess_path),
        "canonical_shape": [501, 256],
        "raw_solver_outputs_included": False,
        "geometry_views_included": False,
        "files": [
            {
                "relative_path": path.relative_to(destination).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
        "promotion_blockers": [
            "Human physical and visible-phase review is still required.",
            "Formal simulation train approval remains a separate dataset-contract action."
        ],
    }
    (destination / "release_manifest.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    report = promote(args.case_dir, args.out_root, replace=args.replace)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
