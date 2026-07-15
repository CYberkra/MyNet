#!/usr/bin/env python3
"""Build or verify a minimal, immutable gprMax solver-evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = (
    ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "02_released_solver_evidence"
).resolve()
ALLOWED_CLASSES = {
    "source_only",
    "rejected_evidence",
    "development_evidence",
    "training_candidate",
    "formal_release",
}
FORBIDDEN_SUFFIXES = {".log", ".vti", ".tmp", ".pid", ".zip"}
RAW_TRACE_OUTPUT = re.compile(r"(?<!merged)\d+\.out$", re.IGNORECASE)
DEVELOPMENT_REQUIRED_ROLES = {
    "one_trace_full_merged",
    "one_trace_control_merged",
    "one_trace_audit",
    "distributed_full_merged",
    "distributed_trace_contract",
    "distributed_morphology_audit",
    "human_review_decision",
    "human_preview",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_spec_sha256(spec: dict[str, object]) -> str:
    """Hash JSON semantics so cross-platform line endings cannot break a release."""
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_repo_path(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    if not _inside(candidate, ROOT):
        raise ValueError(f"path escapes repository: {value}")
    return candidate


def _validate_destination(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"invalid release destination: {value}")
    if candidate.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden release artifact: {value}")
    if RAW_TRACE_OUTPUT.search(candidate.name):
        raise ValueError(f"unmerged per-trace output is forbidden: {value}")
    return candidate


def _inspect_out(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if "rxs" not in handle:
            raise ValueError(f"gprMax output has no receiver group: {path}")
        arrays = 0
        samples = 0
        for receiver in handle["rxs"].values():
            for dataset in receiver.values():
                values = np.asarray(dataset)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite receiver data in {path}")
                arrays += 1
                samples += int(values.size)
        if arrays == 0:
            raise ValueError(f"gprMax output contains no receiver arrays: {path}")
        return {
            "gprmax_version": str(handle.attrs.get("gprMax", "unknown")),
            "iterations": int(handle.attrs.get("Iterations", 0)),
            "dt_s": float(handle.attrs.get("dt", 0.0)),
            "receiver_array_count": arrays,
            "receiver_sample_count": samples,
        }


def load_spec(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gprmax_release_spec_v1":
        raise ValueError("unsupported release spec schema")
    release_class = str(payload.get("release_class", ""))
    if release_class not in ALLOWED_CLASSES:
        raise ValueError(f"unsupported release class: {release_class}")
    if bool(payload.get("formal_training_allowed")) and release_class != "formal_release":
        raise ValueError("only formal_release may set formal_training_allowed=true")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release spec must contain artifacts")
    roles = {str(item.get("role", "")) for item in artifacts}
    if len(roles) != len(artifacts) or "" in roles:
        raise ValueError("artifact roles must be non-empty and unique")
    if release_class == "development_evidence":
        missing = DEVELOPMENT_REQUIRED_ROLES - roles
        if missing:
            raise ValueError(f"development evidence is missing roles: {sorted(missing)}")
        review = payload.get("human_review", {})
        if not isinstance(review, dict) or review.get("decision") != "accepted":
            raise ValueError("development evidence requires accepted human review")
    return payload


def _release_directory(spec: dict[str, object]) -> Path:
    output = _resolve_repo_path(str(spec["output_dir"]))
    if not _inside(output, RELEASE_ROOT):
        raise ValueError("output_dir must be inside 02_released_solver_evidence")
    return output


def package(spec_path: Path) -> Path:
    spec = load_spec(spec_path)
    output = _release_directory(spec)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"release directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for item in spec["artifacts"]:
        source = _resolve_repo_path(str(item["source"]))
        destination = _validate_destination(str(item["destination"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() in FORBIDDEN_SUFFIXES or RAW_TRACE_OUTPUT.search(source.name):
            raise ValueError(f"forbidden source artifact: {source}")
        target = (output / destination).resolve()
        if not _inside(target, output):
            raise ValueError(f"release destination escapes output: {destination}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        record: dict[str, object] = {
            "role": str(item["role"]),
            "path": target.relative_to(output).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
        if target.suffix.lower() == ".out":
            record["gprmax_output"] = _inspect_out(target)
        records.append(record)

    manifest = {
        "schema": "gprmax_release_manifest_v1",
        "release_id": spec["release_id"],
        "case_id": spec["case_id"],
        "release_class": spec["release_class"],
        "created_date": spec["created_date"],
        "formal_training_allowed": bool(spec["formal_training_allowed"]),
        "line9_conditioned": bool(spec.get("line9_conditioned", False)),
        "human_review": spec.get("human_review"),
        "source_spec": spec_path.resolve().relative_to(ROOT).as_posix(),
        "source_spec_sha256": canonical_spec_sha256(spec),
        "artifact_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "artifacts": records,
    }
    manifest_path = output / "RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "FILE_SHA256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "path", "bytes", "sha256"])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})
    return output


def verify(spec_path: Path) -> Path:
    spec = load_spec(spec_path)
    output = _release_directory(spec)
    manifest_path = output / "RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_spec_sha256") != canonical_spec_sha256(spec):
        raise ValueError("release spec hash no longer matches manifest")
    for record in manifest.get("artifacts", []):
        path = (output / str(record["path"])).resolve()
        if not _inside(path, output) or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            raise ValueError(f"release artifact hash mismatch: {path}")
        if path.suffix.lower() == ".out":
            _inspect_out(path)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    result = verify(spec_path) if args.verify_only else package(spec_path)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
