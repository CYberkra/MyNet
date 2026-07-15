#!/usr/bin/env python3
"""Append raw-signal surface proxy channels without split-level statistics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "measured" / "yingshan_v15"
IN_DIR = DATA_ROOT / "terrain_features"
OUT_DIR = DATA_ROOT / "terrain_features_zero_material_v1"
ALL_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1", "LineX1"]
ADDED_FEATURES = ["surface_proxy_z", "surface_confidence_z"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def moving_average_columns(arr: np.ndarray, kernel: int = 9) -> np.ndarray:
    kernel = int(kernel)
    pad = kernel // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    csum = np.vstack([np.zeros((1, padded.shape[1]), dtype=np.float32), np.cumsum(padded, axis=0)])
    return (csum[kernel:] - csum[:-kernel]) / float(kernel)


def estimate_surface_proxy(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = raw.astype(np.float32)
    centered = raw - np.median(raw, axis=0, keepdims=True)
    env = moving_average_columns(np.abs(centered), kernel=11)
    h, w = env.shape
    lo = max(4, int(0.015 * h))
    hi = max(lo + 8, int(0.45 * h))
    search = env[lo:hi]
    local_idx = np.argmax(search, axis=0).astype(np.int64)
    idx = local_idx.astype(np.float32) + float(lo)
    peak = search[local_idx, np.arange(w)]
    floor = np.median(search, axis=0) + 1e-6
    confidence = np.log1p(peak / floor).astype(np.float32)
    return idx, confidence


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads((IN_DIR / "terrain_feature_manifest.json").read_text(encoding="utf-8"))
    out_manifest = {
        "schema_version": "zero_material_features_v2",
        "source_feature_dir": str(IN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "dataset": str(DATA_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "base_feature_names": base_manifest["feature_names"],
        "added_feature_names": ADDED_FEATURES,
        "feature_names": list(base_manifest["feature_names"]) + ADDED_FEATURES,
        "normalization_policy": (
            "Surface sample is mapped to [-1,1] by the fixed time-axis length; confidence uses tanh(log-ratio/3). "
            "No cross-line or test-line statistics are used."
        ),
        "lines": {},
    }

    for line in ALL_LINES:
        line_path = DATA_ROOT / "lines" / f"{line}.npz"
        with np.load(line_path, allow_pickle=False) as line_data:
            raw = np.asarray(line_data["raw_full_normalized"], dtype=np.float32)
        surface, confidence = estimate_surface_proxy(raw)
        h = raw.shape[0]
        surface_z = (2.0 * surface / max(h - 1, 1) - 1.0).astype(np.float32)
        confidence_z = np.tanh(confidence / 3.0).astype(np.float32)

        base_path = IN_DIR / f"{line}_terrain_features.npz"
        with np.load(base_path, allow_pickle=False) as base:
            base_names = [str(v) for v in base["feature_names"]]
            features = np.concatenate(
                [base["features"].astype(np.float32), surface_z[None], confidence_z[None]], axis=0
            )
        names = np.asarray(base_names + ADDED_FEATURES)
        out_path = OUT_DIR / f"{line}_terrain_features.npz"
        np.savez_compressed(
            out_path,
            features=features,
            feature_names=names,
            raw_surface_proxy_sample=surface.astype(np.float32),
            raw_surface_confidence=confidence.astype(np.float32),
            source_line_npz_sha256=np.asarray(sha256(line_path)),
            source_base_feature_sha256=np.asarray(sha256(base_path)),
            normalization_policy=np.asarray("fixed_axis_and_nonlinear_confidence_no_split_statistics"),
        )
        out_manifest["lines"][line] = {
            "width": int(features.shape[1]),
            "feature_file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
            "feature_sha256": sha256(out_path),
            "surface_proxy_sample_median": float(np.median(surface)),
            "surface_confidence_median": float(np.median(confidence)),
        }

    manifest_path = OUT_DIR / "terrain_feature_manifest.json"
    manifest_path.write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
