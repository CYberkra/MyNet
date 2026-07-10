#!/usr/bin/env python3
"""Build leakage-safe terrain/flight metadata channels from canonical line NPZ files.

The canonical line archives are generated from the original YingShan CSV schema:
longitude, latitude, ground elevation, radar amplitude, flight height AGL.
No manual labels or test-line distribution statistics are used here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"
OUT_DIR = DATA_ROOT / "terrain_features"
ALL_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1", "LineX1"]
FEATURE_NAMES = [
    "relative_height_z",
    "ground_elevation_z",
    "altitude_z",
    "terrain_slope_z",
    "trace_position",
]

# Fixed physical scales avoid any train/test distribution leakage.
# Site ground range 350-610 m comes from the project survey description.
PHYSICAL_NORMALIZATION = {
    "relative_height_m": {"center": 11.0, "scale": 9.0, "basis": "planned flight range 2-20 m"},
    "ground_elevation_m": {"center": 480.0, "scale": 130.0, "basis": "site elevation range 350-610 m"},
    "antenna_elevation_m": {"center": 491.0, "scale": 139.0, "basis": "ground center + flight center"},
    "terrain_slope": {"center": 0.0, "scale": 0.5, "basis": "fixed dimensionless slope scale"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def z_fixed(values: np.ndarray, key: str) -> np.ndarray:
    spec = PHYSICAL_NORMALIZATION[key]
    return ((values.astype(np.float32) - float(spec["center"])) / float(spec["scale"])).astype(np.float32)


def terrain_slope(ground: np.ndarray, distance: np.ndarray) -> np.ndarray:
    ground = ground.astype(np.float64)
    distance = distance.astype(np.float64)
    if ground.size < 2:
        return np.zeros_like(ground, dtype=np.float32)
    dg = np.gradient(ground)
    dx = np.gradient(distance)
    slope = dg / np.maximum(np.abs(dx), 1e-3)
    return np.clip(slope, -3.0, 3.0).astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "terrain_features_original_csv_v2",
        "dataset": str(DATA_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "source": "canonical line NPZ metadata imported directly from the original YingShan CSV archive",
        "feature_names": FEATURE_NAMES,
        "normalization": PHYSICAL_NORMALIZATION,
        "normalization_policy": (
            "Fixed physical scales only. No statistics from Line9 or any validation/test line are used."
        ),
        "lines": {},
    }

    for line in ALL_LINES:
        line_path = DATA_ROOT / "lines" / f"{line}.npz"
        with np.load(line_path, allow_pickle=False) as data:
            required = {
                "flight_height_agl_m", "ground_elevation_m", "antenna_elevation_m",
                "gnss_cumulative_distance_m", "source_csv_sha256",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise RuntimeError(f"{line_path} lacks canonical metadata: {missing}")
            flight = np.asarray(data["flight_height_agl_m"], dtype=np.float32)
            ground = np.asarray(data["ground_elevation_m"], dtype=np.float32)
            antenna = np.asarray(data["antenna_elevation_m"], dtype=np.float32)
            distance = np.asarray(data["gnss_cumulative_distance_m"], dtype=np.float64)
            source_csv_sha = str(np.asarray(data["source_csv_sha256"]).item())

        width = flight.size
        if any(values.shape != (width,) for values in (ground, antenna, distance)):
            raise RuntimeError(f"{line}: canonical spatial vectors have inconsistent lengths")
        slope = terrain_slope(ground, distance)
        if distance[-1] > 0:
            position = (2.0 * distance / distance[-1] - 1.0).astype(np.float32)
        else:
            position = np.linspace(-1.0, 1.0, width, dtype=np.float32)

        features = np.stack(
            [
                z_fixed(flight, "relative_height_m"),
                z_fixed(ground, "ground_elevation_m"),
                z_fixed(antenna, "antenna_elevation_m"),
                z_fixed(slope, "terrain_slope"),
                position,
            ],
            axis=0,
        ).astype(np.float32)
        out_path = OUT_DIR / f"{line}_terrain_features.npz"
        np.savez_compressed(
            out_path,
            features=features,
            feature_names=np.asarray(FEATURE_NAMES),
            raw_flight_height_agl_m=flight,
            raw_ground_elevation_m=ground,
            raw_antenna_elevation_m=antenna,
            raw_terrain_slope=slope,
            raw_gnss_cumulative_distance_m=distance,
            source_line_npz_sha256=np.asarray(sha256(line_path)),
            source_csv_sha256=np.asarray(source_csv_sha),
            normalization_policy=np.asarray("fixed_physical_scales_no_split_statistics"),
        )
        manifest["lines"][line] = {
            "width": int(width),
            "feature_file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
            "feature_sha256": sha256(out_path),
            "source_line_npz_sha256": sha256(line_path),
            "source_csv_sha256": source_csv_sha,
            "flight_height_range_m": [float(flight.min()), float(flight.max())],
            "ground_elevation_range_m": [float(ground.min()), float(ground.max())],
            "gnss_distance_m": float(distance[-1]),
        }

    manifest_path = OUT_DIR / "terrain_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
