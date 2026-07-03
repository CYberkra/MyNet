from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"
BY_TRACE = ROOT / "reports" / "full_label_reaudit_terrain_direction" / "by_trace"
OUT_DIR = DATA_ROOT / "terrain_features"
MAIN_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
FEATURE_NAMES = [
    "relative_height_z",
    "ground_elevation_z",
    "altitude_z",
    "terrain_slope_z",
    "trace_position",
]


def read_trace_csv(line):
    path = BY_TRACE / f"{line}_terrain_label_by_trace.csv"
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows.append(
            {
                "trace_idx": int(r["trace_idx"]),
                "relative_height_m": float(r["relative_height_m"]),
                "ground_elevation_est_m": float(r["ground_elevation_est_m"]),
                "altitude_m": float(r["altitude_m"]),
            }
        )
    rows.sort(key=lambda x: x["trace_idx"])
    return rows


def fill_to_width(rows, width, key):
    x = np.array([r["trace_idx"] for r in rows], dtype=np.float32)
    y = np.array([r[key] for r in rows], dtype=np.float32)
    keep = np.isfinite(y)
    if keep.sum() == 0:
        return np.zeros(width, dtype=np.float32)
    traces = np.arange(width, dtype=np.float32)
    return np.interp(traces, x[keep], y[keep]).astype(np.float32)


def robust_stats(values):
    values = np.asarray(values, dtype=np.float32)
    med = float(np.nanmedian(values))
    q25, q75 = np.nanpercentile(values, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return med, scale


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_by_line = {}
    stats_pool = {"relative_height_m": [], "ground_elevation_est_m": [], "altitude_m": [], "terrain_slope": []}
    for line in MAIN_LINES + ["LineX1"]:
        z = np.load(DATA_ROOT / "lines" / f"{line}.npz")
        width = int(z["raw_full_normalized"].shape[1])
        rows = read_trace_csv(line)
        rel = fill_to_width(rows, width, "relative_height_m")
        ground = fill_to_width(rows, width, "ground_elevation_est_m")
        alt = fill_to_width(rows, width, "altitude_m")
        slope = np.gradient(ground).astype(np.float32)
        raw_by_line[line] = {"relative_height_m": rel, "ground_elevation_est_m": ground, "altitude_m": alt, "terrain_slope": slope}
        if line in MAIN_LINES:
            stats_pool["relative_height_m"].append(rel)
            stats_pool["ground_elevation_est_m"].append(ground)
            stats_pool["altitude_m"].append(alt)
            stats_pool["terrain_slope"].append(slope)

    stats = {k: robust_stats(np.concatenate(v)) for k, v in stats_pool.items()}
    manifest = {
        "source": str(BY_TRACE.relative_to(ROOT)).replace("\\", "/"),
        "dataset": str(DATA_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "main_lines_for_normalization": MAIN_LINES,
        "feature_names": FEATURE_NAMES,
        "normalization": {k: {"center": c, "scale": s} for k, (c, s) in stats.items()},
        "policy": "Per-trace flight/terrain metadata channels; normalized on main measured lines only. LineX1 remains review-only.",
    }

    for line, raw in raw_by_line.items():
        width = raw["relative_height_m"].shape[0]
        pos = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        channels = []
        for key in ["relative_height_m", "ground_elevation_est_m", "altitude_m", "terrain_slope"]:
            center, scale = stats[key]
            channels.append(((raw[key] - center) / scale).astype(np.float32))
        channels.append(pos)
        features = np.stack(channels, axis=0).astype(np.float32)
        np.savez_compressed(
            OUT_DIR / f"{line}_terrain_features.npz",
            features=features,
            feature_names=np.array(FEATURE_NAMES),
            raw_relative_height_m=raw["relative_height_m"],
            raw_ground_elevation_est_m=raw["ground_elevation_est_m"],
            raw_altitude_m=raw["altitude_m"],
            raw_terrain_slope=raw["terrain_slope"],
        )
        manifest.setdefault("lines", {})[line] = {
            "width": int(width),
            "feature_file": str((OUT_DIR / f"{line}_terrain_features.npz").relative_to(ROOT)).replace("\\", "/"),
        }

    (OUT_DIR / "terrain_feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR / "terrain_feature_manifest.json")


if __name__ == "__main__":
    main()
