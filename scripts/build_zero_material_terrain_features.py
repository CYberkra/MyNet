from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"
IN_DIR = DATA_ROOT / "terrain_features"
OUT_DIR = DATA_ROOT / "terrain_features_zero_material_v1"
MAIN_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
ALL_LINES = MAIN_LINES + ["LineX1"]
ADDED_FEATURES = ["surface_proxy_z", "surface_confidence_z"]


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


def moving_average_columns(arr, kernel=9):
    kernel = int(kernel)
    pad = kernel // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    csum = np.cumsum(padded, axis=0, dtype=np.float32)
    return (csum[kernel:] - csum[:-kernel]) / float(kernel)


def estimate_surface_proxy(raw):
    raw = raw.astype(np.float32)
    centered = raw - np.median(raw, axis=0, keepdims=True)
    env = moving_average_columns(np.abs(centered), kernel=11)
    h, w = env.shape
    lo = max(4, int(0.015 * h))
    hi = max(lo + 8, int(0.45 * h))
    search = env[lo:hi]
    idx = np.argmax(search, axis=0).astype(np.float32) + float(lo)
    peak = search[idx.astype(np.int64) - lo, np.arange(w)]
    floor = np.median(search, axis=0) + 1e-6
    confidence = np.log1p(peak / floor).astype(np.float32)
    return idx.astype(np.float32), confidence


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads((IN_DIR / "terrain_feature_manifest.json").read_text(encoding="utf-8"))
    raw_added = {}
    stats_pool = {"surface_proxy": [], "surface_confidence": []}

    for line in ALL_LINES:
        z = np.load(DATA_ROOT / "lines" / f"{line}.npz")
        surface, confidence = estimate_surface_proxy(z["raw_full_normalized"])
        raw_added[line] = {"surface_proxy": surface, "surface_confidence": confidence}
        if line in MAIN_LINES:
            stats_pool["surface_proxy"].append(surface)
            stats_pool["surface_confidence"].append(confidence)

    stats = {k: robust_stats(np.concatenate(v)) for k, v in stats_pool.items()}
    out_manifest = {
        "source_feature_dir": str(IN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "dataset": str(DATA_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "main_lines_for_normalization": MAIN_LINES,
        "base_feature_names": base_manifest["feature_names"],
        "added_feature_names": ADDED_FEATURES,
        "feature_names": list(base_manifest["feature_names"]) + ADDED_FEATURES,
        "normalization": {
            "surface_proxy": {"center": stats["surface_proxy"][0], "scale": stats["surface_proxy"][1]},
            "surface_confidence": {"center": stats["surface_confidence"][0], "scale": stats["surface_confidence"][1]},
        },
        "policy": (
            "Zero-material metadata features. Surface proxy is estimated from raw B-scan early-time "
            "envelope only; no geological labels or manual confirmation are used."
        ),
        "lines": {},
    }

    for line in ALL_LINES:
        base = np.load(IN_DIR / f"{line}_terrain_features.npz", allow_pickle=False)
        base_names = [str(v) for v in base["feature_names"]]
        added = raw_added[line]
        sp_center, sp_scale = stats["surface_proxy"]
        sc_center, sc_scale = stats["surface_confidence"]
        surface_z = ((added["surface_proxy"] - sp_center) / sp_scale).astype(np.float32)
        conf_z = ((added["surface_confidence"] - sc_center) / sc_scale).astype(np.float32)
        features = np.concatenate([base["features"].astype(np.float32), surface_z[None], conf_z[None]], axis=0)
        names = np.array(base_names + ADDED_FEATURES)
        np.savez_compressed(
            OUT_DIR / f"{line}_terrain_features.npz",
            features=features,
            feature_names=names,
            raw_surface_proxy_sample=added["surface_proxy"].astype(np.float32),
            raw_surface_confidence=added["surface_confidence"].astype(np.float32),
        )
        out_manifest["lines"][line] = {
            "width": int(features.shape[1]),
            "feature_file": str((OUT_DIR / f"{line}_terrain_features.npz").relative_to(ROOT)).replace("\\", "/"),
            "surface_proxy_sample_median": float(np.median(added["surface_proxy"])),
            "surface_confidence_median": float(np.median(added["surface_confidence"])),
        }

    (OUT_DIR / "terrain_feature_manifest.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR / "terrain_feature_manifest.json")


if __name__ == "__main__":
    main()
