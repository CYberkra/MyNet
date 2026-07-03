from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
REPORT_DIR = ROOT / "reports"

FEATURES_5 = [
    "relative_height_z",
    "ground_elevation_z",
    "altitude_z",
    "terrain_slope_z",
    "trace_position",
]
FEATURES_7 = FEATURES_5 + ["surface_proxy_z", "surface_confidence_z"]


def write_cfg(name, cfg, manifest):
    path = CONFIG_DIR / f"gpu_train_paper_v1_10b_{name}.json"
    cfg["version"] = f"paper_v1_10b_{name}"
    cfg["run_dir"] = f"outputs/run_gpu_paper_v1_10b_{name}"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.append(
        {
            "variant": name,
            "config": str(path.relative_to(ROOT)).replace("\\", "/"),
            "run_dir": cfg["run_dir"],
            "note": cfg["note"],
        }
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads((CONFIG_DIR / "gpu_train_paper_v1_9d_loo_Line9.json").read_text(encoding="utf-8"))
    base["epochs"] = 60
    base["seed"] = 2219
    base["max_preview_val"] = 0
    base["acceptance_note"] = "Line9 LOO ablation; no target-line labels or few-shot calibration."
    base["note"] = "v1.10B ablation base: v1.9D Line9 LOO, 60 epochs."
    manifest = []

    terrain = dict(base)
    terrain["use_terrain_features"] = True
    terrain["terrain_feature_dir"] = "terrain_features"
    terrain["terrain_feature_names"] = FEATURES_5
    terrain["terrain_metadata_dropout_prob"] = 0.20
    terrain["note"] = "Terrain/flight-height conditioning only; no robust normalization or uncertainty changes."
    write_cfg("Line9_terrain_only", terrain, manifest)

    robust = dict(base)
    robust["per_trace_robust_norm"] = True
    robust["per_trace_robust_clip"] = 6.0
    robust["note"] = "Per-trace robust normalization only; no metadata or uncertainty changes."
    write_cfg("Line9_robust_norm_only", robust, manifest)

    uncertain = dict(base)
    loss = dict(uncertain["loss"])
    loss["label_weight_power"] = 1.35
    loss["weak_label_weight_scale"] = 0.45
    loss["weak_presence_target"] = 0.55
    uncertain["loss"] = loss
    uncertain["note"] = "Uncertainty-aware label weighting only; no metadata or robust normalization."
    write_cfg("Line9_uncertainty_only", uncertain, manifest)

    highconf = dict(base)
    loss = dict(highconf["loss"])
    loss["base_pixel_weight"] = 0.02
    loss["label_weight_power"] = 1.75
    loss["weak_label_weight_scale"] = 0.10
    loss["weak_presence_target"] = 0.35
    loss["presence_weight"] = 0.35
    highconf["loss"] = loss
    highconf["note"] = "High-confidence-biased training; weak labels are strongly downweighted."
    write_cfg("Line9_highconf_only", highconf, manifest)

    terrain_no_surface = dict(base)
    terrain_no_surface["use_terrain_features"] = True
    terrain_no_surface["terrain_feature_dir"] = "terrain_features"
    terrain_no_surface["terrain_feature_names"] = FEATURES_5
    terrain_no_surface["terrain_metadata_dropout_prob"] = 0.20
    terrain_no_surface["per_trace_robust_norm"] = True
    terrain_no_surface["per_trace_robust_clip"] = 6.0
    loss = dict(terrain_no_surface["loss"])
    loss["label_weight_power"] = 1.35
    loss["weak_label_weight_scale"] = 0.45
    loss["weak_presence_target"] = 0.55
    terrain_no_surface["loss"] = loss
    terrain_no_surface["note"] = "Combined v1.10B without surface proxy: 5 metadata features + robust norm + uncertainty."
    write_cfg("Line9_combined_no_surface", terrain_no_surface, manifest)

    out = REPORT_DIR / "paper_v1_10b_ablation_config_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
