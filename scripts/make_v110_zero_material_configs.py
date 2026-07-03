from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
REPORT_DIR = ROOT / "reports"
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
FEATURE_NAMES = [
    "relative_height_z",
    "ground_elevation_z",
    "altitude_z",
    "terrain_slope_z",
    "trace_position",
    "surface_proxy_z",
    "surface_confidence_z",
]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    base = json.loads((CONFIG_DIR / "gpu_train_paper_v1_9d_loo_Line9.json").read_text(encoding="utf-8"))
    for heldout in ["Line9", "Line6"]:
        cfg = dict(base)
        cfg["seed"] = 2100 + LINES.index(heldout)
        cfg["train_lines"] = [line for line in LINES if line != heldout]
        cfg["val_lines"] = [heldout]
        cfg["test_lines"] = [heldout]
        cfg["review_lines"] = []
        cfg["epochs"] = 80
        cfg["use_terrain_features"] = True
        cfg["terrain_feature_dir"] = "terrain_features_zero_material_v1"
        cfg["terrain_feature_names"] = FEATURE_NAMES
        cfg["terrain_metadata_dropout_prob"] = 0.20
        cfg["per_trace_robust_norm"] = True
        cfg["per_trace_robust_clip"] = 6.0
        cfg["version"] = f"paper_v1_10_zero_material_loo_{heldout}"
        cfg["run_dir"] = f"outputs/run_gpu_paper_v1_10_zero_material_loo_{heldout}"
        cfg["note"] = (
            "v1.10 zero-material leave-one-line-out: v1.9D backbone plus terrain/flight-height/"
            "surface-proxy conditioning, robust per-trace normalization, and uncertainty-aware label weighting."
        )
        cfg["acceptance_note"] = "No few-shot/manual held-out labels are used for adaptation; X1 remains excluded."
        loss = dict(cfg.get("loss", {}))
        loss["label_weight_power"] = 1.35
        loss["weak_label_weight_scale"] = 0.45
        loss["weak_presence_target"] = 0.55
        loss["centerline_weight"] = 0.12
        loss["continuity_weight"] = 0.045
        cfg["loss"] = loss
        aug = dict(cfg.get("augment", {}))
        aug["amp_scale_min"] = 0.82
        aug["amp_scale_max"] = 1.18
        aug["noise_std"] = 0.00015
        aug["trace_dropout_prob"] = 0.02
        cfg["augment"] = aug

        path = CONFIG_DIR / f"gpu_train_paper_v1_10_zero_material_loo_{heldout}.json"
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append(
            {
                "heldout_line": heldout,
                "config": str(path.relative_to(ROOT)).replace("\\", "/"),
                "run_dir": cfg["run_dir"],
                "feature_names": FEATURE_NAMES,
                "postprocess_plan": "mask+center fusion, breakable-DP, P=0.50 and P=0.20 audit",
            }
        )

    out = REPORT_DIR / "paper_v1_10_zero_material_config_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
