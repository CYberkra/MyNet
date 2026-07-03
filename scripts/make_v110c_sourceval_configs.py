from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
REPORT_DIR = ROOT / "reports"


def write(name, cfg, manifest):
    path = CONFIG_DIR / f"gpu_train_paper_v1_10c_{name}.json"
    cfg["version"] = f"paper_v1_10c_{name}"
    cfg["run_dir"] = f"outputs/run_gpu_paper_v1_10c_{name}"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.append({"variant": name, "config": str(path.relative_to(ROOT)).replace("\\", "/"), "run_dir": cfg["run_dir"]})


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads((CONFIG_DIR / "gpu_train_paper_v1_9d_loo_Line9.json").read_text(encoding="utf-8"))
    base["epochs"] = 80
    base["seed"] = 2319
    base["train_lines"] = ["Line3", "Line6", "LineL1"]
    base["val_lines"] = ["Line7"]
    base["test_lines"] = ["Line9"]
    base["review_lines"] = []
    base["max_preview_val"] = 0
    base["acceptance_note"] = "Strict zero-material Line9 test: Line9 is not used for training, validation, early stopping, or calibration."
    base["note"] = "v1.10C source-validation protocol. Hold out Line9 as test; use Line7 as source-domain validation."
    manifest = []

    robust = dict(base)
    robust["per_trace_robust_norm"] = True
    robust["per_trace_robust_clip"] = 6.0
    robust["note"] = base["note"] + " Variant: robust normalization only."
    write("Line9_sourceval_robust_norm_only", robust, manifest)

    uncertainty = dict(base)
    loss = dict(uncertainty["loss"])
    loss["label_weight_power"] = 1.35
    loss["weak_label_weight_scale"] = 0.45
    loss["weak_presence_target"] = 0.55
    uncertainty["loss"] = loss
    uncertainty["note"] = base["note"] + " Variant: uncertainty weighting only."
    write("Line9_sourceval_uncertainty_only", uncertainty, manifest)

    highconf = dict(base)
    loss = dict(highconf["loss"])
    loss["base_pixel_weight"] = 0.02
    loss["label_weight_power"] = 1.75
    loss["weak_label_weight_scale"] = 0.10
    loss["weak_presence_target"] = 0.35
    loss["presence_weight"] = 0.35
    highconf["loss"] = loss
    highconf["note"] = base["note"] + " Variant: high-confidence-biased labels."
    write("Line9_sourceval_highconf_only", highconf, manifest)

    out = REPORT_DIR / "paper_v1_10c_sourceval_config_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
