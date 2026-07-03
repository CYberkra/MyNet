from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
REPORT_DIR = ROOT / "reports"
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]


def source_val_line(heldout):
    if heldout != "Line7":
        return "Line7"
    return "Line9"


def write_cfg(name, cfg, manifest):
    path = CONFIG_DIR / f"gpu_train_paper_v1_10d_{name}.json"
    cfg["version"] = f"paper_v1_10d_{name}"
    cfg["run_dir"] = f"outputs/run_gpu_paper_v1_10d_{name}"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.append(
        {
            "name": name,
            "heldout_line": cfg["test_lines"][0],
            "val_line": cfg["val_lines"][0],
            "train_lines": cfg["train_lines"],
            "config": str(path.relative_to(ROOT)).replace("\\", "/"),
            "run_dir": cfg["run_dir"],
        }
    )


def highconf_loss(loss):
    loss = dict(loss)
    loss["base_pixel_weight"] = 0.02
    loss["label_weight_power"] = 1.75
    loss["weak_label_weight_scale"] = 0.10
    loss["weak_presence_target"] = 0.35
    loss["presence_weight"] = 0.35
    return loss


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = json.loads((CONFIG_DIR / "gpu_train_paper_v1_9d_loo_Line9.json").read_text(encoding="utf-8"))
    base["epochs"] = 80
    base["max_preview_val"] = 0
    base["review_lines"] = []
    base["acceptance_note"] = "Strict source-validation full LOO: held-out line is not used for train/val/early stopping/calibration."
    manifest = []

    for heldout in LINES:
        val = source_val_line(heldout)
        train = [line for line in LINES if line not in {heldout, val}]
        common = dict(base)
        common["train_lines"] = train
        common["val_lines"] = [val]
        common["test_lines"] = [heldout]
        common["seed"] = 2400 + LINES.index(heldout)
        common["note"] = f"v1.10D source-validation baseline. Held out {heldout}; source validation {val}."
        write_cfg(f"{heldout}_sourceval_v19d_baseline", dict(common), manifest)

        highconf = dict(common)
        highconf["loss"] = highconf_loss(highconf["loss"])
        highconf["seed"] = 2500 + LINES.index(heldout)
        highconf["note"] = f"v1.10D source-validation highconf-only. Held out {heldout}; source validation {val}."
        write_cfg(f"{heldout}_sourceval_highconf_only", highconf, manifest)

    out = REPORT_DIR / "paper_v1_10d_sourceval_full_config_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
