from pathlib import Path
import copy
import json


ROOT = Path(__file__).resolve().parents[1]
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
SEEDS = [1901, 1902, 1903]
BASE_CONFIG = "configs/gpu_train_v3_pilot_mixed.json"
MANIFEST_PATH = ROOT / "reports" / "v3_pilot_mixed_loo_manifest.json"


def make_fold_configs():
    base_path = ROOT / BASE_CONFIG
    base = json.loads(base_path.read_text(encoding="utf-8"))
    manifest = []

    for heldout in LINES:
        train_lines = [line for line in LINES if line != heldout]

        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg["train_lines"] = train_lines
            cfg["train_trace_ranges"] = {}
            cfg["val_lines"] = [heldout]
            cfg["test_lines"] = [heldout]
            cfg["test_trace_ranges"] = {}
            cfg["review_lines"] = ["LineX1"]
            cfg["run_dir"] = f"outputs/run_gpu_v3_pilot_mixed_loo_{heldout}_seed{seed}"
            cfg["version"] = f"v3_pilot_mixed_loo_{heldout}_seed{seed}"
            cfg["note"] = (
                f"v3 pilot mixed LOLO-CV fold; held-out line is {heldout}. "
                f"Train on {', '.join(train_lines)} + 20 Pilot-Mini sim cases. "
                f"Seed {seed}."
            )
            cfg["acceptance_note"] = (
                "LineX1 is review-only. "
                "Full leave-one-line-out cross-validation fold."
            )

            out_name = f"gpu_train_v3_pilot_mixed_loo_{heldout}_seed{seed}.json"
            out_path = ROOT / "configs" / out_name
            out_path.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            manifest.append({
                "heldout_line": heldout,
                "seed": seed,
                "config": str(out_path.relative_to(ROOT)),
                "run_dir": cfg["run_dir"],
                "train_lines": cfg["train_lines"],
            })

            print(f"  {out_name}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Total configs: {len(manifest)}")


if __name__ == "__main__":
    make_fold_configs()
