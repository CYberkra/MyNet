from pathlib import Path
import copy
import json


ROOT = Path(__file__).resolve().parents[1]
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]


def main():
    base_path = ROOT / "configs" / "gpu_train_paper_v1_9d_mambavision_hybrid_final_seed1902_line9holdout.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    manifest = []
    for idx, heldout in enumerate(LINES, start=1):
        cfg = copy.deepcopy(base)
        cfg["seed"] = 1990 + idx
        cfg["train_lines"] = [line for line in LINES if line != heldout]
        cfg["train_trace_ranges"] = {}
        cfg["val_lines"] = [heldout]
        cfg["test_lines"] = [heldout]
        cfg["test_trace_ranges"] = {}
        cfg["review_lines"] = []
        cfg["run_dir"] = f"outputs/run_gpu_paper_v1_9d_loo_{heldout}"
        cfg["version"] = f"paper_v1_9d_loo_{heldout}"
        cfg["note"] = f"v1.9D leave-one-line-out fold; held-out line is {heldout}."
        cfg["acceptance_note"] = "LineX1 excluded; LOO ranks only measured valid lines."
        out = ROOT / "configs" / f"gpu_train_paper_v1_9d_loo_{heldout}.json"
        out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "heldout_line": heldout,
                "config": str(out.relative_to(ROOT)),
                "run_dir": cfg["run_dir"],
                "seed": cfg["seed"],
                "train_lines": cfg["train_lines"],
            }
        )
    mpath = ROOT / "reports" / "paper_v1_9d_loo_config_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(mpath)
    for item in manifest:
        print(item["config"])


if __name__ == "__main__":
    main()
