from pathlib import Path
import copy
import json


ROOT = Path(__file__).resolve().parents[1]
TARGET_LINES = ["Line9", "Line6"]
K_VALUES = [2, 4, 8]


def main():
    base = json.loads((ROOT / "configs" / "gpu_train_paper_v1_9d_mambavision_hybrid_final_seed1902_line9holdout.json").read_text(encoding="utf-8"))
    old_manifest = json.loads((ROOT / "reports" / "paper_v1_7a_fewshot_config_manifest.json").read_text(encoding="utf-8"))
    anchor_map = {(item["line"], int(item["k"])): item["anchors"] for item in old_manifest}
    measured_lines = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
    manifest = []
    for line in TARGET_LINES:
        for k in K_VALUES:
            anchors = anchor_map[(line, k)]
            cfg = copy.deepcopy(base)
            cfg["fewshot_target_line"] = line
            cfg["fewshot_k"] = k
            cfg["fewshot_anchor_sample_ids"] = anchors
            cfg["seed"] = 2900 + (10 if line == "Line9" else 20) + k
            cfg["epochs"] = 60
            cfg["train_lines"] = measured_lines
            cfg["train_trace_ranges"] = {}
            cfg["train_line_sample_ids"] = {line: anchors}
            cfg["val_lines"] = [line]
            cfg["exclude_val_sample_ids"] = anchors
            cfg["test_lines"] = [line]
            cfg["test_trace_ranges"] = {}
            cfg["review_lines"] = []
            cfg["run_dir"] = f"outputs/run_gpu_paper_v1_9d_fewshot_{line}_k{k}"
            cfg["version"] = f"paper_v1_9d_fewshot_{line}_k{k}"
            cfg["note"] = f"v1.9D few-shot calibration: target {line}, k={k} anchor windows."
            cfg["acceptance_note"] = "Few-shot target-line calibration after LOO failure; LineX1 excluded."
            out = ROOT / "configs" / f"gpu_train_paper_v1_9d_fewshot_{line}_k{k}.json"
            out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest.append(
                {
                    "line": line,
                    "k": k,
                    "config": str(out.relative_to(ROOT)),
                    "run_dir": cfg["run_dir"],
                    "anchors": anchors,
                    "seed": cfg["seed"],
                }
            )
    mpath = ROOT / "reports" / "paper_v1_9d_fewshot_config_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(mpath)
    for item in manifest:
        print(item["config"])


if __name__ == "__main__":
    main()
