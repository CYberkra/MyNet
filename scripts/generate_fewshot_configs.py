from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"


BASE_LOSS = {
    "core_weight": 0.55,
    "outside_weight": 0.36,
    "dice_weight": 0.85,
    "presence_weight": 0.42,
    "presence_negative_weight": 2.8,
    "core_threshold": 0.55,
    "outside_margin": 0.05,
    "weak_presence_target": 0.65,
    "positive_pixel_boost": 8.0,
    "hard_negative_weight": 0.24,
    "hard_negative_topk_frac": 0.02,
    "centerline_weight": 0.10,
    "continuity_weight": 0.035,
    "center_valid_min_sum": 0.001,
}


BASE_AUG = {
    "enabled": True,
    "amp_scale_min": 0.88,
    "amp_scale_max": 1.12,
    "noise_std": 0.0001,
    "trace_dropout_prob": 0.015,
    "horizontal_flip_prob": 0.35,
}


def rows_for_line(line):
    rows = [
        r for r in csv.DictReader(open(DATA_ROOT / "window_index.csv", encoding="utf-8"))
        if r["line"] == line
    ]
    rows.sort(key=lambda r: (int(r["start"]), int(r["end"])))
    return rows


def evenly_spaced_ids(line, k):
    rows = rows_for_line(line)
    if k >= len(rows):
        return [r["sample_id"] for r in rows]
    if k == 1:
        return [rows[len(rows) // 2]["sample_id"]]
    idxs = [round(i * (len(rows) - 1) / (k - 1)) for i in range(k)]
    return [rows[i]["sample_id"] for i in idxs]


def config_for(line, k):
    valid_lines = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
    train_lines = [x for x in valid_lines if x != line] + [line]
    anchors = evenly_spaced_ids(line, k)
    return {
        "data_root": "data_corrected_v1_4_terrain_direction",
        "paper_split_file": "configs/paper_splits_v1_6.json",
        "fewshot_target_line": line,
        "fewshot_k": k,
        "fewshot_anchor_sample_ids": anchors,
        "height_resize": 512,
        "width_resize": 256,
        "batch_size": 2,
        "epochs": 60,
        "lr": 0.00055,
        "base_ch": 20,
        "model_arch": "raw_convnext_unet_v17a",
        "model_dropout": 0.06,
        "num_workers": 0,
        "seed": 2400 + k + sum(ord(c) for c in line),
        "train_lines": train_lines,
        "train_trace_ranges": {"Line9": [0, 1407]} if line != "Line9" else {},
        "train_line_sample_ids": {line: anchors},
        "val_lines": [line],
        "exclude_val_sample_ids": anchors,
        "test_lines": [line],
        "review_lines": ["LineX1"],
        "max_preview_val": 0,
        "run_dir": f"outputs/run_gpu_paper_v1_7a_fewshot_{line}_k{k}",
        "loss": BASE_LOSS,
        "augment": BASE_AUG,
        "deterministic": False,
        "input_log_scale": 0.001,
        "no_pick_window_repeats": 2,
        "version": f"paper_v1_7a_fewshot_{line}_k{k}",
    }


def main():
    out_dir = ROOT / "configs"
    manifest = []
    for line in ["Line9", "Line6"]:
        for k in [2, 4, 8]:
            cfg = config_for(line, k)
            out = out_dir / f"gpu_train_paper_v1_7a_fewshot_{line}_k{k}.json"
            out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append({
                "line": line,
                "k": k,
                "config": str(out.relative_to(ROOT)).replace("\\", "/"),
                "anchors": cfg["fewshot_anchor_sample_ids"],
            })
            print(out)
    manifest_path = ROOT / "reports" / "paper_v1_7a_fewshot_config_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
