"""
PGDA-CSNet v1.12: Simulation Pretraining Pipeline

Usage:
1. Generate paired simulation data via SimLab (run_batch_safe_3060.py)
2. Convert to training windows: python scripts/convert_sim_to_training.py
3. Pretrain: python scripts/train_sim_pretrain.py configs/pretrain_sim_v1.json
4. Finetune on real data: use warm_start_from in main training config

Target: ~100+ simulation scenarios with paired raw/basal/background variants.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Pretrain config: train on simulation data, no real data
pretrain_cfg = {
    "data_root": "data/simulation_pretrain_v1",
    "height_resize": 512,
    "width_resize": 256,
    "batch_size": 2,
    "epochs": 50,
    "lr": 0.00055,
    "base_ch": 20,
    "model_dropout": 0.06,
    "num_workers": 0,
    "seed": 42,
    "train_lines": ["sim_line9_zk08", "sim_line3_zk07_zk08", "sim_zk09_line6", "sim_l1_12_20"],
    "train_trace_ranges": {},
    "val_lines": [],
    "test_lines": ["sim_line9_zk08"],
    "test_trace_ranges": {},
    "review_lines": [],
    "loss": {
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
        "centerline_weight": 0.1,
        "continuity_weight": 0.035,
        "center_valid_min_sum": 0.001,
        "spectral_consistency_weight": 0.05,
    },
    "augment": {"enabled": True, "amp_scale_min": 0.9, "amp_scale_max": 1.1, "noise_std": 0.0001, "trace_dropout_prob": 0.02, "horizontal_flip_prob": 0.35},
    "deterministic": False,
    "input_log_scale": 0.001,
    "no_pick_window_repeats": 2,
    "acceptance_note": "Simulation pretrain only. No real data used.",
    "model_arch": "v1_9d_mambavision_hybrid",
    "ssm_kernel": 31,
    "attention_heads": 4,
    "max_preview_val": 0,
    "run_dir": "outputs/pretrain_sim_v1",
    "version": "pretrain_sim_v1",
    "note": "Simulation pretrain on paired gprMax outputs. Will be used for warm-start.",
}

# Finetune config: load sim-pretrained weights, train on real data
finetune_cfg = {
    "data_root": "data_corrected_v1_4_terrain_direction",
    "paper_split_file": "configs/paper_splits_v1_6.json",
    "height_resize": 512,
    "width_resize": 256,
    "batch_size": 2,
    "epochs": 60,
    "lr": 0.0002,  # Lower LR for finetuning
    "base_ch": 20,
    "model_dropout": 0.06,
    "num_workers": 0,
    "seed": 1902,
    "train_lines": ["Line3", "Line6", "LineL1", "Line9"],
    "val_lines": ["Line7"],
    "test_lines": ["Line9"],
    "test_trace_ranges": {"Line9": [1664, 2377]},
    "review_lines": ["LineX1"],
    "loss": {
        "core_weight": 0.55, "outside_weight": 0.36, "dice_weight": 0.85,
        "presence_weight": 0.42, "presence_negative_weight": 2.8,
        "core_threshold": 0.55, "outside_margin": 0.05,
        "weak_presence_target": 0.65, "positive_pixel_boost": 8.0,
        "hard_negative_weight": 0.24, "hard_negative_topk_frac": 0.02,
        "centerline_weight": 0.1, "continuity_weight": 0.035,
        "center_valid_min_sum": 0.001,
        "spectral_consistency_weight": 0.05,
    },
    "augment": {"enabled": True, "amp_scale_min": 0.88, "amp_scale_max": 1.12, "noise_std": 0.0001, "trace_dropout_prob": 0.015, "horizontal_flip_prob": 0.35},
    "deterministic": False,
    "input_log_scale": 0.001,
    "no_pick_window_repeats": 2,
    "warm_start_from": "outputs/pretrain_sim_v1/checkpoint_final.pt",
    "acceptance_note": "Finetune from sim-pretrained on real data.",
    "model_arch": "v1_9d_mambavision_hybrid",
    "ssm_kernel": 31,
    "attention_heads": 4,
    "max_preview_val": 0,
    "run_dir": "outputs/finetune_sim_to_real_v1",
    "version": "finetune_sim_to_real_v1",
    "note": "Finetune from simulation pretrain to real data. Seed-1902.",
}

out_dir = ROOT / "configs"
(out_dir / "pretrain_sim_v1.json").write_text(json.dumps(pretrain_cfg, indent=2), encoding="utf-8")
(out_dir / "finetune_sim_to_real_v1.json").write_text(json.dumps(finetune_cfg, indent=2), encoding="utf-8")

print("Created:")
print(f"  {out_dir / 'pretrain_sim_v1.json'}")
print(f"  {out_dir / 'finetune_sim_to_real_v1.json'}")
