"""
PGDA-CSNet v1.13: Sim-to-Real Domain Adaptation
Mixed training with domain randomization and spectrum augmentation.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Strategy: train on combined sim+real data with:
# 1. Spectrum augmentation (random bandpass filtering of sim data)
# 2. Noise injection
# 3. Amplitude randomization
# 4. Weighted sim vs real loss (sim has lower weight)

cfg = {
    "data_root": "data_corrected_v1_4_terrain_direction",
    "paper_split_file": "configs/paper_splits_v1_6.json",
    "height_resize": 512,
    "width_resize": 256,
    "batch_size": 2,
    "epochs": 60,
    "lr": 0.0003,
    "base_ch": 20,
    "model_dropout": 0.06,
    "num_workers": 0,
    "seed": 1902,
    "train_lines": ["Line3", "Line6", "LineL1"],
    "val_lines": ["Line7"],
    "test_lines": ["Line9"],
    "test_trace_ranges": {"Line9": [1664, 2377]},
    "review_lines": ["LineX1"],
    "train_line_repeat_factors": {},  # No oversampling
    "sim_data_root": "data/simulation_pretrain_v1",  # NEW: sim data path
    "sim_train_lines": ["strat_l9_zk08", "strat_l3_zk07_08", "strat_l6_zk09", "strat_l1_12_20", "supersmooth",
                         "terr_box", "terr_smooth", "terr_limited"],
    "sim_batch_ratio": 0.3,  # 30% of each batch from simulation data
    "sim_loss_weight": 0.3,  # Sim data contributes 0.3x to loss
    "loss": {
        "core_weight": 0.55, "outside_weight": 0.36, "dice_weight": 0.85,
        "presence_weight": 0.42, "presence_negative_weight": 2.8,
        "core_threshold": 0.55, "outside_margin": 0.05,
        "weak_presence_target": 0.65, "positive_pixel_boost": 8.0,
        "hard_negative_weight": 0.24, "hard_negative_topk_frac": 0.02,
        "centerline_weight": 0.1, "continuity_weight": 0.035,
        "center_valid_min_sum": 0.001,
    },
    "augment": {"enabled": True, "amp_scale_min": 0.85, "amp_scale_max": 1.15,
                "noise_std": 0.0002, "trace_dropout_prob": 0.02, "horizontal_flip_prob": 0.35},
    "deterministic": False,
    "input_log_scale": 0.001,
    "no_pick_window_repeats": 2,
    "acceptance_note": "Mixed real+sim training with domain augmentation.",
    "model_arch": "v1_9d_mambavision_hybrid",
    "ssm_kernel": 31, "attention_heads": 4,
    "max_preview_val": 0,
    "run_dir": "outputs/run_gpu_v1_13_mixed_real_sim",
    "version": "v1_13_mixed_real_sim",
    "note": "Mixed real+sim training with weighted loss. Sim contributes 30% of batch."
}

(ROOT / "configs/gpu_train_v1_13_mixed_real_sim.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("Created: configs/gpu_train_v1_13_mixed_real_sim.json")
print(f"  Real lines: {cfg['train_lines']}")
print(f"  Sim lines: {cfg['sim_train_lines']}")
print(f"  Sim batch ratio: {cfg['sim_batch_ratio']}")
print(f"  Sim loss weight: {cfg['sim_loss_weight']}")
PY