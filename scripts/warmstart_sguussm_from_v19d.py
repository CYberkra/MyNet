"""
Warm-start SG-USSM from v1.9D checkpoint.
Loads all matching weights; SGM module gets random init (gamma=0 = identity).
"""
import sys, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pgdacsnet.model_raw_unet import build_model

v19d_ckpt = ROOT / "outputs/run_gpu_paper_v1_9d_mambavision_hybrid_final_seed1902_line9holdout/checkpoint_final.pt"
sguussm_ckpt = ROOT / "outputs/run_gpu_v1_11_sguussm_seed1902_line9holdout/checkpoint_final.pt"
out_ckpt = ROOT / "outputs/run_gpu_v1_11_sguussm_warmstart_from_v19d/checkpoint_last.pt"

# Load v1.9D weights
v19d_state = torch.load(v19d_ckpt, map_location="cpu")["model"]

# Create SG-USSM model
cfg = {"base_ch": 20, "model_dropout": 0.06, "ssm_kernel": 31, "attention_heads": 4, "model_arch": "v1_11_sguussm"}
model = build_model(cfg)
sg_state = model.state_dict()

# Copy matching weights
matched, skipped = [], []
for k, v in sg_state.items():
    if k in v19d_state and v19d_state[k].shape == v.shape:
        sg_state[k] = v19d_state[k]
        matched.append(k)
    else:
        skipped.append(k)

model.load_state_dict(sg_state)

# Save warm-start checkpoint
out_ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": 0, "note": "warm-start from v1.9D, SGM random init"}, out_ckpt)

print(f"v1.9D params loaded: {len(matched)}")
print(f"New/random init:     {len(skipped)} (SGM modules)")
print(f"Skipped: {skipped}")
print(f"Saved: {out_ckpt}")
print(f"SGM gamma: {model.sgm.gamma.data.mean().item():.6f}")
