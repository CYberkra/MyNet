"""PGDA-CSNet 模型预览图生成脚本。
从 3 个 seed checkpoint 加载模型，对 Line9 测试窗口做推理，
输出 raw B-scan / 预测 clean / 预测杂波 / 界面掩码 对比图。
"""
"""PGDA-CSNet 模型预览图生成脚本。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.transform import resize

# === 路径设置 ===
WORKSPACE = Path(
    "D:/Claude/PGDA-CSNet/workspace/transfer_20260627_142748/"
    "PGDA-CSNet_transfer_bundle_20260627_142748/"
    "PGDA_CSNet_v0_9_6_SEARCH_WINDOW_GUARD"
)
DATA_ROOT = WORKSPACE / "data_corrected_v1_4_terrain_direction"
OUT_DIR = Path("C:/Users/17844/Desktop")

sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE))

from pgdacsnet.model_raw_unet import build_model, compress_raw

# === 模型加载 ===
def load_model(run_dir, device='cpu'):
    cfg = json.load(open(run_dir / 'used_config.json'))
    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(run_dir / 'checkpoint_best.pt', map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    # filter out 'module.' prefix
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model, cfg

def preprocess_window(z, cfg):
    """Match DS.__getitem__ preprocessing."""
    x = torch.from_numpy(z['x_raw'][None]).float()
    H, W = cfg['height_resize'], cfg['width_resize']
    x = torch.nn.functional.interpolate(x[None], (H, W), mode='bilinear', align_corners=False)[0]
    x = compress_raw(x, cfg.get('input_log_scale', 1e-3))
    return x

def inference(model, x):
    """Run model and extract Y_clean, C_hat, interface mask."""
    with torch.no_grad():
        out = model(x[None])  # (1, C, H, W)
        if isinstance(out, (tuple, list)):
            pred = out[0]  # (1, 1, H, W) - interface mask
            if len(out) >= 3:
                c_hat = out[1]       # clutter prediction
                y_clean = out[2]     # clean B-scan
            else:
                c_hat = None
                y_clean = None
        else:
            pred = out
            c_hat = None
            y_clean = None
    return pred[0, 0].cpu().numpy(), c_hat, y_clean

# === 选择代表性窗口 ===
test_indices = [
    ('Line9', 0, '起始段 — 浅层信号'),
    ('Line9', 120, '中段 — 连续界面'),
    ('Line9', 250, '中后段 — 深部信号'),
]

# 加载 3-seed 模型
SEEDS = [1901, 1902, 1903]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

models_seeds = []
for s in SEEDS:
    run_dir = WORKSPACE / f"outputs/run_gpu_v3_pilot_mixed_loo_Line9_seed{s}"
    if not (run_dir / 'checkpoint_best.pt').exists():
        print(f"  WARN: seed{s} checkpoint not found, skipping")
        continue
    m, cfg = load_model(run_dir, DEVICE)
    models_seeds.append((m, cfg, s))
    print(f"  seed{s} loaded")

print(f"Total models loaded: {len(models_seeds)}")

# 预处理函数（与 DS.__getitem__ 对齐）
def normalize_raw_channel_3d(x, cfg):
    if not cfg.get('per_trace_robust_norm', False):
        return x
    clip = float(cfg.get('per_trace_robust_clip', 6.0))
    eps = float(cfg.get('per_trace_robust_eps', 1e-4))
    raw = x[:1]
    med = raw.median(dim=1, keepdim=True).values
    mad = (raw - med).abs().median(dim=1, keepdim=True).values
    norm = torch.clamp((raw - med) / (1.4826 * mad + eps), -clip, clip) / clip
    x = x.clone()
    x[:1] = norm
    return x

# 加载窗口数据，运行推理
test_windows = []
with open(DATA_ROOT / 'window_index.csv') as f:
    for line in f.read().strip().split('\n')[1:]:
        parts = line.split(',')
        sid, line_name = parts[0], parts[1]
        if line_name == 'Line9':
            test_windows.append(sid)

print(f"Line9 test windows: {len(test_windows)}")

# select diverse windows
selected = test_windows[:3] + test_windows[len(test_windows)//2:len(test_windows)//2+2] + test_windows[-3:]
selected = list(dict.fromkeys(selected))[:5]  # dedup, limit to 5
print(f"Selected: {selected}")

# === 推理并画图 ===
n_samples = len(selected)
fig, axes = plt.subplots(n_samples, 4, figsize=(16, 3*n_samples))

for i, sid in enumerate(selected):
    z = np.load(DATA_ROOT / 'windows' / f'{sid}.npz')
    raw = z['x_raw']
    y_mask = z['y_mask']

    # Preprocess
    x_t = preprocess_window(z, cfg)

    # Ensemble prediction
    pred_sum = np.zeros((cfg['height_resize'], cfg['width_resize']), dtype=np.float32)
    for m, c, _ in models_seeds:
        pred, c_hat, y_clean = inference(m, x_t.to(DEVICE))
        pred_sum += pred
    pred_avg = pred_sum / len(models_seeds)

    # Resize raw for display
    raw_disp = raw  # (501, 256)

    # Interpolate y_mask to same size as prediction
    y_mask_t = torch.from_numpy(y_mask[None, None]).float()
    y_mask_rs = torch.nn.functional.interpolate(
        y_mask_t, (cfg['height_resize'], cfg['width_resize']),
        mode='bilinear', align_corners=False
    )[0, 0].numpy()

    # Plot
    ax = axes[i] if n_samples > 1 else axes

    # Row title
    ax[0].set_ylabel(f'{sid[:20]}', fontsize=9)

    # 1) Raw B-scan
    im0 = ax[0].imshow(raw_disp, aspect='auto', cmap='gray',
                       norm=Normalize(vmin=np.percentile(raw_disp, 1), vmax=np.percentile(raw_disp, 99)))
    ax[0].set_title('Raw B-scan (Input)', fontsize=10)
    ax[0].set_xlabel('Trace')
    ax[0].set_ylabel('Time (smp)')

    # 2) Predicted clean mask (interface)
    im1 = ax[1].imshow(pred_avg, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax[1].set_title('Predicted Interface Probability', fontsize=10)
    ax[1].set_xlabel('Trace')

    # 3) Ground truth mask
    im2 = ax[2].imshow(y_mask_rs, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax[2].set_title('Ground Truth Mask', fontsize=10)
    ax[2].set_xlabel('Trace')

    # 4) Overlay: predicted contour on raw
    ax[3].imshow(raw_disp, aspect='auto', cmap='gray',
                 norm=Normalize(vmin=np.percentile(raw_disp, 1), vmax=np.percentile(raw_disp, 99)))
    # Overlay prediction as contour
    pred_down = resize(pred_avg, raw_disp.shape, order=1, preserve_range=True)
    contour = ax[3].contour(pred_down, levels=[0.5], colors='cyan', linewidths=0.8, alpha=0.7)
    ax[3].set_title('Prediction Overlay (cyan=0.5)', fontsize=10)
    ax[3].set_xlabel('Trace')

plt.tight_layout()
out_path = OUT_DIR / 'PGDA_CSNet_preview_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
