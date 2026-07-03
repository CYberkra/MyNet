"""Preview for visible_interface_smoke_v2_case_000001"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("D:/Claude/PGDA-CSNet/data/pgda_visible_interface_smoke/PGDA_VISIBLE_INTERFACE_SMOKE_V2_CASE_000001")
OUT = Path("C:/Users/17844/Desktop/visible_interface_preview.png")

geo = ROOT / "geometry"
labels = ROOT / "labels"
preview_dir = ROOT / "preview"

out_files = sorted(geo.glob("raw*.out"), key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
bscan = np.zeros((5937, 128), dtype=np.float32)
for i, f in enumerate(out_files[:128]):
    with h5py.File(f, 'r') as hf:
        bscan[:, i] = hf['rxs']['rx1']['Ez'][:]

old = np.linspace(0, 1, 5937)
new = np.linspace(0, 1, 501)
bscan_501 = np.zeros((501, 128), dtype=np.float32)
for i in range(128):
    bscan_501[:, i] = np.interp(new, old, bscan[:, i])

mask = np.load(labels / "interface_mask_bscan.npy")
y_soft = np.load(labels / "y_soft_501x128.npy")
target_time = np.load(labels / "target_time_ns.npy")

def agc(x, window=50):
    env = np.sqrt(np.convolve(x**2, np.ones(window)/window, mode='same'))
    return x / np.maximum(env, 1e-10)

bscan_agc = np.zeros_like(bscan_501)
for i in range(128):
    bscan_agc[:, i] = agc(bscan_501[:, i])

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
ax = axes.ravel()

# 1. Design contact sheet (from pre-generated preview)
design_img = plt.imread(str(preview_dir / "00_design_contact_sheet.png"))
ax[0].imshow(design_img)
ax[0].set_title("Design Overview")
ax[0].axis('off')

# 2. Raw B-scan
vmax = np.percentile(abs(bscan_501), 99)
ax[1].imshow(bscan_501, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
             extent=[0, 128, 700, 0])
ax[1].plot(range(128), target_time, 'r-', linewidth=1.5, alpha=0.9, label='Target')
ax[1].set_title(f"Raw B-scan (P99={vmax:.2f})")
ax[1].set_xlabel("Trace")
ax[1].set_ylabel("Time (ns)")
ax[1].legend(fontsize=8)

# 3. AGC + target mask
ax[2].imshow(bscan_agc, aspect='auto', cmap='gray', vmin=-3, vmax=3,
             extent=[0, 128, 700, 0])
ax[2].plot(range(128), target_time, 'r-', linewidth=1.5, alpha=0.9, label='Target')
ax[2].set_title("AGC + Target (red)")
ax[2].set_xlabel("Trace")
ax[2].set_ylabel("Time (ns)")
ax[2].legend(fontsize=8)

# 4. Interface mask (hard)
ax[3].imshow(mask, aspect='auto', cmap='Reds', vmin=0, vmax=1,
             extent=[0, 128, 700, 0])
ax[3].set_title("Hard Mask")
ax[3].set_xlabel("Trace")
ax[3].set_ylabel("Time (ns)")

# 5. Soft label
ax[4].imshow(y_soft, aspect='auto', cmap='viridis', vmin=0, vmax=1,
             extent=[0, 128, 700, 0])
ax[4].set_title("Soft Label (y_soft)")
ax[4].set_xlabel("Trace")
ax[4].set_ylabel("Time (ns)")

# 6. AGC + mask overlay
ax[5].imshow(bscan_agc, aspect='auto', cmap='gray', vmin=-3, vmax=3,
             extent=[0, 128, 700, 0])
mask_disp = np.ma.masked_where(y_soft < 0.05, y_soft)
ax[5].imshow(mask_disp, aspect='auto', cmap='Reds', alpha=0.4,
             extent=[0, 128, 700, 0])
ax[5].plot(range(128), target_time, 'r-', linewidth=1, alpha=0.6)
ax[5].set_title("AGC + Soft Label Overlay")
ax[5].set_xlabel("Trace")
ax[5].set_ylabel("Time (ns)")

plt.tight_layout()
plt.savefig(OUT, dpi=100, bbox_inches='tight')
plt.close()
print("Saved:", OUT)
