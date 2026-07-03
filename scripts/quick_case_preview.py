"""Quick preview of case_000001: raw B-scan + AGC + mask overlay."""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

CASE = Path("D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pilot_train_v1_3060/yingshan_pilot_train_3060_v1/models/case_000001")
OUT = Path("C:/Users/17844/Desktop/case_000001_preview.png")

out_files = sorted(CASE.glob("raw*.out"), key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
bscan = np.zeros((5937, 128), dtype=np.float32)
for i, f in enumerate(out_files):
    with h5py.File(f, 'r') as hf:
        bscan[:, i] = hf['rxs']['rx1']['Ez'][:]

mask = np.load(CASE / "labels" / "interface_mask_bscan.npy")

old = np.linspace(0, 1, 5937)
new = np.linspace(0, 1, 501)
bscan_501 = np.zeros((501, 128), dtype=np.float32)
for i in range(128):
    bscan_501[:, i] = np.interp(new, old, bscan[:, i])

def agc(x, window=50):
    env = np.sqrt(np.convolve(x**2, np.ones(window)/window, mode='same'))
    env = np.maximum(env, 1e-10)
    return x / env

bscan_agc = np.zeros_like(bscan_501)
for i in range(128):
    bscan_agc[:, i] = agc(bscan_501[:, i])

mask_center = np.where(mask[:, 63] > 0.5)[0]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
ax = axes.ravel()

model_img = plt.imread(str(CASE / "model_preview.png"))
ax[0].imshow(model_img)
ax[0].set_title("Geometry Cross-Section")
ax[0].axis('off')

vmax = np.percentile(abs(bscan_501), 99)
ax[1].imshow(bscan_501, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
             extent=[0, 128, 700, 0])
ax[1].plot([63.5, 63.5], [0, 700], 'c--', linewidth=0.5, alpha=0.5)
ax[1].set_title("Raw B-scan (P99={:.2f})".format(vmax))
ax[1].set_xlabel("Trace")
ax[1].set_ylabel("Time (ns)")

ax[2].imshow(bscan_agc, aspect='auto', cmap='gray', vmin=-3, vmax=3,
             extent=[0, 128, 700, 0])
if len(mask_center):
    ax[2].plot([0, 128], [mask_center[0]*700/501]*2, 'r-', linewidth=1, alpha=0.8)
ax[2].plot([63.5, 63.5], [0, 700], 'c--', linewidth=0.5, alpha=0.5)
ax[2].set_title("AGC B-scan")
ax[2].set_xlabel("Trace")
ax[2].set_ylabel("Time (ns)")

ct = bscan_501[:, 63]
t = np.linspace(0, 700, 501)
ax[3].plot(t, ct, 'b-', linewidth=0.5)
ax[3].axvline(353.5, color='r', linestyle='--', alpha=0.5, label='Mask')
ax[3].set_title("Center Trace (64) - Raw")
ax[3].set_xlabel("Time (ns)")
ax[3].set_ylabel("Amplitude")
ax[3].set_xlim(0, 700)

ct_agc = bscan_agc[:, 63]
ax[4].plot(t, ct_agc, 'b-', linewidth=0.5)
ax[4].axvline(353.5, color='r', linestyle='--', alpha=0.5, label='Mask')
ax[4].set_title("Center Trace (64) - AGC")
ax[4].set_xlabel("Time (ns)")
ax[4].set_ylabel("Normalized")
ax[4].set_xlim(0, 700)

ax[5].imshow(bscan_agc, aspect='auto', cmap='gray', vmin=-3, vmax=3,
             extent=[0, 128, 700, 0])
if len(mask_center):
    ax[5].plot([0, 128], [mask_center[0]*700/501]*2, 'r-', linewidth=2, alpha=0.9,
               label='Mask')
ax[5].set_title("AGC + Mask (red line)")
ax[5].set_xlabel("Trace")
ax[5].set_ylabel("Time (ns)")
ax[5].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=80, bbox_inches='tight')
plt.close()
print("Saved:", OUT)
