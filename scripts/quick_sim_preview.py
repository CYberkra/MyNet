"""gprMax 仿真 B-scan 预览图"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("D:/Claude/PGDA-CSNet/data/simulation_pretrain_v2/windows")
OUT = Path("C:/Users/17844/Desktop")

cases = [1, 5, 10, 15, 20]
labels = ['Gentle', 'Terrace', 'Wire', 'Anomaly', 'Slope']

fig, axes = plt.subplots(len(cases), 3, figsize=(12, 10))

for i, (c, lbl) in enumerate(zip(cases, labels)):
    sid = f'pilot_case_{c:06d}_w00'
    d = np.load(DATA / f'{sid}.npz')
    raw = d['x_raw']
    mask = d['y_mask']

    # Raw B-scan
    vmin, vmax = np.percentile(raw, [1, 99])
    axes[i, 0].imshow(raw, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f'{lbl} — Raw' if i == 0 else 'Raw', fontsize=9)
    axes[i, 0].set_ylabel(f'Case {c}', fontsize=8)

    # Interface mask
    axes[i, 1].imshow(mask, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axes[i, 1].set_title('Interface Mask' if i == 0 else '', fontsize=9)

    # Overlay
    axes[i, 2].imshow(raw, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, 2].contour(mask, levels=[0.5], colors='cyan', linewidths=0.6, alpha=0.7)
    axes[i, 2].set_title('Overlay (cyan=0.5)' if i == 0 else '', fontsize=9)

plt.tight_layout()
p = OUT / 'gprmax_simulation_preview.png'
fig.savefig(p, dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved: {p} ({fig.get_size_inches()})')
