"""Pilot-Train QC: geometry + raw + AGC + NPZ."""
import json, sys, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, 'D:/Claude/PGDA-CSNet/uavgpr_simlab/src')
from uavgpr_simlab.core.postprocess import merge_available_bscan_for_input

CASE = Path("D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pilot_train_v1/"
            "yingshan_pilot_train_3060_v1/models/case_000001")
OUT = Path("C:/Users/17844/Desktop")
sw = json.load(open(CASE / 'scene_world.json'))

T_MAX = 700
TIME_NS = np.linspace(0, T_MAX, 501)

raw, _ = merge_available_bscan_for_input(CASE / 'raw.in')
raw = raw.reshape(5937, 128)
mask = np.load(CASE / 'labels' / 'interface_mask_bscan.npy')
npz = np.load("D:/Claude/PGDA-CSNet/data/simulation_pretrain_v3_check/"
              "windows/pilot_case_000001_w00.npz")

# Resample raw to 501
old = np.linspace(0, 1, 5937)
new = np.linspace(0, 1, 501)
raw_rs = np.zeros((501, 128))
for i in range(128):
    raw_rs[:, i] = np.interp(new, old, raw[:, i])

# Geometry
gx = np.array(sw['ground_profile']['x_m'])
gy = np.array(sw['ground_profile']['y_m'])
bx = np.array(sw['bedrock_interface']['x_m'])
by = np.array(sw['bedrock_interface']['y_m'])
h = sw['trajectory']['nominal_height_m']
uav_y = gy.max() + h
fgx = np.linspace(gx.min(), gx.max(), 128)
fgy = np.interp(fgx, gx, gy)
fby = np.interp(fgx, bx, by)
gtwt = 2 * (uav_y - fgy) / 0.3

# Mask center TIME (ns) per trace — used for all plots
mask_ns = np.full(128, np.nan)
for j in range(128):
    rows = np.where(mask[:, j] > 0.5)[0]
    if len(rows) > 0:
        mask_ns[j] = rows[len(rows)//2] / 501 * 700

def agc(data, w=100):
    o = np.zeros_like(data)
    for i in range(data.shape[1]):
        s = uniform_filter1d(np.abs(data[:, i]), size=w, mode='reflect')
        o[:, i] = data[:, i] / np.maximum(s, 1e-10)
    return o
agc_rs = agc(raw_rs, 100)

xr, ym = npz['x_raw'], npz['y_mask']
lw = npz['label_weight']

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# (1) Geometry
ax = axes[0, 0]
ax.fill_between(gx, sw['domain']['y_m'], gy, color='lightcyan', alpha=0.4)
ax.fill_between(gx, gy, by, color='sandybrown', alpha=0.5)
ax.fill_between(gx, by, 0, color='rosybrown', alpha=0.5)
ax.plot(gx, gy, 'k-', lw=1.5, label='Ground')
ax.plot(bx, by, 'r-', lw=1.5, label='Bedrock')
ax.axhline(uav_y, c='b', ls='--', lw=0.8, alpha=0.5, label=f'UAV {h:.0f}m')
ax.set_title(f'Geometry: {sw["family"]}', fontsize=10)
ax.set_xlabel('Distance (m)'); ax.set_ylabel('Elevation (m)')
ax.legend(fontsize=6)

# (2) Raw + ground + mask line
ax = axes[0, 1]
vmin, vmax = np.percentile(raw_rs, [1, 99])
ax.imshow(raw_rs, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax,
          extent=[0, 128, 700, 0])
ax.plot(range(128), gtwt, 'b-', lw=1, label='Ground')
ax.plot(range(128), mask_ns, 'r-', lw=1.5, label='Mask')
ax.set_title('Raw + ground(blue) + mask(red)', fontsize=10)
ax.set_xlabel('Trace'); ax.set_ylabel('Time (ns)')
ax.legend(fontsize=6)

# (3) AGC + mask (just plot lines, no contour)
ax = axes[0, 2]
ax.imshow(agc_rs, aspect='auto', cmap='gray', vmin=-1, vmax=3,
          extent=[0, 128, 700, 0])
ax.plot(range(128), gtwt, 'b-', lw=1, label='Ground')
ax.plot(range(128), mask_ns, 'r-', lw=1.5, label='Mask')
ax.set_title('AGC + mask(red)', fontsize=10)
ax.set_xlabel('Trace'); ax.set_ylabel('Time (ns)')
ax.legend(fontsize=6)

# (4) Trace 64
ax = axes[1, 0]
ax.plot(TIME_NS, agc_rs[:, 64], 'r-', lw=0.7)
ax.axvline(gtwt[64], c='b', ls='--', lw=1, label=f'G={gtwt[64]:.0f}ns')
ax.axvline(mask_ns[64], c='r', ls='--', lw=1.5, label=f'Mask={mask_ns[64]:.0f}ns')
ax.set_xlim(50, 500)
ax.set_title('Trace 64 AGC', fontsize=10)
ax.set_xlabel('Time (ns)'); ax.legend(fontsize=6)

# (5) NPZ + mask line
ax = axes[1, 1]
off = 64
vn = np.percentile(xr[:, off:-off], [1, 99])
ax.imshow(xr, aspect='auto', cmap='gray', vmin=vn[0], vmax=vn[1],
          extent=[0, 256, 700, 0])
ax.plot(np.arange(128)+off, gtwt, 'b-', lw=1, label='Ground')
ax.plot(np.arange(128)+off, mask_ns, 'r-', lw=1.5, label='Mask')
ax.set_title('NPZ + mask(red)', fontsize=10)
ax.set_xlabel('Trace (256)'); ax.set_ylabel('Time (ns)')
ax.legend(fontsize=6)

# (6) Status
ax = axes[1, 2]
ax.axis('off')
info = [
    '=== Validation ===',
    '',
    f'Mask center: {np.nanmean(mask_ns):.0f}ns',
    'Red line = mask center (plot)',
    'No contour used (was wrong)',
    '',
    'NPZ padding all zero:',
    f'  y_mask[:,:64]={ym[:,:64].sum():.0f}',
    f'  y_mask[:,192:]={ym[:,192:].sum():.0f}',
    f'  weight[:64]={lw[:64].sum():.0f}',
    f'  weight[192:]={lw[192:].sum():.0f}',
    '',
    'All panels: 501smp = 0-700ns',
]
for i, l in enumerate(info):
    ax.text(0.05, 0.95-i*0.055, l, transform=ax.transAxes, fontsize=9, va='top')

plt.tight_layout()
p = OUT / 'pilot_train_testscene_qc.png'
fig.savefig(p, dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved: {p}')
