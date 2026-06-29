"""gprMax 仿真 4 面板对比图：几何模型 + Raw + 背景抑制 + AGC
仅使用有 merged.out 仿真输出的 case。
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, 'D:/Claude/PGDA-CSNet/uavgpr_simlab/src')
from uavgpr_simlab.core.postprocess import merge_available_bscan_for_input

BASE = Path("D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pgda_batch_v1_3060/models")
OUT = Path("C:/Users/17844/Desktop")

FAMILY_NAMES = {
    'gentle_interbed': 'Gentle (缓坡-砂泥互层)',
    'terrace_paddy': 'Terrace (阶地-水田)',
    'wire_tree_endpoint': 'Wire (含电线杆)',
    'deep_anomaly_21m': 'Anomaly (深部异常体)',
}

# 选有 merged.out 的 4 个不同家族 case
SELECTED = ['case_000001', 'case_000002', 'case_000003', 'case_000004']

def load_bscan(in_path):
    """从 .in 文件加载完整分辨率 B-scan，返回 (n_time, n_traces)."""
    result = merge_available_bscan_for_input(in_path)
    if result is None:
        return None
    bscan, meta = result
    n_time = int(meta['files'][0]['attrs']['Iterations'])
    n_tr = int(meta['files'][0]['attrs']['MergedModelCount'])
    return bscan.reshape(n_time, n_tr)  # (time, traces) — imshow 用

def agc(data, window=40):
    """滑动窗口 AGC on each trace."""
    out = np.zeros_like(data)
    for i in range(data.shape[1]):  # 对每道
        tr = data[:, i]
        env = np.abs(tr)
        smoothed = uniform_filter1d(env, size=window, mode='reflect')
        smoothed = np.maximum(smoothed, 1e-10)
        out[:, i] = tr / smoothed
    return out

def plot_geometry(ax, sw):
    """画几何模型截面."""
    gx = np.array(sw['ground_profile']['x_m'])
    gy = np.array(sw['ground_profile']['y_m'])
    bx = np.array(sw['bedrock_interface']['x_m'])
    by = np.array(sw['bedrock_interface']['y_m'])
    dom = sw['domain']
    top = dom['y_m']
    bot = top - dom['y_m']
    ax.fill_between(gx, top, gy, color='lightcyan', alpha=0.4, label='Air')
    ax.fill_between(gx, gy, by, color='sandybrown', alpha=0.5, label='Cover')
    ax.fill_between(gx, by, bot, color='rosybrown', alpha=0.5, label='Bedrock')
    ax.plot(gx, gy, 'k-', lw=1.5, label='Ground')
    ax.plot(bx, by, 'r-', lw=1.5, label='Bedrock')
    traj = sw.get('trajectory', {})
    if traj and 'nominal_height_m' in traj:
        h = float(traj['nominal_height_m'])
        ax.axhline(gy.mean()+h, color='b', ls='--', lw=0.8, alpha=0.5, label=f'UAV {h:.0f}m')
    ax.set_xlabel('Distance (m)', fontsize=7)
    ax.set_ylabel('Elevation (m)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc='lower right')

fig, axes = plt.subplots(len(SELECTED), 4, figsize=(18, 13))

for row, cid in enumerate(SELECTED):
    cd = BASE / cid
    sw = json.load(open(cd / 'scene_world.json'))
    family = sw['family']
    label = FAMILY_NAMES.get(family, family)

    # 1. 几何模型截面
    plot_geometry(axes[row, 0], sw)
    axes[row, 0].set_ylabel(f'{cid}\n{label}', fontsize=7)

    # 2. Raw B-scan (n_time, n_traces)
    raw = load_bscan(cd / 'raw.in')
    if raw is not None:
        n_time, n_tr = raw.shape
        step = max(1, n_time // 500)
        raw_disp = raw[::step, :]   # 沿时间轴降采样
        vmin, vmax = np.percentile(raw, [1, 99])
        axes[row, 1].imshow(raw_disp, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title('Raw B-scan (FDTD)' if row == 0 else '', fontsize=9)
    else:
        axes[row, 1].text(0.5, 0.5, 'No data', transform=axes[row, 1].transAxes, ha='center')
    axes[row, 1].set_xlabel('Trace', fontsize=7)
    axes[row, 1].set_ylabel(f'Time ({n_time} smp)', fontsize=7)

    # 3. 均值道背景抑制
    if raw is not None:
        bg = np.mean(raw, axis=1, keepdims=True)  # 每时刻取所有道平均 (n_time, 1)
        suppressed = raw - bg
        vmin2, vmax2 = np.percentile(suppressed, [1, 99])
        step2 = max(1, n_time // 500)
        axes[row, 2].imshow(suppressed[::step2, :], aspect='auto', cmap='gray', vmin=vmin2, vmax=vmax2)
        axes[row, 2].set_title('Mean-Trace Subtraction' if row == 0 else '', fontsize=9)
    else:
        axes[row, 2].text(0.5, 0.5, 'No data', transform=axes[row, 2].transAxes, ha='center')
    axes[row, 2].set_xlabel('Trace', fontsize=7)

    # 4. 背景抑制 + AGC
    if raw is not None:
        sup = raw - np.mean(raw, axis=1, keepdims=True)
        sup_agc = agc(sup, window=30)
        step3 = max(1, n_time // 500)
        axes[row, 3].imshow(sup_agc[::step3, :], aspect='auto', cmap='gray', vmin=-1, vmax=3)
        axes[row, 3].set_title('Subtraction + AGC' if row == 0 else '', fontsize=9)
    else:
        axes[row, 3].text(0.5, 0.5, 'No data', transform=axes[row, 3].transAxes, ha='center')
    axes[row, 3].set_xlabel('Trace', fontsize=7)

plt.tight_layout()
p = OUT / 'gprmax_4panel_preview.png'
fig.savefig(p, dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved: {p}')
