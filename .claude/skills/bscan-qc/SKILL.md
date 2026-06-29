---
name: bscan-qc
description: Quick B-scan visualization and quality check from merged.out files. Use when user asks to "画Bscan", "看看效果", "QC", or wants to inspect simulation output.
---

# B-scan QC

Given a `*_merged.out` file path, generate B-scan image and quality metrics.

## Implementation

Read merged HDF5 via h5py, compute envelope, plot 3-panel figure (B-scan + mean trace + spectrum), output PNG and stats.

```python
import numpy as np, h5py, os
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def qc_bscan(merged_path):
    out_dir = os.path.dirname(merged_path)
    stem = os.path.basename(merged_path).replace('_merged.out', '')
    
    with h5py.File(merged_path, 'r') as hf:
        data = np.asarray(hf['rxs']['rx1']['Ez'])
        dt = hf.attrs.get('dt', 0)
        iters = hf.attrs.get('Iterations', 0)
        nx_ny_nz = list(hf.attrs.get('nx_ny_nz', [0,0,0]))
        merged_count = hf.attrs.get('MergedModelCount', '?')
    
    if data.ndim == 1:
        bscan = data.reshape(-1, 1)
    else:
        bscan = data
    
    time_ns = np.arange(bscan.shape[0]) * dt * 1e9
    n_traces = bscan.shape[1]
    
    # Amplitude stats
    early_rms = float(np.std(bscan[:int(50/dt/1e9), :]))
    mid_rms = float(np.std(bscan[int(200/dt/1e9):int(400/dt/1e9), :]))
    late_rms = float(np.std(bscan[int(500/dt/1e9):, :]))
    snr_db = 20 * np.log10(mid_rms / (late_rms + 1e-12))
    
    # Mean trace + envelope
    mean_tr = bscan.mean(axis=1)
    env = np.abs(sg.hilbert(mean_tr))
    
    # Spectrum
    spec = np.abs(np.fft.rfft(mean_tr))
    freqs = np.fft.rfftfreq(len(mean_tr), d=dt)
    peak_freq = freqs[np.argmax(spec[1:])+1] / 1e6
    
    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{stem} | {n_traces}tr x {bscan.shape[0]}samp | grid={nx_ny_nz} | {time_ns[-1]:.0f}ns')
    
    vmax = np.percentile(np.abs(bscan), 99)
    im = axes[0].imshow(bscan, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
                        extent=[0, n_traces, time_ns[-1], 0])
    axes[0].set_title(f'B-scan'); axes[0].set_xlabel('Trace'); axes[0].set_ylabel('Time (ns)')
    plt.colorbar(im, ax=axes[0])
    
    axes[1].plot(mean_tr, time_ns, 'b-', linewidth=0.8, label='Mean')
    axes[1].plot(env, time_ns, 'r-', linewidth=0.8, alpha=0.7, label='Envelope')
    axes[1].set_title('Mean Trace + Envelope'); axes[1].invert_yaxis()
    axes[1].set_xlabel('Amplitude'); axes[1].set_ylabel('Time (ns)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(freqs/1e6, spec/spec.max(), 'k-', linewidth=0.8)
    axes[2].set_title(f'Spectrum (peak={peak_freq:.0f}MHz)')
    axes[2].set_xlabel('Freq (MHz)'); axes[2].set_xlim(0, 250)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'{stem}_qc.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'B-scan: {bscan.shape} | Range: [{bscan.min():.1f}, {bscan.max():.1f}]')
    print(f'Early RMS: {early_rms:.1f} | Mid RMS: {mid_rms:.4f} | Late RMS: {late_rms:.6f}')
    print(f'SNR: {snr_db:.1f}dB | Peak freq: {peak_freq:.0f}MHz')
    print(f'Saved: {fig_path}')
    return fig_path
```

### Multi-file comparison
When user provides multiple `_merged.out` files (e.g. "画出 raw 和 target 对比"), generate a single figure with overlays showing the mean traces of each.
