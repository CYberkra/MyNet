#!/usr/bin/env python3
"""
PGDA_SYNTH_DATASET_V1 — After-run QC

Usage:
    python tools/after_run_qc.py <case_run_dir>

Generates 04_qc/{batch_id}/{CASE_ID}/ with:
  - qc_report.json (structured metrics, machine-readable)
  - qc_metrics.csv  (flat table)
  - qc_preview_full.png (multi-panel overview)
  - qc_target_zoom.png  (target zone zoom)
  - qc_agc_check.png    (AGC check)
  - qc_decision.txt     (GREEN/YELLOW/RED + reason)

Dependencies: numpy, matplotlib, scipy
"""

import sys, json, math, argparse
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROOT = Path(__file__).resolve().parents[1]
HALF_NS = 30  # search half-window for curve support (ns)

# ── QC thresholds (from QC_RULES.md) ──
THRESHOLDS = {
    'target_local_peak_median_green': 0.5,
    'target_local_peak_median_yellow': 0.3,
    'support_ratio_green': 0.6,
    'support_ratio_yellow': 0.4,
    'peak_offset_green_ns': 8.0,
    'peak_offset_yellow_ns': 20.0,
    'dead_trace_ratio_max': 0.05,
    'dominant_x_forbidden': True,
    'target_continuous_required': True,
}


def compute_qc(bscan_path, label_dir, out_dir):
    """
    Run through all QC metrics.
    bscan_path: path to bscan.npy (raw gprMax output, native resolution)
    label_dir:  path to labels/ directory
    out_dir:    output directory for QC artifacts
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    arr = np.load(bscan_path).astype(np.float64)          # (T, W)
    T_full, W = arr.shape
    t_full = np.linspace(0, 700, T_full)

    # Labels (temporally resampled to 501)
    label_dir = Path(label_dir)
    y_soft = np.load(str(label_dir / 'y_soft_501x128.npy')).astype(np.float64)
    target_vis = np.load(str(label_dir / 'target_visible_phase_time_ns.npy'))
    target_geom = np.load(str(label_dir / 'target_geom_time_ns.npy'))
    t_501 = np.linspace(0, 700, 501)

    # Resample B-scan to 501 samples for label comparison
    raw_501 = np.empty((501, W), dtype=np.float64)
    for i in range(W):
        raw_501[:, i] = np.interp(t_501, t_full, arr[:, i])

    # ── 1. Trace variance ──
    diffs = np.array([np.max(np.abs(arr[:, i] - arr[:, 0])) for i in range(W)])
    trace_var = float(np.max(diffs))
    trace_var_mean = float(np.mean(diffs))

    # ── 2. Dead trace ratio ──
    dead_count = 0
    for i in range(1, W):
        cc = np.corrcoef(arr[:, i-1], arr[:, i])[0, 1]
        if cc < 0.8 or not np.isfinite(cc):
            dead_count += 1
    dead_trace_ratio = dead_count / max(W - 1, 1)

    # ── 3. Target visibility ──
    # Follow original audit: curve_support() metric
    # For each trace: ratio = abs(arr[r0]) / max(abs(arr[sel±30ns]))
    # where r0 = time sample closest to target_vis time

    def curve_support(arr, time, curve, half_ns=HALF_NS):
        ratios = []
        offsets = []
        for j, t in enumerate(curve):
            if not np.isfinite(t):
                continue
            # indices within ±half_ns of target time
            sel = np.where((time >= t - half_ns) & (time <= t + half_ns))[0]
            if len(sel) < 2:
                continue
            r0 = int(np.argmin(np.abs(time - t)))  # expected time index
            pk_idx = np.argmax(np.abs(arr[sel, j]))  # peak index within sel
            pk = sel[pk_idx]
            ratio = abs(arr[r0, j]) / max(np.max(np.abs(arr[sel, j])), 1e-12)
            ratios.append(ratio)
            offsets.append(time[pk] - t)
        return float(np.median(ratios)), float(np.mean(np.array(ratios) > 0.5)), float(np.median(offsets))

    # Use background-subtracted data (remove DC per time sample), then t^4
    bg = raw_501 - np.mean(raw_501, axis=1, keepdims=True)
    t4b = bg * ((t_501 / t_501[-1]) ** 4 + 0.01)[:, None]

    target_local_peak_median, support_ratio, peak_offset_median = \
        curve_support(t4b, t_501, target_vis)
    peak_offset_median = float(peak_offset_median)

    # Target zone bounds for plotting
    vis_min = float(np.nanmin(target_vis))
    vis_max = float(np.nanmax(target_vis))
    search_lo = max(0, int((vis_min - 50) / 700 * 501))
    search_hi = min(501, int((vis_max + 50) / 700 * 501))

    # Also compute per-trace offset std from curve_support-like logic
    peak_offset_list = []
    for j in range(W):
        t = target_vis[j] if np.isfinite(target_vis[j]) else np.nan
        if not np.isfinite(t):
            continue
        sel = np.where((t_501 >= t - HALF_NS) & (t_501 <= t + HALF_NS))[0]
        if len(sel) < 2:
            continue
        pk = sel[np.argmax(np.abs(t4b[sel, j]))]
        peak_offset_list.append(t_501[pk] - t)
    peak_offset_std = float(np.std(peak_offset_list)) if peak_offset_list else 0.0

    # ── 4. Energy ratios (shallow/target, mid/target, post/target) ──
    def energy_ratio(zone_ns_start, zone_ns_end, target_start_ns=vis_min, target_end_ns=vis_max):
        """Compute RMS ratio of a zone over target zone."""
        z_lo = int(zone_ns_start / 700 * 501)
        z_hi = int(zone_ns_end / 700 * 501)
        t_lo = int(target_start_ns / 700 * 501)
        t_hi = int(target_end_ns / 700 * 501)
        if z_hi <= z_lo or t_hi <= t_lo:
            return 99.0
        z_rms = np.sqrt(np.mean(raw_501[z_lo:z_hi] ** 2))
        t_rms = np.sqrt(np.mean(raw_501[t_lo:t_hi] ** 2))
        return float(z_rms / max(t_rms, 1e-12))

    early_over_target = energy_ratio(0, 150)
    shallow_over_target = energy_ratio(150, 250)
    mid_over_target = energy_ratio(vis_max + 20, vis_max + 100)
    post_over_target = energy_ratio(vis_max + 100, 700)

    # ── 5. X-pattern check ──
    # Simple: check if early strong energy band (0-150ns) is dominant in AGC
    t2b = bg * ((t_501 / t_501[-1]) ** 2 + 0.01)[:, None]
    early_energy_agc = np.mean(np.abs(t2b[:int(150/700*501), :]))
    target_energy_agc = np.mean(np.abs(t2b[search_lo:search_hi, :]))
    has_dominant_X = bool(early_energy_agc > 3 * target_energy_agc)

    # ── 6. Label alignment ──
    label_alignment_ok = bool(abs(peak_offset_median) < 8.0)

    # ── 7. Polarity check ──
    # slide_zone->weathered_bedrock is negative (R=-0.333): expect dark->bright->dark
    # Simple: check sign of first peak in target zone on mean t⁴ trace
    mean_t4b = np.mean(t4b, axis=1)
    target_region = mean_t4b[search_lo:search_hi]
    first_peak_idx = search_lo + np.argmax(np.abs(target_region))
    polarity = "negative" if mean_t4b[first_peak_idx] < 0 else "positive"
    polarity_match = bool(polarity == "negative")

    # ── GRADE ──
    grade = "GREEN"
    reasons = []

    if target_local_peak_median < THRESHOLDS['target_local_peak_median_green']:
        grade = "YELLOW"
        reasons.append(f"target_local_peak_median {target_local_peak_median:.3f} < {THRESHOLDS['target_local_peak_median_green']}")
    if target_local_peak_median < THRESHOLDS['target_local_peak_median_yellow']:
        grade = "RED"
        reasons.append(f"target_local_peak_median {target_local_peak_median:.3f} < {THRESHOLDS['target_local_peak_median_yellow']}")
    if support_ratio < THRESHOLDS['support_ratio_green']:
        if grade == "GREEN":
            grade = "YELLOW"
        reasons.append(f"support_ratio {support_ratio:.3f} < {THRESHOLDS['support_ratio_green']}")
        if support_ratio < THRESHOLDS['support_ratio_yellow']:
            grade = "RED"
    if abs(peak_offset_median) > THRESHOLDS['peak_offset_yellow_ns']:
        grade = "RED"
        reasons.append(f"peak_offset {peak_offset_median:.1f}ns > {THRESHOLDS['peak_offset_yellow_ns']}ns")
    elif abs(peak_offset_median) > THRESHOLDS['peak_offset_green_ns']:
        if grade == "GREEN":
            grade = "YELLOW"
        reasons.append(f"peak_offset {peak_offset_median:.1f}ns > {THRESHOLDS['peak_offset_green_ns']}ns")
    if dead_trace_ratio > THRESHOLDS['dead_trace_ratio_max']:
        if grade != "RED":
            grade = "YELLOW"
        reasons.append(f"dead_trace_ratio {dead_trace_ratio:.3f} > {THRESHOLDS['dead_trace_ratio_max']}")
    if has_dominant_X:
        grade = "RED"
        reasons.append("Dominant X pattern detected")
    if not label_alignment_ok:
        if grade != "RED":
            grade = "YELLOW"
        reasons.append(f"Label alignment check failed (offset > 8ns)")

    decision = f"QC GRADE: {grade}\n" + "\n".join(f"  - {r}" for r in reasons) if reasons else f"QC GRADE: {grade}\n  All checks passed."

    # ── Assemble metrics ──
    metrics = {
        'trace_count': int(W),
        'trace_var': trace_var,
        'trace_var_mean': trace_var_mean,
        'dead_trace_ratio': dead_trace_ratio,
        'target_local_peak_median': target_local_peak_median,
        'support_ratio': support_ratio,
        'peak_offset_median_ns': peak_offset_median,
        'peak_offset_std_ns': peak_offset_std,
        'early_over_target': early_over_target,
        'shallow_over_target': shallow_over_target,
        'mid_over_target': mid_over_target,
        'post_over_target': post_over_target,
        'has_dominant_X': int(has_dominant_X),
        'label_alignment_ok': int(label_alignment_ok),
        'polarity_match': int(polarity_match),
        'qc_grade': grade,
        'search_lo_sample': int(search_lo),
        'search_hi_sample': int(search_hi),
        'vis_min_ns': vis_min,
        'vis_max_ns': vis_max,
    }

    # Write qc_report.json
    with open(out_dir / 'qc_report.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Write qc_metrics.csv
    with open(out_dir / 'qc_metrics.csv', 'w') as f:
        f.write('metric,value\n')
        for k, v in metrics.items():
            f.write(f'{k},{v}\n')

    # Write qc_decision.txt
    with open(out_dir / 'qc_decision.txt', 'w') as f:
        f.write(decision + '\n')

    # ── Generate figures (if matplotlib available) ──
    if not HAS_MPL:
        print("Matplotlib not available, skipping figure generation")
        return metrics

    _generate_figures(raw_501, y_soft, target_vis, target_geom, t4b, t_501,
                      metrics, out_dir, case_dir=str(bscan_path.parent.parent))

    return metrics


def _parse_geometry(in_path):
    """Parse .in file and return geometry data for plotting."""
    in_path = Path(in_path)
    if not in_path.exists():
        return None
    lines = in_path.read_text().splitlines()

    mats = {}
    for l in lines:
        s = l.strip()
        if s.startswith('#material:'):
            p = s.split()
            mats[p[-1]] = p[1]

    tris = []
    for l in lines:
        s = l.strip()
        if s.startswith('#triangle:'):
            p = s.split()
            if len(p) >= 11:
                tris.append({
                    'xs': [float(p[1]), float(p[4]), float(p[7])],
                    'ys': [float(p[2]), float(p[5]), float(p[8])],
                    'mat': p[-2],
                })

    tx, rx = None, None
    for l in lines:
        s = l.strip()
        if s.startswith('#hertzian_dipole:'):
            p = s.split()
            if len(p) >= 4:
                tx = (float(p[2]), float(p[3]))
        if s.startswith('#rx:'):
            p = s.split()
            if len(p) >= 3:
                rx = (float(p[1]), float(p[2]))

    # Scan range
    src_steps, rx_steps = None, None
    for l in lines:
        s = l.strip()
        if s.startswith('#src_steps:'):
            parts = s.split()
            src_steps = float(parts[1]) if len(parts) > 1 else None
        if s.startswith('#rx_steps:'):
            parts = s.split()
            rx_steps = float(parts[1]) if len(parts) > 1 else None

    # Domain
    domain_x, domain_y = 0, 0
    for l in lines:
        s = l.strip()
        if s.startswith('#domain:'):
            parts = s.split()
            if len(parts) >= 3:
                domain_x = float(parts[1])
                domain_y = float(parts[2])

    n_traces = 128  # default
    scan_start_x = min(tx[0] if tx else 0, rx[0] if rx else 0)
    scan_end_x = (rx[0] if rx else 0) + (rx_steps or 0) * (n_traces - 1)

    return {'mats': mats, 'tris': tris, 'tx': tx, 'rx': rx,
            'src_steps': src_steps, 'rx_steps': rx_steps,
            'scan_start_x': scan_start_x, 'scan_end_x': scan_end_x,
            'domain_x': domain_x, 'domain_y': domain_y}


def _draw_geometry(ax, geo):
    """Draw geometry cross-section with air region, scan range, TX/RX.
    gprMax coordinate: y=0 = top, y=domain_y = bottom (increasing downward).
    invert_yaxis() makes the plot show y=0 at top. So "air above ground"
    means AIR has SMALLER y values (closer to 0) than GROUND."""
    if geo is None or not geo['tris']:
        ax.text(0.5, 0.5, 'No .in file', ha='center', va='center', transform=ax.transAxes)
        return

    # Collect all y-extents per material
    mat_yranges = {}
    for t in geo['tris']:
        mat = t['mat']
        ys = t['ys']
        if mat not in mat_yranges:
            mat_yranges[mat] = {'min': min(ys), 'max': max(ys)}
        else:
            mat_yranges[mat]['min'] = min(mat_yranges[mat]['min'], min(ys))
            mat_yranges[mat]['max'] = max(mat_yranges[mat]['max'], max(ys))

    # Determine ground surface: where non-air materials end (max y)
    # In gprMax, y=0 is domain top, y increases downward.
    # Non-air materials span from surface (y≈0) down to y≈30.
    # Air is from y≈30 to domain bottom (y=45). TX/RX at y≈32 in air.
    material_bottom = 0
    for mat, rng in mat_yranges.items():
        if mat.lower() != 'air':
            material_bottom = max(material_bottom, rng['max'])
    surface_y = material_bottom  # bottom of the ground = start of air region

    unique_mats = sorted(set(t['mat'] for t in geo['tris']))
    cmap = dict(zip(unique_mats, [plt.cm.tab10(i / max(len(unique_mats), 1)) for i in range(len(unique_mats))]))

    for t in geo['tris']:
        ax.fill(t['xs'], t['ys'],
                facecolor=cmap.get(t['mat'], 'gray'),
                edgecolor='black', lw=0.15, alpha=0.95)

    # Air region: from ground surface to domain bottom
    if surface_y < geo['domain_y']:
        ax.axhspan(surface_y, geo['domain_y'], alpha=0.15, color='deepskyblue')
        mid_air = surface_y + (geo['domain_y'] - surface_y) * 0.4
        ax.text(geo['scan_start_x'] + 5, mid_air,
                'Air', fontsize=8, alpha=0.6, ha='center', va='center', style='italic')

    # Ground surface line
    ax.axhline(y=surface_y, color='saddlebrown', lw=1.5, ls='--', alpha=0.6)
    ax.text(geo['domain_x'] - 10, surface_y + 1.5, f'Surface y={surface_y:.0f}m',
            fontsize=6, color='saddlebrown', alpha=0.6, ha='right')

    # Scan range
    ax.axvspan(geo['scan_start_x'], geo['scan_end_x'], alpha=0.06, color='orange')
    ax.text((geo['scan_start_x'] + geo['scan_end_x']) / 2, geo['domain_y'] * 0.97,
            'Scan range', fontsize=6, alpha=0.5, ha='center', va='bottom', style='italic')

    # TX/RX with air gap annotation
    if geo['tx'] and geo['rx']:
        antenna_y = max(geo['tx'][1], geo['rx'][1])
        gap = antenna_y - surface_y
        if gap > 0:
            ax.annotate('', xy=(geo['tx'][0], antenna_y), xytext=(geo['tx'][0], surface_y),
                        arrowprops=dict(arrowstyle='<->', lw=0.6, color='blue', alpha=0.4))
            ax.text(geo['tx'][0] - 5, (antenna_y + surface_y) / 2,
                    f'H={gap:.1f}m', fontsize=6, color='blue', alpha=0.5, ha='right')

    if geo['tx']:
        ax.plot(geo['tx'][0], geo['tx'][1], 'rv', ms=8, zorder=5)
        ax.annotate(f'TX ({geo["tx"][0]:.0f}, {geo["tx"][1]:.1f})',
                    xy=geo['tx'], xytext=(geo['tx'][0] + 3, geo['tx'][1] + 2),
                    fontsize=6, color='red')
    if geo['rx']:
        ax.plot(geo['rx'][0], geo['rx'][1], 'b^', ms=8, zorder=5)
        ax.annotate(f'RX ({geo["rx"][0]:.0f}, {geo["rx"][1]:.1f})',
                    xy=geo['rx'], xytext=(geo['rx'][0] + 3, geo['rx'][1] - 3),
                    fontsize=6, color='blue')

    # Legend
    patches = [mpatches.Patch(color=color, label=mat) for mat, color in cmap.items()]
    if geo['tx']:
        patches.append(Line2D([0], [0], marker='v', color='w', markerfacecolor='r', markersize=8, label='TX'))
    if geo['rx']:
        patches.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='b', markersize=8, label='RX'))
    ax.legend(handles=patches, fontsize=5, loc='lower right', ncol=2)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Geometry Model', fontweight='bold')
    ax.set_xlim(-20, geo['domain_x'] + 20)
    ax.set_ylim(0, geo['domain_y'])  # normal: y=0(ground) at bottom, y=45(air) at top  # domain_y(45) at top = air gap, y=0 at bottom = deep ground


def _save_separate_geometry(geo, out_dir):
    """Save a dedicated large geometry figure with real triangle rendering."""
    if geo is None or not geo['tris']:
        return
    fig, ax = plt.subplots(figsize=(18, 7))

    unique_mats = sorted(set(t['mat'] for t in geo['tris']))
    cmap = dict(zip(unique_mats, [plt.cm.tab10(i / max(len(unique_mats), 1)) for i in range(len(unique_mats))]))

    for t in geo['tris']:
        ax.fill(t['xs'], t['ys'],
                facecolor=cmap.get(t['mat'], 'gray'),
                edgecolor='black', lw=0.2, alpha=0.95)

    # Compute material y-ranges for annotations and legends
    mat_yranges = {}
    for t in geo['tris']:
        mat, ys = t['mat'], t['ys']
        if mat not in mat_yranges:
            mat_yranges[mat] = {'min': min(ys), 'max': max(ys)}
        else:
            mat_yranges[mat]['min'] = min(mat_yranges[mat]['min'], min(ys))
            mat_yranges[mat]['max'] = max(mat_yranges[mat]['max'], max(ys))

    surface_y = max(rng['max'] for rng in mat_yranges.values())

    # Air region
    if surface_y < geo['domain_y']:
        ax.axhspan(surface_y, geo['domain_y'], alpha=0.10, color='deepskyblue')

    # Ground surface line
    ax.axhline(y=surface_y, color='brown', lw=1.5, ls='--', alpha=0.5)

    # TX/RX
    if geo['tx']:
        ax.plot(geo['tx'][0], geo['tx'][1], 'rv', ms=10, mec='black', mew=0.5, zorder=5)
        ax.annotate(f'TX ({geo["tx"][0]:.0f}, {geo["tx"][1]:.1f})',
                    xy=geo['tx'], xytext=(geo['tx'][0] + 5, geo['tx'][1] + 3),
                    fontsize=8, color='red', fontweight='bold')
    if geo['rx']:
        ax.plot(geo['rx'][0], geo['rx'][1], 'b^', ms=10, mec='black', mew=0.5, zorder=5)
        ax.annotate(f'RX ({geo["rx"][0]:.0f}, {geo["rx"][1]:.1f})',
                    xy=geo['rx'], xytext=(geo['rx'][0] + 5, geo['rx'][1] - 3),
                    fontsize=8, color='blue', fontweight='bold')

    # Scan range arrow
    mid_y = surface_y + (geo['domain_y'] - surface_y) * 0.3
    ax.annotate('', xy=(geo['scan_end_x'], mid_y), xytext=(geo['scan_start_x'], mid_y),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='orange'))
    ax.text((geo['scan_start_x'] + geo['scan_end_x']) / 2, mid_y + 1.5,
            f'Scan: {geo["scan_start_x"]:.0f}-{geo["scan_end_x"]:.0f}m',
            fontsize=7, ha='center', color='orange')

    # Legend
    patches = [mpatches.Patch(color=cmap[m], label=m) for m in unique_mats]
    patches.append(Line2D([0], [0], marker='v', color='w', markerfacecolor='r', markersize=8, label='TX'))
    patches.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='b', markersize=8, label='RX'))
    ax.legend(handles=patches, fontsize=7, loc='lower right', ncol=2)

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Geometry Model (triangle-based)', fontweight='bold')
    ax.set_xlim(-20, geo['domain_x'] + 20)
    ax.set_ylim(0, geo['domain_y'])
    ax.grid(alpha=0.1)

    plt.tight_layout()
    fig.savefig(out_dir / 'qc_geometry.png', dpi=160)
    plt.close(fig)
    plt.close(fig)


def _generate_figures(raw_501, y_soft, target_vis, target_geom, t4b, t_501,
                      metrics, out_dir, case_dir=None):
    W = raw_501.shape[1]
    traces = np.arange(W)
    extent = (0, W - 1, t_501[-1], t_501[0])
    search_lo = int(metrics.get('search_lo_sample', 250))
    search_hi = int(metrics.get('search_hi_sample', 300))

    # Parse geometry from .in file
    geo = None
    if case_dir:
        for p in [Path(case_dir) / 'raw' / 'raw.in', Path(case_dir) / 'geometry' / 'raw.in']:
            if p.exists():
                geo = _parse_geometry(str(p))
                if geo and geo['tris']:
                    break

        # ── qc_preview_full.png: 2x3 redesigned layout ──
    # 1: Geometry  2: Raw B-scan  3: BG-suppressed B-scan
    # 4: BG-supp + t^4 gain  5: Target zoom  6: QC Findings
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.28)
    mean_trace = np.mean(raw_501, axis=1, keepdims=True)
    bg_supp = raw_501 - mean_trace
    gain_t4 = (t_501 / t_501[-1]) ** 4
    t4_bg = bg_supp * (gain_t4[:, None] + 0.01)
    v_raw = np.percentile(np.abs(raw_501), 98)
    v_bg = np.percentile(np.abs(bg_supp), 98)
    v_t4 = np.percentile(np.abs(t4_bg), 99.5)
    extent_geo = (0, W - 1, t_501[-1], t_501[0])

    # Panel 1: Geometry
    ax = fig.add_subplot(gs[0, 0])
    if geo and geo['tris']:
        unique_mats = sorted(set(t['mat'] for t in geo['tris']))
        cmap_geo = dict(zip(unique_mats, [plt.cm.tab10(i/max(len(unique_mats),1)) for i in range(len(unique_mats))]))
        for t in geo['tris']:
            ax.fill(t['xs'], t['ys'], facecolor=cmap_geo[t['mat']], edgecolor='black', lw=0.08, alpha=0.92)
        mat_yranges = {}
        for t in geo['tris']:
            m = t['mat']
            if m not in mat_yranges:
                mat_yranges[m] = {'min': min(t['ys']), 'max': max(t['ys'])}
            else:
                mat_yranges[m]['min'] = min(mat_yranges[m]['min'], min(t['ys']))
                mat_yranges[m]['max'] = max(mat_yranges[m]['max'], max(t['ys']))
        surface_y = max(r['max'] for r in mat_yranges.values())
        if surface_y < geo['domain_y']:
            ax.axhspan(surface_y, geo['domain_y'], alpha=0.10, color='deepskyblue')
        ax.axhline(y=surface_y, color='brown', lw=1.2, ls='--', alpha=0.5)
        if geo['tx']: ax.plot(geo['tx'][0], geo['tx'][1], 'rv', ms=8, zorder=5, mec='black', mew=0.5)
        if geo['rx']: ax.plot(geo['rx'][0], geo['rx'][1], 'b^', ms=8, zorder=5, mec='black', mew=0.5)
        ax.set_xlim(-20, geo['domain_x'] + 20); ax.set_ylim(0, geo['domain_y'])
        pg = [mpatches.Patch(color=cmap_geo[m], label=m) for m in unique_mats]
        if geo['tx']: pg.append(Line2D([0],[0], marker='v', color='w', markerfacecolor='r', markersize=8, label='TX'))
        if geo['rx']: pg.append(Line2D([0],[0], marker='^', color='w', markerfacecolor='b', markersize=8, label='RX'))
        ax.legend(handles=pg, fontsize=5, loc='lower right', ncol=2)
    else:
        ax.text(0.5, 0.5, 'No .in file', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('1. Geometry Model', fontweight='bold')

    # Panel 2: Raw B-scan
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(raw_501, aspect='auto', cmap='gray', vmin=-v_raw, vmax=v_raw, extent=extent_geo)
    ax.plot(traces, target_vis, 'c-', lw=1.5, label='target_visible')
    ax.plot(traces, target_geom, 'b--', lw=1, label='target_geom')
    ax.set_title('2. Raw B-scan (full)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(t_501[-1], t_501[0])

    # Panel 3: BG-suppressed
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(bg_supp, aspect='auto', cmap='gray', vmin=-v_bg, vmax=v_bg, extent=extent_geo)
    ax.plot(traces, target_vis, 'c-', lw=1.5, label='target_visible')
    ax.set_title('3. BG-suppressed (full)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(t_501[-1], t_501[0])

    # Panel 4: BG-supp + t^4
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(t4_bg, aspect='auto', cmap='gray', vmin=-v_t4, vmax=v_t4, extent=extent_geo)
    ax.plot(traces, target_vis, 'c-', lw=1.5, label='target_visible')
    ax.set_title('4. BG-supp + t^4 gain (full)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(t_501[-1], t_501[0])

    # Panel 5: Target zoom
    ax = fig.add_subplot(gs[1, 1])
    zoom_lo = max(0, search_lo - 10); zoom_hi = min(501, search_hi + 10)
    z_ext = (0, W - 1, t_501[zoom_hi], t_501[zoom_lo])
    ax.imshow(t4_bg[zoom_lo:zoom_hi], aspect='auto', cmap='gray', vmin=-v_t4, vmax=v_t4, extent=z_ext)
    ax.plot(traces, target_vis, 'c-', lw=1.5, label='target_visible')
    ax.plot(traces, target_geom, 'b--', lw=1, label='target_geom')
    ax.set_title('5. Target zoom (t^4 + BG-supp)', fontweight='bold')
    ax.set_xlabel('Trace'); ax.set_ylabel('Time (ns)')
    ax.legend(fontsize=7)

    # Panel 6: QC Findings
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    grade = metrics['qc_grade']
    gc = {'GREEN':'green','YELLOW':'orange','RED':'red'}.get(grade,'gray')
    ax.text(0.08, 0.92, 'QC Findings', fontsize=13, fontweight='bold', transform=ax.transAxes, va='top')
    ax.text(0.55, 0.92, grade, fontsize=13, fontweight='bold', color=gc, transform=ax.transAxes, va='top')
    findings = []
    lp = metrics['target_local_peak_median']
    findings.append(('+' if lp>0.5 else('~' if lp>0.3 else '-'), 'Visible' if lp>0.5 else 'Weak'))
    sr = metrics['support_ratio']
    findings.append(('+' if sr>0.6 else '~', f'Support {sr:.0%}'))
    po = abs(metrics['peak_offset_median_ns'])
    findings.append(('+' if po<8 else('~' if po<20 else '-'), f'Align {po:.0f}ns'))
    findings.append(('-' if metrics['has_dominant_X'] else '+', 'No X pattern'))
    for i,(ic,txt) in enumerate(findings):
        ax.text(0.08, 0.80-i*0.10, f'{ic}  {txt}', fontsize=9, transform=ax.transAxes, va='top')
    ax.text(0.08, 0.08, 'Target {:.0f}-{:.0f}ns'.format(metrics['vis_min_ns'],metrics['vis_max_ns']),
            fontsize=8, transform=ax.transAxes, style='italic', color='gray')

    fig.savefig(out_dir / 'qc_preview_full.png', dpi=150)
    plt.close(fig)

    # ── qc_target_zoom.png ──
    v4 = np.percentile(np.abs(t4b), 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    zoom_lo = max(0, search_lo - 20)
    zoom_hi = min(501, search_hi + 20)

    for ax, data, title, cmap, vrange in [
        (axes[0], raw_501, 'Raw', 'gray', (-v_raw, v_raw)),
        (axes[1], t4b, 't⁴ gain', 'gray', (-v4, v4)),
        (axes[2], y_soft, 'y_soft + overlay', 'magma', (0, 1)),
    ]:
        zext = (0, W - 1, t_501[zoom_hi], t_501[zoom_lo])
        ax.imshow(data[zoom_lo:zoom_hi], aspect='auto', cmap=cmap,
                  vmin=vrange[0], vmax=vrange[1], extent=zext)
        ax.plot(traces, target_vis, 'c-', lw=1.5, label='target_visible')
        ax.plot(traces, target_geom, 'b--', lw=1, label='target_geom')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Trace')
        ax.set_ylabel('Time (ns)')
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / 'qc_target_zoom.png', dpi=150)
    plt.close(fig)

    # ── qc_agc_check.png ──
    def safe_agc(x, w=31, f=0.05):
        p = w // 2
        y = np.pad(x * x, ((p, p), (0, 0)), 'reflect')
        r = np.empty_like(x)
        for i in range(x.shape[0]):
            r[i] = np.sqrt(np.mean(y[i:i + w], axis=0) + 1e-12)
        return x / np.maximum(r, np.percentile(r, 10) * f)

    agc5 = safe_agc(raw_501, w=31, f=0.05)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    v_agc = np.percentile(np.abs(agc5), 99)

    for ax, data, title in [
        (axes[0], agc5, 'Safe AGC 5%'),
        (axes[1], agc5[zoom_lo:zoom_hi], 'AGC target zoom'),
        (axes[2], t4b[zoom_lo:zoom_hi], 't⁴ target zoom (ref)'),
    ]:
        if data.shape[0] < 50:
            continue
        d_ext = (0, W - 1, t_501[min(zoom_hi, data.shape[0] + zoom_lo - 1)], t_501[zoom_lo]) if data.shape[0] == zoom_hi - zoom_lo else extent
        ax.imshow(data, aspect='auto', cmap='gray', vmin=-v_agc, vmax=v_agc, extent=d_ext)
        ax.plot(traces, target_vis, 'c-', lw=1.5, label='target_visible')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Trace')

    axes[0].set_ylabel('Time (ns)')
    axes[0].set_ylim(600, 200)
    plt.tight_layout()
    fig.savefig(out_dir / 'qc_agc_check.png', dpi=150)
    plt.close(fig)

    print(f"Figures saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="PGDA_SYNTH_DATASET_V1 — After-run QC")
    ap.add_argument("case_run_dir", help="Path to case run directory (e.g. 03_runs/batch_001/CASE_ID/)")
    ap.add_argument("--bscan", default=None, help="Override bscan path; default: <case_run_dir>/raw/bscan.npy")
    ap.add_argument("--label-dir", default=None, help="Override label dir; default: <case_run_dir>/../labels/ or 01_templates/...")
    ap.add_argument("--out-dir", default=None, help="Override QC output dir; default: 04_qc/{batch_id}/{CASE_ID}/")
    args = ap.parse_args()

    run_dir = Path(args.case_run_dir)
    case_id = run_dir.name
    batch_id = run_dir.parent.name

    # Locate bscan
    bscan_path = Path(args.bscan) if args.bscan else run_dir / 'raw' / 'bscan.npy'
    if not bscan_path.exists():
        print(f"ERROR: bscan not found at {bscan_path}")
        sys.exit(1)

    # Locate labels — search order:
    #   1. --label-dir (CLI override)
    #   2. <case_run_dir>/labels (run_batch copies here)
    #   3. 01_templates/{template}/labels/ via run_info.json
    label_dir = None
    if args.label_dir:
        label_dir = Path(args.label_dir)
    else:
        # Priority 1: local labels copied by run_batch
        local_labels = run_dir / 'labels'
        if local_labels.exists() and any(local_labels.iterdir()):
            label_dir = local_labels
        else:
            # Priority 2: template labels via run_info.json
            run_info = run_dir / 'run_info.json'
            if run_info.exists():
                info = json.loads(run_info.read_text())
                template = info.get('template', '')
                if template:
                    cand = ROOT / '01_templates' / template / 'labels'
                    if cand.exists():
                        label_dir = cand
    if label_dir is None or not label_dir.exists():
        print(f"ERROR: labels not found. Specify --label-dir")
        sys.exit(1)

    # QC output
    if args.out_dir:
        qc_out = Path(args.out_dir)
    else:
        qc_out = ROOT / '04_qc' / batch_id / case_id

    print(f"Running QC for {batch_id}/{case_id}")
    print(f"  B-scan: {bscan_path}")
    print(f"  Labels: {label_dir}")
    print(f"  Output: {qc_out}")

    metrics = compute_qc(bscan_path, label_dir, qc_out)

    print(f"\n{'='*50}")
    print(f"  QC GRADE: {metrics['qc_grade']}")
    print(f"  target_local_peak_median: {metrics['target_local_peak_median']:.3f}")
    print(f"  support_ratio:            {metrics['support_ratio']:.1%}")
    print(f"  peak_offset_median:       {metrics['peak_offset_median_ns']:.1f} ns")
    print(f"  trace_var:                {metrics['trace_var']:.2f}")
    print(f"  dead_trace_ratio:         {metrics['dead_trace_ratio']:.3f}")
    print(f"  has_dominant_X:           {bool(metrics['has_dominant_X'])}")
    print(f"{'='*50}")

    # Write to batch summary if exists
    batch_summary = ROOT / '04_qc' / batch_id / 'batch_summary.csv'
    if not batch_summary.exists():
        batch_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(batch_summary, 'w') as f:
            f.write('case_id,qc_grade,target_local_peak_median,support_ratio,peak_offset_ns,trace_var,dead_trace_ratio,has_dominant_X\n')
    with open(batch_summary, 'a') as f:
        f.write(f'{case_id},{metrics["qc_grade"]},{metrics["target_local_peak_median"]:.4f},{metrics["support_ratio"]:.4f},{metrics["peak_offset_median_ns"]:.2f},{metrics["trace_var"]:.2f},{metrics["dead_trace_ratio"]:.4f},{metrics["has_dominant_X"]}\n')


if __name__ == '__main__':
    main()
