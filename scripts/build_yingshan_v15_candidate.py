from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data_corrected_v1_4_terrain_direction'
CROSSING_CSV = ROOT / 'reports' / 'yingshan_direction_profile_audit' / 'line_intersections.csv'
OUT = ROOT / 'data_yingshan_v15_candidate_20260710'
REPORT = ROOT / 'reports' / 'yingshan_v15_candidate_20260710'
C_M_PER_NS = 0.299792458
LINES = ['Line3', 'Line6', 'Line7', 'Line9', 'LineL1', 'LineX1']
REVIEW_RADIUS_M = 6.0


@dataclass(frozen=True)
class ZeroTimeAudit:
    line: str
    direct_wave_peak_ns: float
    direct_wave_relative_offset_ns: float
    direct_wave_bootstrap_std_ns: float
    surface_warp_phase_offset_ns: float
    surface_warp_peak_score: float
    surface_warp_peak_prominence: float
    note: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def centerline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = mask.shape[0]
    yy = np.arange(h, dtype=np.float32)[:, None]
    mass = mask.sum(axis=0)
    center = (mask * yy).sum(axis=0) / np.maximum(mass, 1e-8)
    valid = mass > 1e-4
    center[~valid] = np.nan
    return center, valid


def direct_wave_peak(raw: np.ndarray, time_ns: np.ndarray) -> float:
    common = np.median(raw.astype(np.float64), axis=1)
    valid = (time_ns >= 4.0) & (time_ns <= 30.0)
    idxs = np.where(valid)[0]
    # Use the dominant signed waveform phase, not the analytic-envelope peak.
    # This is stable across all six survey lines at the 1.4 ns sample scale.
    return float(time_ns[idxs[int(np.argmax(np.abs(common[valid])))]] )


def bootstrap_direct_peak(raw: np.ndarray, time_ns: np.ndarray, *, seed: int = 20260710, n_boot: int = 128) -> float:
    rng = np.random.default_rng(seed)
    width = raw.shape[1]
    peaks = []
    block = max(64, width // 8)
    for _ in range(n_boot):
        start = int(rng.integers(0, max(1, width - block + 1)))
        peaks.append(direct_wave_peak(raw[:, start:start + block], time_ns))
    return float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0


def surface_warp_offset(raw: np.ndarray, time_ns: np.ndarray, height_m: np.ndarray) -> tuple[float, float, float]:
    """Find the earliest strong height-linked surface-response envelope phase.

    The returned value is a signal-domain reference phase, not an instrument
    absolute zero. It is only used for cross-line label QC and candidate hints.
    """
    env = np.abs(hilbert(raw.astype(np.float64), axis=0))
    early = time_ns <= 220.0
    scale = np.percentile(env[early], 95.0, axis=0) + 1e-8
    env = env / scale[None, :]
    offsets = np.arange(-40.0, 10.01, 0.2, dtype=np.float64)
    scores = np.empty(offsets.size, dtype=np.float64)
    trace_idx = np.arange(raw.shape[1])
    dt = float(np.median(np.diff(time_ns)))
    for i, off in enumerate(offsets):
        target = off + 2.0 * height_m / C_M_PER_NS
        sample = np.clip(np.rint(target / dt).astype(np.int64), 0, time_ns.size - 1)
        scores[i] = float(np.median(env[sample, trace_idx]))
    smooth = savgol_filter(scores, 15, 3, mode='interp')
    peaks, props = find_peaks(smooth, prominence=0.005, distance=8)
    if peaks.size:
        # Prefer the strongest peak in this physically constrained early range.
        best_local = int(np.argmax(smooth[peaks]))
        idx = int(peaks[best_local])
        prominence = float(props['prominences'][best_local])
    else:
        idx = int(np.argmax(smooth))
        prominence = 0.0
    return float(offsets[idx]), float(smooth[idx]), prominence


def robust_view(raw: np.ndarray, win: int = 13) -> np.ndarray:
    x = raw.astype(np.float32)
    x = x - np.median(x, axis=1, keepdims=True)
    sq = x * x
    pad = win // 2
    padded = np.pad(sq, ((pad, pad), (0, 0)), mode='edge')
    cs = np.vstack([np.zeros((1, padded.shape[1]), np.float32), np.cumsum(padded, axis=0)])
    rms = np.sqrt((cs[win:] - cs[:-win]) / float(win) + 1e-6)
    return x / rms


def trace_range_for_radius(distance_m: np.ndarray, trace: int, radius_m: float) -> tuple[int, int]:
    center = float(distance_m[trace])
    lo = int(np.searchsorted(distance_m, center - radius_m, side='left'))
    hi = int(np.searchsorted(distance_m, center + radius_m, side='right'))
    return max(0, lo), min(distance_m.size, hi)


def current_label_time(z: dict[str, np.ndarray], trace: int) -> float:
    c, valid = centerline(z['soft_mask_train'].astype(np.float32))
    if not bool(valid[trace]):
        return float('nan')
    return float(c[trace] * float(z['dt_ns']))


def surface_reference_time(z: dict[str, np.ndarray], trace: int, offset_ns: float) -> float:
    return float(offset_ns + 2.0 * float(z['flight_height_agl_m'][trace]) / C_M_PER_NS)


def load_line(line: str) -> dict[str, np.ndarray]:
    with np.load(SOURCE / 'lines' / f'{line}.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def zero_time_audit(line_data: dict[str, dict[str, np.ndarray]]) -> dict[str, ZeroTimeAudit]:
    raw_peaks = {}
    boot_std = {}
    surface = {}
    for line, z in line_data.items():
        raw = z['raw_full_normalized'].astype(np.float32)
        time_ns = z['time_ns'].astype(np.float64)
        raw_peaks[line] = direct_wave_peak(raw, time_ns)
        boot_std[line] = bootstrap_direct_peak(raw, time_ns, seed=20260710 + LINES.index(line))
        surface[line] = surface_warp_offset(raw, time_ns, z['flight_height_agl_m'].astype(np.float64))
    reference = float(np.median(list(raw_peaks.values())))
    out = {}
    for line in LINES:
        off, score, prom = surface[line]
        out[line] = ZeroTimeAudit(
            line=line,
            direct_wave_peak_ns=float(raw_peaks[line]),
            direct_wave_relative_offset_ns=float(raw_peaks[line] - reference),
            direct_wave_bootstrap_std_ns=float(boot_std[line]),
            surface_warp_phase_offset_ns=float(off),
            surface_warp_peak_score=float(score),
            surface_warp_peak_prominence=float(prom),
            note=(
                'Direct-wave relative alignment is the zero-time audit. '
                'Surface-warp phase is a separate signal-domain QC reference and must not be treated as absolute instrument zero.'
            ),
        )
    return out




def local_signal_support(z: dict[str, np.ndarray], trace: int, time_value_ns: float) -> float:
    if not np.isfinite(time_value_ns):
        return float('nan')
    raw = z['raw_full_normalized'].astype(np.float64)
    lo = max(0, trace - 50)
    hi = min(raw.shape[1], trace + 51)
    residual = raw[:, trace] - np.median(raw[:, lo:hi], axis=1)
    env = np.abs(hilbert(residual))
    dt = float(z['dt_ns'])
    idx = int(np.clip(round(float(time_value_ns) / dt), 0, env.size - 1))
    lo_i = max(0, idx - 2)
    hi_i = min(env.size, idx + 3)
    return float(np.max(env[lo_i:hi_i]))

def crossing_decision(row: dict[str, str], line_data: dict[str, dict[str, np.ndarray]], zero: dict[str, ZeroTimeAudit]) -> dict[str, Any]:
    a, b = row['line_a'], row['line_b']
    ta, tb = int(row['trace_a']), int(row['trace_b'])
    za, zb = line_data[a], line_data[b]
    label_a = current_label_time(za, ta)
    label_b = current_label_time(zb, tb)
    surf_a = surface_reference_time(za, ta, zero[a].surface_warp_phase_offset_ns)
    surf_b = surface_reference_time(zb, tb, zero[b].surface_warp_phase_offset_ns)
    delay_a = label_a - surf_a
    delay_b = label_b - surf_b
    diff = abs(delay_a - delay_b)
    status_a = int(za['status_code'][ta])
    status_b = int(zb['status_code'][tb])

    # Direct-wave alignment only removes the measured line-to-line sample offset.
    direct_aligned_diff = abs(
        (label_a - 2.0 * float(za['flight_height_agl_m'][ta]) / C_M_PER_NS - zero[a].direct_wave_relative_offset_ns)
        - (label_b - 2.0 * float(zb['flight_height_agl_m'][tb]) / C_M_PER_NS - zero[b].direct_wave_relative_offset_ns)
    )

    # Instrument/direct-wave alignment plus measured air path is the primary
    # conflict metric. Surface-referenced delay is secondary corroborating QC,
    # because the selected surface phase can change with polarity/material.
    primary_diff = direct_aligned_diff
    affected: list[str] = []
    if primary_diff <= 5.0:
        decision = 'PASS'
    elif primary_diff <= 12.0:
        decision = 'REVIEW_KEEP_EXISTING_WEAK_LABELS'
    elif primary_diff <= 20.0 and diff <= 10.0:
        decision = 'REVIEW_PHASE_REFERENCE_DISAGREEMENT_KEEP'
    elif status_a == 1 and status_b == 2:
        decision = 'IGNORE_WEAK_SIDE_PENDING_REVIEW'
        affected = [b]
    elif status_a == 2 and status_b == 1:
        decision = 'IGNORE_WEAK_SIDE_PENDING_REVIEW'
        affected = [a]
    elif status_a == 2 and status_b == 2:
        decision = 'IGNORE_BOTH_WEAK_SIDES_PENDING_REVIEW'
        affected = [a, b]
    else:
        decision = 'IGNORE_BOTH_STRONG_CONFLICT_PENDING_REVIEW'
        affected = [a, b]

    suggested_a = float('nan')
    suggested_b = float('nan')
    suggestion_basis = ''
    if status_a == 1 and status_b == 2:
        suggested_b = surf_b + delay_a
        suggestion_basis = f'{a} strong label surface-referenced delay'
    elif status_b == 1 and status_a == 2:
        suggested_a = surf_a + delay_b
        suggestion_basis = f'{b} strong label surface-referenced delay'

    support_current_a = local_signal_support(za, ta, label_a)
    support_current_b = local_signal_support(zb, tb, label_b)
    support_suggested_a = local_signal_support(za, ta, suggested_a)
    support_suggested_b = local_signal_support(zb, tb, suggested_b)
    ratio_a = support_suggested_a / max(support_current_a, 1e-12) if np.isfinite(support_suggested_a) else float('nan')
    ratio_b = support_suggested_b / max(support_current_b, 1e-12) if np.isfinite(support_suggested_b) else float('nan')
    finite_ratios = [x for x in (ratio_a, ratio_b) if np.isfinite(x)]
    if not finite_ratios:
        suggestion_grade = 'not_applicable'
    elif min(finite_ratios) >= 0.8:
        suggestion_grade = 'signal_supported_for_manual_review'
    elif min(finite_ratios) >= 0.5:
        suggestion_grade = 'weak_signal_support_review_only'
    else:
        suggestion_grade = 'poor_signal_support_do_not_apply'

    return {
        'crossing': row['crossing'],
        'line_a': a,
        'trace_a': ta,
        'status_a': status_a,
        'label_time_a_ns': label_a,
        'direct_wave_peak_a_ns': zero[a].direct_wave_peak_ns,
        'surface_reference_a_ns': surf_a,
        'surface_referenced_delay_a_ns': delay_a,
        'suggested_label_time_a_ns': suggested_a,
        'line_b': b,
        'trace_b': tb,
        'status_b': status_b,
        'label_time_b_ns': label_b,
        'direct_wave_peak_b_ns': zero[b].direct_wave_peak_ns,
        'surface_reference_b_ns': surf_b,
        'surface_referenced_delay_b_ns': delay_b,
        'suggested_label_time_b_ns': suggested_b,
        'direct_wave_aligned_air_corrected_difference_ns': direct_aligned_diff,
        'surface_referenced_difference_ns': diff,
        'primary_conflict_metric': 'direct_wave_aligned_air_corrected_difference_ns',
        'decision_v15_candidate': decision,
        'affected_lines': ';'.join(affected),
        'review_radius_m': REVIEW_RADIUS_M,
        'suggestion_basis': suggestion_basis,
        'current_signal_support_a': support_current_a,
        'suggested_signal_support_a': support_suggested_a,
        'suggested_to_current_support_ratio_a': ratio_a,
        'current_signal_support_b': support_current_b,
        'suggested_signal_support_b': support_suggested_b,
        'suggested_to_current_support_ratio_b': ratio_b,
        'suggestion_signal_grade': suggestion_grade,
        'nearest_separation_m': float(row['nearest_separation_m']),
        'note': 'Candidate suggestions are review aids only; no automatic centerline relocation is applied.',
    }


def make_crossing_workbench(
    decision: dict[str, Any],
    line_data: dict[str, dict[str, np.ndarray]],
    zero: dict[str, ZeroTimeAudit],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
    for col, side in enumerate(('a', 'b')):
        line = str(decision[f'line_{side}'])
        trace = int(decision[f'trace_{side}'])
        z = line_data[line]
        raw = z['raw_full_normalized'].astype(np.float32)
        time_ns = z['time_ns'].astype(np.float64)
        distance = z['gnss_cumulative_distance_m'].astype(np.float64)
        lo, hi = trace_range_for_radius(distance, trace, 10.0)
        sub = raw[:, lo:hi]
        processed = robust_view(sub)
        c, valid = centerline(z['soft_mask_train'].astype(np.float32))
        local_dist = distance[lo:hi] - distance[trace]
        center_ns = c[lo:hi] * float(z['dt_ns'])
        status = z['status_code'][lo:hi].astype(np.int16)
        vmax = max(float(np.percentile(np.abs(sub), 99.0)), 1e-6)
        pmax = max(float(np.percentile(np.abs(processed), 99.0)), 1e-6)

        ax = axes[0, col]
        ax.imshow(sub, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
                  extent=(local_dist[0], local_dist[-1], time_ns[-1], time_ns[0]))
        ax.plot(local_dist[valid[lo:hi]], center_ns[valid[lo:hi]], lw=1.3, label='V14 center')
        ax.axvline(0.0, ls='--', lw=1)
        ax.set_title(f'{line} raw | trace {trace} | status {int(z["status_code"][trace])}')
        ax.set_ylabel('time (ns)')
        ax.legend(loc='upper right')

        ax = axes[1, col]
        ax.imshow(processed, aspect='auto', cmap='gray', vmin=-pmax, vmax=pmax,
                  extent=(local_dist[0], local_dist[-1], time_ns[-1], time_ns[0]))
        # Strong and weak labels are separated visually.
        strong = status == 1
        weak = status == 2
        if strong.any():
            ax.plot(local_dist[strong], center_ns[strong], lw=1.5, label='strong')
        if weak.any():
            ax.plot(local_dist[weak], center_ns[weak], lw=1.2, ls='--', label='weak')
        ax.axvline(0.0, ls='--', lw=1)
        suggested = float(decision[f'suggested_label_time_{side}_ns'])
        if np.isfinite(suggested):
            ax.axhline(suggested, ls=':', lw=1.5, label='cross-line suggestion')
        ax.set_title('background-suppressed + AGC(13)')
        ax.set_ylabel('time (ns)')
        ax.legend(loc='upper right')

        trace_raw = raw[:, trace].astype(np.float64)
        env = np.abs(hilbert(trace_raw))
        ax = axes[2, col]
        scale = max(float(np.max(np.abs(trace_raw))), 1e-8)
        env_scale = max(float(np.max(env)), 1e-8)
        ax.plot(time_ns, trace_raw / scale, lw=0.9, label='A-scan')
        ax.plot(time_ns, env / env_scale, lw=1.0, label='envelope')
        ax.axvline(float(decision[f'label_time_{side}_ns']), lw=1.5, label='current label')
        ax.axvline(float(decision[f'direct_wave_peak_{side}_ns']), ls='--', lw=1.0, label='direct peak')
        ax.axvline(float(decision[f'surface_reference_{side}_ns']), ls=':', lw=1.5, label='surface reference')
        if np.isfinite(suggested):
            ax.axvline(suggested, ls='-.', lw=1.4, label='suggested review time')
        ax.set_xlim(0, 550)
        ax.set_title('crossing A-scan signal references')
        ax.set_xlabel('time (ns)')
        ax.legend(loc='upper right', fontsize=8)

        ax = axes[3, col]
        ground = z['ground_elevation_m'][lo:hi].astype(np.float64)
        flight = z['flight_height_agl_m'][lo:hi].astype(np.float64)
        ax.plot(local_dist, ground - np.median(ground), label='ground elevation (relative m)')
        ax.plot(local_dist, flight, label='flight height AGL (m)')
        ax.axvline(0.0, ls='--', lw=1)
        ax.set_title(
            f'direct={zero[line].direct_wave_peak_ns:.1f} ns; '
            f'surface phase offset={zero[line].surface_warp_phase_offset_ns:.1f} ns'
        )
        ax.set_xlabel('GNSS distance from crossing (m)')
        ax.legend(loc='best', fontsize=8)

    fig.suptitle(
        f"{decision['crossing']} | {decision['decision_v15_candidate']} | "
        f"surface-ref diff={decision['surface_referenced_difference_ns']:.2f} ns | "
        f"direct/air diff={decision['direct_wave_aligned_air_corrected_difference_ns']:.2f} ns",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def apply_candidate_policy(
    line_data: dict[str, dict[str, np.ndarray]],
    decisions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    outputs = {}
    audit_rows: list[dict[str, Any]] = []
    for line, z in line_data.items():
        arrays = {k: np.array(v, copy=True) for k, v in z.items()}
        review_mask = arrays['soft_mask_train'].astype(np.float32).copy()
        train_mask = review_mask.copy()
        status = arrays['status_code'].astype(np.int16).copy()
        weight = arrays['label_weight'].astype(np.float32).copy()
        ignore = np.zeros_like(train_mask, dtype=np.float32)
        decision_code = np.array(['KEEP_V14_PENDING_GLOBAL_REVIEW'] * train_mask.shape[1], dtype='U64')
        review_reason = np.array([''] * train_mask.shape[1], dtype='U160')

        for dec in decisions:
            affected = [x for x in str(dec['affected_lines']).split(';') if x]
            if line not in affected:
                continue
            side = 'a' if dec['line_a'] == line else 'b'
            trace = int(dec[f'trace_{side}'])
            lo, hi = trace_range_for_radius(arrays['gnss_cumulative_distance_m'].astype(np.float64), trace, REVIEW_RADIUS_M)
            ignore[:, lo:hi] = 1.0
            train_mask[:, lo:hi] = 0.0
            status[lo:hi] = 2
            weight[lo:hi] = 0.0
            code = str(dec['decision_v15_candidate'])
            reason = f"{dec['crossing']}: {code}; surface-ref diff={dec['surface_referenced_difference_ns']:.2f} ns"
            decision_code[lo:hi] = 'IGNORE_CROSSING_PENDING_MANUAL_REVIEW'
            review_reason[lo:hi] = reason
            audit_rows.append({
                'line': line,
                'crossing': dec['crossing'],
                'center_trace': trace,
                'trace_start': lo,
                'trace_end_inclusive': hi - 1,
                'gnss_radius_m': REVIEW_RADIUS_M,
                'decision': code,
                'reason': reason,
            })

        arrays['soft_mask_review_v15'] = review_mask
        arrays['soft_mask_train'] = train_mask
        arrays['ignore_mask'] = ignore
        arrays['status_code'] = status
        arrays['label_weight'] = weight
        arrays['v15_decision_code'] = decision_code
        arrays['v15_review_reason'] = review_reason
        arrays['v15_candidate_version'] = np.array('YINGSHAN_V15_CANDIDATE_20260710')
        arrays['v15_training_policy'] = np.array(
            'V14 centerline retained; unresolved high-risk crossing neighborhoods are excluded from all label losses pending manual review.'
        )
        outputs[line] = arrays
    return outputs, audit_rows


def window_starts(width: int, window: int = 256, stride: int = 128) -> list[int]:
    starts = list(range(0, max(1, width - window + 1), stride))
    last = max(0, width - window)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def write_dataset(outputs: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'lines').mkdir(parents=True)
    (OUT / 'windows').mkdir()
    (OUT / 'manifests').mkdir()
    for feature_dir in ('terrain_features', 'terrain_features_zero_material_v1'):
        src = SOURCE / feature_dir
        if src.exists():
            shutil.copytree(src, OUT / feature_dir)

    rows = []
    for line in LINES:
        arrays = outputs[line]
        np.savez_compressed(OUT / 'lines' / f'{line}.npz', **arrays)
        width = int(arrays['raw_full_normalized'].shape[1])
        for start in window_starts(width):
            end = start + 256
            sl = slice(start, end)
            sample_id = f'{line}_tr{start:04d}_{end-1:04d}'
            ignore_col = arrays['ignore_mask'][:, sl].mean(axis=0) > 0.5
            active = ~ignore_col
            status = arrays['status_code'][sl]
            np.savez_compressed(
                OUT / 'windows' / f'{sample_id}.npz',
                x_raw=arrays['raw_full_normalized'][:, sl].astype(np.float32),
                y_mask=arrays['soft_mask_train'][:, sl].astype(np.float32),
                status_code=status.astype(np.int16),
                label_weight=arrays['label_weight'][sl].astype(np.float32),
                ignore_mask=arrays['ignore_mask'][:, sl].astype(np.float32),
                line=np.array(line),
                start_trace=np.array(start, np.int32),
                end_trace=np.array(end - 1, np.int32),
            )
            rows.append({
                'sample_id': sample_id,
                'line': line,
                'start': start,
                'end': end - 1,
                'split': str(arrays['split']),
                'present': int(((status == 1) & active).sum()),
                'weak': int(((status == 2) & active).sum()),
                'no_pick': int(((status == 0) & active).sum()),
                'ignore': int(ignore_col.sum()),
                'source_line_path': f'lines/{line}.npz',
                'label_version': 'YINGSHAN_V15_CANDIDATE_20260710',
            })
    write_csv(OUT / 'window_index.csv', rows)
    return rows


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    workbench = REPORT / 'crossing_workbench'
    if workbench.exists():
        shutil.rmtree(workbench)
    workbench.mkdir(parents=True)

    line_data = {line: load_line(line) for line in LINES}
    zero = zero_time_audit(line_data)
    zero_rows = [asdict(zero[line]) for line in LINES]
    write_csv(REPORT / 'zero_time_audit.csv', zero_rows)

    source_rows = list(csv.DictReader(CROSSING_CSV.open(encoding='utf-8')))
    decisions = [crossing_decision(row, line_data, zero) for row in source_rows]
    write_csv(REPORT / 'crossing_v15_decisions.csv', decisions)
    for i, decision in enumerate(decisions, start=1):
        make_crossing_workbench(
            decision,
            line_data,
            zero,
            workbench / f'{i:02d}_{decision["crossing"]}.png',
        )

    outputs, ignored_rows = apply_candidate_policy(line_data, decisions)
    write_csv(REPORT / 'v15_ignored_crossing_segments.csv', ignored_rows)
    window_rows = write_dataset(outputs)

    line_summary = []
    for line in LINES:
        arr = outputs[line]
        ignore_cols = arr['ignore_mask'].mean(axis=0) > 0.5
        line_summary.append({
            'line': line,
            'trace_count': int(arr['raw_full_normalized'].shape[1]),
            'strong_active': int(((arr['status_code'] == 1) & ~ignore_cols).sum()),
            'weak_active': int(((arr['status_code'] == 2) & ~ignore_cols).sum()),
            'ignored_traces': int(ignore_cols.sum()),
            'label_geometry_changed': False,
            'formal_training_allowed': False,
        })
    write_csv(OUT / 'manifests' / 'v15_line_summary.csv', line_summary)

    critical = [d for d in decisions if str(d['affected_lines']).strip()]
    manifest = {
        'dataset': str(OUT.relative_to(ROOT)),
        'version': 'YINGSHAN_V15_CANDIDATE_20260710',
        'source_dataset': str(SOURCE.relative_to(ROOT)),
        'source_crossing_registry': str(CROSSING_CSV.relative_to(ROOT)),
        'source_crossing_registry_sha256': _sha256(CROSSING_CSV),
        'direct_wave_zero_time_policy': (
            'Estimate relative line alignment from the common direct-wave envelope peak. '
            'No whole-line label translation is applied because the measured offsets are at most one 1.4 ns sample.'
        ),
        'surface_reference_policy': (
            'Height-warped surface response is used only as a cross-line QC reference, not as absolute instrument zero or geological depth.'
        ),
        'label_policy': (
            'Retain V14 centerline geometry. Exclude unresolved high-risk crossing neighborhoods with ignore_mask and zero label weight. '
            'Cross-line suggested times are review aids and are not applied automatically.'
        ),
        'review_radius_m': REVIEW_RADIUS_M,
        'critical_crossing_count': len(critical),
        'formal_training_allowed': False,
        'blockers': [
            'Manual review of ignored crossing neighborhoods is incomplete.',
            'Confirmed true-negative measured windows remain unavailable.',
            'Formal non-Line9-conditioned simulations remain unavailable.',
        ],
        'outputs': {
            'zero_time_audit': str((REPORT / 'zero_time_audit.csv').relative_to(ROOT)),
            'crossing_decisions': str((REPORT / 'crossing_v15_decisions.csv').relative_to(ROOT)),
            'ignored_segments': str((REPORT / 'v15_ignored_crossing_segments.csv').relative_to(ROOT)),
            'workbench': str(workbench.relative_to(ROOT)),
            'window_index': str((OUT / 'window_index.csv').relative_to(ROOT)),
        },
    }
    (OUT / 'manifests' / 'v15_candidate_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (REPORT / 'audit_summary.json').write_text(
        json.dumps({'zero_time': zero_rows, 'crossings': decisions, 'line_summary': line_summary}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (OUT / 'DATASET_README.md').write_text(
        '# YingShan V15 Candidate Labels\n\n'
        'This is a conservative review dataset, not a formally released training dataset.\n\n'
        '- Canonical trace order and V14 centerline geometry are preserved.\n'
        '- `soft_mask_review_v15` stores the unchanged V14 review geometry.\n'
        '- `soft_mask_train` is zeroed only in unresolved crossing neighborhoods.\n'
        '- `ignore_mask` excludes those pixels from segmentation and curve losses.\n'
        '- Affected traces are weak-status with zero label weight, so they do not supervise presence.\n'
        '- Candidate cross-line times are written only to the audit CSV/workbench and are not applied automatically.\n',
        encoding='utf-8',
    )
    print(OUT)
    print(REPORT)
    print(f'windows={len(window_rows)} critical_crossings={len(critical)}')


if __name__ == '__main__':
    main()
