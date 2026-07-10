from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, savgol_filter

from build_yingshan_v15_candidate import (
    C_M_PER_NS,
    CROSSING_CSV,
    LINES,
    ROOT,
    SOURCE,
    centerline,
    load_line,
    robust_view,
    trace_range_for_radius,
    window_starts,
    write_csv,
    zero_time_audit,
)

OUT = ROOT / 'data_yingshan_v15_final_20260710'
REPORT = ROOT / 'reports' / 'yingshan_v15_final_20260710'
VERSION = 'YINGSHAN_V15_FINAL_20260710'
SIGMA_NS = 8.0

# These are the visual decisions accepted in the interactive audit.
# The trusted Line9/L1 anchor is used only to define a local weak-label target;
# it never modifies the trusted anchor line itself.
RELABEL_RULES: tuple[dict[str, Any], ...] = (
    {
        'crossing': 'Line3-Line9',
        'line': 'Line3',
        'center_trace': 167,
        'radius_m': 6.0,
        'core_radius_m': 4.0,
        'target_time_ns': 453.013482,
        'target_band_ns': 10.0,
        'confidence_cap': 0.42,
        'trusted_anchor': 'Line9',
        'decision_code': 'RELABEL_WEAK_LINE9_ANCHORED',
        'rationale': (
            'Line9 is the highest-quality survey line. The deeper Line3 event near 450-455 ns is more laterally continuous '
            'and cross-line consistent than the prior ~396 ns weak label. The relabel remains weak; outer transition collars are ignored.'
        ),
        'ignore_transition_collar': True,
    },
    {
        'crossing': 'LineL1-LineX1',
        'line': 'LineX1',
        'center_trace': 659,
        'radius_m': 6.0,
        'core_radius_m': 4.0,
        'target_time_ns': 327.528667,
        'target_band_ns': 10.0,
        'confidence_cap': 0.48,
        'trusted_anchor': 'LineL1',
        'decision_code': 'RELABEL_WEAK_L1_ANCHORED',
        'rationale': (
            'The ~327-330 ns X1 event connects smoothly to adjacent labels and matches the strong L1 crossing anchor. '
            'X1 remains review-only and the relabel remains weak.'
        ),
        'ignore_transition_collar': False,
    },
)

IGNORE_RULES: tuple[dict[str, Any], ...] = (
    {
        'crossing': 'Line6-Line9',
        'line': 'Line6',
        'center_trace': 523,
        'radius_m': 6.0,
        'decision_code': 'IGNORE_LINE6_AMBIGUOUS_KEEP_LINE9',
        'rationale': (
            'Line9 current label is visually the more reliable event. The Line6 crossing event may be a shallower interface or adjacent phase; '
            'no defensible replacement curve is available, so Line6 supervision is excluded locally.'
        ),
    },
    {
        'crossing': 'Line9-LineX1',
        'line': 'LineX1',
        'center_trace': 197,
        'radius_m': 6.0,
        'decision_code': 'IGNORE_X1_AMBIGUOUS_KEEP_LINE9',
        'rationale': (
            'Line9 is retained as the trusted reference. X1 contains multiple plausible events and remains review-only; '
            'the crossing neighborhood is excluded from supervised losses.'
        ),
    },
)

FINAL_CROSSING_DECISIONS: dict[str, str] = {
    'Line3-Line9': 'RELABEL_LINE3_WEAK_TO_LINE9_ANCHOR',
    'Line3-LineL1': 'KEEP_EXISTING_WEAK_LABELS',
    'Line3-Line7': 'PASS_KEEP',
    'Line6-Line9': 'KEEP_LINE9_IGNORE_LINE6_AMBIGUOUS',
    'Line6-LineL1': 'KEEP_EXISTING_SURFACE_REFERENCE_CONSISTENT',
    'Line6-Line7': 'PASS_KEEP',
    'Line9-LineX1': 'KEEP_LINE9_IGNORE_X1_REVIEW_ONLY',
    'LineL1-LineX1': 'RELABEL_X1_WEAK_TO_L1_ANCHOR',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def gaussian_columns(path_samples: np.ndarray, confidence: np.ndarray, height: int, sigma_samples: float) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32)[:, None]
    mask = np.exp(-0.5 * ((yy - path_samples[None, :]) / float(sigma_samples)) ** 2)
    return (mask * confidence[None, :]).astype(np.float32)


def cosine_blend(distance_m: np.ndarray, center_distance_m: float, core_radius_m: float, radius_m: float) -> np.ndarray:
    delta = np.abs(distance_m - center_distance_m)
    alpha = np.zeros_like(delta, dtype=np.float64)
    alpha[delta <= core_radius_m] = 1.0
    transition = (delta > core_radius_m) & (delta < radius_m)
    u = (radius_m - delta[transition]) / max(radius_m - core_radius_m, 1e-6)
    alpha[transition] = 0.5 - 0.5 * np.cos(np.pi * u)
    return alpha.astype(np.float32)


def track_ridge_near_target(
    z: dict[str, np.ndarray],
    lo: int,
    hi: int,
    target_time_ns: float,
    *,
    band_ns: float,
    max_step_samples: int = 2,
    target_penalty: float = 2.0,
) -> np.ndarray:
    """Track a smooth envelope ridge close to a manually accepted time target.

    The target penalty is deliberately strong: this is a visual/manual relabel,
    not an unconstrained automatic repick. The signal term only snaps the curve
    to the nearest locally supported phase within the accepted narrow band.
    """
    raw = z['raw_full_normalized'][:, lo:hi].astype(np.float64)
    dt = float(z['dt_ns'])
    width = raw.shape[1]
    processed = robust_view(raw.astype(np.float32))
    envelope = np.abs(hilbert(processed.astype(np.float64), axis=0))

    target_samples = np.full(width, float(target_time_ns) / dt, dtype=np.float64)
    band_samples = max(2.0, float(band_ns) / dt)
    sample_lo = max(0, int(math.floor(float(target_samples.min() - band_samples))))
    sample_hi = min(envelope.shape[0] - 1, int(math.ceil(float(target_samples.max() + band_samples))))
    samples = np.arange(sample_lo, sample_hi + 1, dtype=np.int32)

    local = envelope[sample_lo:sample_hi + 1]
    scale = np.percentile(local, 95.0, axis=0) + 1e-6
    signal = local / scale[None, :]
    target_cost = ((samples[:, None] - target_samples[None, :]) / band_samples) ** 2
    score = signal - float(target_penalty) * target_cost

    n_state = samples.size
    dp = np.full((n_state, width), -1e12, dtype=np.float64)
    prev = np.full((n_state, width), -1, dtype=np.int16)
    dp[:, 0] = score[:, 0]
    for x in range(1, width):
        for state in range(n_state):
            a = max(0, state - max_step_samples)
            b = min(n_state, state + max_step_samples + 1)
            candidates = dp[a:b, x - 1] - 0.06 * np.abs(np.arange(a, b) - state)
            local_idx = int(np.argmax(candidates))
            prev[state, x] = np.int16(a + local_idx)
            dp[state, x] = score[state, x] + candidates[local_idx]

    state = int(np.argmax(dp[:, -1]))
    path = np.empty(width, dtype=np.float64)
    path[-1] = samples[state]
    for x in range(width - 1, 0, -1):
        state = int(prev[state, x])
        path[x - 1] = samples[state]

    smooth_window = min(21, width if width % 2 else width - 1)
    if smooth_window >= 7:
        path = savgol_filter(path, smooth_window, 2, mode='interp')
    return (path * dt).astype(np.float32)


def apply_final_policy(
    line_data: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]]]:
    outputs: dict[str, dict[str, np.ndarray]] = {}
    segment_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for line in LINES:
        z = line_data[line]
        arrays = {k: np.array(v, copy=True) for k, v in z.items()}
        old_mask = arrays['soft_mask_train'].astype(np.float32).copy()
        old_status = arrays['status_code'].astype(np.int16).copy()
        old_weight = arrays['label_weight'].astype(np.float32).copy()
        old_path_samples, old_valid = centerline(old_mask)
        dt = float(arrays['dt_ns'])
        old_time = old_path_samples * dt
        height, width = old_mask.shape

        review_mask = old_mask.copy()
        train_mask = old_mask.copy()
        status = old_status.copy()
        weight = old_weight.copy()
        ignore = np.zeros_like(old_mask, dtype=np.float32)
        final_time = old_time.astype(np.float32).copy()
        changed = np.zeros(width, dtype=np.uint8)
        decision_code = np.array(['KEEP_V14_AS_V15_FINAL'] * width, dtype='U80')
        review_reason = np.array([''] * width, dtype='U240')
        distance = arrays['gnss_cumulative_distance_m'].astype(np.float64)
        sigma_samples = max(3, int(round(SIGMA_NS / dt)))

        for rule in RELABEL_RULES:
            if rule['line'] != line:
                continue
            center_trace = int(rule['center_trace'])
            radius = float(rule['radius_m'])
            core_radius = float(rule['core_radius_m'])
            lo, hi = trace_range_for_radius(distance, center_trace, radius)
            candidate_time = track_ridge_near_target(
                arrays,
                lo,
                hi,
                float(rule['target_time_ns']),
                band_ns=float(rule['target_band_ns']),
            )
            alpha = cosine_blend(distance[lo:hi], float(distance[center_trace]), core_radius, radius)
            blended_time = (1.0 - alpha) * old_time[lo:hi] + alpha * candidate_time
            final_time[lo:hi] = blended_time.astype(np.float32)
            changed[lo:hi] = (np.abs(blended_time - old_time[lo:hi]) > 0.25).astype(np.uint8)

            region_conf = np.minimum(old_weight[lo:hi], float(rule['confidence_cap'])).astype(np.float32)
            region_conf = np.maximum(region_conf, 0.34).astype(np.float32)
            new_columns = gaussian_columns(blended_time / dt, region_conf, height, sigma_samples)
            review_mask[:, lo:hi] = new_columns
            train_mask[:, lo:hi] = new_columns
            status[lo:hi] = 2
            weight[lo:hi] = region_conf
            decision_code[lo:hi] = str(rule['decision_code'])
            review_reason[lo:hi] = str(rule['rationale'])

            core = np.abs(distance[lo:hi] - float(distance[center_trace])) <= core_radius
            if bool(rule['ignore_transition_collar']):
                collar = ~core
                if collar.any():
                    global_cols = np.arange(lo, hi)[collar]
                    ignore[:, global_cols] = 1.0
                    train_mask[:, global_cols] = 0.0
                    status[global_cols] = 2
                    weight[global_cols] = 0.0
                    decision_code[global_cols] = 'IGNORE_RELABEL_TRANSITION_COLLAR'
                    review_reason[global_cols] = (
                        f"{rule['crossing']}: transition between V14 and accepted weak V15 ridge is retained for review but excluded from losses."
                    )

            segment_rows.append({
                'line': line,
                'crossing': rule['crossing'],
                'action': 'weak_relabel',
                'center_trace': center_trace,
                'trace_start': lo,
                'trace_end_inclusive': hi - 1,
                'core_radius_m': core_radius,
                'outer_radius_m': radius,
                'trusted_anchor': rule['trusted_anchor'],
                'manual_target_time_ns': float(rule['target_time_ns']),
                'old_center_time_ns': float(old_time[center_trace]),
                'final_center_time_ns': float(final_time[center_trace]),
                'center_shift_ns': float(final_time[center_trace] - old_time[center_trace]),
                'confidence_cap': float(rule['confidence_cap']),
                'decision_code': rule['decision_code'],
                'rationale': rule['rationale'],
            })

        for rule in IGNORE_RULES:
            if rule['line'] != line:
                continue
            center_trace = int(rule['center_trace'])
            lo, hi = trace_range_for_radius(distance, center_trace, float(rule['radius_m']))
            ignore[:, lo:hi] = 1.0
            train_mask[:, lo:hi] = 0.0
            status[lo:hi] = 2
            weight[lo:hi] = 0.0
            decision_code[lo:hi] = str(rule['decision_code'])
            review_reason[lo:hi] = str(rule['rationale'])
            segment_rows.append({
                'line': line,
                'crossing': rule['crossing'],
                'action': 'ignore_unresolved',
                'center_trace': center_trace,
                'trace_start': lo,
                'trace_end_inclusive': hi - 1,
                'core_radius_m': '',
                'outer_radius_m': float(rule['radius_m']),
                'trusted_anchor': 'Line9',
                'manual_target_time_ns': '',
                'old_center_time_ns': float(old_time[center_trace]),
                'final_center_time_ns': float(final_time[center_trace]),
                'center_shift_ns': 0.0,
                'confidence_cap': 0.0,
                'decision_code': rule['decision_code'],
                'rationale': rule['rationale'],
            })

        ignored_cols = ignore.mean(axis=0) > 0.5
        for trace in np.where((changed > 0) | ignored_cols)[0]:
            trace_rows.append({
                'line': line,
                'trace': int(trace),
                'gnss_distance_m': float(distance[trace]),
                'old_time_ns': float(old_time[trace]) if bool(old_valid[trace]) else float('nan'),
                'final_time_ns': float(final_time[trace]) if np.isfinite(final_time[trace]) else float('nan'),
                'delta_ns': float(final_time[trace] - old_time[trace]) if bool(old_valid[trace]) else float('nan'),
                'old_status': int(old_status[trace]),
                'final_status': int(status[trace]),
                'old_weight': float(old_weight[trace]),
                'final_weight': float(weight[trace]),
                'ignored': bool(ignored_cols[trace]),
                'decision_code': str(decision_code[trace]),
                'reason': str(review_reason[trace]),
            })

        arrays['soft_mask_v14_original'] = old_mask
        arrays['soft_mask_review_v15_final'] = review_mask.astype(np.float32)
        arrays['soft_mask_train'] = train_mask.astype(np.float32)
        arrays['ignore_mask'] = ignore.astype(np.float32)
        arrays['status_code_v14_original'] = old_status
        arrays['label_weight_v14_original'] = old_weight
        arrays['status_code'] = status.astype(np.int16)
        arrays['label_weight'] = weight.astype(np.float32)
        arrays['v15_final_center_time_ns'] = final_time.astype(np.float32)
        arrays['v15_final_changed_trace'] = changed.astype(np.uint8)
        arrays['v15_final_ignore_trace'] = ignored_cols.astype(np.uint8)
        arrays['v15_final_decision_code'] = decision_code
        arrays['v15_final_review_reason'] = review_reason
        arrays['v15_final_version'] = np.array(VERSION)
        arrays['v15_final_label_semantics'] = np.array(
            'visible-phase centerline; Line9 is the primary crossing anchor; accepted cross-line relocations remain weak; unresolved ambiguity is excluded.'
        )
        arrays['v15_final_release_status'] = np.array('final_label_release_not_formal_training_release')
        arrays['label_source'] = np.array(
            'V14 source labels plus user-accepted visual crossing decisions recorded in reports/yingshan_v15_final_20260710.'
        )
        outputs[line] = arrays

    return outputs, segment_rows, trace_rows


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
    for name in ('trace_direction_registry.csv', 'orientation_contract.json'):
        src = SOURCE / name
        if src.exists():
            shutil.copy2(src, OUT / name)

    with (SOURCE / 'window_index.csv').open(encoding='utf-8', newline='') as handle:
        source_index = {row['sample_id']: row for row in csv.DictReader(handle)}

    rows: list[dict[str, Any]] = []
    for line in LINES:
        arrays = outputs[line]
        np.savez_compressed(OUT / 'lines' / f'{line}.npz', **arrays)
        width = int(arrays['raw_full_normalized'].shape[1])
        for start in window_starts(width):
            end = start + 256
            sl = slice(start, end)
            sample_id = f'{line}_tr{start:04d}_{end-1:04d}'
            ignore_col = arrays['v15_final_ignore_trace'][sl].astype(bool)
            changed_col = arrays['v15_final_changed_trace'][sl].astype(bool)
            active = ~ignore_col
            status = arrays['status_code'][sl]
            np.savez_compressed(
                OUT / 'windows' / f'{sample_id}.npz',
                x_raw=arrays['raw_full_normalized'][:, sl].astype(np.float32),
                y_mask=arrays['soft_mask_train'][:, sl].astype(np.float32),
                y_mask_review_v15_final=arrays['soft_mask_review_v15_final'][:, sl].astype(np.float32),
                y_mask_v14_original=arrays['soft_mask_v14_original'][:, sl].astype(np.float32),
                status_code=status.astype(np.int16),
                label_weight=arrays['label_weight'][sl].astype(np.float32),
                ignore_mask=arrays['ignore_mask'][:, sl].astype(np.float32),
                v15_final_changed_trace=changed_col.astype(np.uint8),
                v15_final_decision_code=arrays['v15_final_decision_code'][sl],
                line=np.array(line),
                start_trace=np.array(start, np.int32),
                end_trace=np.array(end - 1, np.int32),
                label_version=np.array(VERSION),
            )
            base = dict(source_index.get(sample_id, {}))
            if not base:
                base = {
                    'sample_id': sample_id,
                    'line': line,
                    'start': start,
                    'end': end - 1,
                    'split': str(arrays['split']),
                }
            base.update({
                'present': int(((status == 1) & active).sum()),
                'weak': int(((status == 2) & active).sum()),
                'no_pick': int(((status == 0) & active).sum()),
                'ignore': int(ignore_col.sum()),
                'relabelled': int((changed_col & active).sum()),
                'source_line_path': f'lines/{line}.npz',
                'label_version': VERSION,
            })
            rows.append(base)
    write_csv(OUT / 'window_index.csv', rows)
    return rows


def make_change_preview(outputs: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    panels = [
        ('Line3', 167, 'Line3-Line9: accepted weak relabel'),
        ('Line6', 523, 'Line6-Line9: Line6 ignored, Line9 retained'),
        ('LineX1', 197, 'Line9-LineX1: X1 ignored, Line9 retained'),
        ('LineX1', 659, 'LineL1-LineX1: accepted weak relabel'),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), constrained_layout=True)
    for ax, (line, trace, title) in zip(axes, panels):
        arr = outputs[line]
        distance = arr['gnss_cumulative_distance_m'].astype(np.float64)
        lo, hi = trace_range_for_radius(distance, trace, 10.0)
        raw = arr['raw_full_normalized'][:, lo:hi].astype(np.float32)
        view = robust_view(raw)
        scale = max(float(np.percentile(np.abs(view), 99.0)), 1e-6)
        time_ns = arr['time_ns'].astype(np.float64)
        x = distance[lo:hi] - distance[trace]
        old_c, old_v = centerline(arr['soft_mask_v14_original'][:, lo:hi].astype(np.float32))
        new_c, new_v = centerline(arr['soft_mask_review_v15_final'][:, lo:hi].astype(np.float32))
        ax.imshow(view, aspect='auto', cmap='gray', vmin=-scale, vmax=scale, extent=(x[0], x[-1], time_ns[-1], time_ns[0]))
        ax.plot(x[old_v], old_c[old_v] * float(arr['dt_ns']), ls='--', lw=1.3, label='V14')
        ax.plot(x[new_v], new_c[new_v] * float(arr['dt_ns']), lw=1.7, label='V15 final review geometry')
        ignored = arr['v15_final_ignore_trace'][lo:hi].astype(bool)
        if ignored.any():
            ax.fill_between(x, time_ns[0], time_ns[-1], where=ignored, alpha=0.15, label='ignored from losses')
        ax.axvline(0.0, ls=':', lw=1.0)
        ax.set_ylim(530, 250)
        ax.set_title(title)
        ax.set_xlabel('GNSS distance from crossing (m)')
        ax.set_ylabel('time (ns)')
        ax.legend(loc='best')
    fig.suptitle('YingShan V15 final crossing decisions', fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def update_contract_files() -> None:
    contract_path = ROOT / 'data' / 'dataset_contract_v2' / 'dataset_manifest.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract['audit_commit'] = '799f229 (source candidate; final release is the next git commit)'
    contract['formal_training_allowed'] = False
    contract['label_semantics'] = 'visible_phase_v15_final_with_weak_crossing_relabels_and_explicit_ignore'
    contract['current_real_label_dataset'] = {
        'path': str(OUT.relative_to(ROOT)),
        'version': VERSION,
        'release_status': 'final_label_release_not_formal_training_release',
        'line9_policy': 'Line9 preserved as primary crossing anchor and test-only line.',
        'x1_policy': 'X1 remains review-only/excluded.',
    }
    contract['blockers'] = [
        'no confirmed true-negative real windows',
        'Line9-conditioned simulations are quarantined and no formal independent simulation is approved',
        'formal line-level train/validation split remains unassigned',
        'Batch 3 requires case-wise geometry review',
    ]
    observations = list(contract.get('observations', []))
    note = (
        'YingShan V15 final labels resolve crossing supervision by two accepted weak relabels '
        '(Line3 at Line9 crossing; X1 at L1 crossing) and two explicit local exclusions '
        '(Line6 at Line9 crossing; X1 at Line9 crossing).'
    )
    if note not in observations:
        observations.append(note)
    contract['observations'] = observations
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')

    semantics_path = ROOT / 'data' / 'dataset_contract_v2' / 'label_semantics.json'
    semantics = json.loads(semantics_path.read_text(encoding='utf-8'))
    semantics['v4_status'] = 'superseded_by_v15_final_for_current_real_label_release'
    semantics['v15_status'] = 'released_final_labels_not_formal_training_release'
    semantics['v15_version'] = VERSION
    semantics['v15_policy'] = {
        'primary_anchor': 'Line9 at cross-line conflicts',
        'accepted_weak_relabels': ['Line3-Line9: Line3', 'LineL1-LineX1: LineX1'],
        'explicit_ignores': ['Line6-Line9: Line6', 'Line9-LineX1: LineX1'],
        'unresolved_ambiguity_is_never_promoted_to_strong': True,
    }
    semantics_path.write_text(json.dumps(semantics, ensure_ascii=False, indent=2), encoding='utf-8')

    policy_path = SOURCE / 'dataset_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    policy['reason'] = (
        'Canonical full-line arrays come from the original CSV archive. V15 final labels are released with ambiguous crossing regions '
        'excluded, but formal training remains blocked by missing true negatives, independent simulations, and a formal line-level split.'
    )
    policy['current_label_release'] = {
        'path': str(OUT.relative_to(ROOT)),
        'version': VERSION,
        'training_allowed': False,
    }
    policy['crossing_label_review'] = {
        'status': 'resolved_for_supervision',
        'resolution': (
            'Two accepted weak relabels and two explicit local ignores; trusted Line9 labels were not moved. '
            'Underlying ambiguous geology remains documented but no longer contaminates supervised losses.'
        ),
        'final_decision_csv': str((REPORT / 'v15_final_crossing_decisions.csv').relative_to(ROOT)),
        'trace_change_log': str((REPORT / 'v15_final_trace_changes.csv').relative_to(ROOT)),
    }
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    if REPORT.exists():
        shutil.rmtree(REPORT)
    REPORT.mkdir(parents=True)

    line_data = {line: load_line(line) for line in LINES}
    outputs, segment_rows, trace_rows = apply_final_policy(line_data)
    window_rows = write_dataset(outputs)
    write_csv(REPORT / 'v15_final_segments.csv', segment_rows)
    write_csv(REPORT / 'v15_final_trace_changes.csv', trace_rows)

    source_crossings = list(csv.DictReader(CROSSING_CSV.open(encoding='utf-8')))
    final_crossing_rows = []
    for row in source_crossings:
        crossing = row['crossing']
        line_a, line_b = row['line_a'], row['line_b']
        trace_a, trace_b = int(row['trace_a']), int(row['trace_b'])
        arr_a, arr_b = outputs[line_a], outputs[line_b]
        time_a = float(arr_a['v15_final_center_time_ns'][trace_a])
        time_b = float(arr_b['v15_final_center_time_ns'][trace_b])
        delay_a = time_a - 2.0 * float(arr_a['flight_height_agl_m'][trace_a]) / C_M_PER_NS
        delay_b = time_b - 2.0 * float(arr_b['flight_height_agl_m'][trace_b]) / C_M_PER_NS
        ignored_a = bool(arr_a['v15_final_ignore_trace'][trace_a])
        ignored_b = bool(arr_b['v15_final_ignore_trace'][trace_b])
        final_crossing_rows.append({
            'crossing': crossing,
            'nearest_separation_m': float(row['nearest_separation_m']),
            'line_a': line_a,
            'trace_a': trace_a,
            'final_time_a_ns': time_a,
            'final_status_a': int(arr_a['status_code'][trace_a]),
            'ignored_a': ignored_a,
            'line_b': line_b,
            'trace_b': trace_b,
            'final_time_b_ns': time_b,
            'final_status_b': int(arr_b['status_code'][trace_b]),
            'ignored_b': ignored_b,
            'air_corrected_v14_difference_ns': float(row['air_corrected_delay_abs_difference_ns']),
            'air_corrected_v15_difference_ns': abs(delay_a - delay_b),
            'v15_final_decision': FINAL_CROSSING_DECISIONS[crossing],
            'supervision_conflict_resolved': True,
            'note': (
                'Resolved means labels are either accepted/relabelled or explicitly excluded from supervised losses; '
                'it does not claim ambiguous geological truth was recovered.'
            ),
        })
    write_csv(REPORT / 'v15_final_crossing_decisions.csv', final_crossing_rows)

    zero = zero_time_audit(line_data)
    zero_rows = [vars(zero[line]) for line in LINES]
    write_csv(REPORT / 'v15_final_zero_time_audit.csv', zero_rows)

    line_summary = []
    for line in LINES:
        arr = outputs[line]
        ignored = arr['v15_final_ignore_trace'].astype(bool)
        changed = arr['v15_final_changed_trace'].astype(bool)
        status = arr['status_code'].astype(np.int16)
        line_summary.append({
            'line': line,
            'trace_count': int(status.size),
            'changed_trace_count': int(changed.sum()),
            'ignored_trace_count': int(ignored.sum()),
            'active_strong': int(((status == 1) & ~ignored).sum()),
            'active_weak': int(((status == 2) & ~ignored).sum()),
            'split': str(arr['split']),
            'formal_training_allowed': False,
        })
    write_csv(OUT / 'manifests' / 'v15_final_line_summary.csv', line_summary)

    manifest = {
        'dataset': str(OUT.relative_to(ROOT)),
        'version': VERSION,
        'release_status': 'final_label_release_not_formal_training_release',
        'source_dataset': str(SOURCE.relative_to(ROOT)),
        'source_candidate_commit': '799f229',
        'source_crossing_registry': str(CROSSING_CSV.relative_to(ROOT)),
        'source_crossing_registry_sha256': sha256(CROSSING_CSV),
        'decision_provenance': (
            'Visual interpretation by the assistant, using Line9 as the user-confirmed highest-quality line; '
            'the user accepted the proposed decisions before finalization.'
        ),
        'label_semantics': 'visible-phase centerline',
        'line9_policy': 'Preserve Line9 labels; Line9 remains test-only and is the primary crossing anchor.',
        'x1_policy': 'X1 remains review-only/excluded from train, validation, and test.',
        'crossing_supervision_resolved': True,
        'formal_training_allowed': False,
        'blockers': [
            'Confirmed true-negative measured windows remain unavailable.',
            'Formal non-Line9-conditioned simulations remain unavailable.',
            'Formal line-level train/validation split remains unassigned.',
        ],
        'relabel_rules': list(RELABEL_RULES),
        'ignore_rules': list(IGNORE_RULES),
        'outputs': {
            'window_index': str((OUT / 'window_index.csv').relative_to(ROOT)),
            'line_summary': str((OUT / 'manifests' / 'v15_final_line_summary.csv').relative_to(ROOT)),
            'crossing_decisions': str((REPORT / 'v15_final_crossing_decisions.csv').relative_to(ROOT)),
            'segments': str((REPORT / 'v15_final_segments.csv').relative_to(ROOT)),
            'trace_changes': str((REPORT / 'v15_final_trace_changes.csv').relative_to(ROOT)),
            'preview': str((REPORT / 'v15_final_crossing_preview.png').relative_to(ROOT)),
        },
    }
    (OUT / 'manifests' / 'v15_final_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (OUT / 'dataset_policy.json').write_text(
        json.dumps(
            {
                'dataset_id': 'data_yingshan_v15_final_20260710',
                'label_version': VERSION,
                'training_allowed': False,
                'release_status': 'final_label_release_not_formal_training_release',
                'reason': (
                    'V15 labels are finalized and ambiguous crossings are excluded, but formal training remains blocked by '
                    'missing confirmed true negatives, independent simulations, and a formal line-level split.'
                ),
                'line9_split': 'test',
                'x1_split': 'exclude',
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    (OUT / 'DATASET_README.md').write_text(
        '# YingShan V15 Final Labels\n\n'
        'This is the final audited label release, but it is not a formal training release.\n\n'
        '- Original CSV waveform, GNSS, ground elevation, flight height, and acquisition order are preserved.\n'
        '- `soft_mask_v14_original` preserves V14 exactly.\n'
        '- `soft_mask_review_v15_final` stores the complete V15 review geometry.\n'
        '- `soft_mask_train` excludes ambiguous and transition regions through `ignore_mask`.\n'
        '- Accepted cross-line relocations remain weak labels.\n'
        '- Line9 labels were not moved and Line9 remains test-only.\n'
        '- X1 remains review-only/excluded.\n'
        '- Formal training remains blocked by missing true negatives, independent simulations, and a formal line split.\n',
        encoding='utf-8',
    )

    make_change_preview(outputs, REPORT / 'v15_final_crossing_preview.png')
    update_contract_files()

    report_lines = [
        '# YingShan V15 Final Label Release',
        '',
        f'- Version: `{VERSION}`',
        '- Release status: final label release; not a formal training release.',
        '- Line9: preserved as the highest-quality, test-only crossing anchor.',
        '- X1: remains review-only/excluded.',
        '',
        '## Final crossing decisions',
        '',
        '- Line3-Line9: weak relabel Line3 to the locally supported ~453 ns ridge; transition collars ignored.',
        '- Line6-Line9: preserve Line9; ignore the ambiguous Line6 neighborhood.',
        '- Line9-LineX1: preserve Line9; ignore the ambiguous X1 neighborhood.',
        '- LineL1-LineX1: weak relabel X1 to the locally supported ~327.5 ns ridge.',
        '- Other four crossings retain existing labels.',
        '',
        '## Safety properties',
        '',
        '- No whole-line time shift was applied.',
        '- Accepted relocations remain weak; no ambiguous label was promoted to strong.',
        '- Unresolved regions have zero label weight and are excluded from all supervised label losses.',
        '- V14 geometry is preserved in every final line NPZ for rollback/audit.',
        '',
        '## Remaining formal-training blockers',
        '',
        '- No confirmed true-negative measured windows.',
        '- No approved simulation independent of Line9 conditioning.',
        '- No final line-level train/validation split.',
    ]
    (REPORT / 'V15_FINAL_RELEASE_REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    print(OUT)
    print(REPORT)
    print(f'windows={len(window_rows)} changed_trace_rows={len(trace_rows)} segments={len(segment_rows)}')


if __name__ == '__main__':
    main()
