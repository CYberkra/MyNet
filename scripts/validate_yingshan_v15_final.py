from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data_corrected_v1_4_terrain_direction'
DATA = ROOT / 'data_yingshan_v15_final_20260710'
REPORT = ROOT / 'reports' / 'yingshan_v15_final_20260710'
LINES = ['Line3', 'Line6', 'Line7', 'Line9', 'LineL1', 'LineX1']
VERSION = 'YINGSHAN_V15_FINAL_20260710'
C_M_PER_NS = 0.299792458


def centerline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yy = np.arange(mask.shape[0], dtype=np.float32)[:, None]
    mass = mask.sum(axis=0)
    center = (mask * yy).sum(axis=0) / np.maximum(mass, 1e-8)
    valid = mass > 1e-4
    center[~valid] = np.nan
    return center, valid


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    facts: dict[str, object] = {}

    manifest_path = DATA / 'manifests' / 'v15_final_manifest.json'
    policy_path = DATA / 'dataset_policy.json'
    if not manifest_path.is_file():
        failures.append('missing v15_final_manifest.json')
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('version') != VERSION:
            failures.append(f"manifest version mismatch: {manifest.get('version')}")
        if manifest.get('formal_training_allowed') is not False:
            failures.append('V15 final manifest must keep formal_training_allowed=false')
        if manifest.get('crossing_supervision_resolved') is not True:
            failures.append('crossing supervision must be explicitly resolved')
    if not policy_path.is_file():
        failures.append('missing V15 final dataset_policy.json')
    else:
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
        if policy.get('training_allowed') is not False:
            failures.append('V15 final dataset policy must block formal training')

    line_facts: dict[str, dict[str, object]] = {}
    final_lines: dict[str, dict[str, np.ndarray]] = {}
    source_lines: dict[str, dict[str, np.ndarray]] = {}
    for line in LINES:
        src_path = SOURCE / 'lines' / f'{line}.npz'
        dst_path = DATA / 'lines' / f'{line}.npz'
        if not dst_path.is_file():
            failures.append(f'{line}: missing final line NPZ')
            continue
        with np.load(src_path, allow_pickle=False) as src_npz, np.load(dst_path, allow_pickle=False) as dst_npz:
            src = {k: src_npz[k] for k in src_npz.files}
            dst = {k: dst_npz[k] for k in dst_npz.files}
        source_lines[line] = src
        final_lines[line] = dst
        required = {
            'soft_mask_v14_original', 'soft_mask_review_v15_final', 'soft_mask_train', 'ignore_mask',
            'status_code_v14_original', 'label_weight_v14_original', 'status_code', 'label_weight',
            'v15_final_center_time_ns', 'v15_final_changed_trace', 'v15_final_ignore_trace',
            'v15_final_decision_code', 'v15_final_review_reason', 'v15_final_version',
        }
        missing = required - set(dst)
        if missing:
            failures.append(f'{line}: missing arrays {sorted(missing)}')
            continue
        if str(dst['v15_final_version'].item()) != VERSION:
            failures.append(f'{line}: v15_final_version mismatch')
        if not np.array_equal(dst['soft_mask_v14_original'], src['soft_mask_train']):
            failures.append(f'{line}: V14 mask rollback copy differs from source')
        if not np.array_equal(dst['status_code_v14_original'], src['status_code']):
            failures.append(f'{line}: V14 status rollback copy differs from source')
        if not np.array_equal(dst['label_weight_v14_original'], src['label_weight']):
            failures.append(f'{line}: V14 weight rollback copy differs from source')

        review = dst['soft_mask_review_v15_final'].astype(np.float32)
        train = dst['soft_mask_train'].astype(np.float32)
        ignore = dst['ignore_mask'].astype(np.float32)
        ignore_trace = dst['v15_final_ignore_trace'].astype(bool)
        changed = dst['v15_final_changed_trace'].astype(bool)
        status = dst['status_code'].astype(np.int16)
        weight = dst['label_weight'].astype(np.float32)
        if ignore.shape != train.shape or review.shape != train.shape:
            failures.append(f'{line}: mask shapes differ')
        if not np.array_equal(ignore_trace, ignore.mean(axis=0) > 0.5):
            failures.append(f'{line}: v15_final_ignore_trace inconsistent with ignore_mask')
        if ignore_trace.any():
            if not np.all(train[:, ignore_trace] == 0.0):
                failures.append(f'{line}: ignored traces retain training mask')
            if not np.all(weight[ignore_trace] == 0.0):
                failures.append(f'{line}: ignored traces retain label weight')
            if not np.all(status[ignore_trace] == 2):
                failures.append(f'{line}: ignored traces must have weak/unknown status 2')
        if not np.isfinite(review).all() or not np.isfinite(train).all() or not np.isfinite(ignore).all():
            failures.append(f'{line}: non-finite V15 arrays')

        final_center, final_valid = centerline(review)
        stored_center = dst['v15_final_center_time_ns'].astype(np.float32) / float(dst['dt_ns'])
        if not np.allclose(final_center[final_valid], stored_center[final_valid], atol=2e-3, rtol=0.0):
            failures.append(f'{line}: stored V15 centerline does not match review mask')
        line_facts[line] = {
            'changed_traces': int(changed.sum()),
            'ignored_traces': int(ignore_trace.sum()),
            'active_strong': int(((status == 1) & ~ignore_trace).sum()),
            'active_weak': int(((status == 2) & ~ignore_trace).sum()),
        }

    # Trusted lines and unaffected lines must remain unchanged.
    for line in ('Line7', 'Line9', 'LineL1'):
        if line not in final_lines:
            continue
        dst, src = final_lines[line], source_lines[line]
        if not np.array_equal(dst['soft_mask_review_v15_final'], src['soft_mask_train']):
            failures.append(f'{line}: trusted/unaffected review geometry changed')
        if bool(dst['v15_final_ignore_trace'].any()):
            failures.append(f'{line}: trusted/unaffected line unexpectedly ignored')

    # Exact accepted decisions.
    if 'Line3' in final_lines:
        z = final_lines['Line3']
        t = float(z['v15_final_center_time_ns'][167])
        if not (450.0 <= t <= 455.0):
            failures.append(f'Line3-Line9: final Line3 center {t:.3f} ns not in accepted 450-455 ns range')
        if int(z['status_code'][167]) != 2 or bool(z['v15_final_ignore_trace'][167]):
            failures.append('Line3-Line9: crossing center must be active weak, not ignored')
        if str(z['v15_final_decision_code'][167]) != 'RELABEL_WEAK_LINE9_ANCHORED':
            failures.append('Line3-Line9: decision code mismatch')
        if line_facts['Line3']['ignored_traces'] <= 0:
            failures.append('Line3-Line9: transition collar must be ignored')

    if 'Line6' in final_lines and 'Line9' in final_lines:
        if not bool(final_lines['Line6']['v15_final_ignore_trace'][523]):
            failures.append('Line6-Line9: Line6 crossing must be ignored')
        if bool(final_lines['Line9']['v15_final_ignore_trace'][1391]):
            failures.append('Line6-Line9: trusted Line9 crossing must remain active')
        if not np.array_equal(final_lines['Line9']['soft_mask_review_v15_final'], source_lines['Line9']['soft_mask_train']):
            failures.append('Line6-Line9: Line9 geometry changed')

    if 'LineX1' in final_lines and 'Line9' in final_lines:
        x1 = final_lines['LineX1']
        if not bool(x1['v15_final_ignore_trace'][197]):
            failures.append('Line9-LineX1: X1 crossing must be ignored')
        if bool(final_lines['Line9']['v15_final_ignore_trace'][853]):
            failures.append('Line9-LineX1: Line9 crossing must remain active')
        t = float(x1['v15_final_center_time_ns'][659])
        if not (326.0 <= t <= 330.0):
            failures.append(f'LineL1-LineX1: final X1 center {t:.3f} ns not in accepted 326-330 ns range')
        if int(x1['status_code'][659]) != 2 or bool(x1['v15_final_ignore_trace'][659]):
            failures.append('LineL1-LineX1: X1 relabel must be active weak')
        if str(x1['v15_final_decision_code'][659]) != 'RELABEL_WEAK_L1_ANCHORED':
            failures.append('LineL1-LineX1: decision code mismatch')
        if str(x1['split'].item()) != 'exclude':
            failures.append('X1 must remain excluded')

    # Crossing-level final differences: accepted relabels should materially resolve their conflicts.
    if all(line in final_lines for line in ('Line3', 'Line9')):
        a, b = final_lines['Line3'], final_lines['Line9']
        da = float(a['v15_final_center_time_ns'][167]) - 2.0 * float(a['flight_height_agl_m'][167]) / C_M_PER_NS
        db = float(b['v15_final_center_time_ns'][324]) - 2.0 * float(b['flight_height_agl_m'][324]) / C_M_PER_NS
        facts['Line3-Line9_final_air_corrected_difference_ns'] = abs(da - db)
        if abs(da - db) > 10.0:
            failures.append('Line3-Line9 accepted weak relabel remains >10 ns inconsistent')
    if all(line in final_lines for line in ('LineL1', 'LineX1')):
        a, b = final_lines['LineL1'], final_lines['LineX1']
        da = float(a['v15_final_center_time_ns'][568]) - 2.0 * float(a['flight_height_agl_m'][568]) / C_M_PER_NS
        db = float(b['v15_final_center_time_ns'][659]) - 2.0 * float(b['flight_height_agl_m'][659]) / C_M_PER_NS
        facts['LineL1-LineX1_final_air_corrected_difference_ns'] = abs(da - db)
        if abs(da - db) > 5.0:
            failures.append('LineL1-LineX1 accepted weak relabel remains >5 ns inconsistent')

    # Window slices must exactly match final full-line arrays.
    index_path = DATA / 'window_index.csv'
    if not index_path.is_file():
        failures.append('missing final window_index.csv')
        rows = []
    else:
        rows = list(csv.DictReader(index_path.open(encoding='utf-8')))
    if len(rows) != 78:
        failures.append(f'expected 78 final windows, got {len(rows)}')
    for row in rows:
        line = row['line']
        if line not in final_lines:
            continue
        start, end = int(row['start']), int(row['end']) + 1
        path = DATA / 'windows' / f"{row['sample_id']}.npz"
        if not path.is_file():
            failures.append(f"missing window {row['sample_id']}")
            continue
        with np.load(path, allow_pickle=False) as w:
            line_z = final_lines[line]
            checks = (
                ('x_raw', w['x_raw'], line_z['raw_full_normalized'][:, start:end]),
                ('y_mask', w['y_mask'], line_z['soft_mask_train'][:, start:end]),
                ('status_code', w['status_code'], line_z['status_code'][start:end]),
                ('label_weight', w['label_weight'], line_z['label_weight'][start:end]),
                ('ignore_mask', w['ignore_mask'], line_z['ignore_mask'][:, start:end]),
            )
            for key, got, expected in checks:
                if not np.array_equal(got, expected):
                    failures.append(f"{row['sample_id']}: {key} differs from full line")
            if str(w['label_version'].item()) != VERSION:
                failures.append(f"{row['sample_id']}: label version mismatch")

    decisions_path = REPORT / 'v15_final_crossing_decisions.csv'
    if not decisions_path.is_file():
        failures.append('missing v15_final_crossing_decisions.csv')
    else:
        decisions = list(csv.DictReader(decisions_path.open(encoding='utf-8')))
        if len(decisions) != 8:
            failures.append(f'expected 8 final crossing decisions, got {len(decisions)}')
        if not all(row.get('supervision_conflict_resolved') == 'True' for row in decisions):
            failures.append('not all crossing supervision decisions are resolved')

    facts['line_summary'] = line_facts
    facts['window_count'] = len(rows)
    facts['formal_training_allowed'] = False
    result = {
        'ok': not failures,
        'label_release_complete': not failures,
        'formal_training_allowed': False,
        'failures': failures,
        'warnings': warnings,
        'facts': facts,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'v15_final_validation.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
