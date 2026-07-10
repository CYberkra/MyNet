from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data_corrected_v1_4_terrain_direction'
DATA = ROOT / 'data_yingshan_v15_candidate_20260710'
REPORT = ROOT / 'reports' / 'yingshan_v15_candidate_20260710'
LINES = ['Line3', 'Line6', 'Line7', 'Line9', 'LineL1', 'LineX1']


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    line_rows = []

    manifest_path = DATA / 'manifests' / 'v15_candidate_manifest.json'
    if not manifest_path.exists():
        failures.append('missing v15_candidate_manifest.json')
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('formal_training_allowed') is not False:
            failures.append('candidate manifest must keep formal_training_allowed=false')

    decisions = list(csv.DictReader((REPORT / 'crossing_v15_decisions.csv').open(encoding='utf-8')))
    if len(decisions) != 8:
        failures.append(f'expected 8 crossing decisions, got {len(decisions)}')
    workbench = list((REPORT / 'crossing_workbench').glob('*.png'))
    if len(workbench) != 8:
        failures.append(f'expected 8 crossing workbench images, got {len(workbench)}')

    for line in LINES:
        src_path = SOURCE / 'lines' / f'{line}.npz'
        dst_path = DATA / 'lines' / f'{line}.npz'
        if not dst_path.exists():
            failures.append(f'{line}: missing candidate line NPZ')
            continue
        with np.load(src_path, allow_pickle=False) as src, np.load(dst_path, allow_pickle=False) as dst:
            required = {'soft_mask_review_v15', 'soft_mask_train', 'ignore_mask', 'status_code', 'label_weight'}
            missing = required - set(dst.files)
            if missing:
                failures.append(f'{line}: missing arrays {sorted(missing)}')
                continue
            review = dst['soft_mask_review_v15'].astype(np.float32)
            train = dst['soft_mask_train'].astype(np.float32)
            ignore = dst['ignore_mask'].astype(np.float32)
            status = dst['status_code'].astype(np.int16)
            weight = dst['label_weight'].astype(np.float32)
            ignored = ignore.mean(axis=0) > 0.5
            if not np.array_equal(review, src['soft_mask_train'].astype(np.float32)):
                failures.append(f'{line}: V14 review geometry changed')
            if ignored.any():
                if not np.all(train[:, ignored] == 0.0):
                    failures.append(f'{line}: ignored columns retain training mask energy')
                if not np.all(weight[ignored] == 0.0):
                    failures.append(f'{line}: ignored columns retain label weight')
                if not np.all(status[ignored] == 2):
                    failures.append(f'{line}: ignored columns must be weak/unknown status 2')
            active = ~ignored
            if not np.array_equal(train[:, active], review[:, active]):
                failures.append(f'{line}: active centerline geometry differs from V14')
            if not np.isfinite(train).all() or not np.isfinite(ignore).all():
                failures.append(f'{line}: non-finite candidate arrays')
            line_rows.append({
                'line': line,
                'trace_count': int(train.shape[1]),
                'ignored_trace_count': int(ignored.sum()),
                'active_strong': int(((status == 1) & active).sum()),
                'active_weak': int(((status == 2) & active).sum()),
                'review_geometry_preserved': True,
            })

    index_rows = list(csv.DictReader((DATA / 'window_index.csv').open(encoding='utf-8')))
    if len(index_rows) != 78:
        failures.append(f'expected 78 windows, got {len(index_rows)}')
    for row in index_rows:
        p = DATA / 'windows' / f"{row['sample_id']}.npz"
        if not p.exists():
            failures.append(f"missing window {row['sample_id']}")
            continue
        with np.load(p, allow_pickle=False) as z:
            ignore_col = z['ignore_mask'].mean(axis=0) > 0.5
            if int(row['ignore']) != int(ignore_col.sum()):
                failures.append(f"{row['sample_id']}: ignore count mismatch")

    result = {
        'ok': not failures,
        'formal_training_allowed': False,
        'failures': failures,
        'warnings': warnings,
        'line_summary': line_rows,
        'crossing_decisions': len(decisions),
        'workbench_images': len(workbench),
        'window_count': len(index_rows),
    }
    out = REPORT / 'validation_result.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
