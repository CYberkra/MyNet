from pathlib import Path
import csv
import importlib.util
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / 'scripts' / 'build_yingshan_v15_final.py'
DATA = ROOT / 'data_yingshan_v15_final_20260710'
REPORT = ROOT / 'reports' / 'yingshan_v15_final_20260710'
SOURCE = ROOT / 'data_corrected_v1_4_terrain_direction'


def _module():
    spec = importlib.util.spec_from_file_location('build_yingshan_v15_final', BUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='session')
def built_context():
    mod = _module()
    source = {line: mod.load_line(line) for line in mod.LINES}
    outputs, segments, changes = mod.apply_final_policy(source)
    return mod, source, outputs, segments, changes


def test_final_policy_preserves_v14_rollback_and_line9(built_context):
    mod, source, outputs, _, _ = built_context
    for line in mod.LINES:
        assert np.array_equal(outputs[line]['soft_mask_v14_original'], source[line]['soft_mask_train'])
        assert np.array_equal(outputs[line]['status_code_v14_original'], source[line]['status_code'])
        assert np.array_equal(outputs[line]['label_weight_v14_original'], source[line]['label_weight'])
    assert np.array_equal(outputs['Line9']['soft_mask_review_v15_final'], source['Line9']['soft_mask_train'])
    assert not outputs['Line9']['v15_final_ignore_trace'].any()
    assert str(outputs['Line9']['split'].item()) == 'test'


def test_line3_line9_final_relabel_is_weak_and_consistent(built_context):
    _, _, outputs, _, _ = built_context
    line3 = outputs['Line3']
    line9 = outputs['Line9']
    t3 = float(line3['v15_final_center_time_ns'][167])
    assert 450.0 <= t3 <= 455.0
    assert int(line3['status_code'][167]) == 2
    assert not bool(line3['v15_final_ignore_trace'][167])
    assert str(line3['v15_final_decision_code'][167]) == 'RELABEL_WEAK_LINE9_ANCHORED'
    c = 0.299792458
    delay3 = t3 - 2.0 * float(line3['flight_height_agl_m'][167]) / c
    delay9 = float(line9['v15_final_center_time_ns'][324]) - 2.0 * float(line9['flight_height_agl_m'][324]) / c
    assert abs(delay3 - delay9) <= 10.0
    assert line3['v15_final_ignore_trace'].sum() > 0  # transition collar


def test_line6_line9_keeps_line9_and_ignores_only_line6(built_context):
    _, source, outputs, _, _ = built_context
    assert bool(outputs['Line6']['v15_final_ignore_trace'][523])
    assert not bool(outputs['Line9']['v15_final_ignore_trace'][1391])
    assert np.array_equal(outputs['Line9']['soft_mask_review_v15_final'], source['Line9']['soft_mask_train'])


def test_x1_decisions_are_conservative(built_context):
    _, _, outputs, _, _ = built_context
    x1 = outputs['LineX1']
    assert bool(x1['v15_final_ignore_trace'][197])
    t = float(x1['v15_final_center_time_ns'][659])
    assert 326.0 <= t <= 330.0
    assert int(x1['status_code'][659]) == 2
    assert not bool(x1['v15_final_ignore_trace'][659])
    assert str(x1['v15_final_decision_code'][659]) == 'RELABEL_WEAK_L1_ANCHORED'
    assert str(x1['split'].item()) == 'exclude'


def test_ignored_traces_carry_no_supervision(built_context):
    mod, _, outputs, _, _ = built_context
    for line in mod.LINES:
        arr = outputs[line]
        ignored = arr['v15_final_ignore_trace'].astype(bool)
        if ignored.any():
            assert np.all(arr['soft_mask_train'][:, ignored] == 0.0)
            assert np.all(arr['label_weight'][ignored] == 0.0)
            assert np.all(arr['status_code'][ignored] == 2)


def test_written_final_dataset_has_78_exact_windows():
    rows = list(csv.DictReader((DATA / 'window_index.csv').open(encoding='utf-8')))
    assert len(rows) == 78
    for row in rows:
        line = row['line']
        start, end = int(row['start']), int(row['end']) + 1
        with np.load(DATA / 'lines' / f'{line}.npz', allow_pickle=False) as full, np.load(
            DATA / 'windows' / f"{row['sample_id']}.npz", allow_pickle=False
        ) as win:
            assert np.array_equal(win['x_raw'], full['raw_full_normalized'][:, start:end])
            assert np.array_equal(win['y_mask'], full['soft_mask_train'][:, start:end])
            assert np.array_equal(win['ignore_mask'], full['ignore_mask'][:, start:end])
            assert str(win['label_version'].item()) == 'YINGSHAN_V15_FINAL_20260710'


def test_final_release_manifest_is_not_a_training_release():
    import json

    manifest = json.loads((DATA / 'manifests' / 'v15_final_manifest.json').read_text(encoding='utf-8'))
    policy = json.loads((DATA / 'dataset_policy.json').read_text(encoding='utf-8'))
    assert manifest['crossing_supervision_resolved'] is True
    assert manifest['formal_training_allowed'] is False
    assert policy['training_allowed'] is False
    decisions = list(csv.DictReader((REPORT / 'v15_final_crossing_decisions.csv').open(encoding='utf-8')))
    assert len(decisions) == 8
    assert all(row['supervision_conflict_resolved'] == 'True' for row in decisions)
