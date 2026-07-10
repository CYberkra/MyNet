from pathlib import Path
import csv
import importlib.util
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'build_yingshan_v15_candidate.py'


def _module():
    spec = importlib.util.spec_from_file_location('build_yingshan_v15_candidate', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='session')
def context():
    mod = _module()
    data = {line: mod.load_line(line) for line in mod.LINES}
    zero = mod.zero_time_audit(data)
    rows = list(csv.DictReader(mod.CROSSING_CSV.open(encoding='utf-8')))
    decisions = {row['crossing']: mod.crossing_decision(row, data, zero) for row in rows}
    return mod, data, zero, decisions


def test_direct_wave_audit_is_sample_scale_only(context):
    mod, _, zero, _ = context
    peaks = np.array([zero[line].direct_wave_peak_ns for line in mod.LINES])
    assert peaks.min() >= 12.0
    assert peaks.max() <= 17.0
    assert peaks.max() - peaks.min() <= 1.5


def test_crossing_policy_does_not_move_centerlines(context):
    mod, data, _, decisions = context
    outputs, _ = mod.apply_candidate_policy(data, list(decisions.values()))
    for line in mod.LINES:
        assert np.array_equal(outputs[line]['soft_mask_review_v15'], data[line]['soft_mask_train'])
        ignored = outputs[line]['ignore_mask'].mean(axis=0) > 0.5
        assert np.all(outputs[line]['soft_mask_train'][:, ignored] == 0.0)
        assert np.array_equal(
            outputs[line]['soft_mask_train'][:, ~ignored],
            data[line]['soft_mask_train'][:, ~ignored],
        )


def test_known_consistent_crossings_are_not_ignored(context):
    _, _, _, decisions = context
    assert decisions['Line3-Line7']['decision_v15_candidate'] == 'PASS'
    assert decisions['Line6-Line7']['decision_v15_candidate'] == 'PASS'
    assert decisions['Line3-Line7']['affected_lines'] == ''
    assert decisions['Line6-Line7']['affected_lines'] == ''


def test_critical_weak_crossings_are_excluded_conservatively(context):
    _, _, _, decisions = context
    assert 'Line3' in decisions['Line3-Line9']['affected_lines']
    assert 'Line9' in decisions['Line6-Line9']['affected_lines']
    assert 'LineX1' in decisions['LineL1-LineX1']['affected_lines']


def test_cross_line_suggestions_are_never_applied_automatically(context):
    _, _, _, decisions = context
    assert decisions['Line3-Line9']['suggestion_signal_grade'] in {
        'weak_signal_support_review_only', 'poor_signal_support_do_not_apply'
    }
    assert decisions['Line6-Line9']['suggestion_signal_grade'] == 'poor_signal_support_do_not_apply'
    assert decisions['LineL1-LineX1']['suggestion_signal_grade'] == 'signal_supported_for_manual_review'
