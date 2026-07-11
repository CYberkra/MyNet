from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.eval_full_line import distribution_path_metrics, normalise_trace_distribution, write_metrics


def test_trace_distribution_normalisation():
    arr = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    dist, valid = normalise_trace_distribution(arr)
    assert valid.tolist() == [True, False]
    assert dist[:, 0].sum() == pytest.approx(1.0)
    assert dist[:, 1].sum() == pytest.approx(0.0)


def test_path_metrics_are_independent_from_mask_segmentation(tmp_path: Path):
    h, w = 8, 3
    gt = np.zeros((h, w), np.float32); gt[4, :] = 1.0
    mask_pred = np.zeros_like(gt); path_pred = gt.copy()
    pres_pred = np.array([0.9, 0.9, 0.9], np.float32)
    status = np.array([1, 0, 2], np.int16)
    label_w = np.ones(w, np.float32)
    write_metrics(
        tmp_path, "Line9", mask_pred, path_pred, pres_pred, gt, status, label_w, 1.0,
        path_source="curve_distribution_dp",
    )
    rows = {}
    with (tmp_path / "Line9_full_metrics.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f): rows[row["metric"]] = row["value"]
    assert float(rows["mask_soft_dice_weighted"]) == pytest.approx(0.0)
    assert float(rows["path_expected_mae_ns"]) == pytest.approx(0.0)
    assert int(float(rows["presence_known_trace_count"])) == 2
    assert int(float(rows["presence_true_negative_trace_count"])) == 1
    assert float(rows["presence_false_pick_rate_confirmed_negative"]) == pytest.approx(1.0)
    assert "soft_dice_weighted" not in rows
    assert "weighted_bce" not in rows


def test_distribution_path_metrics_exact_curve():
    gt = np.zeros((6, 4), np.float32); gt[3, :] = 1.0
    metrics = distribution_path_metrics(gt.copy(), gt, dt_ns=2.0)
    assert metrics["path_expected_mae_ns"] == pytest.approx(0.0)
    assert metrics["path_hit_rate_le_5ns"] == pytest.approx(1.0)
    assert metrics["path_emd"] == pytest.approx(0.0)


def test_uncertainty_and_no_pick_remain_separate_artifacts(tmp_path: Path):
    h, w = 8, 4
    gt = np.zeros((h, w), np.float32); gt[3, :] = 1.0
    pred = gt.copy(); status = np.array([1, 1, 0, 0], np.int16)
    metrics = write_metrics(
        tmp_path, "LineA", pred, pred, np.ones(w, np.float32), gt, status,
        np.ones(w, np.float32), 1.0,
        cdp=np.full(w, 3.0, np.float32), vdp=np.ones(w, bool),
        cgt=np.full(w, 3.0, np.float32), vgt=np.ones(w, bool),
        path_log_variance=np.array([-2.0, -1.0, 0.0, 1.0], np.float32),
        no_pick_prob=np.array([0.1, 0.2, 0.8, 0.9], np.float32), no_pick_thr=0.5,
    )
    assert metrics["uncertainty_available"] is True
    assert metrics["no_pick_reject_rate"] == pytest.approx(0.5)
    assert "mask_iou_thr_0.5" in metrics
