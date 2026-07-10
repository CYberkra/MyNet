import csv

import numpy as np

from scripts.eval_full_line import normalise_time_distribution, write_metrics


def test_curve_distribution_is_not_scored_as_segmentation(tmp_path):
    h, w = 8, 4
    gt = np.zeros((h, w), np.float32)
    gt[3, :] = 1.0
    mask = gt.copy()
    curve = np.zeros_like(gt)
    curve[5, :] = 1.0
    curve = normalise_time_distribution(curve)
    path = curve.copy()
    pres = np.ones(w, np.float32)
    status = np.ones(w, np.int16)
    label_w = np.ones(w, np.float32)

    metrics = write_metrics(
        tmp_path,
        "case",
        mask,
        path,
        pres,
        gt,
        status,
        label_w,
        1.0,
        curve_prob=curve,
        curve_source="curve_distribution_dp",
    )

    assert metrics["mask_soft_dice_weighted"] > 0.99
    assert "curve_nll" in metrics
    assert "soft_dice_weighted" not in metrics
    assert "iou_thr_0.5" not in metrics
    assert metrics["path_source"] == "curve_distribution_dp"

    rows = dict(csv.reader((tmp_path / "case_full_metrics.csv").open(encoding="utf-8")))
    assert "mask_soft_dice_weighted" in rows
    assert "curve_nll" in rows
