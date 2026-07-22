from __future__ import annotations

import numpy as np

from scripts.audit_real_negative_candidates import audit


def _write_line(path, *, status, weight, mask, ignore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        status_code=np.asarray(status, dtype=np.int16),
        label_weight=np.asarray(weight, dtype=np.float32),
        soft_mask_train=np.asarray(mask, dtype=np.float32),
        ignore_mask=np.asarray(ignore, dtype=np.uint8),
    )


def test_only_explicit_clean_status_zero_is_a_true_negative(tmp_path) -> None:
    data_root = tmp_path / "measured"
    _write_line(
        data_root / "lines" / "Line3.npz",
        status=[1, 0, 2, 0],
        weight=[1.0, 0.0, 0.0, 0.0],
        mask=[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        ignore=[[0, 0, 1, 0], [0, 0, 1, 0]],
    )

    result, rows = audit(data_root, contract_root=None)

    assert result["confirmed_true_negative_trace_count"] == 1
    assert result["invalid_status_zero_trace_count"] == 1
    assert result["formal_negative_supervision_ready"] is False
    assert [row["category"] for row in rows] == ["confirmed_true_negative", "invalid_status_zero", "ambiguous_or_ignore"]
    assert rows[0]["trace_start"] == rows[0]["trace_end"] == 1


def test_ignored_trace_is_never_inferred_as_negative(tmp_path) -> None:
    data_root = tmp_path / "measured"
    _write_line(
        data_root / "lines" / "Line9.npz",
        status=[2, 2],
        weight=[0.0, 0.0],
        mask=[[0.0, 0.0], [0.0, 0.0]],
        ignore=[[1, 1], [1, 1]],
    )

    result, rows = audit(data_root, contract_root=None)

    assert result["confirmed_true_negative_trace_count"] == 0
    assert result["decision"] == "blocked_no_confirmed_real_true_negative_traces"
    assert rows[0]["category"] == "ambiguous_or_ignore"
