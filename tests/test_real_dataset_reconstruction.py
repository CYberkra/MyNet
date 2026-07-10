from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reconstruct_real_dataset_from_windows.py"
SPEC = importlib.util.spec_from_file_location("real_reconstruct", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_window(root: Path, name: str, start: int, raw: np.ndarray, *, status: np.ndarray | None = None) -> None:
    width = raw.shape[1]
    end = start + width - 1
    status = np.ones(width, np.int16) if status is None else status.astype(np.int16)
    np.savez_compressed(
        root / "windows" / f"{name}_tr{start:04d}_{end:04d}.npz",
        x_raw=raw.astype(np.float32),
        y_mask=(raw * 0.1).astype(np.float32),
        status_code=status,
        label_weight=np.full(width, 0.8, np.float32),
    )


def test_reconstructs_exact_overlapping_windows(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "windows").mkdir(parents=True)
    full = np.arange(24, dtype=np.float32).reshape(4, 6)
    write_window(root, "LineA", 0, full[:, :4], status=np.array([1, 1, 2, 2]))
    write_window(root, "LineA", 2, full[:, 2:], status=np.array([2, 2, 1, 1]))

    records = MODULE.parse_windows(root / "windows")
    arrays = MODULE.reconstruct_line(records)
    np.testing.assert_array_equal(arrays["raw_full_normalized"], full)
    np.testing.assert_array_equal(arrays["soft_mask_train"], full * 0.1)
    np.testing.assert_array_equal(arrays["status_code"], np.array([1, 1, 2, 2, 1, 1], np.int16))


def test_rejects_overlap_conflict(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "windows").mkdir(parents=True)
    first = np.zeros((4, 4), np.float32)
    second = np.zeros((4, 4), np.float32)
    second[:, 0] = 1.0
    write_window(root, "LineA", 0, first)
    write_window(root, "LineA", 2, second)
    records = MODULE.parse_windows(root / "windows")
    with pytest.raises(MODULE.ReconstructionError, match="overlap conflict"):
        MODULE.reconstruct_line(records)


def test_reconstruct_dataset_writes_line_and_index(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "windows").mkdir(parents=True)
    full = np.arange(20, dtype=np.float32).reshape(4, 5)
    write_window(root, "LineA", 0, full[:, :3], status=np.array([1, 2, 2]))
    write_window(root, "LineA", 2, full[:, 2:], status=np.array([2, 1, 1]))
    report = MODULE.reconstruct_dataset(
        root,
        dt_ns=1.4,
        dt_source="unit_test",
        update_governance=False,
    )
    assert report["window_count"] == 2
    line = np.load(root / "lines" / "LineA.npz", allow_pickle=False)
    np.testing.assert_array_equal(line["raw_full_normalized"], full)
    assert float(line["dt_ns"]) == pytest.approx(1.4)
    index_text = (root / "window_index.csv").read_text(encoding="utf-8")
    assert "LineA_tr0000_0002" in index_text
    assert (root / "reconstruction_manifest.json").is_file()


def test_window_cache_reconstruction_refuses_to_overwrite_original_csv_canonical(tmp_path):
    from scripts.reconstruct_real_dataset_from_windows import ReconstructionError, reconstruct_dataset

    dataset = tmp_path / "dataset"
    (dataset / "windows").mkdir(parents=True)
    (dataset / "lines").mkdir(parents=True)
    np.savez_compressed(
        dataset / "lines" / "Line3.npz",
        canonical_source=np.asarray("original_yingshan_csv"),
    )
    with pytest.raises(ReconstructionError, match="refusing to overwrite canonical"):
        reconstruct_dataset(dataset, dt_ns=1.4, dt_source="test", update_governance=False)
