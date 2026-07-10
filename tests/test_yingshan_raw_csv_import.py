from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from scripts.import_yingshan_raw_csv import (
    build_line_npz, load_label_source_from_windows, load_raw_line_from_zip,
)


def _write_tiny_zip(path: Path) -> None:
    rows = [
        "Number of Samples = 3,,",
        "Time windows (ns) = 2,,",
        "Number of Traces = 2,,",
        "Trace interval (m) = 1.5,,",
        "106.0,31.0,440.0,1.0,8.0",
        "106.0,31.0,440.0,2.0,8.0",
        "106.0,31.0,440.0,3.0,8.0",
        "106.00001,31.0,439.0,4.0,9.0",
        "106.00001,31.0,439.0,5.0,9.0",
        "106.00001,31.0,439.0,6.0,9.0",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("营山测线数据/Line3origin(36).csv", "\n".join(rows) + "\n")


def test_raw_csv_schema_maps_ground_and_flight_height(tmp_path: Path):
    archive_path = tmp_path / "raw.zip"
    _write_tiny_zip(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        raw = load_raw_line_from_zip(archive, "营山测线数据/Line3origin(36).csv")
    assert raw.raw_amplitude.shape == (3, 2)
    assert np.array_equal(raw.ground_elevation_m, np.array([440.0, 439.0]))
    assert np.array_equal(raw.flight_height_agl_m, np.array([8.0, 9.0]))
    assert np.array_equal(raw.raw_amplitude[:, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_build_line_npz_preserves_labels_and_adds_spatial_metadata(tmp_path: Path):
    archive_path = tmp_path / "raw.zip"
    _write_tiny_zip(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        raw = load_raw_line_from_zip(archive, "营山测线数据/Line3origin(36).csv")
    scale = float(np.percentile(np.abs(raw.raw_amplitude.astype(np.float64)), 99.0))
    label_source = tmp_path / "labels.npz"
    np.savez_compressed(
        label_source,
        raw_full_normalized=(raw.raw_amplitude / scale).astype(np.float32),
        soft_mask_train=np.ones((3, 2), dtype=np.float32),
        status_code=np.array([1, 2], dtype=np.int16),
        label_weight=np.array([1.0, 0.5], dtype=np.float32),
    )
    output = tmp_path / "Line3.npz"
    result = build_line_npz(raw, label_source_path=label_source, output_path=output, source_zip_sha256="abc")
    assert result.reconstruction_correlation > 0.999999
    with np.load(output, allow_pickle=False) as data:
        assert np.array_equal(data["status_code"], np.array([1, 2], dtype=np.int16))
        assert np.allclose(data["antenna_elevation_m"], np.array([448.0, 448.0]))
        assert np.array_equal(data["flight_height_agl_m"], np.array([8.0, 9.0], dtype=np.float32))
        assert str(data["canonical_source"]) == "original_yingshan_csv"
        assert bool(data["profile_display_flip"]) is True
        assert str(data["orientation_contract"]) == "canonical arrays remain acquisition order; profile flip is display-only"


def test_window_cache_reconstruction_detects_and_preserves_overlaps(tmp_path: Path):
    root = tmp_path / "data"
    (root / "windows").mkdir(parents=True)
    rows = [
        ("Line3_tr0000_0002", 0, 2, np.array([1.0, 2.0, 3.0])),
        ("Line3_tr0002_0004", 2, 4, np.array([3.0, 4.0, 5.0])),
    ]
    with (root / "window_index.csv").open("w", encoding="utf-8") as handle:
        handle.write("sample_id,line,start,end\n")
        for sample_id, start, end, _ in rows:
            handle.write(f"{sample_id},Line3,{start},{end}\n")
    for sample_id, start, end, values in rows:
        width = end - start + 1
        raw = np.tile(values[None, :], (3, 1)).astype(np.float32)
        mask = np.tile((values / 5.0)[None, :], (3, 1)).astype(np.float32)
        np.savez_compressed(
            root / "windows" / f"{sample_id}.npz",
            x_raw=raw, y_mask=mask,
            status_code=np.ones(width, dtype=np.int16),
            label_weight=np.ones(width, dtype=np.float32),
        )
    rebuilt = load_label_source_from_windows(root, line="Line3", samples=3, traces=5)
    assert np.array_equal(rebuilt["raw_full_normalized"][0], np.arange(1, 6, dtype=np.float32))
    assert rebuilt["status_code"].shape == (5,)
