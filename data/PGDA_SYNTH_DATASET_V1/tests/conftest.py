"""Shared fixtures for PGDA_SYNTH_DATASET_V1 tool tests."""

import pytest
import numpy as np
from pathlib import Path

SAMPLE_IN_CONTENT = """#title: PGDA test_case depth=12.0m flat
#domain: 480 45 0.05
#dx_dy_dz: 0.05 0.05 0.05
#time_window: 7.0e-07
#pml_cells: 60 60 0 60 60 0
#material: 1.0 0.0 1 0 air
#material: 13.5 0.001 1 0 moist_silty_clay
#material: 12.9 0.001 1 0 weak_cover_band
#material: 24.0 0.003 1 0 slide_zone
#material: 6.0 0.001 1 0 weathered_bedrock
#waveform: ricker 1 1e+08 uavgpr_wavelet
#hertzian_dipole: z 119.300 31.900 0.025 uavgpr_wavelet
#rx: 120.700 31.900 0.025
#src_steps: 1.701898 0 0
#rx_steps: 1.701898 0 0
#geometry_view: 0 0 0 480 45 0.05 0.05 0.05 0.05 geometry_raw n
#triangle: 0.00 30.000 0 1.00 30.000 0 1.00 32.000 0 0.05 moist_silty_clay y
#triangle: 0.00 30.000 0 1.00 32.000 0 0.00 32.000 0 0.05 moist_silty_clay y
"""


@pytest.fixture
def sample_in_path(tmp_path):
    p = tmp_path / "raw.in"
    p.write_text(SAMPLE_IN_CONTENT)
    return p


@pytest.fixture
def sample_in_with_comments(sample_in_path):
    content = sample_in_path.read_text()
    content += "\n#: this is a comment line without colon issue\n"
    sample_in_path.write_text(content)
    return sample_in_path


@pytest.fixture
def sample_case_dir(tmp_path, sample_in_path):
    """Create a case directory with raw.in and labels/."""
    geo_dir = tmp_path / "geometry"
    geo_dir.mkdir()
    (geo_dir / "raw.in").write_text(SAMPLE_IN_CONTENT)

    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    time = np.linspace(0, 700, 501, dtype=np.float32)
    vis = np.full(128, 150.0, dtype=np.float32) + np.random.RandomState(42).uniform(-5, 5, 128).astype(np.float32)
    geom = vis - 10
    y_soft = np.zeros((501, 128), dtype=np.float32)
    for i in range(128):
        c = int(vis[i] / 700 * 501)
        g = np.exp(-((time - vis[i]) ** 2) / (2 * (10) ** 2))
        y_soft[:, i] = g / g.max()

    np.save(label_dir / "time_501_ns.npy", time)
    np.save(label_dir / "trace_x_m.npy", np.arange(128, dtype=np.float32))
    np.save(label_dir / "target_visible_phase_time_ns.npy", vis)
    np.save(label_dir / "target_geom_time_ns.npy", geom)
    np.save(label_dir / "y_soft_501x128.npy", y_soft)
    np.save(label_dir / "interface_mask_bscan.npy", np.zeros((501, 128), dtype=np.float32))

    return tmp_path


@pytest.fixture
def sample_bscan():
    """128-trace synthetic B-scan with a linear target ramp 100-200ns."""
    np.random.seed(42)
    W = 128
    T = 501
    t = np.linspace(0, 700, T)
    arr = np.zeros((T, W), dtype=np.float64)
    for i in range(W):
        target_ns = 100 + i * 0.8  # ramp 100-202ns
        c = int(target_ns / 700 * T)
        arr[max(0, c - 5):min(T, c + 5), i] = -1.0
    arr += np.random.randn(T, W).astype(np.float64) * 0.05
    return arr, t


@pytest.fixture
def sample_labels():
    """Label arrays matching a linear ramp 100-202ns."""
    W = 128
    t = np.linspace(0, 700, 501)
    vis = np.array([100 + i * 0.8 for i in range(W)], dtype=np.float64)
    geom = vis - 10
    y_soft = np.zeros((501, W), dtype=np.float64)
    for i in range(W):
        c = int(vis[i] / 700 * 501)
        g = np.exp(-((t - vis[i]) ** 2) / (2 * 10 ** 2))
        y_soft[:, i] = g / g.max()
    return vis, geom, y_soft, t
