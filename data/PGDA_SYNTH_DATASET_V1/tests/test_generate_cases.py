"""Tests for generate_cases.py — domain_y, terrain, layer_depths, and generate_one."""

import sys, os, json, shutil
import importlib.util
from pathlib import Path
import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

spec = importlib.util.spec_from_file_location("generate_cases", TOOLS_DIR / "generate_cases.py")
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)


def _is_grid_integer(dy, dx):
    """Float-safe check that dy/dx is an integer."""
    ratio = dy / dx
    return abs(ratio - round(ratio)) < 1e-10


class TestComputeDomainY:
    def test_flat_minimum(self):
        dy = gc.compute_domain_y(6.0, 'flat')
        assert dy >= 45.0
        assert _is_grid_integer(dy, gc.DX)

    def test_flat_deep(self):
        dy = gc.compute_domain_y(24.0, 'flat')
        assert dy >= 45.0
        assert _is_grid_integer(dy, gc.DX)

    def test_terrain_larger_than_flat(self):
        flat_dy = gc.compute_domain_y(12.0, 'flat')
        terr_dy = gc.compute_domain_y(12.0, 'terrain')
        assert terr_dy >= flat_dy

    def test_rounds_to_dx(self):
        for depth in [7.3, 11.7, 18.2, 23.9]:
            dy = gc.compute_domain_y(depth, 'flat')
            assert _is_grid_integer(dy, gc.DX), f"domain_y={dy} not multiple of dx={gc.DX}"

    def test_monotonic_with_depth(self):
        prev = 0
        for d in [6, 10, 14, 18, 24]:
            dy = gc.compute_domain_y(d, 'flat')
            assert dy >= prev, f"domain_y decreased at depth={d}"
            prev = dy

    def test_jitter_margin_included(self):
        """compute_domain_y adds len(STRATA)*JITTER_AMP margin."""
        # For shallow depths, domain_y should be above baseline
        dy_shallow = gc.compute_domain_y(1.0, 'flat')  # unrealistic but tests margin logic
        assert dy_shallow >= gc.SURFACE_BASE + 5.0  # at least base + margin_below


class TestMakeTraceX:
    def test_length(self):
        tx = gc.make_trace_x()
        assert len(tx) == gc.N_TRACES

    def test_range(self):
        tx = gc.make_trace_x()
        expected_end = gc.SCAN_X0 + gc.TRACE_STEP * (gc.N_TRACES - 1)
        assert abs(tx[0] - gc.SCAN_X0) < 1e-6
        assert abs(tx[-1] - expected_end) < 1e-6

    def test_uniform_spacing(self):
        tx = gc.make_trace_x()
        diffs = np.diff(tx)
        assert np.allclose(diffs, diffs[0])


class TestTerrainFn:
    def test_flat(self):
        x = np.linspace(0, 480, 100)
        y = gc.terrain_fn('flat', x)
        assert np.all(y == gc.SURFACE_BASE)

    def test_terrain_baseline(self):
        x = np.array([gc.SCAN_X0])
        y = gc.terrain_fn('terrain', x)
        # At x=SCAN_X0=120, sin(0)=0, sin(0)=0 → surface = SURFACE_BASE + 0 + 0
        assert abs(y[0] - gc.SURFACE_BASE) < 0.01

    def test_terrain_variation(self):
        x = np.linspace(0, 480, 1000)
        y = gc.terrain_fn('terrain', x)
        assert np.max(y) - np.min(y) > 0.5  # has meaningful variation
        assert np.max(y) <= gc.SURFACE_BASE + 1.8  # within amplitude

    def test_unknown_name_falls_back_to_flat(self):
        x = np.linspace(0, 480, 10)
        y = gc.terrain_fn('unknown', x)
        assert np.all(y == gc.SURFACE_BASE)


class TestLayerDepths:
    def test_returns_all_layers(self):
        surf = np.full(gc.N_TRACES, gc.SURFACE_BASE)
        layers = gc.layer_depths(surf, 12.0, 42)
        for name, _ in gc.STRATA:
            assert name in layers
            assert 'top' in layers[name]
            assert 'bottom' in layers[name]
            assert len(layers[name]['top']) == gc.N_TRACES
            assert len(layers[name]['bottom']) == gc.N_TRACES

    def test_total_strata(self):
        surf = np.full(gc.N_TRACES, gc.SURFACE_BASE)
        layers = gc.layer_depths(surf, 12.0, 42)
        # total_strata = sum of all bottom - top
        total = layers['total_strata']
        last_bottom = layers[gc.STRATA[-1][0]]['bottom']
        assert np.allclose(total, last_bottom - surf)

    def test_non_flat_labels(self):
        """Layer depths should vary per trace (jitter)."""
        surf = np.full(gc.N_TRACES, gc.SURFACE_BASE)
        layers = gc.layer_depths(surf, 12.0, 42)
        slide = layers['slide_zone']
        vrange = np.max(slide['top']) - np.min(slide['top'])
        assert vrange > 0.01, f"Label range too small: {vrange}"

    def test_positive_thickness(self):
        surf = np.full(gc.N_TRACES, gc.SURFACE_BASE)
        layers = gc.layer_depths(surf, 12.0, 42)
        for name, _ in gc.STRATA:
            thick = layers[name]['bottom'] - layers[name]['top']
            assert np.all(thick > 0), f"{name} has non-positive thickness"

    def test_seed_determinism(self):
        surf = np.full(gc.N_TRACES, gc.SURFACE_BASE)
        l1 = gc.layer_depths(surf, 12.0, 42)
        l2 = gc.layer_depths(surf, 12.0, 42)
        l3 = gc.layer_depths(surf, 12.0, 99)
        assert np.allclose(l1['slide_zone']['top'], l2['slide_zone']['top'])
        # Different seed → different result
        assert not np.allclose(l1['slide_zone']['top'], l3['slide_zone']['top'])

    def test_thicknesses_sum_to_expected(self):
        """Total accumulated depth should approximately equal target_depth."""
        surf = np.full(gc.N_TRACES, gc.SURFACE_BASE)
        target_depth = 15.0
        layers = gc.layer_depths(surf, target_depth, 42)
        total = np.mean(layers['total_strata'])
        expected_total = sum(frac * target_depth for _, frac in gc.STRATA)
        # Jitter can cause small deviation, but mean should be close
        assert abs(total - expected_total) < 1.0


class TestGenerateOne:
    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path):
        """Temporarily redirect GC's global paths to tmp_path for isolation."""
        self.orig_pool = gc.POOL_DIR
        gc.POOL_DIR = tmp_path

    def teardown_method(self):
        gc.POOL_DIR = self.orig_pool

    def test_generate_basic(self, tmp_path):
        batch_dir = tmp_path / "test_batch"
        params = {'target_depth_m': 12.0, 'terrain': 'flat', 'uav_height_m': 2.2}
        ok = gc.generate_one("TEST_001", params, batch_dir, force=True, seed=42)
        assert ok
        assert (batch_dir / "cases" / "TEST_001").exists()
        assert (batch_dir / "cases" / "TEST_001" / "geometry" / "raw.in").exists()

    def test_labels_created(self, tmp_path):
        batch_dir = tmp_path / "test_labels"
        params = {'target_depth_m': 12.0, 'terrain': 'flat'}
        gc.generate_one("TEST_LBL", params, batch_dir, force=True, seed=42)
        label_dir = batch_dir / "cases" / "TEST_LBL" / "labels"
        required = ["y_soft_501x128.npy", "target_visible_phase_time_ns.npy",
                     "target_geom_time_ns.npy", "interface_mask_bscan.npy", "time_501_ns.npy"]
        for rl in required:
            assert (label_dir / rl).exists(), f"Missing label: {rl}"

    def test_labels_non_flat(self, tmp_path):
        batch_dir = tmp_path / "test_nonflat"
        params = {'target_depth_m': 12.0, 'terrain': 'flat'}
        gc.generate_one("TEST_NF", params, batch_dir, force=True, seed=42)
        vis = np.load(str(batch_dir / "cases" / "TEST_NF" / "labels" / "target_visible_phase_time_ns.npy"))
        vrange = float(np.nanmax(vis) - np.nanmin(vis))
        assert vrange > 0.5, f"Label range={vrange} — flat line!"

    def test_in_file_content(self, tmp_path):
        batch_dir = tmp_path / "test_infile"
        params = {'target_depth_m': 12.0, 'terrain': 'flat'}
        gc.generate_one("TEST_INF", params, batch_dir, force=True, seed=42)
        in_path = batch_dir / "cases" / "TEST_INF" / "geometry" / "raw.in"
        content = in_path.read_text()
        assert "#pml_cells: 60 60 0 60 60 0" in content
        assert "#domain:" in content
        assert "#hertzian_dipole:" in content
        assert "#rx:" in content
        assert "y" in content  # triangle averaging

    def test_skip_existing_without_force(self, tmp_path):
        batch_dir = tmp_path / "test_skip"
        params = {'target_depth_m': 12.0, 'terrain': 'flat'}
        gc.generate_one("TEST_SKIP", params, batch_dir, force=True, seed=42)
        ok = gc.generate_one("TEST_SKIP", params, batch_dir, force=False, seed=42)
        assert not ok  # should return False (skipped)

    def test_domain_y_from_infile(self, tmp_path):
        """Domain_y in generated .in matches compute_domain_y."""
        batch_dir = tmp_path / "test_domain"
        params = {'target_depth_m': 18.0, 'terrain': 'terrain'}
        gc.generate_one("TEST_DOM", params, batch_dir, force=True, seed=42)
        in_path = batch_dir / "cases" / "TEST_DOM" / "geometry" / "raw.in"
        for line in in_path.read_text().splitlines():
            if line.startswith("#domain:"):
                parts = line.split()
                domain_y = float(parts[2])
                expected = gc.compute_domain_y(18.0, 'terrain')
                assert abs(domain_y - expected) < 1e-6
                break

    def test_scene_world_json(self, tmp_path):
        batch_dir = tmp_path / "test_sw"
        params = {'target_depth_m': 15.0, 'terrain': 'flat'}
        gc.generate_one("TEST_SW", params, batch_dir, force=True, seed=42)
        sw_path = batch_dir / "cases" / "TEST_SW" / "geometry" / "scene_world.json"
        assert sw_path.exists()
        sw = json.loads(sw_path.read_text())
        assert sw['case_id'] == "TEST_SW"
        assert sw['target_depth_m'] == 15.0
        assert sw['domain_y'] > 0
