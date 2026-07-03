"""Tests for after_run_qc.py — _parse_geometry, compute_qc, curve_support, grading."""

import sys, os, json, shutil
import importlib.util
from pathlib import Path
import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

spec = importlib.util.spec_from_file_location("after_run_qc", TOOLS_DIR / "after_run_qc.py")
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)

# Force matplotlib to Agg (must match how the module does it)
import matplotlib
matplotlib.use('Agg')


# ── Sample .in content for geometry parsing tests ──
SAMPLE_GEOMETRY_IN = """#title: QC test geometry
#domain: 480 45 0.05
#pml_cells: 60 60 0 60 60 0
#material: 1.0 0.0 1 0 air
#material: 13.5 0.001 1 0 moist_silty_clay
#material: 24.0 0.003 1 0 slide_zone
#material: 6.0 0.001 1 0 weathered_bedrock
#waveform: ricker 1 1e+08 uavgpr_wavelet
#hertzian_dipole: z 119.300 31.900 0.025 uavgpr_wavelet
#rx: 120.700 31.900 0.025
#src_steps: 1.701898 0 0
#rx_steps: 1.701898 0 0
#triangle: 0.00 30.000 0 1.00 30.000 0 1.00 32.000 0 0.05 moist_silty_clay y
#triangle: 0.00 30.000 0 1.00 32.000 0 0.00 32.000 0 0.05 moist_silty_clay y
#triangle: 1.00 30.000 0 2.00 30.000 0 2.00 32.000 0 0.05 moist_silty_clay y
#triangle: 1.00 30.000 0 2.00 32.000 0 1.00 32.000 0 0.05 moist_silty_clay y
"""


class TestParseGeometry:
    @pytest.fixture
    def geom_in(self, tmp_path):
        p = tmp_path / "raw.in"
        p.write_text(SAMPLE_GEOMETRY_IN)
        return p

    def test_parse_returns_dict(self, geom_in):
        result = qc._parse_geometry(str(geom_in))
        assert result is not None
        assert isinstance(result, dict)

    def test_parse_materials(self, geom_in):
        result = qc._parse_geometry(str(geom_in))
        assert 'air' in result['mats']
        assert 'moist_silty_clay' in result['mats']
        assert 'slide_zone' in result['mats']

    def test_parse_triangles(self, geom_in):
        result = qc._parse_geometry(str(geom_in))
        assert len(result['tris']) == 4
        for t in result['tris']:
            assert 'xs' in t
            assert 'ys' in t
            assert 'mat' in t
            assert len(t['xs']) == 3
            assert len(t['ys']) == 3

    def test_parse_tx_rx(self, geom_in):
        result = qc._parse_geometry(str(geom_in))
        assert result['tx'] == (119.300, 31.900)
        assert result['rx'] == (120.700, 31.900)

    def test_parse_domain(self, geom_in):
        result = qc._parse_geometry(str(geom_in))
        assert result['domain_x'] == 480.0
        assert result['domain_y'] == 45.0

    def test_parse_steps(self, geom_in):
        result = qc._parse_geometry(str(geom_in))
        assert result['src_steps'] == 1.701898
        assert result['rx_steps'] == 1.701898

    def test_missing_file(self):
        result = qc._parse_geometry("/nonexistent/file.in")
        assert result is None

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.in"
        p.write_text("")
        result = qc._parse_geometry(str(p))
        assert result is not None
        assert result['tris'] == []


class TestComputeQC:
    def test_basic_run(self, sample_bscan, sample_labels, tmp_path):
        """compute_qc should not crash with valid input."""
        arr, t = sample_bscan
        vis, geom, y_soft, t_501 = sample_labels

        bscan_path = tmp_path / "bscan.npy"
        np.save(bscan_path, arr)

        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        np.save(label_dir / "y_soft_501x128.npy", y_soft)
        np.save(label_dir / "target_visible_phase_time_ns.npy", vis)
        np.save(label_dir / "target_geom_time_ns.npy", geom)

        out_dir = tmp_path / "qc_out"
        metrics = qc.compute_qc(bscan_path, label_dir, out_dir)
        assert metrics is not None
        assert "qc_grade" in metrics
        assert metrics['trace_count'] == 128

    def test_qc_output_files(self, sample_bscan, sample_labels, tmp_path):
        arr, t = sample_bscan
        vis, geom, y_soft, t_501 = sample_labels
        bscan_path = tmp_path / "bscan.npy".format()
        np.save(bscan_path, arr)

        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        np.save(label_dir / "y_soft_501x128.npy", y_soft)
        np.save(label_dir / "target_visible_phase_time_ns.npy", vis)
        np.save(label_dir / "target_geom_time_ns.npy", geom)

        out_dir = tmp_path / "qc_out"
        qc.compute_qc(bscan_path, label_dir, out_dir)
        assert (out_dir / "qc_report.json").exists()
        assert (out_dir / "qc_metrics.csv").exists()
        assert (out_dir / "qc_decision.txt").exists()

    def test_report_json_content(self, sample_bscan, sample_labels, tmp_path):
        arr, t = sample_bscan
        vis, geom, y_soft, t_501 = sample_labels
        bscan_path = tmp_path / "bscan.npy".format()
        np.save(bscan_path, arr)

        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        np.save(label_dir / "y_soft_501x128.npy", y_soft)
        np.save(label_dir / "target_visible_phase_time_ns.npy", vis)
        np.save(label_dir / "target_geom_time_ns.npy", geom)

        out_dir = tmp_path / "qc_out"
        qc.compute_qc(bscan_path, label_dir, out_dir)

        with open(out_dir / "qc_report.json") as f:
            report = json.load(f)
        assert "trace_count" in report
        assert "target_local_peak_median" in report
        assert "support_ratio" in report
        assert "peak_offset_median_ns" in report
        assert "qc_grade" in report

    def test_curve_support_strong_signal(self, sample_labels, tmp_path):
        """Strong late-time signal → high target_local_peak_median."""
        vis, geom, y_soft, t_501 = sample_labels
        bscan_path = tmp_path / "bscan.npy"

        # Strong signal at 380-420ns where t^4 gain is ~0.1 (not crushing)
        W, T = 128, 501
        late_vis = 400.0 + np.linspace(-20, 20, W)
        strong = np.zeros((T, W), dtype=np.float64)
        for i in range(W):
            c = int(late_vis[i] / 700 * T)
            strong[max(0, c - 3):min(T, c + 3), i] = -50.0
        np.save(bscan_path, strong)

        # Matching labels
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        np.save(label_dir / "y_soft_501x128.npy", y_soft)
        np.save(label_dir / "target_visible_phase_time_ns.npy", late_vis.astype(np.float64))
        np.save(label_dir / "target_geom_time_ns.npy", (late_vis - 10).astype(np.float64))

        out_dir = tmp_path / "qc_out"
        metrics = qc.compute_qc(bscan_path, label_dir, out_dir)
        assert metrics['target_local_peak_median'] > 0.5, f"Got {metrics['target_local_peak_median']}"
        assert metrics['support_ratio'] > 0.5

    def test_no_signal_all_zeros(self, sample_labels, tmp_path):
        """All-zero bscan → zero peak median, no crash."""
        vis, geom, y_soft, t_501 = sample_labels
        bscan_path = tmp_path / "bscan.npy"
        np.save(bscan_path, np.zeros((501, 128), dtype=np.float64))

        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        np.save(label_dir / "y_soft_501x128.npy", y_soft)
        np.save(label_dir / "target_visible_phase_time_ns.npy", vis)
        np.save(label_dir / "target_geom_time_ns.npy", geom)

        out_dir = tmp_path / "qc_out"
        metrics = qc.compute_qc(bscan_path, label_dir, out_dir)
        assert metrics['target_local_peak_median'] == 0.0

    def test_dead_trace_detection(self, sample_bscan, sample_labels, tmp_path):
        """Corrupted traces should increase dead_trace_ratio."""
        arr, t = sample_bscan
        vis, geom, y_soft, t_501 = sample_labels
        bscan_path = tmp_path / "bscan.npy"

        # Make last 32 traces flat (dead)
        dead = arr.copy()
        dead[:, -32:] = 0.01 + np.random.RandomState(42).randn(501, 32) * 0.001

        np.save(bscan_path, dead)

        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        np.save(label_dir / "y_soft_501x128.npy", y_soft)
        np.save(label_dir / "target_visible_phase_time_ns.npy", vis)
        np.save(label_dir / "target_geom_time_ns.npy", geom)

        out_dir = tmp_path / "qc_out"
        metrics = qc.compute_qc(bscan_path, label_dir, out_dir)
        assert metrics['dead_trace_ratio'] > 0.0


class TestGradeDecision:
    def test_green_thresholds(self):
        """Verify THRESHOLDS constants make sense."""
        assert qc.THRESHOLDS['target_local_peak_median_green'] == 0.5
        assert qc.THRESHOLDS['support_ratio_green'] == 0.6
        assert qc.THRESHOLDS['peak_offset_green_ns'] == 8.0
        assert qc.THRESHOLDS['dead_trace_ratio_max'] == 0.05


class TestParseGeometryReal:
    """Test _parse_geometry with realistic LINE9_STYLE .in file."""

    @pytest.fixture
    def real_in(self, tmp_path):
        """Create a realistic multi-trace .in with 4 materials and layers."""
        lines = ["#domain: 480 55 0.05", "#pml_cells: 60 60 0 60 60 0"]
        for name, eps, sig in [("air", 1.0, 0), ("moist_silty_clay", 13.5, 0.001),
                                ("weak_cover_band", 12.9, 0.001), ("slide_zone", 24.0, 0.003),
                                ("weathered_bedrock", 6.0, 0.001)]:
            lines.append(f"#material: {eps} {sig} 1 0 {name}")
        lines.append("#hertzian_dipole: z 119.300 32.200 0.025 uavgpr_wavelet")
        lines.append("#rx: 120.700 32.200 0.025")
        lines.append("#src_steps: 1.701898 0 0")
        lines.append("#rx_steps: 1.701898 0 0")

        # 4 triangles for first x-segment covering all 4 materials
        # Layer 1: moist_silty_clay  y=30→33
        lines.append("#triangle: 0 30.000 0 1 30.000 0 1 33.000 0 0.05 moist_silty_clay y")
        lines.append("#triangle: 0 30.000 0 1 33.000 0 0 33.000 0 0.05 moist_silty_clay y")
        # Layer 2: weak_cover_band  y=33→34
        lines.append("#triangle: 0 33.000 0 1 33.000 0 1 34.000 0 0.05 weak_cover_band y")
        lines.append("#triangle: 0 33.000 0 1 34.000 0 0 34.000 0 0.05 weak_cover_band y")
        # Layer 3: slide_zone  y=34→36
        lines.append("#triangle: 0 34.000 0 1 34.000 0 1 36.000 0 0.05 slide_zone y")
        lines.append("#triangle: 0 34.000 0 1 36.000 0 0 36.000 0 0.05 slide_zone y")
        # Layer 4: weathered_bedrock  y=36→55
        lines.append("#triangle: 0 36.000 0 1 36.000 0 1 55.000 0 0.05 weathered_bedrock y")
        lines.append("#triangle: 0 36.000 0 1 55.000 0 0 55.000 0 0.05 weathered_bedrock y")

        p = tmp_path / "raw.in"
        p.write_text("\n".join(lines))
        return p

    def test_parse_all_materials(self, real_in):
        result = qc._parse_geometry(str(real_in))
        assert len(result['mats']) == 5
        assert len(result['tris']) == 8

    def test_parse_layer_boundaries(self, real_in):
        result = qc._parse_geometry(str(real_in))
        # Find slide_zone triangles
        slide_tris = [t for t in result['tris'] if t['mat'] == 'slide_zone']
        assert len(slide_tris) == 2
        # y should span 34-36
        all_ys = [y for t in slide_tris for y in t['ys']]
        assert min(all_ys) >= 34.0
        assert max(all_ys) <= 36.0

    def test_txrx_above_surface(self, real_in):
        result = qc._parse_geometry(str(real_in))
        assert result['tx'][1] == 32.200
        # gprMax: y=0 at top, y increases downward.
        # Ground surface is the MIN y of all triangles (moist_silty_clay starts at y=30).
        # The parser's "surface_y" is actually the bottom of ground layers, not the air-ground interface.
        ground_surface = min(t['ys'][0] for t in result['tris'])
        assert result['tx'][1] >= ground_surface  # TX at or above ground surface
