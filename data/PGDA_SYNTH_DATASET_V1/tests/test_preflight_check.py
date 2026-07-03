"""Tests for preflight_check.py — parse, find, check, and run_preflight."""

import sys, os, json
import importlib.util
from pathlib import Path
import numpy as np
import pytest

# ── Import preflight_check module ──
TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

spec = importlib.util.spec_from_file_location("preflight_check", TOOLS_DIR / "preflight_check.py")
pfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfc)


class TestParseInFile:
    def test_parse_basic(self, sample_in_path):
        cmds = pfc.parse_in_file(sample_in_path)
        types = [t for t, _ in cmds]
        assert "domain" in types
        assert "pml_cells" in types
        assert "material" in types
        assert "hertzian_dipole" in types
        assert "rx" in types
        assert "triangle" in types

    def test_parse_commands_count(self, sample_in_path):
        cmds = pfc.parse_in_file(sample_in_path)
        assert len(cmds) > 5
        mats = [a for t, a in cmds if t == "material"]
        assert len(mats) == 5  # 5 materials

    def test_parse_with_comments(self, sample_in_with_comments):
        cmds = pfc.parse_in_file(sample_in_with_comments)
        # #: comment lines should not appear
        assert not any("comment" in str(a) for t, a in cmds)

    def test_parse_empty_file(self, tmp_path):
        p = tmp_path / "empty.in"
        p.write_text("")
        cmds = pfc.parse_in_file(p)
        assert cmds == []

    def test_parse_malformed(self, tmp_path):
        p = tmp_path / "weird.in"
        p.write_text("not a command\n#no_colon\n#:comment\n")
        cmds = pfc.parse_in_file(p)
        assert len(cmds) == 1
        assert cmds[0][0] == "no_colon"


class TestFindCommand:
    def test_finds_existing(self, sample_in_path):
        cmds = pfc.parse_in_file(sample_in_path)
        dom = pfc.find_command(cmds, "domain")
        assert dom is not None
        assert "480" in dom

    def test_returns_none_for_missing(self, sample_in_path):
        cmds = pfc.parse_in_file(sample_in_path)
        assert pfc.find_command(cmds, "nonexistent") is None

    def test_finds_first_only(self, sample_in_path):
        cmds = pfc.parse_in_file(sample_in_path)
        mat = pfc.find_command(cmds, "material")
        assert mat is not None
        assert "air" in mat  # first material


class TestFindAllCommands:
    def test_finds_all_materials(self, sample_in_path):
        cmds = pfc.parse_in_file(sample_in_path)
        mats = pfc.find_all_commands(cmds, "material")
        assert len(mats) == 5

    def test_finds_all_triangles(self, sample_in_path):
        cmds = pfc.parse_in_file(sample_in_path)
        tris = pfc.find_all_commands(cmds, "triangle")
        assert len(tris) == 2

    def test_returns_empty_list(self, sample_in_path):
        cmds = pfc.parse_in_file(sample_in_path)
        assert pfc.find_all_commands(cmds, "box") == []


class TestCheck:
    def test_check_appends_result(self):
        pfc.RESULTS.clear()
        pfc.check("test_check", "PASS", "detail text")
        assert len(pfc.RESULTS) == 1
        assert pfc.RESULTS[0] == ("test_check", "PASS", "detail text")

    def test_check_returns_status(self):
        pfc.RESULTS.clear()
        r = pfc.check("test_return", "FAIL", "oops")
        assert r == "FAIL"


class TestFindInFile:
    def test_finds_raw_in(self, sample_case_dir):
        path = pfc.find_in_file(sample_case_dir / "geometry")
        assert path is not None
        assert path.name == "raw.in"

    def test_falls_back_to_any_in(self, tmp_path):
        (tmp_path / "other.in").write_text("")
        path = pfc.find_in_file(tmp_path)
        assert path is not None
        assert path.suffix == ".in"

    def test_returns_none_for_empty(self, tmp_path):
        path = pfc.find_in_file(tmp_path)
        assert path is None


class TestRunPreflight:
    # All run_preflight tests need to override ROOT since temp paths
    # aren't under the real ROOT.
    _ROOT_BACKUP = None

    def _patch_root(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pfc, 'ROOT', tmp_path)

    def test_missing_dir_exits(self):
        with pytest.raises(SystemExit) as e:
            pfc.run_preflight("/nonexistent/path")
        assert e.value.code == 1

    def test_no_in_file(self, tmp_path, monkeypatch):
        self._patch_root(monkeypatch, tmp_path)
        # No .in file → early return, no sys.exit
        pfc.run_preflight(str(tmp_path))

    def test_valid_case_passes(self, sample_case_dir, monkeypatch):
        self._patch_root(monkeypatch, sample_case_dir)
        try:
            pfc.run_preflight(str(sample_case_dir / "geometry"))
        except SystemExit as e:
            assert e.code in (0, 2), f"Unexpected exit code: {e.code}"

    def test_h5_detected(self, tmp_path, monkeypatch):
        self._patch_root(monkeypatch, tmp_path)
        p = tmp_path / "raw.in"
        p.write_text("#geometry_objects_read: some.h5\n")
        with pytest.raises(SystemExit) as e:
            pfc.run_preflight(str(tmp_path))
        assert e.value.code == 1

    def test_missing_pml_fails(self, tmp_path, monkeypatch):
        self._patch_root(monkeypatch, tmp_path)
        p = tmp_path / "raw.in"
        p.write_text("#domain: 480 45 0.05\n#hertzian_dipole: z 120 32 0 1.0\n")
        with pytest.raises(SystemExit) as e:
            pfc.run_preflight(str(tmp_path))
        assert e.value.code == 1

    def test_label_non_flat_pass(self, sample_case_dir, monkeypatch):
        """Labels with range > 0.5ns should pass."""
        self._patch_root(monkeypatch, sample_case_dir)
        try:
            pfc.run_preflight(str(sample_case_dir / "geometry"))
        except SystemExit as e:
            assert e.code in (0, 2), f"Unexpected: {e.code}"

    def test_label_non_flat_fail(self, tmp_path, monkeypatch):
        """Flat labels should cause FAIL exit."""
        self._patch_root(monkeypatch, tmp_path)
        geo = tmp_path / "geometry"
        geo.mkdir()
        (geo / "raw.in").write_text("#domain: 480 45 0.05\n#pml_cells: 60 60 0 60 60 0\n")
        lbl = tmp_path / "labels"
        lbl.mkdir()
        vis = np.full(128, 150.0, dtype=np.float32)  # completely flat!
        np.save(lbl / "target_visible_phase_time_ns.npy", vis)
        np.save(lbl / "y_soft_501x128.npy", np.zeros((501, 128), dtype=np.float32))
        np.save(lbl / "interface_mask_bscan.npy", np.zeros((501, 128), dtype=np.float32))
        np.save(lbl / "target_geom_time_ns.npy", vis - 5)
        with pytest.raises(SystemExit) as e:
            pfc.run_preflight(str(tmp_path))
        assert e.value.code == 1

    def test_triangles_exceed_domain(self, tmp_path, monkeypatch):
        """Triangles exceeding domain_y should fail."""
        self._patch_root(monkeypatch, tmp_path)
        p = tmp_path / "raw.in"
        p.write_text(
            "#domain: 480 30 0.05\n"
            "#pml_cells: 60 60 0 60 60 0\n"
            "#triangle: 0 0 0 1 0 0 1 35 0 0.05 test_mat n\n"
            "#material: 1.0 0.0 1 0 test_mat\n"
        )
        with pytest.raises(SystemExit) as e:
            pfc.run_preflight(str(tmp_path))
        assert e.value.code == 1

    def test_txrx_buried(self, tmp_path, monkeypatch):
        """TX/RX within 1m of surface should fail."""
        self._patch_root(monkeypatch, tmp_path)
        p = tmp_path / "raw.in"
        p.write_text(
            "#domain: 480 45 0.05\n"
            "#pml_cells: 60 60 0 60 60 0\n"
            "#material: 1.0 0.0 1 0 air\n"
            "#material: 13.5 0.001 1 0 soil\n"
            "#hertzian_dipole: z 120 30.5 0.025 uavgpr_wavelet\n"
            "#rx: 121 30.5 0.025\n"
            "#triangle: 0 30.0 0 1 30.0 0 1 33.0 0 0.05 soil n\n"
            "#triangle: 0 30.0 0 1 33.0 0 0 33.0 0 0.05 soil n\n"
        )
        with pytest.raises(SystemExit) as exc:
            pfc.run_preflight(str(tmp_path))
        # TX at y=30.5, surface at y=30.0, gap=0.5m < 1.0m → FAIL
        assert exc.value.code == 1, f"Expected FAIL for buried TX/RX, got {exc.value.code}"
