from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / 'data' / 'PGDA_SYNTH_DATASET_V2' / '00_controls'


def test_all_control_manifests_preserve_sampling_and_governance():
    index = json.loads((CONTROLS/'control_index.json').read_text(encoding='utf-8'))
    assert index['formal_training_allowed'] is False
    assert index['line9_conditioned'] is False
    assert index['case_count'] == 4
    for item in index['cases']:
        case = CONTROLS/item['case_id']
        m = json.loads((case/'scene_manifest.json').read_text(encoding='utf-8'))
        g = m['grid']
        assert m['formal_training_allowed'] is False
        assert m['line9_conditioned'] is False
        assert m['reference_line'] is None
        assert g['trace_count'] == 256
        assert math.isclose(g['trace_spacing_m'], 0.09)
        assert math.isclose(g['trace_span_m'], 22.95)
        assert g['canonical_output_samples'] == 501
        assert math.isclose(g['canonical_output_dt_ns'], 1.4)
        assert g['pml_cells'] == [20,20,0,20,20,0]
        assert g['guard_cells'] >= 20
        assert math.isclose(g['canonical_time_window_ns'], 700.0)
        assert math.isclose(g['solver_time_window_ns'], 701.0)
        t = np.load(case/'labels'/'time_501_ns.npy', allow_pickle=False)
        assert t.shape == (501,) and math.isclose(float(t[-1]),700.0)
        assert '-n 256 --geometry-fixed' in (case/'RUN_COMMANDS.md').read_text(encoding='utf-8')


def test_legacy_quarantine_remains_non_trainable():
    with (ROOT/'data'/'simulation_contract_v2'/'legacy_quarantine.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert rows
    assert all(r['legacy_status']=='legacy_quarantine' for r in rows)
    assert all(r['formal_training_allowed'].lower()=='false' for r in rows)
    assert all(r['line9_conditioned'].lower()=='true' for r in rows)


def test_ctrl02_ctrl04_pair_is_reciprocal_and_upper_geometry_identical():
    pos_dir = CONTROLS / "CTRL02_FLAT_DEEP_MODERATE_POS"
    neg_dir = CONTROLS / "CTRL04_MATCHED_BACKGROUND_NEG"
    pos = json.loads((pos_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    neg = json.loads((neg_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    assert pos["geometry"]["matched_negative_case_id"] == neg["case_id"]
    assert neg["geometry"]["matched_positive_case_id"] == pos["case_id"]
    assert pos["target_presence"] is True
    assert neg["target_presence"] is False
    assert pos["materials"]["set"] == neg["materials"]["set"]
    assert pos["grid"]["trace_count"] == neg["grid"]["trace_count"]
    for name in (
        "ground_y_m.npy",
        "flight_height_agl_m.npy",
        "antenna_y_m.npy",
        "cover_thickness_m.npy",
        "weathered_thickness_m.npy",
        "basal_interface_depth_m.npy",
    ):
        a = np.load(pos_dir / "labels" / name, allow_pickle=False)
        b = np.load(neg_dir / "labels" / name, allow_pickle=False)
        assert np.array_equal(a, b), name


def test_contract_json_matches_generated_grid_and_arrival_semantics():
    contract = json.loads((ROOT / "data" / "simulation_contract_v2" / "simulation_contract_v2.json").read_text(encoding="utf-8"))
    grid = contract["fdtd_reference_grid"]
    assert grid["guard_cells"] == 20
    assert math.isclose(grid["solver_time_window_ns"], 701.0)
    assert math.isclose(grid["canonical_resample_time_end_ns"], 700.0)
    policy = contract["label_policy"]
    assert "exact horizontal" in policy["flat_reference_arrival_time_ns"]
    assert "not claimed to be an exact" in policy["curved_reference_arrival_time_ns"]
