from pathlib import Path

import numpy as np
import pytest

from scripts import generate_macro04_deeper_dropout_voxel as macro04
from scripts.generate_macro05_nightly_batch import (
    C0,
    FAMILIES,
    GridSpec,
    equivalence_geometry,
    family_properties,
    make_lenses,
    make_profiles,
    material_rows,
    reference_arrival,
    write_readme,
    write_runners,
)


def test_macro05_domain_protects_target_window_and_reduces_cost():
    spec = GridSpec()
    right_guard = (
        spec.domain_x_m
        - spec.pml_cells * spec.dl_m
        - (spec.scan_start_x_m + spec.scan_span_m + spec.tx_rx_offset_m)
    )
    assert spec.physical_guard_m == 80.0
    assert right_guard == pytest.approx(80.0)
    assert 2e9 * min(spec.physical_guard_m, right_guard) / C0 > 500.0
    assert spec.domain_x_m / macro04.Spec().domain_x_m < 0.80


def test_macro05_ten_families_are_independent_and_fit_protected_window():
    spec = GridSpec()
    assert len(FAMILIES) == 10
    assert len({family.case_id for family in FAMILIES}) == 10
    assert len({family.field_seed for family in FAMILIES}) == 10
    assert len({family.profile_seed for family in FAMILIES}) == 10
    assert all("line9" not in family.design_note.lower() for family in FAMILIES)
    for family in FAMILIES[1:]:
        profile = make_profiles(spec, family)
        labels = reference_arrival(spec, profile, family_properties(family))
        assert labels["geometric_reference_arrival_time_ns"].max() < 500.0
        assert labels["basal_interface_depth_m"].min() > 12.0
        assert labels["basal_interface_depth_m"].max() < 18.0


def test_macro05_strict_pair_changes_only_transition_and_bedrock():
    spec = GridSpec()
    for family in FAMILIES[1:]:
        properties = family_properties(family)
        lenses = make_lenses(spec, family)
        full = material_rows(properties, lenses, control=False)
        control = material_rows(properties, lenses, control=True)
        changed = [
            index
            for index, (left, right) in enumerate(zip(full, control))
            if left != right
        ]
        assert changed == list(range(10, 30))


def test_macro05_f01_is_exact_shifted_macro04_scan_contract():
    spec = GridSpec()
    _data, profile, properties, _lenses, _full, _control, _thresholds = equivalence_geometry(spec)
    new = reference_arrival(spec, profile, properties)
    old_spec = macro04.Spec()
    old = macro04.reference_arrival(old_spec, macro04.profiles(old_spec))
    for name in (
        "ground_y_m",
        "flight_height_m",
        "basal_interface_depth_m",
        "transition_thickness_m",
        "geometric_reference_arrival_time_ns",
    ):
        assert np.array_equal(new[name], old[name])
    assert np.allclose(new["source_x_m"], old["source_x_m"] - 37.0, atol=2e-5, rtol=0.0)


def test_macro05_portable_runner_supports_direct_case_and_resume(tmp_path: Path):
    write_runners(tmp_path)
    write_readme(tmp_path)
    runner = (tmp_path / "RUN_NIGHTLY_GPU.cmd").read_text(encoding="ascii")
    one = (tmp_path / "RUN_ONE_CASE_GPU.cmd").read_text(encoding="ascii")
    assert 'if not "%~1"==""' in runner
    assert "pair_complete.marker" in runner
    assert "nightly_case_order.all.txt" not in one
    assert 'RUN_NIGHTLY_GPU.cmd" "%~1"' in one
    assert "533.7 ns" in (tmp_path / "README_NIGHTLY.md").read_text(encoding="ascii")
