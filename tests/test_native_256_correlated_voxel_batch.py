from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from scripts.generate_native_256_correlated_voxel_batch import (
    DEFAULT_CATALOG,
    Q_OFFSETS,
    REGIONS,
    Spec,
    build_indices,
    build_profiles,
    correlated_quantiles,
    generate_case,
    lens_definitions,
    material_rows,
    morphology_metrics,
)


def _catalog_cases() -> list[dict[str, object]]:
    return json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))["cases"]


def _tiny_spec() -> Spec:
    return Spec(
        domain_x_m=12.96,
        domain_y_m=25.92,
        dl_m=0.09,
        pml_cells=2,
        trace_count=32,
        trace_spacing_m=0.18,
        scan_start_x_m=3.42,
        source_y_m=24.3,
        tx_rx_offset_m=0.18,
    )


def test_correlated_property_field_is_deterministic_and_multibin() -> None:
    spec = _tiny_spec()
    first, thresholds_first = correlated_quantiles(spec, 12345)
    second, thresholds_second = correlated_quantiles(spec, 12345)
    assert np.array_equal(first, second)
    assert thresholds_first == thresholds_second
    assert first.shape == (spec.nx, spec.ny)
    assert set(np.unique(first)) == set(range(len(Q_OFFSETS)))


def test_strict_pair_changes_only_transition_and_bedrock_materials() -> None:
    full = material_rows("balanced", control=False)
    control = material_rows("balanced", control=True)
    changed = [
        int(a["index"])
        for a, b in zip(full, control)
        if (a["epsilon_r"], a["conductivity_s_per_m"])
        != (b["epsilon_r"], b["conductivity_s_per_m"])
    ]
    assert changed == list(range(2 * len(Q_OFFSETS), len(REGIONS) * len(Q_OFFSETS)))
    assert full[: 2 * len(Q_OFFSETS)] == control[: 2 * len(Q_OFFSETS)]
    assert full[len(REGIONS) * len(Q_OFFSETS) :] == control[len(REGIONS) * len(Q_OFFSETS) :]


def test_true_negative_geometry_contains_no_target_region_indices() -> None:
    spec = _tiny_spec()
    negative = next(case for case in _catalog_cases() if not case["target_presence"])
    quantiles, _ = correlated_quantiles(spec, int(negative["field_seed"]))
    profiles = build_profiles(spec, negative)
    lenses = lens_definitions(spec, str(negative["lens_family"]))
    data = build_indices(spec, negative, quantiles, profiles, lenses)
    target_indices = set(range(2 * len(Q_OFFSETS), len(REGIONS) * len(Q_OFFSETS)))
    assert not target_indices.intersection(set(np.unique(data)))
    assert "basal_depth_m" not in profiles


def test_finite_lenses_taper_without_vertical_partition_walls() -> None:
    spec = _tiny_spec()
    negative = next(case for case in _catalog_cases() if not case["target_presence"])
    quantiles, _ = correlated_quantiles(spec, int(negative["field_seed"]))
    profiles = build_profiles(spec, negative)
    lenses = [
        {"material": 0, "x0": 3.6, "x1": 8.1, "depth": 6.1, "thickness": 0.54},
        {"material": 1, "x0": 4.5, "x1": 10.8, "depth": 7.8, "thickness": 0.45},
    ]
    data = build_indices(spec, negative, quantiles, profiles, lenses)[:, :, 0]
    lens_start = len(REGIONS) * len(Q_OFFSETS)
    for lens in lenses:
        material = lens_start + int(lens["material"])
        counts = np.sum(data == material, axis=1)
        active = np.flatnonzero(counts)
        assert active.size > 2
        assert counts[active[0]] <= 1
        assert counts[active[-1]] <= 1
        assert int(np.max(np.abs(np.diff(counts)))) <= 2


def test_native_positive_scan_crops_are_not_single_quadratic_bowls() -> None:
    spec = Spec()
    for case in _catalog_cases():
        if not case["target_presence"]:
            continue
        profiles = build_profiles(spec, case)
        metrics = morphology_metrics(spec, profiles, True, str(case["material_family"]))
        assert metrics["broad_morphology_gate_ok"] is True
        assert metrics["single_quadratic_bowl_rejected"] is True
        assert metrics["smoothed_extrema_count"] >= 2


def test_tiny_negative_deck_writes_compressed_hdf5_and_no_control(tmp_path: Path) -> None:
    negative = next(case for case in _catalog_cases() if not case["target_presence"])
    manifest = generate_case(
        tmp_path,
        negative,
        overwrite=False,
        spec=_tiny_spec(),
        catalog_path=DEFAULT_CATALOG,
    )
    case_dir = tmp_path / str(negative["case_id"])
    with h5py.File(case_dir / "geology_indices.h5", "r") as handle:
        dataset = handle["data"]
        assert dataset.compression == "gzip"
        assert dataset.shape == tuple(manifest["geometry"]["index_shape"])
        assert tuple(handle.attrs["dx_dy_dz"]) == (_tiny_spec().dl_m,) * 3
    assert (case_dir / "full_scene.in").is_file()
    assert (case_dir / "air_reference.in").is_file()
    assert not (case_dir / "no_basal_contrast_control.in").exists()
    assert manifest["formal_training_allowed"] is False
    assert manifest["line9_conditioned"] is False


def test_tiny_positive_deck_writes_reference_compatibility_aliases(tmp_path: Path) -> None:
    positive = next(case for case in _catalog_cases() if case["target_presence"])
    generate_case(
        tmp_path,
        positive,
        overwrite=False,
        spec=_tiny_spec(),
        catalog_path=DEFAULT_CATALOG,
    )
    labels = tmp_path / str(positive["case_id"]) / "labels"
    canonical = np.load(labels / "geometric_reference_arrival_time_ns.npy")
    assert np.array_equal(np.load(labels / "reference_arrival_time_ns.npy"), canonical)
    assert np.array_equal(np.load(labels / "geometric_arrival_time_ns.npy"), canonical)
