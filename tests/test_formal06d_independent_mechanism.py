from pathlib import Path

from scripts.generate_formal06c_subtle_interface_development import DESIGN
from scripts.generate_formal06d_independent_mechanism_development import (
    FIELD_SEED,
    PROFILE_SEED,
    default_spec,
    generate,
)


def test_formal06d_locks_formal06c_measurement_contract_but_changes_seeds() -> None:
    spec = default_spec()
    assert spec.profile_seed == PROFILE_SEED
    assert spec.field_seed == FIELD_SEED
    assert DESIGN.bulk_long_x_scale_m == 6.7
    assert DESIGN.bulk_long_y_scale_m == 3.0
    assert DESIGN.bulk_meso_x_scale_m == 1.8
    assert DESIGN.bulk_meso_y_scale_m == 0.96
    assert DESIGN.bulk_meso_weight == 0.18


def test_formal06d_marks_new_geometry_as_development_only(tmp_path: Path) -> None:
    source = generate(tmp_path)
    manifest = __import__("json").loads((source / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formal_training_allowed"] is False
    assert manifest["line9_conditioned"] is True
    assert manifest["geometry"]["independent_geometry"]["arrays_reused_from_formal06c"] is False
    assert manifest["geometry"]["independent_geometry"]["profile_seed"] == PROFILE_SEED
    assert (source / "full_scene.in").is_file()
    assert (source / "no_basal_contrast_control.in").is_file()
    assert (source / "source_waveform.txt").is_file()
