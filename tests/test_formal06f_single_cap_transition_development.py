import json
from pathlib import Path

from scripts.generate_formal06f_single_cap_transition_development import (
    default_spec,
    generate,
    one_cap_material_rows,
)


def test_formal06f_has_one_full_contrast_weathered_cap() -> None:
    spec = default_spec()
    assert spec.transition_levels == 1
    full = one_cap_material_rows(spec, control=False)
    control = one_cap_material_rows(spec, control=True)
    assert len(full) == spec.cover_bins * 3
    assert full[spec.cover_bins].epsilon_r == 13.0
    assert control[spec.cover_bins].epsilon_r == control[0].epsilon_r


def test_formal06f_records_transition_as_the_only_changed_factor(tmp_path: Path) -> None:
    source = generate(tmp_path)
    manifest = json.loads((source / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formal_training_allowed"] is False
    assert manifest["ablation"]["predecessor_case_id"] == "FORMAL06D_INDEPENDENT_MECHANISM_DEVELOPMENT"
    assert len(manifest["ablation"]["changed"]) == 1
    assert manifest["transition_diagnosis"]["candidate_transition_levels"] == 1
    assert (source / "full_scene.in").is_file()
    assert (source / "no_basal_contrast_control.in").is_file()
