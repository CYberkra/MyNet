import json
from pathlib import Path

from scripts.generate_formal06d_independent_mechanism_development import default_spec
from scripts.generate_formal06e_nonlayered_cover_development import (
    COVER_COVARIANCE,
    build_bulk_field,
    generate,
)


def test_formal06e_cover_covariance_is_near_isotropic_and_nonlayered() -> None:
    _, _, stats = build_bulk_field(default_spec())
    gates = stats["gate_results"]
    assert all(gates.values())
    assert COVER_COVARIANCE["broad_xy_m"][0] / COVER_COVARIANCE["broad_xy_m"][1] < 1.2
    assert stats["sinusoidal_stratigraphy"] is False
    assert stats["isolated_inclusions"] == 0


def test_formal06e_is_development_only_and_records_the_single_change(tmp_path: Path) -> None:
    source = generate(tmp_path)
    manifest = json.loads((source / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formal_training_allowed"] is False
    assert manifest["line9_conditioned"] is True
    assert manifest["ablation"]["predecessor_case_id"] == "FORMAL06D_INDEPENDENT_MECHANISM_DEVELOPMENT"
    assert manifest["ablation"]["changed"] == [
        "non-target cover latent-field covariance: elongated to near-isotropic"
    ]
    assert "full_scene_target_to_local_background_rms_max" not in manifest["visibility_gate"]
    assert (source / "full_scene.in").is_file()
    assert (source / "no_basal_contrast_control.in").is_file()
