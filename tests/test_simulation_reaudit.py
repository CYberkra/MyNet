from scripts.revalidate_existing_simulations import _development_decision, _semantic_decision


def test_shifted_soft_target_requires_rebuild_before_development():
    metrics = {
        "curve_training_contract_ok": False,
        "soft_target_semantics": "GEOMETRY_TO_VISIBLE_BAND_OR_SHIFTED",
        "automatic_signal_grade": "SUPPORTED",
    }
    assert _semantic_decision(metrics) == "REBUILD_VISIBLE_PHASE_DISTRIBUTION"
    assert _development_decision(metrics, {"visual_decision": "VISUAL_PASS"}) == "REBUILD_LABEL_BEFORE_DEVELOPMENT"


def test_shifted_soft_target_with_phase_artifact_requires_both_repairs():
    metrics = {
        "curve_training_contract_ok": False,
        "soft_target_semantics": "GEOMETRY_TO_VISIBLE_BAND_OR_SHIFTED",
        "automatic_signal_grade": "SUPPORTED",
    }
    assert _development_decision(metrics, {"visual_decision": "VISUAL_REVIEW_LOCAL_ARTIFACT"}) == "REBUILD_LABEL_AND_LOCAL_IGNORE"
