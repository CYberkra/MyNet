from scripts.audit_independent_v2_promotion import audit


def test_independent_promotion_audit_keeps_release_blocked() -> None:
    result = audit()

    assert result["formal_training_allowed"] is False
    assert result["promotion_decision"] == "blocked_pending_new_independent_native256_release"
    by_id = {family["family"]: family for family in result["families"]}
    assert by_id["F01"]["disposition"] == "blocked_integrity_mismatch"
    assert by_id["F01"]["line9_conditioned"] is False
    assert by_id["F02"]["disposition"] == "blocked_integrity_mismatch"
    assert by_id["F02"]["line9_conditioned"] is True
    assert by_id["F03"]["disposition"] == "blocked_integrity_mismatch"
    assert all("hash mismatch" in error for family in result["families"] for error in family["errors"])
