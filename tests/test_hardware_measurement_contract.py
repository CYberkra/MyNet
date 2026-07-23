from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.validate_hardware_measurement_contract import READY_STATUS, validate_contract


def _contract() -> dict:
    path = Path(__file__).resolve().parents[1] / "data" / "contracts" / "simulation_v2" / "hardware_measurement_contract_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_pending_contract_blocks_a_finite_antenna_run() -> None:
    errors, warnings = validate_contract(_contract(), Path.cwd(), require_ready=True)
    assert "contract is not ready_for_3d_local_preflight" in errors
    assert warnings


def test_ready_contract_requires_traceable_pulse_evidence(tmp_path: Path) -> None:
    payload = _contract()
    pulse = tmp_path / "air_reference.csv"
    pulse.write_text("time_s,amplitude\n0,0\n1e-9,1\n", encoding="utf-8")
    recipe = tmp_path / "preprocessing.json"
    recipe.write_text("{}\n", encoding="utf-8")
    payload["status"] = READY_STATUS
    payload["hardware_identity"].update({"system_model": "example", "antenna_model": "example"})
    payload["antenna_geometry"].update({
        "element_type": "documented_example",
        "physical_length_m": 1.0,
        "tx_rx_separation_m": 0.2,
        "polarization": "documented_example",
    })
    payload["source_measurement"].update({
        "direct_or_air_reference_raw_path": str(pulse),
        "direct_or_air_reference_sha256": hashlib.sha256(pulse.read_bytes()).hexdigest(),
        "sample_interval_s": 1e-9,
        "preprocessing_manifest_path": str(recipe),
    })
    errors, _ = validate_contract(payload, tmp_path, require_ready=True)
    assert errors == []
