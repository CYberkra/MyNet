import json
from pathlib import Path

import h5py
import numpy as np

from scripts.generate_formal06d_independent_mechanism_development import generate as generate_formal06d
from scripts.generate_formal06h_source_temporal_development import SOURCE, generate


def test_formal06h_changes_only_the_temporal_source_contract(tmp_path: Path) -> None:
    predecessor = generate_formal06d(tmp_path)
    candidate = generate(tmp_path)
    with h5py.File(predecessor / "geology_indices.h5", "r") as previous_handle:
        previous = np.asarray(previous_handle["data"])
    with h5py.File(candidate / "geology_indices.h5", "r") as candidate_handle:
        current = np.asarray(candidate_handle["data"])
    assert np.array_equal(previous, current)

    manifest = json.loads((candidate / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["kind"] == "ricker"
    assert manifest["source"]["center_frequency_hz"] == SOURCE.center_frequency_hz
    assert len(manifest["ablation"]["changed"]) == 1
    assert manifest["formal_training_allowed"] is False
    assert (candidate / "preview_source_waveforms.png").is_file()
