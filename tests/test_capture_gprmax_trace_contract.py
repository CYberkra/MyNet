from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def write_trace(path: Path, source_x: float, receiver_x: float) -> None:
    trace = path
    with h5py.File(trace, "w") as handle:
        handle.attrs["gprMax"] = "3.1.7"
        handle.attrs["Iterations"] = 5
        handle.attrs["dt"] = 1e-10
        source = handle.create_group("srcs/src1")
        source.attrs["Position"] = (source_x, 29.65, 0.0)
        receiver = handle.create_group("rxs/rx1")
        receiver.attrs["Position"] = (receiver_x, 29.65, 0.0)
        receiver.create_dataset("Ez", data=np.arange(5, dtype=np.float32))


def run_capture(tmp_path: Path, output: Path, expected: int, timeout: float = 2.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "capture_gprmax_trace_contract.py"),
            str(tmp_path),
            "--prefix",
            "paired",
            "--expected",
            str(expected),
            "--output",
            str(output),
            "--poll-seconds",
            "0.01",
            "--timeout-seconds",
            str(timeout),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_capture_trace_contract_preserves_positions_and_shapes(tmp_path: Path) -> None:
    write_trace(tmp_path / "paired.out", 120.0, 120.2)
    output = tmp_path / "contract.json"
    result = run_capture(tmp_path, output, expected=1)
    assert result.returncode == 0, result.stderr
    contract = json.loads(output.read_text(encoding="utf-8"))
    assert contract["complete"] is True
    assert contract["captured_trace_count"] == 1
    captured = contract["traces"][0]
    assert captured["source_positions_m"] == [[120.0, 29.65, 0.0]]
    assert captured["receiver_positions_m"] == [[120.2, 29.65, 0.0]]
    assert captured["receiver_shapes"]["rx1"]["Ez"] == [5]
    assert len(captured["sha256"]) == 64


def test_capture_trace_contract_resumes_partial_report(tmp_path: Path) -> None:
    write_trace(tmp_path / "paired1.out", 120.0, 120.2)
    output = tmp_path / "contract.json"
    partial = run_capture(tmp_path, output, expected=2, timeout=0.05)
    assert partial.returncode == 2
    first = json.loads(output.read_text(encoding="utf-8"))
    assert first["captured_trace_count"] == 1
    assert first["complete"] is False

    write_trace(tmp_path / "paired2.out", 121.7, 121.9)
    resumed = run_capture(tmp_path, output, expected=2)
    assert resumed.returncode == 0, resumed.stderr
    final = json.loads(output.read_text(encoding="utf-8"))
    assert final["captured_trace_count"] == 2
    assert final["complete"] is True
    assert [row["trace_index"] for row in final["traces"]] == [1, 2]
    assert final["traces"][1]["source_positions_m"] == [[121.7, 29.65, 0.0]]


def test_capture_trace_contract_reloads_changed_trace(tmp_path: Path) -> None:
    trace = tmp_path / "paired.out"
    write_trace(trace, 120.0, 120.2)
    output = tmp_path / "contract.json"
    first = run_capture(tmp_path, output, expected=1)
    assert first.returncode == 0, first.stderr
    old_hash = json.loads(output.read_text(encoding="utf-8"))["traces"][0]["sha256"]

    write_trace(trace, 130.0, 130.2)
    second = run_capture(tmp_path, output, expected=1)
    assert second.returncode == 0, second.stderr
    updated = json.loads(output.read_text(encoding="utf-8"))["traces"][0]
    assert updated["sha256"] != old_hash
    assert updated["source_positions_m"] == [[130.0, 29.65, 0.0]]
