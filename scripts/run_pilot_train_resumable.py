#!/usr/bin/env python3
"""Resumable batch runner using SafeGprMaxRunner. Skips cases with 128 .out files."""
import csv
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "uavgpr_simlab" / "src"
sys.path.insert(0, str(SRC))

from uavgpr_simlab.core.runner import SafeGprMaxRunner, resolve_manifest_path

PYTHON_EXE = r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
MANIFEST = Path("D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pilot_train_v1_3060/yingshan_pilot_train_3060_v1/datasets/yingshan_pilot_train_3060_v1_manifest.csv")
WORKSPACE_ROOT = MANIFEST.resolve().parent.parent
MODELS_ROOT = WORKSPACE_ROOT / "models"
GPU_TIMEOUT_S = 14400
EXPECTED = 128
N_TRACES = 128

log_dir = WORKSPACE_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(log_dir / "resumable_runner.log", "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y%m%d_%H%M%S')}] {msg}\n")


def safe_input_file(row):
    raw_value = row.get("input_file", "")
    raw_path = Path(raw_value)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError(f"invalid manifest input_file for {row.get('case_id')}: {raw_value}")

    case_id = row.get("case_id", "")
    variant = row.get("variant", "")
    expected = Path("models") / case_id / f"{variant}.in"
    if raw_path.as_posix() != expected.as_posix():
        raise ValueError(f"unexpected manifest input_file for {case_id}: {raw_value}")

    input_file = resolve_manifest_path(raw_value, manifest)
    resolved = input_file.resolve()
    if MODELS_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"manifest path escapes models root for {case_id}: {resolved}")
    if resolved.parent.name != case_id or resolved.name != f"{variant}.in":
        raise ValueError(f"manifest path does not match case/variant for {case_id}: {resolved}")
    return resolved

# Read manifest, filter to raw only, check completion
manifest = MANIFEST.resolve()
pending = []
with manifest.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("variant", "") != "raw":
            continue
        input_file = safe_input_file(row)
        case_dir = input_file.parent
        out_files = sorted(case_dir.glob("raw*.out"))
        n_existing = len(out_files)
        # Validate beyond count: check first and last .out are readable
        valid = False
        if n_existing >= EXPECTED:
            try:
                import h5py, numpy as np
                with h5py.File(out_files[0], "r") as h:
                    first_shape = np.asarray(h["rxs"]["rx1"]["Ez"]).shape
                with h5py.File(out_files[-1], "r") as h:
                    last_shape = np.asarray(h["rxs"]["rx1"]["Ez"]).shape
                valid = (first_shape == last_shape and first_shape[0] > 0)
            except Exception:
                valid = False
        if valid:
            log(f"SKIP {row['case_id']}: {n_existing}/{EXPECTED} .out files validated")
            continue
        # Clean partial
        for f in case_dir.glob("raw*.out"):
            f.unlink()
        for f in case_dir.glob("raw*.vti"):
            f.unlink()
        pending.append(row)

log(f"\nPending: {len(pending)} cases")
if not pending:
    log("All done!")
    sys.exit(0)

results = {"total": 0, "success": 0, "timeout": 0, "failed": 0, "tasks": []}

for i, row in enumerate(pending):
    input_file = resolve_manifest_path(row["input_file"], manifest)
    case_id = row["case_id"]
    task_key = f"{case_id}_raw"
    results["total"] += 1

    log(f"\n[{i+1}/{len(pending)}] {task_key} ({N_TRACES} traces)")
    cmd = [PYTHON_EXE, "-m", "gprMax", str(input_file), "-gpu", "-n", str(N_TRACES), "--geometry-fixed"]

    runner = SafeGprMaxRunner(cmd=cmd, cwd=str(log_dir), timeout=GPU_TIMEOUT_S, log_dir=log_dir)
    result = runner.run()

    n_final = len(list(input_file.parent.glob("raw*.out")))
    emoji = {"success": "[OK]", "timeout": "[TIMEOUT]", "failed": "[FAIL]"}
    log(f"  {emoji.get(result.status, '?')} in {result.elapsed:.0f}s ({n_final}/{EXPECTED} files)")

    task_result = {
        "task_key": task_key, "case_id": case_id,
        "status": result.status, "return_code": result.return_code,
        "elapsed": result.elapsed, "files": n_final,
    }
    results["tasks"].append(task_result)
    results[result.status] = results.get(result.status, 0) + 1

    # Audit every 5 cases
    if (i + 1) % 5 == 0:
        n_ok = sum(1 for t in results["tasks"] if t["status"] == "success" and t["files"] >= EXPECTED)
        log(f"\n[AUDIT] {i+1}/{len(pending)} done: {n_ok} success")

# Final summary
summary_path = log_dir / f"resumable_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
ok = results["success"]
log(f"\n=== Done: {ok}/{results['total']} success ===")
