#!/usr/bin/env python3
"""Post-process gprMax .out files to merged .npy B-scans for Pilot-Train batch.

Merges individual raw*.out → outputs/raw_bscan_native.npy (5937, 128)
Also copies shared artifacts from the first completed case.

Usage:
  "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe" scripts/postprocess_simulation_batch.py
"""
import sys
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "uavgpr_simlab" / "src"
sys.path.insert(0, str(SRC))

from uavgpr_simlab.core.postprocess import merge_bscan_from_outputs

MODELS = Path(
    "D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pilot_train_v1_3060/"
    "yingshan_pilot_train_3060_v1/models"
)
N_TRACES = 128
NATIVE_SAMPLES = 5937


def sort_out_files(case_dir, prefix="raw"):
    """Sort raw*.out files by trace number."""
    files = sorted(case_dir.glob(f"{prefix}*.out"),
                   key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
    return files


def process_case(case_dir):
    """Merge .out → raw_bscan_native.npy for one case."""
    case_id = case_dir.name

    # Check if already done
    out_dir = case_dir / "outputs"
    out_npy = out_dir / "raw_bscan_native.npy"
    if out_npy.exists():
        print(f"  {case_id}: raw_bscan_native.npy exists, skip")
        return True

    out_files = sort_out_files(case_dir, "raw")
    if len(out_files) < N_TRACES:
        print(f"  {case_id}: only {len(out_files)}/{N_TRACES} .out files, skipping")
        return False

    try:
        bscan, meta = merge_bscan_from_outputs(out_files)
    except Exception as e:
        print(f"  {case_id}: merge error: {e}")
        return False

    if bscan.shape != (NATIVE_SAMPLES, N_TRACES):
        print(f"  {case_id}: merged shape {bscan.shape}, expected ({NATIVE_SAMPLES},{N_TRACES})")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, bscan.astype(np.float32))
    print(f"  {case_id}: {out_npy.name} saved ({bscan.shape})")
    return True


def main():
    case_dirs = sorted(MODELS.glob("case_*"))
    print(f"Found {len(case_dirs)} case directories")

    ok = fail = skip = 0
    for c in case_dirs:
        if process_case(c):
            ok += 1
        else:
            # Check if it has outputs/raw_bscan_native.npy already
            if (c / "outputs" / "raw_bscan_native.npy").exists():
                skip += 1
            else:
                fail += 1

    print(f"\nDone: {ok} processed, {skip} existed, {fail} incomplete")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
