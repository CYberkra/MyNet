"""Batch postprocess Pilot-Mini: merge gprMax .out files into .npy bscan arrays.

WARNING: clutter_gt = raw - target_only is LEGACY semantics. The current
PGDA-CSNet signal model defines:
  X_clean = Y_target - Y_air = S + G
  C_gt    = Y_full - X_clean = raw - (target_only - air_only) = A + E
For new conversions use the air_only subtraction. See CLAUDE.md signal model.

For each case (000001-000020) and variant (raw/target_only/background_only/air_only),
calls merge_available_bscan_for_input() to merge matching .out files and saves
to outputs/{variant}_bscan.npy. Also computes clutter_gt = raw - target_only.

Skips variants where .out files are missing or incomplete (e.g. air_only still
being simulated). Run this now for raw/target_only/background_only which are
complete, and re-run after air_only finishes.
"""
import sys
from pathlib import Path

import numpy as np

from uavgpr_simlab.core.postprocess import merge_available_bscan_for_input

WORKSPACE = Path(
    "D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pilot_mini_v1/"
    "yingshan_pilot_3060_v1"
)
MODELS = WORKSPACE / "models"
VARIANTS = ["raw", "target_only", "background_only", "air_only"]


def main():
    case_dirs = sorted(MODELS.glob("case_*"))
    print(f"Found {len(case_dirs)} case directories\n")

    counts: dict[str, int] = {v: 0 for v in VARIANTS}
    errors: list[str] = []

    for cdir in case_dirs:
        case_id = cdir.name
        out_dir = cdir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        for variant in VARIANTS:
            input_file = cdir / f"{variant}.in"
            if not input_file.exists():
                print(f"  {case_id}/{variant}: .in file missing, skipping")
                continue

            result = merge_available_bscan_for_input(input_file)
            if result is None:
                print(f"  {case_id}/{variant}: no .out files found, skipping")
                continue

            bscan, meta = result
            out_path = out_dir / f"{variant}_bscan.npy"
            np.save(str(out_path), bscan.astype(np.float32))
            n_traces = bscan.shape[1]
            n_samples = bscan.shape[0]
            counts[variant] += 1
            print(
                f"  {case_id}/{variant}: {n_samples}x{n_traces} "
                f"-> {out_path.name} "
                f"({n_traces}/{n_traces} traces)"
            )

    # Compute clutter_gt = raw - target_only for each case
    print("\n--- Computing clutter_gt ---")
    clutter_count = 0
    for cdir in case_dirs:
        out_dir = cdir / "outputs"
        raw_path = out_dir / "raw_bscan.npy"
        target_path = out_dir / "target_only_bscan.npy"
        if not raw_path.exists() or not target_path.exists():
            continue
        raw = np.load(str(raw_path))
        target = np.load(str(target_path))
        clutter_gt = raw - target
        np.save(str(out_dir / "clutter_gt_bscan.npy"), clutter_gt.astype(np.float32))
        clutter_count += 1
        print(f"  {cdir.name}: clutter_gt shape {clutter_gt.shape}, "
              f"range [{clutter_gt.min():.4f}, {clutter_gt.max():.4f}]")

    # Summary
    print(f"\n{'='*60}")
    print(f"Postprocessing complete:")
    for v in VARIANTS:
        print(f"  {v}: {counts[v]}/20 cases")
    print(f"  clutter_gt: {clutter_count}/20 cases")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
