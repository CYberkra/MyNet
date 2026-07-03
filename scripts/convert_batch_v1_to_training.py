"""DEPRECATED: Convert batch v1 extracted B-scans to training .npz windows.
WARNING: This script uses per-case P99 normalization, NOT the global P99
used by current training pipeline (convert_pilot_to_training.py).
The clutter_gt semantics are also legacy (raw - target_only, missing air_only).
Use convert_pilot_to_training.py for new conversions.

Format matches existing simulation_pretrain_v1: (501, 256) per window,
P99-normalized raw, interface mask as y_target, reflect-padded to 256 traces.
"""
import csv, sys
from pathlib import Path

import numpy as np

MODELS = Path("D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pgda_batch_v1_3060/models")
OUT = Path("D:/Claude/PGDA-CSNet/data/simulation_pretrain_v1/windows")
IDX = OUT.parent / "window_index.csv"
WINDOW_WIDTH = 256
N_SAMPLES = 501
CENTER_PAD = True  # center 64 traces within 256-wide window; False = left-align


def p99_normalize(raw: np.ndarray) -> np.ndarray:
    """Same P99 normalization as real data preprocessing."""
    p99 = float(np.percentile(np.abs(raw), 99))
    if p99 < 1e-12:
        p99 = 1e-12
    return raw / p99


def create_window(raw: np.ndarray, mask: np.ndarray, pad_mode: str = "reflect") -> tuple:
    """Pad (N, 64) raw+mask to (N, 256) and derive status_code + label_weight.

    Padded traces have their y_mask, status_code, and label_weight zeroed
    so the loss function ignores them.
    """
    nx = raw.shape[1]
    if nx >= WINDOW_WIDTH:
        start = np.random.randint(0, nx - WINDOW_WIDTH + 1)
        rw = raw[:, start:start + WINDOW_WIDTH]
        mw = mask[:, start:start + WINDOW_WIDTH]
        original_cols = np.ones(WINDOW_WIDTH, dtype=bool)
    else:
        if CENTER_PAD:
            pad_left = (WINDOW_WIDTH - nx) // 2
            pad_right = WINDOW_WIDTH - nx - pad_left
        else:
            pad_left = 0
            pad_right = WINDOW_WIDTH - nx
        rw = np.pad(raw, ((0, 0), (pad_left, pad_right)), mode=pad_mode)
        mw = np.pad(mask, ((0, 0), (pad_left, pad_right)), mode=pad_mode)
        # Track which columns are original (non-padded)
        original_cols = np.zeros(WINDOW_WIDTH, dtype=bool)
        original_cols[pad_left:pad_left + nx] = True

    # Zero out padded regions in the mask
    mw[:, ~original_cols] = 0.0

    # status_code: 1 where any mask peak > 0.05, else 0
    peak_per_trace = mw.max(axis=0)
    status = (peak_per_trace > 0.05).astype(np.int16)

    # label_weight: peak value with 0.3 floor
    weight = np.maximum(peak_per_trace, 0.3).astype(np.float32)

    # Zero out padded traces in status and weight
    status[~original_cols] = 0
    weight[~original_cols] = 0.0

    return rw, mw, status, weight


def main():
    case_dirs = sorted(MODELS.glob("case_*"))
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    converted = 0
    skipped = 0

    for cdir in case_dirs:
        case_id = cdir.name
        raw_path = cdir / "outputs" / "raw_bscan.npy"
        mask_path = cdir / "labels" / "interface_mask_bscan.npy"
        if not raw_path.exists() or not mask_path.exists():
            skipped += 1
            continue

        raw = np.load(str(raw_path)).astype(np.float32)
        mask = np.load(str(mask_path)).astype(np.float32)

        # Skip NaN-filled cases
        if not np.isfinite(raw).all() or raw.size == 0:
            skipped += 1
            continue

        # P99 normalize (matching real data preprocessing)
        raw = p99_normalize(raw)

        # Clip mask to [0, 1]
        mask = np.clip(mask, 0.0, 1.0)

        # Create padded window
        rw, mw, sc, lw = create_window(raw, mask)
        assert rw.shape == (N_SAMPLES, WINDOW_WIDTH), f"raw shape mismatch: {rw.shape}"
        assert mw.shape == (N_SAMPLES, WINDOW_WIDTH), f"mask shape mismatch: {mw.shape}"
        assert sc.shape == (WINDOW_WIDTH,), f"status_code shape: {sc.shape}"
        assert lw.shape == (WINDOW_WIDTH,), f"label_weight shape: {lw.shape}"

        # Save
        sid = f"batch_v1_{case_id}_w00"
        npz_path = OUT / f"{sid}.npz"
        np.savez_compressed(
            npz_path,
            x_raw=rw.astype(np.float32),
            y_mask=mw.astype(np.float32),
            status_code=sc,
            label_weight=lw,
        )

        # Window index row
        n_present = int((sc == 1).sum())
        n_no_pick = WINDOW_WIDTH - n_present
        all_rows.append({
            "sample_id": sid,
            "line": case_id,
            "start": 0,
            "end": WINDOW_WIDTH - 1,
            "split": "train",
            "present": n_present,
            "weak": 0,
            "no_pick": n_no_pick,
        })
        converted += 1

        print(
            f"  {case_id}: raw range [{raw.min():.2f}, {raw.max():.2f}]  "
            f"present={n_present}/{WINDOW_WIDTH}  "
            f"label_weight=[{lw.min():.2f}, {lw.max():.2f}]"
        )

    # Write window index
    if all_rows:
        idx_path = OUT.parent / "window_index.csv"
        fieldnames = ["sample_id", "line", "start", "end", "split", "present", "weak", "no_pick"]
        with idx_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} entries to {idx_path}")
    else:
        print("\nNo valid cases converted!")

    print(f"\nSummary: {converted} converted, {skipped} skipped/NaN")
    return 0 if converted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
