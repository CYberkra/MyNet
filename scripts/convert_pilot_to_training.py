"""Convert Pilot-Train B-scans to training .npz windows (v2).

Pipeline:
  simlab workspace (case_*/outputs/raw_bscan_native.npy)
  → linear interp to 501 samples
  → global P99 normalization across all cases
  → pad 128 → 256 traces (zero in padding)
  → load fixed interface_mask_bscan.npy
  → create soft mask (Gaussian, sigma=8)
  → save NPZ (x_raw, y_mask, y_soft, status_code, label_weight, label_weight_2d)
  → save metadata (time axes, resample_config, velocity_model)

Fixes vs v1:
  - No FFT resample (linear interp, no ringing)
  - No per-case P99 (global P99 across all cases)
  - Padding zeros in x_raw, y_mask, label_weight
  - y_soft mask for training gradient
  - label_weight_2d in NPZ
  - Full metadata lineage
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

WORKSPACE = Path(
    "D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pilot_train_v1_3060/"
    "yingshan_pilot_train_3060_v1"
)
MODELS = WORKSPACE / "models"
OUT = Path("D:/Claude/PGDA-CSNet/data/simulation_pretrain_v3/windows")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_WIDTH = 256
N_SAMPLES = 501
T_WINDOW_NS = 700.0
NATIVE_SAMPLES = 5937
N_TRACES = 128
PAD = (WINDOW_WIDTH - N_TRACES) // 2  # 64


def compute_global_p99(case_dirs):
    """Compute unified P99 across all cases (from the resampled 501 data)."""
    old = np.linspace(0, 1, NATIVE_SAMPLES)
    new = np.linspace(0, 1, N_SAMPLES)
    all_vals = []
    for c in case_dirs:
        raw = np.load(c / "outputs" / "raw_bscan_native.npy").astype(np.float32)
        if raw.shape != (NATIVE_SAMPLES, N_TRACES):
            print(f"  WARN: {c.name} raw shape {raw.shape}, skipping")
            continue
        rs = np.zeros((N_SAMPLES, N_TRACES), dtype=np.float32)
        for i in range(N_TRACES):
            rs[:, i] = np.interp(new, old, raw[:, i])
        all_vals.append(np.abs(rs).ravel())
    all_vals = np.concatenate(all_vals)
    p99 = float(np.percentile(all_vals, 99))
    if p99 < 1e-12:
        p99 = 1e-12
    return p99


def create_windows(raw_501, mask, global_p99):
    """Create NPZ data: resample, normalize, pad, add metadata."""
    # P99 normalize
    rw = raw_501 / global_p99

    # Pad 128 → 256 (center-pad)
    x_raw = np.pad(rw, ((0, 0), (PAD, PAD)), mode="reflect")
    y_mask = np.pad(mask.astype(np.float32), ((0, 0), (PAD, PAD)), mode="reflect")

    # Zero padding
    x_raw[:, :PAD] = 0.0
    x_raw[:, -PAD:] = 0.0
    y_mask[:, :PAD] = 0.0
    y_mask[:, -PAD:] = 0.0

    # Soft mask: Gaussian blur on valid (128) traces, then pad to 256, normalize peak=1.0
    y_soft_valid = gaussian_filter(mask.astype(np.float32), sigma=(8, 1))
    y_soft_valid = y_soft_valid.clip(0, 1)
    y_soft_valid = y_soft_valid / max(y_soft_valid.max(), 1e-10)
    y_soft = np.zeros((N_SAMPLES, WINDOW_WIDTH), dtype=np.float32)
    y_soft[:, PAD:PAD + N_TRACES] = y_soft_valid
    # Padding area stays 0

    # status_code from hard mask
    peak = y_mask[:, PAD:-PAD].max(axis=0)
    status = np.full(WINDOW_WIDTH, 2, dtype=np.int16)
    status[PAD:-PAD][peak > 0.5] = 1
    status[PAD:-PAD][peak < 0.1] = 0
    status[:PAD] = 0
    status[-PAD:] = 0

    # label_weight (trace-level)
    weight = np.full(WINDOW_WIDTH, 0.3, dtype=np.float32)
    weight[PAD:-PAD] = np.maximum(peak, 0.3)
    weight[:PAD] = 0.0
    weight[-PAD:] = 0.0

    # label_weight_2d
    weight_2d = np.broadcast_to(weight[None, :], (N_SAMPLES, WINDOW_WIDTH)).copy()

    return x_raw, y_mask, y_soft, status, weight, weight_2d


def main():
    case_dirs = sorted(MODELS.glob("case_*"))
    print(f"Found {len(case_dirs)} case directories")

    # Phase 1: compute global P99
    print("\nPhase 1: Computing global P99...")
    global_p99 = compute_global_p99(case_dirs)
    print(f"  Global P99 = {global_p99:.4f}")

    # Phase 2: convert each case
    print("\nPhase 2: Converting cases...")
    time_501 = np.linspace(0, T_WINDOW_NS, N_SAMPLES, dtype=np.float32)
    time_native = np.arange(NATIVE_SAMPLES, dtype=np.float32) * (T_WINDOW_NS / NATIVE_SAMPLES)

    all_rows = []
    converted = 0

    for c in case_dirs:
        cid = c.name

        # Load raw native
        raw_path = c / "outputs" / "raw_bscan_native.npy"
        if not raw_path.exists():
            print(f"  {cid}: raw_bscan_native.npy missing, skipping")
            continue
        raw = np.load(raw_path).astype(np.float32)

        # Load mask
        mask_path = c / "labels" / "interface_mask_bscan.npy"
        if not mask_path.exists():
            print(f"  {cid}: interface_mask_bscan.npy missing, skipping")
            continue
        mask = np.load(mask_path)

        # Validate
        if raw.shape != (NATIVE_SAMPLES, N_TRACES):
            print(f"  {cid}: raw shape {raw.shape}, expected ({NATIVE_SAMPLES},{N_TRACES}), skipping")
            continue
        if mask.shape != (N_SAMPLES, N_TRACES):
            print(f"  {cid}: mask shape {mask.shape}, expected ({N_SAMPLES},{N_TRACES}), skipping")
            continue
        if not np.isfinite(raw).all():
            print(f"  {cid}: NaN/inf in raw, skipping")
            continue

        # Resample raw 5937→501 (linear interp)
        old_ax = np.linspace(0, 1, NATIVE_SAMPLES)
        new_ax = np.linspace(0, 1, N_SAMPLES)
        raw_501 = np.zeros((N_SAMPLES, N_TRACES), dtype=np.float32)
        for i in range(N_TRACES):
            raw_501[:, i] = np.interp(new_ax, old_ax, raw[:, i])

        # Create windows
        x_raw, y_mask, y_soft, status, weight, weight_2d = create_windows(
            raw_501, mask, global_p99
        )

        # Save NPZ (1 window per case, padded to 256 traces)
        sid = f"pilot_{cid}_w00"
        npz_path = OUT / f"{sid}.npz"
        np.savez_compressed(
            npz_path,
            x_raw=x_raw.astype(np.float32),
            y_mask=y_mask,
            y_soft=y_soft,
            status_code=status,
            label_weight=weight,
            label_weight_2d=weight_2d,
        )

        # Window index row
        n_present = int((status == 1).sum())
        n_weak = int((status == 2).sum())
        n_absent = int((status == 0).sum())
        all_rows.append({
            "sample_id": sid,
            "line": f"pilot_{cid}",
            "start": 0,
            "end": WINDOW_WIDTH - 1,
            "split": "train",
            "present": n_present,
            "weak": n_weak,
            "no_pick": n_absent,
        })
        converted += 1

        print(f"  {cid}: x_raw=[{x_raw.min():.3f},{x_raw.max():.3f}] "
              f"present={n_present}/{WINDOW_WIDTH}")

    # Phase 3: write index CSV
    if all_rows:
        idx_path = OUT.parent / "window_index.csv"
        fieldnames = ["sample_id", "line", "start", "end", "split",
                      "present", "weak", "no_pick"]
        with open(idx_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} entries to {idx_path}")

    # Phase 4: save metadata
    print("\nPhase 4: Writing metadata...")
    np.save(OUT.parent / "time_501_ns.npy", time_501)
    np.save(OUT.parent / "time_native_ns.npy", time_native)

    json.dump({
        "method": "linear_interpolation",
        "native_samples": NATIVE_SAMPLES,
        "target_samples": N_SAMPLES,
        "time_window_ns": T_WINDOW_NS,
        "P99_normalization": f"global_P99_across_{len(case_dirs)}_cases",
        "global_P99_value": round(global_p99, 4),
    }, open(OUT.parent / "resample_config.json", "w"), indent=2)

    json.dump({
        "description": "Training metadata for Pilot-Train v3 NPZ files",
        "n_cases": converted,
        "n_traces_effective": N_TRACES,
        "n_traces_padded": WINDOW_WIDTH,
        "valid_trace_slice": [PAD, PAD + N_TRACES],
        "padding_value": 0.0,
        "y_mask_convention": "geometric_interface_position",
        "y_soft_convention": "Gaussian_blur_sigma8rows",
        "y_soft_note": "Training target, buffers ~41ns phase delay",
        "mask_source": "scene_world geometry -> air_twt + effective_avg_velocity -> interface_mask_bscan",
        "raw_source": "raw_bscan_native.npy -> linear_interp -> raw_bscan_501.npy",
        "time_axis": "time_501_ns.npy (501 samples, 0-700ns)",
    }, open(OUT.parent / "training_metadata.json", "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"Done: {converted}/{len(case_dirs)} cases converted")
    print(f"NPZ: {OUT}")
    print(f"Index: {OUT.parent / 'window_index.csv'}")
    return 0 if converted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
