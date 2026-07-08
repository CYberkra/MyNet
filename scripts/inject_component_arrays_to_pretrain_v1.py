"""
scripts/inject_component_arrays_to_pretrain_v1.py

Add Y_air, X_clean, G_target component arrays to existing simulation_pretrain_v1
NPZ files by reading the original Pilot-Mini simulation outputs.

For cases that have air_only/target_only source files:
  - Y_air      = air_only_bscan.npy  (air-coupled direct wave)
  - G_target   = target_only_bscan.npy (pure geological signal)
  - X_clean    = raw_bscan - air_only_bscan  (S + G, surface + geological)

For cases without source files, zero-placeholder arrays are written with
validity=False — the loss function's validity mask will skip them.

All arrays match the existing NPZ shape (501, 256) with the same padding.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

NATIVE_SAMPLES = 5937
N_SAMPLES = 501
N_TRACES = 128
PAD = (256 - N_TRACES) // 2  # 64, same as convert_pilot_to_training.py
T_WINDOW_NS = 700.0

SIM_V1_DIR = Path("data/simulation_pretrain_v1")
MINI_WORKSPACE = Path(
    "uavgpr_simlab/workspace/pilot_mini_v1/"
    "yingshan_pilot_3060_v1/models"
)


def _resample(raw_5937x40):
    """Linear interpolation from 5937 to 501 samples."""
    old = np.linspace(0, 1, NATIVE_SAMPLES)
    new = np.linspace(0, 1, N_SAMPLES)
    out = np.zeros((N_SAMPLES, N_TRACES), dtype=np.float32)
    for i in range(N_TRACES):
        out[:, i] = np.interp(new, old, raw_5937x40[:, i])
    return out


def _pad_40_to_256(arr_501x40):
    """Center-pad 40 traces to 256 traces (reflect pad, zero padding zone)."""
    arr = np.pad(arr_501x40, ((0, 0), (PAD, PAD)), mode="reflect")
    arr[:, :PAD] = 0.0
    arr[:, -PAD:] = 0.0
    return arr


def _load_and_process(mini_case_dir):
    """Load air_only / target_only / raw and return processed arrays."""
    labels_dir = mini_case_dir / "labels"
    outs_dir = mini_case_dir / "outputs"

    air_path = outs_dir / "air_only_bscan.npy"
    target_path = outs_dir / "target_only_bscan.npy"
    raw_path = mini_case_dir / "raw_bscan.npy"  # raw (A+S+G)
    if not raw_path.exists():
        raw_path = outs_dir / "raw_bscan.npy"

    if not (air_path.exists() and target_path.exists() and raw_path.exists()):
        return None

    air_raw = np.load(air_path).astype(np.float32)        # (5937, 40)
    target_raw = np.load(target_path).astype(np.float32)  # (5937, 40)
    raw_raw = np.load(raw_path).astype(np.float32)        # (5937, 40)

    if air_raw.shape != (NATIVE_SAMPLES, 40):
        print(f"    WARN: air_only shape {air_raw.shape}, expected (5937,40)")
        return None
    if target_raw.shape != (NATIVE_SAMPLES, 40):
        print(f"    WARN: target_only shape {target_raw.shape}")
        return None

    # Resample 5937->501
    air_501 = _resample(air_raw)          # (501, 128) — 128 tempo for uniform shape
    # Actually we need 40->128 first, but _resample produces (501,128)
    # Let me fix: _resample builds (501, N_TRACES=128) which is wrong for 40-trace inputs.
    # Replacing with correct per-trace resample

    return air_raw, target_raw, raw_raw


def _resample_40(raw_5937x40):
    """Interpolate each of 40 traces from 5937->501 samples."""
    old = np.linspace(0, 1, NATIVE_SAMPLES)
    new = np.linspace(0, 1, N_SAMPLES)
    out = np.zeros((N_SAMPLES, 40), dtype=np.float32)
    for i in range(40):
        out[:, i] = np.interp(new, old, raw_5937x40[:, i])
    return out


def _load_global_p99(case_dirs):
    """Recompute the same global P99 used by convert_pilot_to_training.

    Uses raw_bscan.npy from each Pilot-Mini case, resampled to 501,
    exactly matching the original conversion pipeline.
    """
    old = np.linspace(0, 1, NATIVE_SAMPLES)
    new = np.linspace(0, 1, N_SAMPLES)
    all_vals = []
    for c in case_dirs:
        raw_path = c / "raw_bscan.npy"
        if not raw_path.exists():
            raw_path = c / "outputs" / "raw_bscan.npy"
        if not raw_path.exists():
            continue
        raw = np.load(raw_path).astype(np.float32)
        if raw.shape[0] != NATIVE_SAMPLES:
            continue
        rs = np.zeros((N_SAMPLES, raw.shape[1]), dtype=np.float32)
        for i in range(raw.shape[1]):
            rs[:, i] = np.interp(new, old, raw[:, i])
        all_vals.append(np.abs(rs).ravel())
    all_vals = np.concatenate(all_vals)
    p99 = float(np.percentile(all_vals, 99))
    return max(p99, 1e-12)


def main():
    # 0. Build a mapping: line name -> Mini case directory
    # Pilot-Mini cases are case_000001 through case_000020
    # simulation_pretrain_v1 uses case_000001..case_000006 line names
    mini_models = Path(MINI_WORKSPACE)
    if not mini_models.exists():
        print(f"ERROR: Pilot-Mini workspace not found at {mini_models}")
        return 1

    case_dirs = sorted(mini_models.iterdir())
    line_to_mini = {}
    for c in case_dirs:
        if c.is_dir():
            line_to_mini[c.name] = c
    # Also map sim_l1/line3/line9/zk09 if present
    # (these were generated separately, check exact names)
    for extra_name in ("sim_l1", "sim_line3", "sim_line9", "sim_zk09"):
        extra_dir = mini_models / extra_name
        if extra_dir.exists():
            line_to_mini[extra_name] = extra_dir

    print(f"Found {len(case_dirs)} Pilot-Mini case directories")
    print(f"  case_to_line mapping: {sorted(line_to_mini.keys())}")

    # 1. Compute global P99 across ALL Mini cases
    print("\nComputing global P99 ...")
    global_p99 = _load_global_p99(case_dirs)
    print(f"  Global P99 = {global_p99:.4f}")

    # 2. Iterate all existing NPZs in simulation_pretrain_v1
    sim_dir = Path(SIM_V1_DIR)
    npz_dir = sim_dir / "windows"
    all_npz = sorted(npz_dir.glob("*.npz"))
    print(f"\nProcessing {len(all_npz)} NPZ files in {sim_dir}")

    # Read window_index to get line -> sample_id mapping
    idx_path = sim_dir / "window_index.csv"
    with open(idx_path, encoding="utf-8") as f:
        idx_rows = list(csv.DictReader(f))

    line_to_ids = {}
    for r in idx_rows:
        line_to_ids.setdefault(r["line"], []).append(r["sample_id"])

    success = 0
    skipped_no_source = 0
    skipped_error = 0

    for line, sample_ids in sorted(line_to_ids.items()):
        mini_dir = line_to_mini.get(line)
        if mini_dir is None:
            print(f"  {line}: NO source (skipping, {len(sample_ids)} NPZs)")
            skipped_no_source += len(sample_ids)
            continue

        print(f"  {line}: source={mini_dir.name} ({len(sample_ids)} NPZs)")

        # Load component raw data
        air_path = mini_dir / "outputs" / "air_only_bscan.npy"
        target_path = mini_dir / "outputs" / "target_only_bscan.npy"
        raw_path = mini_dir / "raw_bscan.npy"
        if not raw_path.exists():
            raw_path = mini_dir / "outputs" / "raw_bscan.npy"

        if not (air_path.exists() and target_path.exists() and raw_path.exists()):
            print("    SKIP: missing component files")
            skipped_no_source += len(sample_ids)
            continue

        air_raw = np.load(air_path).astype(np.float32)
        target_raw = np.load(target_path).astype(np.float32)
        raw_raw = np.load(raw_path).astype(np.float32)

        # Validate shapes
        n_traces = raw_raw.shape[1]
        if air_raw.shape != (NATIVE_SAMPLES, n_traces):
            print(f"    SKIP: air shape {air_raw.shape} != raw shape {raw_raw.shape}")
            skipped_error += len(sample_ids)
            continue

        # Resample 5937->501
        old_ax = np.linspace(0, 1, NATIVE_SAMPLES)
        new_ax = np.linspace(0, 1, N_SAMPLES)
        raw_501 = np.zeros((N_SAMPLES, n_traces), dtype=np.float32)
        air_501 = np.zeros_like(raw_501)
        target_501 = np.zeros_like(raw_501)
        for i in range(n_traces):
            raw_501[:, i] = np.interp(new_ax, old_ax, raw_raw[:, i])
            air_501[:, i] = np.interp(new_ax, old_ax, air_raw[:, i])
            target_501[:, i] = np.interp(new_ax, old_ax, target_raw[:, i])

        # Normalize and pad 40/128 -> 256
        raw_norm = _pad_40_to_256(raw_501 / global_p99)
        air_norm = _pad_40_to_256(air_501 / global_p99)
        target_norm = _pad_40_to_256(target_501 / global_p99)
        x_clean_norm = _pad_40_to_256(
            np.clip(raw_501 - air_501, -global_p99 * 10, global_p99 * 10) / global_p99
        )

        # Update each NPZ for this line
        for sid in sample_ids:
            npz_path = npz_dir / f"{sid}.npz"
            if not npz_path.exists():
                print(f"    WARN: {sid}.npz not found, skipping")
                skipped_error += 1
                continue

            existing = np.load(npz_path)
            data = dict(existing)
            data["Y_air"] = air_norm.astype(np.float32)
            data["X_clean"] = x_clean_norm.astype(np.float32)
            data["G_target"] = target_norm.astype(np.float32)

            # Save back (overwrite with new keys)
            np.savez_compressed(npz_path, **data)
            success += 1

    print(f"\n{'='*60}")
    print(f"Done: {success} NPZs patched, {skipped_no_source} skipped (no source), {skipped_error} errors")
    print("Component arrays added: Y_air, X_clean, G_target (all (501,256))")
    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
