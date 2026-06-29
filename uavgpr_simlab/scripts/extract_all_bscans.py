"""Extract B-scans from existing gprMax .out HDF5 files for all 50 batch v1 cases.

The .out files use the _merged.out naming convention (all 64 traces in one HDF5),
so merge_available_bscan_for_input (expecting one file per trace) won't work.
We read h5 files directly and resample.
"""
import json, sys, time, traceback
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print("h5py not found — install with: pip install h5py")
    sys.exit(1)

WORKSPACE = Path("D:/Claude/PGDA-CSNet/uavgpr_simlab/workspace/pgda_batch_v1_3060")
MODELS = WORKSPACE / "models"

sys.path.insert(0, str(Path("D:/Claude/PGDA-CSNet/uavgpr_simlab/src")))
from uavgpr_simlab.services.sceneworld_bscan_service import resample_bscan, build_case_bscan_qc

VARIANTS = ["raw", "target_only", "background_only", "air_only"]
EXPECTED_SHAPE = (501, 64)
COMPONENT = "Ez"
RX = "rx1"


def read_merged_bscan(h5_path: Path, rx: str = RX, component: str = COMPONENT) -> np.ndarray | None:
    """Read a _merged.out HDF5 file and return a (time, traces) float32 array."""
    if not h5_path.exists():
        return None
    try:
        with h5py.File(str(h5_path), "r") as f:
            ds_path = f"rxs/{rx}/{component}"
            if ds_path not in f:
                # Try fallback components
                rx_group = f.get(f"rxs/{rx}")
                if rx_group is None:
                    return None
                candidates = [k for k in rx_group.keys() if k != "Positions"]
                if not candidates:
                    return None
                ds_path = f"rxs/{rx}/{candidates[0]}"
            data = f[ds_path][()]  # (time, traces)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    except Exception:
        return None


def extract_one_variant(case_dir: Path, variant: str) -> dict:
    rep = {"variant": variant, "status": "pending"}
    out_dir = case_dir / "outputs"
    merged_path = case_dir / f"{variant}_merged.out"
    inp_path = case_dir / f"{variant}.in"
    bscan_path = out_dir / f"{variant}_bscan.npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if bscan already has finite data of the right shape
    if bscan_path.exists():
        try:
            existing = np.load(bscan_path)
            if np.isfinite(existing).all() and existing.shape == EXPECTED_SHAPE:
                rep.update({"status": "skipped", "reason": "already valid"})
                return rep
        except Exception:
            pass

    # Read from _merged.out
    data = read_merged_bscan(merged_path)
    if data is None and inp_path.exists():
        # Fall back to candidate_out_files (for non-merged naming)
        from uavgpr_simlab.core.postprocess import merge_available_bscan_for_input
        result = merge_available_bscan_for_input(str(inp_path), rx=RX, component=COMPONENT)
        if result is not None:
            data = result[0]
        else:
            rep.update({"status": "failed", "error": f"no readable data from {merged_path.name} or .in fallback"})
            return rep
    elif data is None:
        rep.update({"status": "failed", "error": f"h5 file not found: {merged_path}"})
        return rep

    native_shape = list(data.shape)
    resampled = resample_bscan(data, EXPECTED_SHAPE[0], EXPECTED_SHAPE[1])
    np.save(bscan_path, resampled)

    finites = np.isfinite(resampled).sum()
    total = resampled.size
    rep.update({
        "status": "success",
        "native_shape": native_shape,
        "saved_shape": list(resampled.shape),
        "finite_frac": float(finites / total),
        "min": float(np.nanmin(resampled)),
        "max": float(np.nanmax(resampled)),
        "mean": float(np.nanmean(resampled)),
    })
    return rep


def main():
    case_dirs = sorted(MODELS.glob("case_*"))
    print(f"Found {len(case_dirs)} case directories\n")

    variant_counts = {v: {"ok": 0, "fail": 0, "skip": 0} for v in VARIANTS}
    case_reports = {}
    successes = 0
    failures = 0

    for case_dir in case_dirs:
        cid = case_dir.name
        print(f"── {cid} ──")
        case_ok = True
        reports = {}
        for v in VARIANTS:
            rep = extract_one_variant(case_dir, v)
            reports[v] = rep
            s = rep["status"]
            variant_counts[v][{"success": "ok", "failed": "fail", "skipped": "skip"}.get(s, "fail")] += 1

            ffrac = rep.get("finite_frac", 0)
            if "saved_shape" in rep:
                print(f"  {v:15s}: {s:8s}  shape={rep['saved_shape']}  finite={ffrac:.3f}")
            else:
                print(f"  {v:15s}: {s:8s}  {rep.get('error', '')}")

            if s == "failed":
                case_ok = False

        case_reports[cid] = reports

        if case_ok:
            try:
                qc = build_case_bscan_qc(case_dir, variants=VARIANTS, expected_shape=EXPECTED_SHAPE)
                qc_status = qc.get("status", "unknown")
                print(f"  {'QC':15s}: {qc_status}")
                if qc_status == "success":
                    successes += 1
                else:
                    failures += 1
                    print(f"    QC details: {json.dumps(qc, indent=2)[:500]}")
            except Exception as exc:
                print(f"  {'QC':15s}: error: {exc}")
                failures += 1
        else:
            failures += 1
            print(f"  {'QC':15s}: skipped (variant failure)")
        print()

    # Summary
    print("=" * 60)
    print(f"Cases: {len(case_dirs)} total, {successes} success, {failures} failed")
    for v in VARIANTS:
        c = variant_counts[v]
        print(f"  {v}: {c['ok']} ok, {c['fail']} fail, {c['skip']} skip")

    report_path = WORKSPACE / "reports" / "bscan_batch_extract_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "total": len(case_dirs), "success": successes, "failed": failures,
        "variant_counts": variant_counts, "case_reports": case_reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    sys.exit(rc)
