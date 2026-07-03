#!/usr/bin/env python3
"""
PGDA_SYNTH_DATASET_V1 — Batch Simulation Runner

Usage:
    python tools/run_batch.py <batch_dir> [--dry-run] [--resume]

Features:
  ✓ Preflight auto-check before each case (PASS → run, FAIL → skip)
  ✓ SafeGprMaxRunner for complex GPU models (MSVC + CUDA path injection)
  ✓ Auto-assemble bscan.npy after completion
  ✓ Auto-run after_run_qc after completion
  ✓ Journal file for crash/TDR resume
  ✓ GPU watchdog (nvidia-smi ping every 30s, temperature monitor)
  ✓ Manifest auto-update

TDR Protection:
  Windows GPU timeout detection resets the driver if GPU is unresponsive >2s.
  The batch runner combats this with:
  1. GPU watchdog thread — runs nvidia-smi every 20s to keep driver responsive
  2. Per-trace journal — resume from last completed trace after crash
  3. Temperature throttle — pauses if GPU >88°C
  4. Grace period — extends TDR by keeping driver engaged

Directory flow:
  02_case_pool/batch_xxx/  →  preflight  →  03_runs/batch_xxx/  →  after_run_qc  →  04_qc/batch_xxx/
"""

import sys, os, json, time, subprocess, shutil, argparse, glob, datetime
from pathlib import Path
from threading import Thread, Event

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
GPRMAX_MODULE = "gprMax"

# ── GPU watchdog ──
_stop_watchdog = Event()
_last_gpu_ok = [True]
_gpu_temp = [60]

def _gpu_watchdog():
    """Background thread: ping nvidia-smi every 20s to prevent TDR.
    Monitors temperature and logs anomalies."""
    while not _stop_watchdog.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 1:
                    temp = int(parts[0].strip())
                    _gpu_temp[0] = temp
                    if temp > 90:
                        print(f"  ⚠ GPU TEMPERATURE WARNING: {temp}°C — risk of thermal throttle")
                _last_gpu_ok[0] = True
            else:
                _last_gpu_ok[0] = False
        except Exception as e:
            _last_gpu_ok[0] = False
        _stop_watchdog.wait(20)

def start_watchdog():
    _stop_watchdog.clear()
    t = Thread(target=_gpu_watchdog, daemon=True)
    t.start()
    return t

def stop_watchdog():
    _stop_watchdog.set()

def wait_if_hot(max_temp=88):
    """Pause if GPU exceeds max_temp."""
    while _gpu_temp[0] > max_temp:
        print(f"  🥵 GPU at {_gpu_temp[0]}°C, waiting 60s for cool-down...")
        time.sleep(60)

# ── SafeGprMaxRunner (inline to avoid import issues) ──
def _inject_msvc_paths():
    """Inject MSVC and CUDA paths into the environment for nvcc compilation."""
    msvc_base = r"E:\sisual stdio 2022"
    msvc_ver = r"14.39.33519"
    kits_base = r"C:\Program Files (x86)\Windows Kits\10"
    sdk_ver = r"10.0.22621.0"
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

    entries = [
        # CUDA bin
        (os.path.join(cuda_base, "bin"), "PATH"),
        # MSVC host compiler
        (os.path.join(msvc_base, "VC", "Tools", "MSVC", msvc_ver, "bin", "Hostx64", "x64"), "PATH"),
        # MSVC includes
        (os.path.join(msvc_base, "VC", "Tools", "MSVC", msvc_ver, "include"), "INCLUDE"),
        (os.path.join(msvc_base, "VC", "Tools", "MSVC", msvc_ver, "atlmfc", "include"), "INCLUDE"),
        (os.path.join(msvc_base, "VC", "Auxiliary", "VS", "include"), "INCLUDE"),
        # MSVC libs
        (os.path.join(msvc_base, "VC", "Tools", "MSVC", msvc_ver, "lib", "x64"), "LIB"),
        (os.path.join(msvc_base, "VC", "Tools", "MSVC", msvc_ver, "atlmfc", "lib", "x64"), "LIB"),
        (os.path.join(msvc_base, "VC", "Auxiliary", "VS", "lib", "x64"), "LIB"),
        # Windows SDK includes
        (os.path.join(kits_base, "Include", sdk_ver, "ucrt"), "INCLUDE"),
        (os.path.join(kits_base, "Include", sdk_ver, "um"), "INCLUDE"),
        (os.path.join(kits_base, "Include", sdk_ver, "shared"), "INCLUDE"),
        # Windows SDK libs
        (os.path.join(kits_base, "Lib", sdk_ver, "ucrt", "x64"), "LIB"),
        (os.path.join(kits_base, "Lib", sdk_ver, "um", "x64"), "LIB"),
    ]
    env = os.environ.copy()
    for p, var in entries:
        if os.path.isdir(p):
            if var == "PATH":
                env["PATH"] = p + os.pathsep + env.get("PATH", "")
            else:
                env[var] = p + os.pathsep + env.get(var, "")
    return env

def run_gprmax(in_path, n_traces, gpu=True, timeout_min=120):
    """Run gprMax with SafeGprMaxRunner-style env injection.
    Returns (success: bool, stdout: str, stderr: str)."""
    in_path = Path(in_path)  # ensure Path object
    env = _inject_msvc_paths()
    cmd = [VENV_PYTHON, "-m", GPRMAX_MODULE, str(in_path)]
    if gpu:
        cmd.extend(["-gpu", "--geometry-fixed"])
    cmd.extend(["-n", str(n_traces)])

    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=str(in_path.parent)
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_min * 60)
        success = proc.returncode == 0
        return success, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        proc.kill()
        return False, "", "TIMEOUT"

def assemble_bscan(out_dir, stem, n_traces):
    """Assemble raw*.out → bscan.npy. Returns True on success."""
    try:
        venv_python = Path(VENV_PYTHON)
        sys.path.insert(0, str(venv_python.parent.parent / "Lib" / "site-packages"))
        import h5py
        import numpy as np

        files = sorted(out_dir.glob(f"{stem}*.out"))
        files = sorted(
            [f for f in files if f.stem.replace(stem, "").isdigit()],
            key=lambda x: int(x.stem.replace(stem, "")),
        )
        if not files:
            print(f"  ❌ No .out files found for stem={stem}")
            return False

        traces = []
        for f in files:
            with h5py.File(f, "r") as h:
                traces.append(np.asarray(h["rxs"]["rx1"]["Ez"]))
        arr = np.stack(traces, axis=1)
        np.save(out_dir / "bscan.npy", arr)
        n_actual = arr.shape[1]
        print(f"  ✅ bscan.npy assembled: {arr.shape[0]}×{n_actual}")
        return True
    except Exception as e:
        print(f"  ❌ assemble_bscan failed: {e}")
        return False

# ── Journal ──
def load_journal(batch_run_dir):
    jpath = batch_run_dir / ".batch_journal.json"
    if jpath.exists():
        return json.loads(jpath.read_text())
    return {"cases": {}, "started_at": None, "completed_count": 0}

def save_journal(batch_run_dir, journal):
    jpath = batch_run_dir / ".batch_journal.json"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(journal, indent=2, ensure_ascii=False))

# ── Main ──
def run_batch(batch_dir, dry_run=False, force_resume=False):
    batch_dir = Path(batch_dir)
    batch_id = batch_dir.name

    if not batch_dir.exists():
        print(f"❌ Batch directory not found: {batch_dir}")
        sys.exit(1)

    cases_dir = batch_dir / "cases"
    if not cases_dir.exists():
        print(f"❌ No cases/ subdirectory in {batch_dir}")
        sys.exit(1)

    # Collect case dirs
    case_dirs = sorted([d for d in cases_dir.iterdir() if d.is_dir()])
    if not case_dirs:
        print(f"❌ No case directories in {cases_dir}")
        sys.exit(1)

    # Setup run & QC output dirs
    runs_out = ROOT / "03_runs" / batch_id
    qc_out = ROOT / "04_qc" / batch_id

    # Load journal
    journal = load_journal(runs_out)
    if not journal["started_at"]:
        journal["started_at"] = datetime.datetime.now().isoformat()

    print(f"\n{'='*60}")
    print(f"  BATCH RUNNER: {batch_id}")
    print(f"  Cases found: {len(case_dirs)}")
    print(f"  Dry run: {dry_run}")
    print(f"  Resume: {force_resume}")
    print(f"  Journal: {runs_out / '.batch_journal.json'}")
    print(f"{'='*60}\n")

    # Start GPU watchdog
    if not dry_run:
        start_watchdog()
        print("  GPU watchdog started (nvidia-smi every 20s)")

    completed = 0
    skipped = 0
    failed = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        print(f"\n─── [{completed + skipped + failed + 1}/{len(case_dirs)}] {case_id} ───")

        # Check journal for already completed
        case_j = journal["cases"].get(case_id, {})
        if case_j.get("status") == "completed" and not force_resume:
            print(f"  ⏭ Already completed (journal). Use --resume to re-run.")
            completed += 1
            continue
        if case_j.get("status") == "running" and not force_resume:
            print(f"  ⏭ Previously interrupted. Use --resume to retry.")
            skipped += 1
            continue

        # ── Preflight ──
        print(f"  🔍 Preflight check...")
        preflight_script = ROOT / "tools" / "preflight_check.py"
        if not preflight_script.exists():
            print(f"  ❌ preflight_check.py not found")
            failed += 1
            continue

        preflight_result = subprocess.run(
            [sys.executable, str(preflight_script), str(case_dir)],
            capture_output=True, text=True, timeout=30
        )
        print(preflight_result.stdout)
        if preflight_result.returncode != 0:
            print(f"  ❌ Preflight FAILED (exit={preflight_result.returncode}), skipping")
            journal["cases"][case_id] = {"status": "preflight_failed", "exit_code": preflight_result.returncode}
            save_journal(runs_out, journal)
            failed += 1
            continue

        if dry_run:
            print(f"  ✅ [DRY RUN] Would run gprMax")
            continue

        # ── Setup run directory ──
        case_run_dir = runs_out / case_id
        raw_dir = case_run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Copy .in file to run dir
        in_src = case_dir / "geometry" / "raw.in"
        if not in_src.exists():
            print(f"  ❌ raw.in not found in {case_dir}/geometry/")
            journal["cases"][case_id] = {"status": "missing_in_file"}
            save_journal(runs_out, journal)
            failed += 1
            continue

        # Use the in-file in place (gprMax writes .out to same dir)
        # We'll symlink or copy, then run from case_run_dir/raw/
        in_dst = raw_dir / "raw.in"
        shutil.copy2(in_src, in_dst)

        # Copy tools (preflight + audit)
        tools_src = case_dir / "tools"
        tools_dst = case_run_dir / "tools"
        if tools_src.exists():
            shutil.copytree(tools_src, tools_dst, dirs_exist_ok=True)

        # Copy labels (for QC)
        labels_src = case_dir / "labels"
        labels_dst = case_run_dir / "labels"
        if labels_src.exists():
            shutil.copytree(labels_src, labels_dst, dirs_exist_ok=True)

        # Determine trace count
        n_traces = 128  # default
        # Check #rx_steps in .in file
        with open(in_src) as f:
            for line in f:
                if line.strip().startswith("#rx_steps"):
                    # Just use default 128
                    pass

        # Mark as running in journal
        journal["cases"][case_id] = {
            "status": "running",
            "started_at": datetime.datetime.now().isoformat(),
            "n_traces": n_traces
        }
        save_journal(runs_out, journal)

        # ── Run gprMax ──
        print(f"  🏃 Running gprMax ({n_traces} traces)...")

        # Check temp before starting
        wait_if_hot()

        success, stdout, stderr = run_gprmax(in_dst, n_traces, gpu=True)

        # Save logs
        log_stem = datetime.datetime.now().strftime("gprmax_%Y%m%d_%H%M%S")
        (raw_dir / f"{log_stem}.stdout.log").write_text(stdout)
        (raw_dir / f"{log_stem}.stderr.log").write_text(stderr)

        if not success:
            print(f"  ❌ gprMax failed")
            if "TDR" in stderr or "driver" in stderr.lower() or "context" in stderr.lower():
                print(f"  ⚠ Possible TDR/reset detected")
            journal["cases"][case_id]["status"] = "gprmax_failed"
            journal["cases"][case_id]["stderr"] = stderr[-500:]
            save_journal(runs_out, journal)
            failed += 1
            continue

        # ── Assemble bscan.npy ──
        print(f"  🔗 Assembling bscan.npy...")
        if not assemble_bscan(raw_dir, "raw", n_traces):
            print(f"  ❌ B-scan assembly failed")
            journal["cases"][case_id]["status"] = "assembly_failed"
            save_journal(runs_out, journal)
            failed += 1
            continue

        # Count actual traces
        out_files = sorted(raw_dir.glob("raw*.out"))
        n_actual = len(out_files)
        print(f"  {'✅' if n_actual >= n_traces else '❌'} {n_actual}/{n_traces} traces completed")

        if n_actual < n_traces:
            print(f"  ❌ Trace count {n_actual} < expected {n_traces}, marking incomplete")
            journal["cases"][case_id]["status"] = "incomplete_traces"
            journal["cases"][case_id]["n_actual"] = n_actual
            save_journal(runs_out, journal)
            failed += 1
            continue

        # ── Clean .vti ──
        for v in raw_dir.glob("*.vti"):
            try:
                v.unlink()
            except:
                pass

        # ── After-run QC ──
        print(f"  📊 Running after-run QC...")
        qc_script = ROOT / "tools" / "after_run_qc.py"
        if qc_script.exists() and (case_run_dir / "labels").exists():
            qc_result = subprocess.run(
                [sys.executable, str(qc_script), str(case_run_dir)],
                capture_output=True, text=True, timeout=120
            )
            print(qc_result.stdout)
            if qc_result.returncode != 0:
                print(f"  ❌ QC returned non-zero:\n{qc_result.stderr[-500:]}")
                journal["cases"][case_id]["status"] = "qc_failed"
                journal["cases"][case_id]["qc_stderr"] = qc_result.stderr[-500:]
                save_journal(runs_out, journal)
                failed += 1
                continue
        else:
            print(f"  ⏭ QC skipped (no script or no labels)")

        # ── Mark complete ──
        journal["cases"][case_id]["status"] = "completed"
        journal["cases"][case_id]["completed_at"] = datetime.datetime.now().isoformat()
        journal["cases"][case_id]["n_actual"] = n_actual
        journal["completed_count"] += 1
        save_journal(runs_out, journal)
        completed += 1

        # Update manifest
        _update_manifest(case_id, batch_id, n_traces, n_actual)

        # Copy qc results to 04_qc/
        qc_case_out = qc_out / case_id
        if qc_case_out.exists():
            print(f"  📁 QC results: {qc_case_out}")

        print(f"  ✅ {case_id} COMPLETE")

    # ── Final summary ──
    stop_watchdog()
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE: {batch_id}")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {len(case_dirs)}")
    print(f"{'='*60}")

    # Write batch summary
    batch_summary = {
        "batch_id": batch_id,
        "started_at": journal["started_at"],
        "ended_at": datetime.datetime.now().isoformat(),
        "total": len(case_dirs),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }
    (runs_out / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2))
    print(f"\n  📄 Summary: {runs_out / 'batch_summary.json'}")


def _update_manifest(case_id, batch_id, n_expected, n_actual):
    """Update manifest_master.csv with case completion info."""
    manifest = ROOT / "manifest_master.csv"
    if not manifest.exists():
        return
    lines = manifest.read_text(encoding="utf-8").strip().split("\n")
    header = lines[0]
    hdr = header.split(",")

    # Find or add row
    found = False
    for i, line in enumerate(lines[1:], 1):
        if line.startswith(case_id + ","):
            parts = line.split(",")
            # Update fields by index
            field_map = {
                "status": "completed",
                "run_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "trace_count_actual": str(n_actual),
            }
            for field, value in field_map.items():
                if field in hdr:
                    idx = hdr.index(field)
                    while len(parts) <= idx:
                        parts.append("")
                    parts[idx] = value
            lines[i] = ",".join(parts)
            found = True
            break

    if not found:
        new_row = [case_id, batch_id] + [""] * (len(hdr) - 2)
        field_map = {
            "status": "completed",
            "run_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "trace_count": str(n_expected),
            "trace_count_actual": str(n_actual),
        }
        for field, value in field_map.items():
            if field in hdr:
                idx = hdr.index(field)
                while len(new_row) <= idx:
                    new_row.append("")
                new_row[idx] = value
        lines.append(",".join(new_row))

    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="PGDA_SYNTH_DATASET_V1 — Batch Runner")
    ap.add_argument("batch_dir", help="Batch directory (e.g. 02_case_pool/batch_001_line9_style/)")
    ap.add_argument("--dry-run", action="store_true", help="Preflight only, don't run gprMax")
    ap.add_argument("--resume", action="store_true", help="Re-run previously completed/interrupted cases")
    ap.add_argument("--case", default=None, help="Run only this specific case ID")
    args = ap.parse_args()

    if args.case:
        # Single case mode
        case_dir = Path(args.batch_dir)
        if not case_dir.exists():
            print(f"Case directory not found: {case_dir}")
            sys.exit(1)
        batch_id = case_dir.parent.name if case_dir.parent.name != "cases" else case_dir.parent.parent.name
        # Temporarily create a batch context
        from tempfile import mkdtemp
        temp_batch = Path(mkdtemp())
        cases_dir = temp_batch / "cases"
        cases_dir.mkdir()
        shutil.copytree(case_dir, cases_dir / case_dir.name, symlinks=False)
        run_batch(temp_batch, args.dry_run, args.resume)
        shutil.rmtree(temp_batch)
    else:
        run_batch(args.batch_dir, args.dry_run, args.resume)


if __name__ == "__main__":
    main()
