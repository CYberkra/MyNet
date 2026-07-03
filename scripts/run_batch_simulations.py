"""
PGDA-CSNet batch simulation runner.
Runs gprMax SceneWorld simulations sequentially using SafeGprMaxRunner
to avoid CUDA context pollution between runs.
"""
import sys, os, subprocess, json, csv
from pathlib import Path

sys.path.insert(0, r"D:\Claude\PGDA-CSNet\uavgpr_simlab\src")
from uavgpr_simlab.core.runner import SafeGprMaxRunner

ROOT = Path(r"D:\Claude\PGDA-CSNet\uavgpr_simlab\workspace\pgda_batch_v1_3060")
MANIFEST = ROOT / "datasets" / "pgda_batch_v1_3060_manifest.csv"
PYTHON_EXE = r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
MAX_CASES = 6  # 6 cases × 4 variants = 24 sims ≈ 6 hours

# Read manifest and extract unique case IDs
cases = set()
with open(MANIFEST, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cases.add(row["case_id"])
cases = sorted(cases)[:MAX_CASES]
print(f"Running {len(cases)} cases: {cases[0]} to {cases[-1]}")

results = {}
for cid in cases:
    results[cid] = {}
    for variant in ["raw", "target_only", "background_only", "air_only"]:
        infile = ROOT / "models" / cid / f"{variant}.in"
        out_file = ROOT / "models" / cid / f"{variant}1.out"
        merged_file = ROOT / "models" / cid / f"{variant}_merged.out"

        if merged_file.exists():
            print(f"  SKIP {cid}/{variant} (merged exists)")
            results[cid][variant] = "skipped"
            continue

        print(f"  RUN  {cid}/{variant} (64 traces, GPU)...", end=" ", flush=True)
        runner = SafeGprMaxRunner(
            cmd=[PYTHON_EXE, '-m', 'gprMax', str(infile), '-gpu', '-n', '64', '--geometry-fixed'],
            cwd=str(infile.parent),
            timeout=3600,
        )
        r = runner.run()
        if r.status == "success":
            # Merge
            subprocess.run(
                [PYTHON_EXE, '-m', 'tools.outputfiles_merge', str(infile.stem), '--remove-files'],
                cwd=str(infile.parent), capture_output=True, text=True, timeout=120
            )
            # Check merge output
            if merged_file.exists():
                sz = merged_file.stat().st_size / 1024 / 1024
                print(f"OK ({sz:.0f} MB)", flush=True)
                results[cid][variant] = "ok"
            else:
                print(f"MERGE FAILED", flush=True)
                results[cid][variant] = "merge_failed"
        else:
            print(f"FAILED ({r.error[:60] if r.error else 'unknown'})", flush=True)
            results[cid][variant] = "failed"

# Write summary
summary_path = ROOT / "reports" / "batch_run_summary.json"
summary_path.parent.mkdir(exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

ok = sum(1 for r in results.values() for v in r.values() if v == "ok")
total = sum(len(r) for r in results.values())
print(f"\nDone: {ok}/{total} simulations succeeded -> {summary_path}")
