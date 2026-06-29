#!/usr/bin/env python3
"""Safe batch gprMax runner for 3060 laptop (6 GB VRAM).

Uses SafeGprMaxRunner (process isolation, timeout tree-kill, MSVC+CUDA PATH injection).
Supports skip-completed and summary reporting.

Usage:
  python scripts/run_batch_safe_3060.py --manifest <path> [--variants raw] [--limit 3] [--dry-run]
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from uavgpr_simlab.core.runner import SafeGprMaxRunner, resolve_manifest_path

PYTHON_EXE = r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
N_TRACES_DEFAULT = 20
GPU_TIMEOUT_S = 14400


def run_manifest(manifest_csv, variants=None, limit=0, dry_run=False, n_traces=N_TRACES_DEFAULT):
    manifest = Path(manifest_csv)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    workspace = manifest.resolve().parent.parent
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(variants) if variants else None
    results = {"total": 0, "success": 0, "timeout": 0, "failed": 0, "skipped": 0, "tasks": []}

    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row.get("variant", "raw")
            if wanted and variant not in wanted:
                continue

            input_file = resolve_manifest_path(row["input_file"], manifest)
            case_id = row.get("case_id", "")
            n = int(float(row.get("n_traces", n_traces) or n_traces))
            task_key = f"{case_id}_{variant}"
            results["total"] += 1

            if dry_run:
                print(f"[DRY-RUN] {task_key} ({n} traces) -> {input_file}")
                results["skipped"] += 1
                continue

            print(f"\n[RUN] {task_key} ({n} traces)")
            cmd = [PYTHON_EXE, "-m", "gprMax", str(input_file), "-gpu", "-n", str(n), "--geometry-fixed"]

            runner = SafeGprMaxRunner(cmd=cmd, cwd=str(log_dir), timeout=GPU_TIMEOUT_S, log_dir=log_dir)
            result = runner.run()

            task_result = {
                "task_key": task_key, "case_id": case_id, "variant": variant,
                "input_file": str(input_file), "status": result.status,
                "return_code": result.return_code, "elapsed": result.elapsed,
                "stdout_log": result.stdout_log, "stderr_log": result.stderr_log,
            }
            results["tasks"].append(task_result)
            results[result.status] = results.get(result.status, 0) + 1

            emoji = {"success": "[OK]", "timeout": "[TIMEOUT]", "failed": "[FAIL]", "cancelled": "[CANCEL]"}
            print(f"  {emoji.get(result.status, result.status)} in {result.elapsed:.1f}s")

            if limit and results["total"] >= limit:
                break

    summary_path = log_dir / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    ok, to, fail = results["success"], results.get("timeout", 0), results.get("failed", 0)
    print(f"\n=== Summary: {ok}/{results['total']} success, {to} timeout, {fail} failed ===")
    print(f"Report: {summary_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Safe batch gprMax runner for 3060")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--variants", default="raw", help="Comma-separated variants")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-traces", type=int, default=N_TRACES_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run_manifest(args.manifest, variants=variants, limit=args.limit, dry_run=args.dry_run, n_traces=args.n_traces)


if __name__ == "__main__":
    main()
