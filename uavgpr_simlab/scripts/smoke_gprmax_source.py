from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from an unpacked project tree without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavgpr_simlab.services.gprmax_smoke_service import run_gprmax_source_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal CPU gprMax smoke test from a local source tree.")
    parser.add_argument("--gprmax-root", required=True, help="Path to the gprMax source root containing gprMax/__main__.py and setup.py.")
    parser.add_argument("--work-dir", default=str(ROOT / "workspace" / "gprmax_source_smoke"), help="Output directory for the tiny input, .out file and JSON report.")
    parser.add_argument("--build", action="store_true", help="Run `python setup.py build_ext --inplace` before the smoke test.")
    parser.add_argument("--omp-threads", type=int, default=1, help="OMP_NUM_THREADS for the tiny CPU run.")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per subprocess in seconds.")
    args = parser.parse_args()

    report = run_gprmax_source_smoke(
        Path(args.gprmax_root).expanduser(),
        Path(args.work_dir).expanduser(),
        build=args.build,
        omp_threads=args.omp_threads,
        timeout=args.timeout,
    )
    print(report.report_path)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
