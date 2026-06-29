from __future__ import annotations

import argparse
import json
from uavgpr_simlab.services.sceneworld_bscan_service import run_sceneworld_bscan_from_manifest


def _split(text: str) -> list[str]:
    return [x.strip() for x in text.replace(';', ',').split(',') if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run gprMax for SceneWorld manifest rows and replace B-scan placeholders.")
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--gprmax-root', required=True)
    ap.add_argument('--variants', default='raw,target_only,background_only,clutter_only,air_only')
    ap.add_argument('--one-case-per-family', action='store_true')
    ap.add_argument('--max-cases', type=int, default=0)
    ap.add_argument('--python-executable', default='python')
    ap.add_argument('--omp-threads', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--failed-only', action='store_true', help='Run only manifest rows marked failed.')
    ap.add_argument('--no-skip-completed', action='store_true', help='Do not skip existing success rows unless --force is used.')
    ns = ap.parse_args()
    rep = run_sceneworld_bscan_from_manifest(
        ns.manifest,
        gprmax_root=ns.gprmax_root,
        variants=_split(ns.variants),
        one_case_per_family=ns.one_case_per_family,
        max_cases=ns.max_cases,
        python_executable=ns.python_executable,
        omp_threads=ns.omp_threads,
        timeout_sec=ns.timeout,
        no_gpu=True,
        force=ns.force,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep.get('ok') else 2


if __name__ == '__main__':
    raise SystemExit(main())
