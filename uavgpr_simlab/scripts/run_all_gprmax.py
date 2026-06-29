from __future__ import annotations

import argparse
import json
from pathlib import Path

from uavgpr_simlab.services.sceneworld_bscan_service import run_sceneworld_bscan_from_manifest


def _split_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.replace(';', ',').split(',') if x.strip()]


def _find_manifest(workspace: Path) -> Path:
    dataset_dir = workspace / 'datasets'
    found = sorted(dataset_dir.glob('*_manifest.csv'))
    if not found:
        raise FileNotFoundError(f'No *_manifest.csv found under {dataset_dir}')
    return found[-1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Run all requested gprMax variants for a SceneWorld workspace and replace B-scan placeholders.'
    )
    ap.add_argument('--workspace', required=True, help='SceneWorld dataset workspace, e.g. workspace/yingshan_sceneworld_smoke_v080a3')
    ap.add_argument('--manifest', default='', help='Optional manifest path. If omitted, the latest *_manifest.csv under <workspace>/datasets is used.')
    ap.add_argument('--gprmax-source-dir', required=True, help='gprMax source root containing gprMax/__main__.py and setup.py')
    ap.add_argument('--conda-env', default='', help='Optional conda environment name. When set, gprMax is executed through conda run -n <env> python.')
    ap.add_argument('--python-executable', default='python', help='Python executable when --conda-env is not used.')
    ap.add_argument('--gpu-ids', default='', help='Comma-separated GPU ids. Empty means CPU mode / no -gpu.')
    ap.add_argument('--variants', default='raw,target_only,background_only,clutter_only,air_only')
    ap.add_argument('--one-case-per-family', action='store_true', help='Run only the first case per family. v080a3 smoke has exactly one case per family.')
    ap.add_argument('--max-cases', type=int, default=0)
    ap.add_argument('--omp-threads', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--failed-only', action='store_true', help='Run only manifest rows marked failed.')
    ap.add_argument('--no-skip-completed', action='store_true', help='Do not skip existing success rows unless --force is used.')
    ap.add_argument('--allow-resample', action='store_true', help='Explicitly align/resample gprMax native B-scans to the manifest target grid. Use for chain-validation smoke profiles; keep disabled for pilot/formal acceptance unless a separate ML export step is intended.')
    ns = ap.parse_args()

    workspace = Path(ns.workspace).expanduser().resolve()
    manifest = Path(ns.manifest).expanduser().resolve() if ns.manifest else _find_manifest(workspace)
    variants = _split_csv(ns.variants)
    gpu_ids = [int(x) for x in _split_csv(ns.gpu_ids)]
    if ns.conda_env:
        python_executable = ['conda', 'run', '-n', ns.conda_env, 'python']
    else:
        python_executable = ns.python_executable

    rep = run_sceneworld_bscan_from_manifest(
        manifest,
        gprmax_root=ns.gprmax_source_dir,
        variants=variants,
        one_case_per_family=bool(ns.one_case_per_family),
        max_cases=int(ns.max_cases or 0),
        python_executable=python_executable,
        omp_threads=int(ns.omp_threads),
        timeout_sec=int(ns.timeout),
        no_gpu=(len(gpu_ids) == 0),
        gpu_ids=gpu_ids,
        force=bool(ns.force),
        allow_resample=bool(ns.allow_resample),
        progress_callback=lambda msg: print(msg, flush=True),
        skip_completed=not bool(ns.no_skip_completed),
        rerun_failed_only=bool(ns.failed_only),
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep.get('ok') else 2


if __name__ == '__main__':
    raise SystemExit(main())
