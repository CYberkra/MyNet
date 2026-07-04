"""Preflight-check PGDA-CSNet dataset contract before training.

Verifies: required base keys exist (x_raw, y_mask, status_code, label_weight),
and when decomposition training is enabled, checks for clean/clutter arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_BASE_KEYS = ('x_raw', 'y_mask', 'status_code', 'label_weight')
OPTIONAL_GROUPS = {
    'x_clean': ('x_clean', 'clean', 'clean_bscan', 'x_target_clean'),
    'x_clutter': ('x_clutter', 'clutter_gt', 'c_gt', 'clutter', 'x_background', 'background_only'),
}
DECOMP_WEIGHT_KEYS = (
    'clean_recon_weight',
    'clutter_recon_weight',
    'clean_consistency_weight',
    'clutter_consistency_weight',
)


def _load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, encoding='utf-8') as f:
        return json.load(f)


def _resolve_data_root(cfg: dict, cfg_path: Path) -> Path:
    data_root = Path(cfg.get('data_root', 'data'))
    if data_root.is_absolute():
        return data_root
    return (ROOT / data_root) if cfg_path is None else (cfg_path.parent.parent / data_root).resolve()


def _iter_npz_files(data_root: Path) -> Iterable[Path]:
    window_dir = data_root / 'windows'
    return sorted(window_dir.glob('*.npz'))


def decomposition_enabled(cfg: dict) -> bool:
    loss_cfg = cfg.get('loss', {})
    return any(float(loss_cfg.get(key, 0.0)) > 0 for key in DECOMP_WEIGHT_KEYS)


def resolve_optional_aliases(cfg: dict, target_key: str) -> list[str]:
    aliases = list(OPTIONAL_GROUPS.get(target_key, (target_key,)))
    aliases.extend(cfg.get('dataset_contract', {}).get(target_key, []))
    aliases.extend(cfg.get('paired_array_aliases', {}).get(target_key, []))
    deduped = []
    seen = set()
    for key in aliases:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def inspect_npz(npz_path: Path, cfg: dict) -> list[str]:
    errors = []
    with np.load(npz_path, allow_pickle=False) as z:
        for key in REQUIRED_BASE_KEYS:
            if key not in z.files:
                errors.append(f'missing required key {key}')
        if not decomposition_enabled(cfg):
            return errors
        raw_shape = z['x_raw'].shape if 'x_raw' in z.files else None
        for target_key in OPTIONAL_GROUPS:
            aliases = resolve_optional_aliases(cfg, target_key)
            matched = next((alias for alias in aliases if alias in z.files), None)
            if matched is None:
                errors.append(f'missing decomposition array for {target_key}; checked {aliases}')
                continue
            if raw_shape is not None and z[matched].shape != raw_shape:
                errors.append(f'shape mismatch for {matched}: expected {raw_shape}, got {z[matched].shape}')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Preflight-check PGDA dataset contract before training.')
    parser.add_argument('config', help='Path to training config JSON.')
    parser.add_argument('--max-errors', type=int, default=20, help='Stop after this many errors.')
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_cfg(cfg_path)
    data_root = _resolve_data_root(cfg, cfg_path)
    npz_files = list(_iter_npz_files(data_root))
    if not npz_files:
        print(f'ERROR: no NPZ files found under {data_root / "windows"}')
        return 1

    enabled = decomposition_enabled(cfg)
    total_errors = 0
    for npz_path in npz_files:
        errors = inspect_npz(npz_path, cfg)
        for err in errors:
            print(f'{npz_path.name}: {err}')
            total_errors += 1
            if total_errors >= args.max_errors:
                print(f'Reached max errors ({args.max_errors}), stopping.')
                return 1
    if total_errors > 0:
        print(f'\nFAILED: {total_errors} errors found across {len(npz_files)} NPZ files.')
        return 1
    print(f'\nOK: all {len(npz_files)} NPZ files pass preflight check (decomp_enabled={enabled}).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
