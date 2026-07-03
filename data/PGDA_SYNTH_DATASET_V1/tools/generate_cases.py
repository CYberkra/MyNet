#!/usr/bin/env python3
"""
PGDA_SYNTH_DATASET_V1 — Case Generator v2

Generates gprMax simulation cases with varying target depths, non-flat labels,
correct domain sizing, and valid TX/RX placement.

Usage:
    python tools/generate_cases.py batch_002_depth_30cases --n-cases 30 --depth-range 6 24
    python tools/generate_cases.py --dry-run --n-cases 3  # preview
"""
import sys, os, json, math, random, csv, shutil
from pathlib import Path
from typing import Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = ROOT / '02_case_pool'

# ── Physical constants ──
C_AIR = 0.3
N_TIME = 501
TW_NS = 700.0
DT_NS = TW_NS / N_TIME

# ── Materials ──
MATERIALS = {
    'air':               {'eps': 1.0,  'sig': 0.0,    'v': 0.300, 'rho': 0},
    'moist_silty_clay':  {'eps': 13.5, 'sig': 0.001,  'v': 0.082, 'rho': 1},
    'weak_cover_band':   {'eps': 12.9, 'sig': 0.001,  'v': 0.084, 'rho': 2},
    'slide_zone':        {'eps': 24.0, 'sig': 0.003,  'v': 0.061, 'rho': 3},
    'weathered_bedrock': {'eps': 6.0,  'sig': 0.001,  'v': 0.122, 'rho': 4},
}
BACKGROUND = 'moist_silty_clay'

# ── Geometry defaults ──
DOMAIN_X = 480.0
DX = 0.05
PML = [60, 60, 0, 60, 60, 0]
UAV_H = 2.2
N_TRACES = 128
TRACE_STEP = 1.701898
SCAN_X0 = 120.0
TX_RX_OFFSET = 1.4

# ── Stratigraphy fractions ──
STRATA = [
    ('moist_silty_clay',    0.30),   # top soil
    ('weak_cover_band',     0.06),   # transition
    ('slide_zone',          0.12),   # TARGET
    ('weathered_bedrock',   0.25),   # basement
]
MARGIN_BELOW = 5.0  # extra space below deepest layer (m)
SURFACE_BASE = 0.0   # ground surface y (domain top) — matches LINE9 template convention

# ── Jitter ──
JITTER_AMP = 0.05  # ±m per trace, makes labels non-flat


def compute_domain_y(target_depth_m: float, terrain: str = 'flat') -> float:
    """Compute minimum domain_y needed. Matches LINE9 template convention:
    ground near y=0, air region below ground, TX in the air."""
    max_surface = 0.0 + (1.8 if terrain == 'terrain' else 0.0)
    total_strata = sum(frac * target_depth_m for _, frac in STRATA) + MARGIN_BELOW
    total_strata += len(STRATA) * JITTER_AMP
    strata_bottom = max_surface + total_strata
    # Leave 15m air gap below strata (consistent with LINE9 ~12m air gap)
    needed = strata_bottom + 15.0
    return max(45.0, math.ceil(needed / DX) * DX)


def make_trace_x():
    return np.linspace(SCAN_X0, SCAN_X0 + TRACE_STEP * (N_TRACES - 1), N_TRACES)


def terrain_fn(name: str, x: np.ndarray) -> np.ndarray:
    """Ground surface elevation near y=0 (domain top)."""
    if name == 'terrain':
        return 0.0 + 1.5 * np.sin(2 * np.pi * (x - 120) / 200) \
               + 0.3 * np.sin(2 * np.pi * (x - 120) / 30)
    return np.full_like(x, 0.0)


def layer_depths(surface_y: np.ndarray, target_depth_m: float, seed: int) -> dict:
    """
    For each trace, compute depth to top/bottom of each stratum.
    Adds random jitter so labels are NOT flat lines.
    """
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(-JITTER_AMP, JITTER_AMP, N_TRACES)

    # Cumulative thickness per trace with per-trace variation
    cum = np.zeros(N_TRACES)
    layers = {}
    for name, frac in STRATA:
        thick = frac * target_depth_m + jitter
        thick = np.maximum(thick, 0.01)  # ensure positive
        top = surface_y + cum
        bottom = top + thick
        layers[name] = {'top': top.copy(), 'bottom': bottom.copy()}
        cum += thick

    layers['total_strata'] = cum.copy()
    return layers


def generate_one(case_id: str, params: dict, batch_dir: Path, force: bool = False, seed: int = 42):
    """Generate one complete case directory. Returns True on success."""
    case_dir = batch_dir / 'cases' / case_id
    if case_dir.exists():
        if not force:
            print(f'  ⏭ {case_id}: exists (use --force)')
            return False
        shutil.rmtree(case_dir)

    td = params['target_depth_m']
    terrain = params['terrain']
    dom_y = compute_domain_y(td, terrain)
    uav_h = params.get('uav_height_m', UAV_H)
    rng_seed = seed

    # ── Geometry ──
    tx = np.linspace(SCAN_X0 - TX_RX_OFFSET/2,
                     SCAN_X0 + TRACE_STEP*(N_TRACES-1) - TX_RX_OFFSET/2, N_TRACES)
    rx = tx + TX_RX_OFFSET

    surf = terrain_fn(terrain, tx)  # ground surface y per trace (near y=0)
    layers = layer_depths(surf, td, rng_seed)

    # Place antenna below deepest ground stratum, in the air region
    # LINE9 convention: ground near y=0, air below, TX ~2.4m into air gap
    strata_bottom_y = float(np.max(surf + layers['total_strata']))
    antenna_y = strata_bottom_y + 2.4

    # ── Domain ──
    lines = []
    lines.append(f'#title: PGDA {case_id} depth={td}m {terrain}')
    lines.append(f'#domain: {DOMAIN_X} {dom_y} {DX}')
    lines.append(f'#dx_dy_dz: {DX} {DX} {DX}')
    lines.append(f'#time_window: {TW_NS*1e-9:.1e}')
    lines.append(f'#pml_cells: {" ".join(map(str, PML))}')
    for name, p in MATERIALS.items():
        lines.append(f'#material: {p["eps"]} {p["sig"]} 1 0 {name}')
    lines.append(f'#waveform: ricker 1 1e+08 uavgpr_wavelet')
    lines.append(f'#hertzian_dipole: z {tx[0]:.3f} {antenna_y:.3f} 0.025 uavgpr_wavelet')
    lines.append(f'#rx: {rx[0]:.3f} {antenna_y:.3f} 0.025')
    lines.append(f'#src_steps: {TRACE_STEP} 0 0')
    lines.append(f'#rx_steps: {TRACE_STEP} 0 0')
    lines.append(f'#geometry_view: 0 0 0 {DOMAIN_X} {dom_y} {DX} {DX} {DX} {DX} geometry_raw n')

    # ── Triangles ──
    # For each x segment (1m), create triangles for each stratum
    x_edges = np.arange(0, int(DOMAIN_X), 1.0)
    dx_tri = DX

    for i in range(len(x_edges) - 1):
        x1, x2 = x_edges[i], x_edges[i+1]
        x_mid = (x1 + x2) / 2
        ti = int(np.clip((x_mid - tx[0]) / TRACE_STEP, 0, N_TRACES - 2))

        for name_key, lyr in layers.items():
            if name_key == 'total_strata':
                continue
            ly_top = float(lyr['top'][ti])
            ly_bot = float(lyr['bottom'][ti])
            ly_top2 = float(lyr['top'][min(ti+1, N_TRACES-1)])
            ly_bot2 = float(lyr['bottom'][min(ti+1, N_TRACES-1)])
            # Use nearest-trace values for triangle vertices
            lines.append(f'#triangle: {x1:.2f} {ly_top:.3f} 0 {x2:.2f} {ly_top2:.3f} 0 {x2:.2f} {ly_bot2:.3f} 0 {dx_tri} {name_key} y')
            lines.append(f'#triangle: {x1:.2f} {ly_top:.3f} 0 {x2:.2f} {ly_bot2:.3f} 0 {x1:.2f} {ly_bot:.3f} 0 {dx_tri} {name_key} y')

        # Background below strata
        bg_top = float(surf[ti]) + float(layers['total_strata'][ti]) + MARGIN_BELOW
        lines.append(f'#triangle: {x1:.2f} {bg_top:.3f} 0 {x2:.2f} {bg_top:.3f} 0 {x2:.2f} {dom_y:.2f} 0 {dx_tri} {BACKGROUND} y')
        lines.append(f'#triangle: {x1:.2f} {bg_top:.3f} 0 {x2:.2f} {dom_y:.2f} 0 {x1:.2f} {dom_y:.2f} 0 {dx_tri} {BACKGROUND} y')

    in_text = '\n'.join(lines)

    # ── Labels ──
    slide_top = layers['slide_zone']['top']
    slide_bottom = layers['slide_zone']['bottom']
    slide_mid = (slide_top + slide_bottom) / 2

    target_vis = np.full(N_TRACES, np.nan, dtype=np.float32)
    target_geom = np.full(N_TRACES, np.nan, dtype=np.float32)
    for i in range(N_TRACES):
        if not np.isfinite(slide_mid[i]):
            continue
        # Two-way travel time
        air_twt = 2 * uav_h / C_AIR
        soil_depth = slide_mid[i] - surf[i]
        soil_v = MATERIALS['moist_silty_clay']['v']
        soil_twt = 2 * soil_depth / soil_v
        total_twt = air_twt + soil_twt
        # Visible phase = slightly inside the slide_zone
        vis_shift = 2 * (slide_mid[i] - slide_top[i]) / MATERIALS['slide_zone']['v']
        target_vis[i] = total_twt  # at top of slide zone
        target_geom[i] = total_twt - 2 * 0.5 / soil_v  # ~0.5m above slide

    time_501 = np.linspace(0, TW_NS, N_TIME, dtype=np.float32)
    y_soft = np.zeros((N_TIME, N_TRACES), dtype=np.float32)
    for i in range(N_TRACES):
        if not np.isfinite(target_vis[i]):
            continue
        center = target_vis[i]
        g = np.exp(-((time_501 - center) ** 2) / (2 * (8 * DT_NS) ** 2))
        g /= g.max() + 1e-10
        y_soft[:, i] = g.astype(np.float32)

    # Interface masks
    imask = np.zeros((N_TIME, N_TRACES), dtype=np.float32)
    for i in range(N_TRACES):
        if np.isfinite(target_vis[i]):
            c = int(target_vis[i] / DT_NS)
            imask[max(0, c-3):min(N_TIME, c+4), i] = 1.0

    # ── Save files ──
    # geometry
    (case_dir / 'geometry').mkdir(parents=True)
    (case_dir / 'geometry' / 'raw.in').write_text(in_text)
    mat_text = '\n'.join(f'{n}  eps={p["eps"]} sig={p["sig"]} v={p["v"]}' for n, p in MATERIALS.items())
    (case_dir / 'geometry' / 'materials.txt').write_text(mat_text)
    sw = {
        'case_id': case_id, 'target_depth_m': td, 'terrain': terrain,
        'domain_x': DOMAIN_X, 'domain_y': dom_y, 'dx': DX,
        'uav_height_m': uav_h, 'n_traces': N_TRACES,
    }
    (case_dir / 'geometry' / 'scene_world.json').write_text(json.dumps(sw, indent=2))

    # labels
    (case_dir / 'labels').mkdir(parents=True)
    np.save(case_dir / 'labels' / 'time_501_ns.npy', time_501)
    np.save(case_dir / 'labels' / 'trace_x_m.npy', tx.astype(np.float32))
    np.save(case_dir / 'labels' / 'y_soft_501x128.npy', y_soft)
    np.save(case_dir / 'labels' / 'interface_mask_bscan.npy', imask)
    np.save(case_dir / 'labels' / 'target_visible_phase_time_ns.npy', target_vis)
    np.save(case_dir / 'labels' / 'target_geom_time_ns.npy', target_geom)

    # tables
    (case_dir / 'tables').mkdir(parents=True)
    with open(case_dir / 'tables' / 'design_metrics.csv', 'w') as f:
        f.write(f'case_id,target_depth_m,domain_y,n_traces,target_visible_median_ns\n')
        f.write(f'{case_id},{td},{dom_y},{N_TRACES},{float(np.nanmedian(target_vis)):.1f}\n')

    # README
    (case_dir / 'README.md').write_text(
        f'# {case_id}\nDepth={td}m  Terrain={terrain}  UAV_H={uav_h:.1f}m\n')

    print(f'  ✅ {case_id}: depth={td:.1f}m {terrain:8s} domain_y={dom_y}  '
          f'vis={np.nanmedian(target_vis):.0f}ns  range={np.nanmax(target_vis)-np.nanmin(target_vis):.1f}ns')
    return True


def generate_batch(batch_id, n_cases=12, depth_range=(6.0, 24.0), seed=42, force=False, dry_run=False):
    batch_dir = POOL_DIR / batch_id
    if not dry_run:
        (batch_dir / 'cases').mkdir(parents=True)

    print(f'Batch: {batch_id}  ({n_cases} cases, depth {depth_range[0]}-{depth_range[1]}m)')
    rng = random.Random(seed)
    npre = rng.randint(0, 9999)

    if dry_run:
        for i in range(n_cases):
            td = rng.uniform(*depth_range)
            terr = 'flat' if rng.random() < 0.7 else 'terrain'
            dy = compute_domain_y(td, terr)
            print(f'  DEPTH_{i+1:03d}: depth={td:.1f}m {terr} domain_y={dy:.0f}')
        print(f'\nDry-run: {n_cases} cases')
        return

    ok = 0
    for i in range(n_cases):
        td = round(rng.uniform(*depth_range), 1)
        terr = 'flat' if rng.random() < 0.7 else 'terrain'
        cid = f'LINE9_STYLE_DEPTH_{i+1:03d}'
        params = {'target_depth_m': td, 'terrain': terr, 'uav_height_m': UAV_H}
        if generate_one(cid, params, batch_dir, force=force, seed=seed + i):
            ok += 1

    # Batch manifest
    with open(batch_dir / 'batch_manifest.csv', 'w') as f:
        f.write('case_id,batch_id,target_depth_m,terrain,domain_y\n')
        for cid in sorted(os.listdir(batch_dir / 'cases')):
            p = batch_dir / 'cases' / cid / 'tables' / 'design_metrics.csv'
            if p.exists():
                with open(p) as f2:
                    line2 = f2.read().splitlines()[1]
                f.write(f'{cid},{batch_id},{line2}\n')

    print(f'\n✅ Generated: {ok}/{n_cases} cases in {batch_dir}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('batch_id', nargs='?', default='batch_002_generated')
    ap.add_argument('--n-cases', type=int, default=12)
    ap.add_argument('--depth-range', type=float, nargs=2, default=[6.0, 24.0])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    generate_batch(args.batch_id, args.n_cases, tuple(args.depth_range),
                   args.seed, args.force, args.dry_run)


if __name__ == '__main__':
    main()
