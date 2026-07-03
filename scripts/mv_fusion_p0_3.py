"""P0-3 Multi-view fusion: explore fusion strategies."""
from pathlib import Path
import sys, math, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_full_line import (
    breakable_dp_ridge_centerline, centerline,
    write_centerline_csv, write_metrics,
)

HOLDOUT = slice(1664, 2378)
PRESENCE_THR = 0.45
PATH_PROB_THR = 0.50
DP_MAX_JUMP = 6
DP_SMOOTH_WEIGHT = 0.16
DP_MIN_SEGMENT = 16
SEARCH_MIN_NS = 320.0
SEARCH_MAX_NS = 560.0
CFW = 0.5

ENSEMBLE_DIR = ROOT / 'outputs/eval_3seed_ensemble_line9_holdout'
ROBUST_DIR = ROOT / 'outputs/eval_v11_frozen_Line9_robust_pp020'
OUT_DIR = ROOT / 'outputs/eval_mvfusion_line9_holdout'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('=== Loading masks ===')
pred_e3 = np.load(ENSEMBLE_DIR / 'Line9_holdout_tr1664_2377_pred_softmask.npy')
center_e3 = np.load(ENSEMBLE_DIR / 'Line9_holdout_tr1664_2377_center_softmask.npy')
pres_e3 = np.load(ENSEMBLE_DIR / 'Line9_holdout_tr1664_2377_presence_prob.npy')

pred_r = np.load(ROBUST_DIR / 'Line9_pred_softmask.npy')[:, HOLDOUT]
center_r = np.load(ROBUST_DIR / 'Line9_center_softmask.npy')[:, HOLDOUT]
pres_r = np.load(ROBUST_DIR / 'Line9_presence_prob.npy')[HOLDOUT]

print('=== Loading ground truth ===')
data_root = ROOT / 'data_corrected_v1_4_terrain_direction'
line_npz = np.load(data_root / 'lines' / 'Line9.npz')
gt = line_npz['soft_mask_train'].astype(np.float32)[:, HOLDOUT]
dt_ns = float(line_npz['dt_ns'])
H, W = gt.shape

def dp_metric(cdp, vdp):
    cgt, vgt = centerline(gt, 1e-3)
    both = vgt & vdp & np.isfinite(cdp)
    if both.any():
        mae = float(np.nanmean(np.abs(cdp[both] - cgt[both]))) * dt_ns
    else:
        mae = float('nan')
    return mae, float(vdp.mean()), int(vdp.sum())

# ── Baseline: 3-seed ensemble ─────────────────────────────
print('\n=== Baseline: 3-seed ensemble ===')
path_e3 = (1 - CFW) * pred_e3 + CFW * center_e3
cdp_e3, vdp_e3 = breakable_dp_ridge_centerline(
    path_e3, pres_e3, presence_thr=PRESENCE_THR, path_prob_thr=PATH_PROB_THR,
    min_segment=DP_MIN_SEGMENT, max_jump=DP_MAX_JUMP, smooth_weight=DP_SMOOTH_WEIGHT,
    search_min_sample=int(SEARCH_MIN_NS / dt_ns),
    search_max_sample=int(SEARCH_MAX_NS / dt_ns),
)
mae_e3, pr_e3, n_e3 = dp_metric(cdp_e3, vdp_e3)
print(f'  MAE={mae_e3:.3f}  PR={pr_e3:.3f}  valid={n_e3}')

# ── Strategy 1: Center-only fusion (fine sweep around 0.25) ──
print('\n=== Strategy 1: Center-only fusion — fine sweep ===')
# Use ensemble pred + presence, but fuse center with robust norm
# Initial sweep showed best near 0.25
best_config = None
for rw in np.linspace(0.05, 0.45, 9):
    rw = round(rw, 2)
    center_fused = (1 - rw) * center_e3 + rw * center_r
    path_pred = (1 - CFW) * pred_e3 + CFW * center_fused
    cdp, vdp = breakable_dp_ridge_centerline(
        path_pred, pres_e3, presence_thr=PRESENCE_THR, path_prob_thr=PATH_PROB_THR,
        min_segment=DP_MIN_SEGMENT, max_jump=DP_MAX_JUMP, smooth_weight=DP_SMOOTH_WEIGHT,
        search_min_sample=int(SEARCH_MIN_NS / dt_ns),
        search_max_sample=int(SEARCH_MAX_NS / dt_ns),
    )
    mae, pr, n = dp_metric(cdp, vdp)
    delta_mae = mae - mae_e3
    delta_pr = pr - pr_e3
    print(f'  rw={rw:.2f}: MAE={mae:.3f} ({delta_mae:+.3f})  PR={pr:.3f} ({delta_pr:+.3f})  valid={n}')
    if best_config is None or mae < best_config[1]:
        best_config = (rw, mae, pr, n, cdp, vdp)

best_rw, best_mae, best_pr, best_n, best_cdp, best_vdp = best_config
print(f'\n  ** Best: rw={best_rw:.2f} MAE={best_mae:.3f} PR={best_pr:.3f} valid={best_n}')

# ── Strategy 2: Late DP fusion (average centerlines) ──────
print('\n=== Strategy 2: Late DP fusion ===')
cdp_r, vdp_r = breakable_dp_ridge_centerline(
    (1 - CFW) * pred_r + CFW * center_r, pres_r,
    presence_thr=PRESENCE_THR, path_prob_thr=PATH_PROB_THR,
    min_segment=DP_MIN_SEGMENT, max_jump=DP_MAX_JUMP, smooth_weight=DP_SMOOTH_WEIGHT,
    search_min_sample=int(SEARCH_MIN_NS / dt_ns),
    search_max_sample=int(SEARCH_MAX_NS / dt_ns),
)
both_valid = vdp_e3 & vdp_r & np.isfinite(cdp_e3) & np.isfinite(cdp_r)
cdp_fused = cdp_e3.copy()
cdp_fused[both_valid] = (cdp_e3[both_valid] + cdp_r[both_valid]) * 0.5
vdp_fused = both_valid
mae_lf, pr_lf, n_lf = dp_metric(cdp_fused, vdp_fused)
delta_lf = mae_lf - mae_e3
print(f'  Both-valid: MAE={mae_lf:.3f} ({delta_lf:+.3f})  PR={pr_lf:.3f}  valid={n_lf}')

# ── Strategy 3: Combined center + mask fusion ──────────────
print('\n=== Strategy 3: Combined center + mask fusion ===')
# Apply center fusion at best_rw AND small mask fusion
for mw in [0.05, 0.10, 0.15]:
    center_fused = (1 - best_rw) * center_e3 + best_rw * center_r
    pred_fused = (1 - mw) * pred_e3 + mw * pred_r
    path_pred = (1 - CFW) * pred_fused + CFW * center_fused
    cdp, vdp = breakable_dp_ridge_centerline(
        path_pred, pres_e3, presence_thr=PRESENCE_THR, path_prob_thr=PATH_PROB_THR,
        min_segment=DP_MIN_SEGMENT, max_jump=DP_MAX_JUMP, smooth_weight=DP_SMOOTH_WEIGHT,
        search_min_sample=int(SEARCH_MIN_NS / dt_ns),
        search_max_sample=int(SEARCH_MAX_NS / dt_ns),
    )
    mae, pr, n = dp_metric(cdp, vdp)
    print(f'  center_rw={best_rw:.2f} + mask_mw={mw:.2f}: MAE={mae:.3f} ({mae-mae_e3:+.3f})  PR={pr:.3f} ({pr-pr_e3:+.3f})  valid={n}')

# ── Final: save best config outputs ─────────────────────────
print(f'\n=== Saving best config: center_rw={best_rw:.2f} ===')
center_fused = (1 - best_rw) * center_e3 + best_rw * center_r
path_best = (1 - CFW) * pred_e3 + CFW * center_fused
eval_name = f'Line9_holdout_tr1664_2377_centerfusion{best_rw:.2f}'

np.save(OUT_DIR / f'{eval_name}_pred_softmask.npy', pred_e3)
np.save(OUT_DIR / f'{eval_name}_center_softmask.npy', center_fused)
np.save(OUT_DIR / f'{eval_name}_presence_prob.npy', pres_e3)
np.save(OUT_DIR / f'{eval_name}_path_softmask.npy', path_best)

cmean, vmean, cdp_best, vdp_best, cgt, vgt, path_prob = write_centerline_csv(
    OUT_DIR, eval_name, path_best, pres_e3, gt, dt_ns,
    search_min_ns=SEARCH_MIN_NS, search_max_ns=SEARCH_MAX_NS,
    presence_thr=PRESENCE_THR, path_prob_thr=PATH_PROB_THR,
    trace_offset=1664,
    dp_max_jump=DP_MAX_JUMP, dp_smooth_weight=DP_SMOOTH_WEIGHT,
    dp_breakable=True, dp_min_segment=DP_MIN_SEGMENT,
)

write_metrics(
    OUT_DIR, eval_name, pred_e3, pres_e3, gt,
    np.ones(W, dtype=np.int16), np.ones(W, dtype=np.float32), dt_ns,
    cmean=cmean, vmean=vmean, cdp=cdp_best, vdp=vdp_best, cgt=cgt, vgt=vgt,
    path_prob=path_prob,
    presence_thr=PRESENCE_THR, path_prob_thr=PATH_PROB_THR,
    trace_start=1664, trace_end=2377,
    dp_max_jump=DP_MAX_JUMP, dp_smooth_weight=DP_SMOOTH_WEIGHT,
    curve_source=f'centerfusion_{best_rw:.2f}_dp',
    dp_breakable=True, dp_min_segment=DP_MIN_SEGMENT,
)

# ── Summary ──────────────────────────────────────────────
print(f'\n========== P0-3 SUMMARY ==========')
print(f'  3-seed ensemble:  MAE={mae_e3:.3f}  PR={pr_e3:.3f}')
print(f'  Center fusion rw={best_rw:.2f}:  MAE={best_mae:.3f}  PR={best_pr:.3f}')
print(f'  Late DP fusion:   MAE={mae_lf:.3f}  PR={pr_lf:.3f}')
delta_mae = best_mae - mae_e3
delta_pr = best_pr - pr_e3
print(f'\n  ** Best: center fusion rw={best_rw:.2f} — MAE {delta_mae:+.3f} ({delta_mae/mae_e3*100:+.1f}%), PR {delta_pr:+.3f} ({delta_pr/pr_e3*100:+.1f}%)')
print(f'  ** Saved to {OUT_DIR / eval_name}')
print('P0-3_DONE')
