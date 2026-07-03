"""
Traditional baseline comparison for PGDA-CSNet.
Implements SVD rank-1, f-k filter, and RPCA on raw B-scans,
then applies the same DP picking to compare end-to-end interface detection.
"""
from pathlib import Path
import csv, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"
LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
OUT = ROOT / "outputs" / "traditional_baselines"


def load_line(line):
    data = np.load(DATA_ROOT / "lines" / f"{line}.npz")
    raw = data["raw_full_normalized"].astype(np.float32)  # (501, N_traces)
    mask = data["soft_mask_train"].astype(np.float32)
    gt_valid = data["status_code"] > 0  # 1=strong, 2=weak
    dt_ns = float(data["dt_ns"])
    return raw, mask, gt_valid, dt_ns


# ── Traditional methods ────────────────────────────────────────

def svd_rank1(raw, rank=1):
    """SVD rank-1 reconstruction (dominant horizontal event = clutter)."""
    U, S, Vt = np.linalg.svd(raw, full_matrices=False)
    cleaned = raw - U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    return cleaned


def fk_filter(raw, dt_ns, velocity=0.1, cutoff_vel=0.06):
    """Simple f-k domain filter: remove low-velocity (steep dip) events."""
    from numpy.fft import fft2, ifft2, fftshift
    nt, nx = raw.shape
    fk = fftshift(fft2(raw))
    freqs = np.fft.fftfreq(nt, d=dt_ns * 1e-9)  # Hz
    freqs = np.fft.fftshift(freqs)
    wavenums = np.fft.fftfreq(nx, d=0.4)  # 0.4m trace interval
    wavenums = np.fft.fftshift(wavenums)
    F, K = np.meshgrid(freqs, wavenums, indexing="ij")
    # Apparent velocity mask: keep events with |v| > cutoff
    with np.errstate(divide="ignore", invalid="ignore"):
        apparent_v = np.where(np.abs(K) > 1e-10, np.abs(F) / np.abs(K), np.inf)
    keep = (apparent_v >= cutoff_vel) | (np.abs(F) < 5e6)  # always keep DC/low freq
    fk_filtered = fk * keep
    cleaned = np.real(ifft2(np.fft.ifftshift(fk_filtered)))
    return cleaned.astype(np.float32)


def rpca(raw, lam=None, max_iter=200, tol=1e-7):
    """Robust PCA via ADMM: decompose into low-rank (clutter) + sparse (target)."""
    n, m = raw.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, m))
    L = np.zeros_like(raw)
    S = np.zeros_like(raw)
    Y = np.zeros_like(raw)
    mu = 1.0
    rho = 1.5
    for _ in range(max_iter):
        X = raw - S + Y / mu
        U, sig, Vt = np.linalg.svd(X, full_matrices=False)
        L_new = U @ np.diag(np.maximum(sig - 1.0 / mu, 0)) @ Vt
        Z = raw - L_new + Y / mu
        S_new = np.sign(Z) * np.maximum(np.abs(Z) - lam / mu, 0)
        residual = np.linalg.norm(raw - L_new - S_new, "fro") / (np.linalg.norm(raw, "fro") + 1e-10)
        Y = Y + mu * (raw - L_new - S_new)
        mu = min(mu * rho, 1e10)
        L, S = L_new, S_new
        if residual < tol:
            break
    return S  # sparse component = target-like events


# ── DP picking (simplified) ────────────────────────────────────

def dp_pick(image, dt_ns, search_start_ns=320, search_end_ns=560, path_prob_thr=0.5):
    """Simple DP ridge extraction on a 2D image within a search window."""
    nt, nx = image.shape
    t0 = max(0, int(search_start_ns / dt_ns))
    t1 = min(nt, int(search_end_ns / dt_ns))
    region = image[t0:t1, :]
    # Soft probability: normalize to [0, 1]
    pmin, pmax = np.percentile(region, 5), np.percentile(region, 95)
    prob = np.clip((region - pmin) / (pmax - pmin + 1e-10), 0, 1)
    # DP
    n_traces = prob.shape[1]
    n_rows = prob.shape[0]
    dp = np.zeros_like(prob)
    bp = np.zeros_like(prob, dtype=np.int32)
    dp[:, 0] = prob[:, 0]
    max_jump = 6
    for j in range(1, n_traces):
        for i in range(n_rows):
            best_val, best_k = -1, i
            for di in range(-max_jump, max_jump + 1):
                k = i + di
                if 0 <= k < n_rows:
                    val = dp[k, j - 1]
                    if val > best_val:
                        best_val, best_k = val, k
            dp[i, j] = prob[i, j] + best_val * 0.95
            bp[i, j] = best_k
    # Trace back
    picks = np.full(n_traces, np.nan, dtype=np.float32)
    valid = np.zeros(n_traces, dtype=bool)
    j = n_traces - 1
    i = int(np.argmax(dp[:, j]))
    while j >= 0:
        picks[j] = i + t0
        valid[j] = dp[i, j] > path_prob_thr * n_rows * 0.1
        i = bp[i, j]
        j -= 1
    return picks, valid


# ── Evaluation ─────────────────────────────────────────────────

def evaluate(picks, valid, gt_mask, dt_ns):
    """Compute MAE and pick rate against soft_mask ground truth."""
    nt = gt_mask.shape[0]
    n_traces = gt_mask.shape[1]
    errors = []
    for j in range(n_traces):
        if not valid[j] or not np.isfinite(picks[j]):
            continue
        pred_sample = picks[j]
        # GT: peak of soft mask
        gt_col = gt_mask[:, j]
        gt_sample = int(np.argmax(gt_col))
        if gt_col[gt_sample] < 0.01:
            continue
        err_ns = abs(pred_sample - gt_sample) * dt_ns
        errors.append(err_ns)
    if not errors:
        return {"mae_ns": float("nan"), "pick_rate": 0.0, "n_picked": 0}
    return {
        "mae_ns": float(np.mean(errors)),
        "pick_rate": sum(valid) / n_traces,
        "n_picked": sum(valid),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    methods = {
        "raw_no_processing": lambda raw, dt_ns: raw,
        "svd_rank1_subtract": lambda raw, dt_ns: svd_rank1(raw, rank=1),
        "svd_rank2_subtract": lambda raw, dt_ns: svd_rank1(raw, rank=2),
        "fk_filter": lambda raw, dt_ns: fk_filter(raw, dt_ns),
        "rpca_sparse": lambda raw, dt_ns: rpca(raw),
    }

    all_rows = []
    for line in LINES:
        print(f"\n=== {line} ===")
        raw, gt_mask, gt_valid, dt_ns = load_line(line)
        print(f"  shape: {raw.shape}, dt: {dt_ns:.4f} ns")

        for name, method_fn in methods.items():
            cleaned = method_fn(raw, dt_ns)
            picks, valid = dp_pick(cleaned, dt_ns)
            metrics = evaluate(picks, valid, gt_mask, dt_ns)
            row = {"line": line, "method": name, **metrics}
            all_rows.append(row)
            print(f"  {name:25s}: MAE={metrics['mae_ns']:.3f} ns, pick_rate={metrics['pick_rate']:.3f}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Method':30s} {'Avg MAE ns':>10s} {'Avg Pick Rate':>14s}")
    print("-" * 80)
    methods_list = list(methods.keys())
    for method in methods_list:
        subset = [r for r in all_rows if r["method"] == method and np.isfinite(r["mae_ns"])]
        if subset:
            avg_mae = np.mean([r["mae_ns"] for r in subset])
            avg_pr = np.mean([r["pick_rate"] for r in subset])
            print(f"{method:30s} {avg_mae:10.3f} {avg_pr:14.3f}")

    write_csv(OUT / "traditional_baseline_comparison.csv", all_rows)
    print(f"\nSaved: {OUT / 'traditional_baseline_comparison.csv'}")
    print("TRADITIONAL_BASELINES_DONE")


if __name__ == "__main__":
    main()
