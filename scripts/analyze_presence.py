from pathlib import Path
import json

import numpy as np


def auc_score(values, labels):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))


root = Path(__file__).resolve().parents[1]
thresholds = json.loads((root / "reports/pick_thresholds.json").read_text(encoding="utf-8"))

for line_path in sorted((root / "data/lines").glob("*.npz")):
    line = np.load(line_path)
    raw = line["raw_full_normalized"]
    status = line["status_code"]
    dt_ns = float(line["dt_ns"])
    lo = max(0, int(round(320.0 / dt_ns)))
    hi = min(raw.shape[0], int(round(560.0 / dt_ns)) + 1)
    search = raw[lo:hi]
    trace_energy = np.mean(np.abs(search), axis=0)
    smooth_energy = np.convolve(trace_energy, np.ones(31) / 31, mode="same")
    hard = status != 2
    print(line_path.stem, "energy_auc", auc_score(trace_energy[hard], status[hard] == 1),
          "smooth_energy_auc", auc_score(smooth_energy[hard], status[hard] == 1))
    print(line_path.stem, "raw_abs_quantiles", np.quantile(np.abs(raw), [0.5, 0.9, 0.95, 0.99, 0.999, 1.0]))
    stats = []
    for code, name in ((0, "no_pick"), (1, "present"), (2, "weak")):
        values = search[:, status == code]
        stats.append((name, values.shape[1], float(np.abs(values).mean()), float(values.std())))
    print(line_path.stem, "raw_search_stats", stats)

for fold in thresholds["fold_hashes"]:
    stem = "_".join(
        (
            Path(fold["run_dir"]).name,
            fold["line"],
            fold["checkpoint_sha256_12"],
            fold["line_sha256_12"],
        )
    )
    presence = np.load(root / "reports/calibration_cache" / f"{stem}_presence.npy")
    pred = np.load(root / "reports/calibration_cache" / f"{stem}_pred.npy")
    line = np.load(root / "data/lines" / f"{fold['line']}.npz")
    status = line["status_code"]
    dt_ns = float(line["dt_ns"])
    lo = max(0, int(round(320.0 / dt_ns)))
    hi = min(pred.shape[0], int(round(560.0 / dt_ns)) + 1)
    path_peak = pred[lo:hi].max(axis=0)
    print(fold["line"], "all", np.quantile(presence, [0, 0.1, 0.5, 0.9, 1]))
    for code, name in ((0, "no_pick"), (1, "present"), (2, "weak")):
        values = presence[status == code]
        peaks = path_peak[status == code]
        print(name, len(values), "presence", np.quantile(values, [0, 0.1, 0.5, 0.9, 1]))
        print(name, len(values), "path_peak", np.quantile(peaks, [0, 0.1, 0.5, 0.9, 1]))
