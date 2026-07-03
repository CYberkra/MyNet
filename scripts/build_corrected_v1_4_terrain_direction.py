from pathlib import Path
import argparse
import csv
import json
import math
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VELOCITY_M_PER_NS = 0.074


CORRECTION_WINDOWS = {
    "Line3": [
        {
            "name": "line9_crossing_soft_adjust",
            "start": 90,
            "end": 260,
            "anchors": [(90, 13.9), (164, 14.7), (260, 14.2)],
            "depth_margin_m": 0.9,
            "confidence": 0.44,
            "status": 2,
            "reason": "Line3-Line9 crossing is inconsistent, but Line3's own ZK08-side anchor is shallower than Line9. Apply only a soft local correction.",
        },
        {
            "name": "line7_crossing_pdf_anchor",
            "start": 1180,
            "end": 1740,
            "anchors": [(1180, 14.7), (1447, 17.0), (1740, 17.2)],
            "depth_margin_m": 1.2,
            "confidence": 0.66,
            "status": 1,
            "reason": "Line3-Line7 GPS crossing is constrained by UavGPR Line7 page to about 17 m.",
        },
    ],
    "Line6": [
        {
            "name": "line9_to_lineL1_continuous_pdf_anchor",
            "start": 390,
            "end": 1050,
            "anchors": [(390, 11.3), (520, 14.6), (720, 16.2), (850, 18.0), (955, 18.0), (1050, 17.2)],
            "depth_margin_m": 1.3,
            "confidence": 0.68,
            "status": 1,
            "reason": "Line6-Line9 crossing is too shallow in v1, while Line6-LineL1 is a correct about-18 m anchor. Redraw one continuous segment to avoid window-boundary kinks.",
        },
    ],
    "Line7": [
        {
            "name": "line3_crossing_pdf_anchor",
            "start": 1040,
            "end": 1500,
            "anchors": [(1040, 13.4), (1282, 17.0), (1500, 15.5)],
            "depth_margin_m": 1.4,
            "confidence": 0.66,
            "status": 1,
            "reason": "UavGPR Line7 page states the basal/interface approaches about 17 m near the Line3 crossing.",
        },
    ],
    "LineL1": [
        {
            "name": "line3_crossing_pdf_anchor",
            "start": 0,
            "end": 150,
            "anchors": [(0, 12.4), (60, 13.0), (150, 12.5)],
            "depth_margin_m": 0.8,
            "confidence": 0.55,
            "status": 2,
            "reason": "LineL1 at the Line3 crossing is slightly shallow; apply a weak local correction toward about 13 m.",
        },
        {
            "name": "line6_crossing_pdf_anchor",
            "start": 860,
            "end": 1450,
            "anchors": [(860, 13.2), (1000, 16.6), (1103, 18.0), (1250, 17.0), (1450, 14.8)],
            "depth_margin_m": 1.4,
            "confidence": 0.68,
            "status": 1,
            "reason": "LineL1 is too shallow at the Line6 crossing; PDF/UavGPR pages constrain this crossing to about 18 m.",
        },
    ],
    "LineX1": [
        {
            "name": "line9_crossing_review_anchor",
            "start": 120,
            "end": 275,
            "anchors": [(120, 12.1), (193, 15.0), (275, 12.4)],
            "depth_margin_m": 1.5,
            "confidence": 0.48,
            "status": 2,
            "reason": "X1 has no engineering profile PDF. Its Line9 crossing is inconsistent, so use a conservative weak correction.",
        },
        {
            "name": "lineL1_crossing_review_anchor",
            "start": 600,
            "end": 720,
            "anchors": [(600, 12.0), (658, 13.0), (720, 12.1)],
            "depth_margin_m": 0.9,
            "confidence": 0.48,
            "status": 2,
            "reason": "X1-L1 crossing should be in the 12-15 m band; move the weak label to about 13 m.",
        },
    ],
}


PDF_ANCHOR_CHECKS = [
    ("C01", "Line3-Line7", "Line3", 1447, 16.0, 18.0),
    ("C01", "Line3-Line7", "Line7", 1282, 16.0, 18.0),
    ("C02", "Line6-Line9", "Line6", 520, 13.0, 16.0),
    ("C02", "Line6-Line9", "Line9", 1366, 13.0, 16.0),
    ("C03", "Line6-LineL1", "Line6", 955, 17.0, 19.5),
    ("C03", "Line6-LineL1", "LineL1", 1103, 17.0, 19.5),
    ("C04", "Line3-LineL1", "LineL1", 60, 12.0, 14.5),
    ("C05", "LineL1-LineX1", "LineL1", 566, 12.0, 15.0),
    ("C05", "LineL1-LineX1", "LineX1", 658, 12.0, 15.0),
]


CROSSING_PAIRS = [
    ("Line3", 1447, "Line7", 1282),
    ("Line3", 164, "Line9", 319),
    ("Line3", 585, "LineL1", 60),
    ("Line6", 1627, "Line7", 210),
    ("Line6", 520, "Line9", 1366),
    ("Line6", 955, "LineL1", 1103),
    ("Line9", 849, "LineX1", 193),
    ("LineL1", 566, "LineX1", 658),
]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def centerline(mask, min_sum=1e-4):
    h, _ = mask.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    s = mask.sum(axis=0)
    c = (mask * yy).sum(axis=0) / np.maximum(s, 1e-6)
    v = s > min_sum
    c[~v] = np.nan
    return c, v


def moving_average_axis(a, win, axis):
    if win <= 1:
        return a
    pad = win // 2
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad, pad)
    ap = np.pad(a, pad_width, mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis, ap)


def depth_to_sample(depth_m, dt_ns):
    return (2.0 * np.asarray(depth_m, dtype=np.float32) / VELOCITY_M_PER_NS) / float(dt_ns)


def sample_to_depth(sample, dt_ns):
    return np.asarray(sample, dtype=np.float32) * float(dt_ns) * VELOCITY_M_PER_NS * 0.5


def make_score(raw):
    x = raw.astype(np.float32)
    x = x - np.nanmedian(x, axis=1, keepdims=True)
    score = np.abs(x)
    scale = float(np.nanpercentile(score, 98.0))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    score = np.clip(score / scale, 0.0, 3.0)
    score = moving_average_axis(score, 5, axis=0)
    score = moving_average_axis(score, 9, axis=1)
    return score.astype(np.float32)


def dp_pick_variable_band(score, target_sample, margin_sample, max_jump=7, smooth_weight=0.045, target_weight=0.9):
    h, w = score.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    tgt = target_sample.astype(np.float32)[None, :]
    prior = np.exp(-0.5 * ((yy - tgt) / max(float(margin_sample) * 0.45, 2.0)) ** 2).astype(np.float32)
    work = score + float(target_weight) * prior
    lo = np.maximum(0, np.floor(target_sample - margin_sample).astype(int))
    hi = np.minimum(h - 1, np.ceil(target_sample + margin_sample).astype(int))
    for x in range(w):
        if lo[x] > 0:
            work[: lo[x], x] = -1e6
        if hi[x] + 1 < h:
            work[hi[x] + 1 :, x] = -1e6
    dp = np.empty((h, w), np.float32)
    back = np.zeros((h, w), np.int16)
    dp[:, 0] = work[:, 0]
    offsets = np.arange(-max_jump, max_jump + 1, dtype=np.int16)
    for x in range(1, w):
        prev = dp[:, x - 1]
        cand = np.full((len(offsets), h), -1e9, dtype=np.float32)
        for oi, off in enumerate(offsets):
            penalty = np.float32(smooth_weight * (int(off) ** 2))
            if off < 0:
                cand[oi, -off:] = prev[:off] - penalty
            elif off > 0:
                cand[oi, :-off] = prev[off:] - penalty
            else:
                cand[oi, :] = prev
        arg = np.argmax(cand, axis=0).astype(np.int16)
        dp[:, x] = work[:, x] + cand[arg, np.arange(h)]
        back[:, x] = np.clip(np.arange(h, dtype=np.int32) + offsets[arg].astype(np.int32), 0, h - 1).astype(np.int16)
    path = np.zeros(w, np.float32)
    y = int(np.argmax(dp[:, -1]))
    path[-1] = y
    for x in range(w - 1, 0, -1):
        y = int(back[y, x])
        path[x - 1] = y
    return path


def taper(n, edge_frac=0.16):
    t = np.ones(n, dtype=np.float32)
    edge = max(1, int(round(n * edge_frac)))
    if edge * 2 >= n:
        x = np.linspace(0.0, math.pi, n, dtype=np.float32)
        return np.sin(x).astype(np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, edge, dtype=np.float32))
    t[:edge] = ramp
    t[-edge:] = ramp[::-1]
    return t


def interp_anchors(anchors, xs):
    ax = np.asarray([a[0] for a in anchors], dtype=np.float32)
    ay = np.asarray([a[1] for a in anchors], dtype=np.float32)
    return np.interp(xs.astype(np.float32), ax, ay).astype(np.float32)


def make_mask_from_path(path_sample, confidence, h, sigma_samples):
    yy = np.arange(h, dtype=np.float32)[:, None]
    mask = np.exp(-0.5 * ((yy - path_sample[None, :]) / float(sigma_samples)) ** 2)
    return (mask * confidence[None, :]).astype(np.float32)


def gained_bscan(raw):
    x = raw.astype(np.float32)
    x = x - np.nanmedian(x, axis=1, keepdims=True)
    env = np.sqrt(moving_average_axis(x * x, 41, axis=0))
    base = float(np.nanpercentile(np.abs(x), 50.0))
    y = x / (env + max(base * 0.08, 1e-6))
    y = moving_average_axis(y, 3, axis=1)
    vmax = float(np.nanpercentile(np.abs(y), 99.2))
    if not np.isfinite(vmax) or vmax <= 1e-6:
        vmax = 1.0
    return np.clip(y / vmax, -1.0, 1.0).astype(np.float32)


def apply_line_corrections(line, z):
    arrays = {k: z[k] for k in z.files}
    raw = z["raw_full_normalized"].astype(np.float32)
    old_mask = z["soft_mask_train"].astype(np.float32)
    old_status = z["status_code"].astype(np.int16)
    old_weight = z["label_weight"].astype(np.float32)
    dt_ns = float(z["dt_ns"])
    h, w = raw.shape
    old_path, valid = centerline(old_mask, 1e-3)
    old_depth = sample_to_depth(old_path, dt_ns)
    new_path = old_path.copy()
    new_status = old_status.copy()
    new_weight = old_weight.copy()
    review_code = np.zeros(w, dtype=np.int16)
    score = make_score(raw)
    notes = []
    for win in CORRECTION_WINDOWS.get(line, []):
        s = max(0, int(win["start"]))
        e = min(w - 1, int(win["end"]))
        if e < s:
            continue
        xs = np.arange(s, e + 1, dtype=np.int32)
        target_depth = interp_anchors(win["anchors"], xs)
        target_sample = depth_to_sample(target_depth, dt_ns)
        margin_sample = max(5, int(round(float(win["depth_margin_m"]) * 2.0 / VELOCITY_M_PER_NS / dt_ns)))
        picked = dp_pick_variable_band(score[:, s : e + 1], target_sample, margin_sample)
        blend = taper(len(xs))
        new_path[xs] = (1.0 - blend) * new_path[xs] + blend * picked
        active = blend > 0.08
        active_x = xs[active]
        new_status[active_x] = int(win["status"])
        new_weight[active_x] = np.maximum(np.minimum(new_weight[active_x], 0.72), float(win["confidence"]))
        if int(win["status"]) == 2:
            new_weight[active_x] = np.minimum(new_weight[active_x], float(win["confidence"]))
        review_code[active_x] = np.maximum(review_code[active_x], 1 if int(win["status"]) == 1 else 2)
        notes.append(
            {
                "line": line,
                "window": win["name"],
                "start_trace": s,
                "end_trace": e,
                "status": int(win["status"]),
                "confidence": float(win["confidence"]),
                "reason": win["reason"],
            }
        )
    sigma_ns = float(z["correction_sigma_ns"]) if "correction_sigma_ns" in z.files else 8.0
    sigma_samples = max(3, int(round(sigma_ns / dt_ns)))
    conf = np.clip(new_weight, 0.0, 0.86)
    new_mask = make_mask_from_path(new_path.astype(np.float32), conf.astype(np.float32), h, sigma_samples)
    arrays["soft_mask_train"] = new_mask.astype(np.float32)
    arrays["status_code"] = new_status.astype(np.int16)
    arrays["label_weight"] = conf.astype(np.float32)
    arrays["review_code_v1_4_terrain_direction"] = review_code.astype(np.int16)
    arrays["correction_v1_4_note"] = np.array(
        json.dumps(
            {
                "version": "data_corrected_v1_4_terrain_direction",
                "source_report": "reports/full_label_reaudit_terrain_direction/FULL_LABEL_REAUDIT_TERRAIN_DIRECTION_REPORT.md",
                "policy": "Local redraw using PDF arrows, GPS crossing constraints, terrain-aware audit, and raw B-scan ridge search near anchored depth windows.",
                "line_notes": notes,
            },
            ensure_ascii=False,
        )
    )
    return arrays, old_path, new_path, review_code, notes


def window_starts(w, width=256, stride=128):
    starts = list(range(0, max(1, w - width + 1), stride))
    last = max(0, w - width)
    if starts[-1] != last:
        starts.append(last)
    return starts


def save_windows(lines_dir, windows_dir, index_path):
    if windows_dir.exists():
        shutil.rmtree(windows_dir)
    windows_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line_path in sorted(lines_dir.glob("*.npz")):
        z = np.load(line_path, allow_pickle=False)
        line = line_path.stem
        raw = z["raw_full_normalized"].astype(np.float32)
        mask = z["soft_mask_train"].astype(np.float32)
        status = z["status_code"].astype(np.int16)
        weight = z["label_weight"].astype(np.float32)
        split = str(z["split"].item() if z["split"].shape == () else z["split"])
        _, w = raw.shape
        for s in window_starts(w):
            e = s + 255
            sample_id = f"{line}_tr{s:04d}_{e:04d}"
            np.savez_compressed(
                windows_dir / f"{sample_id}.npz",
                x_raw=raw[:, s : e + 1],
                y_mask=mask[:, s : e + 1],
                status_code=status[s : e + 1],
                label_weight=weight[s : e + 1],
            )
            st = status[s : e + 1]
            rows.append(
                {
                    "sample_id": sample_id,
                    "line": line,
                    "start": s,
                    "end": e,
                    "split": split,
                    "present": int((st == 1).sum()),
                    "weak": int((st == 2).sum()),
                    "no_pick": int((st == 0).sum()),
                }
            )
    write_csv(index_path, rows, ["sample_id", "line", "start", "end", "split", "present", "weak", "no_pick"])


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_preview(line, old_z, new_z, old_path, new_path, review_code, out_path):
    raw = old_z["raw_full_normalized"].astype(np.float32)
    old_mask = old_z["soft_mask_train"].astype(np.float32)
    new_mask = new_z["soft_mask_train"].astype(np.float32)
    dt_ns = float(old_z["dt_ns"])
    gain = gained_bscan(raw)
    h, w = raw.shape
    t = np.arange(h, dtype=np.float32) * dt_ns
    extent = (0, w - 1, t[-1], t[0])
    x = np.arange(w)
    vmax = max(0.08, float(np.nanpercentile(np.abs(raw), 98)))
    fig, ax = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    ax = ax.ravel()
    ax[0].imshow(raw, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=extent)
    ax[0].plot(x, old_path * dt_ns, color="#facc15", lw=0.8)
    ax[0].set_title(f"{line} raw B-scan with v1 centerline")
    ax[1].imshow(gain, aspect="auto", cmap="gray", vmin=-1, vmax=1, extent=extent)
    ax[1].plot(x, old_path * dt_ns, color="#facc15", lw=0.8, label="v1")
    ax[1].plot(x, new_path * dt_ns, color="#38bdf8", lw=0.9, label="v1.4")
    ax[1].set_title("gained B-scan with old/new centerlines")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].imshow(gain, aspect="auto", cmap="gray", vmin=-1, vmax=1, extent=extent)
    ax[2].imshow(old_mask, aspect="auto", cmap="viridis", alpha=np.clip(old_mask * 0.75, 0, 0.55), extent=extent)
    ax[2].set_title("corrected-v1 mask")
    ax[3].imshow(gain, aspect="auto", cmap="gray", vmin=-1, vmax=1, extent=extent)
    ax[3].imshow(new_mask, aspect="auto", cmap="magma", alpha=np.clip(new_mask * 0.75, 0, 0.55), extent=extent)
    if review_code.max() > 0:
        marked = review_code > 0
        y = np.full(marked.sum(), t[-1] * 0.96, dtype=np.float32)
        ax[3].scatter(x[marked], y, s=1.0, color="#22c55e", alpha=0.55)
    ax[3].set_title("corrected-v1.4 terrain/direction mask")
    for a in ax:
        a.set_xlabel("trace")
        a.set_ylabel("time ns")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def depth_at(line_arrays, line, trace):
    z = line_arrays[line]
    path, valid = centerline(z["soft_mask_train"].astype(np.float32), 1e-3)
    if trace < 0 or trace >= path.size or not np.isfinite(path[trace]):
        return float("nan")
    return float(sample_to_depth(path[trace], float(z["dt_ns"])))


def postcheck(line_arrays):
    anchor_rows = []
    for cid, pair, line, trace, lo, hi in PDF_ANCHOR_CHECKS:
        depth = depth_at(line_arrays, line, trace)
        if not np.isfinite(depth):
            verdict = "missing"
            delta = ""
        elif lo <= depth <= hi:
            verdict = "ok"
            delta = 0.0
        elif depth < lo:
            verdict = "too_shallow"
            delta = round(depth - lo, 3)
        else:
            verdict = "too_deep"
            delta = round(depth - hi, 3)
        anchor_rows.append(
            {
                "constraint_id": cid,
                "line_pair": pair,
                "line": line,
                "trace_idx": trace,
                "depth_m": "" if not np.isfinite(depth) else round(depth, 3),
                "expected_low_m": lo,
                "expected_high_m": hi,
                "verdict": verdict,
                "delta_to_band_m": delta,
            }
        )
    crossing_rows = []
    for line_a, trace_a, line_b, trace_b in CROSSING_PAIRS:
        da = depth_at(line_arrays, line_a, trace_a)
        db = depth_at(line_arrays, line_b, trace_b)
        diff = abs(da - db) if np.isfinite(da) and np.isfinite(db) else float("nan")
        if not np.isfinite(diff):
            verdict = "missing"
        elif diff <= 1.0:
            verdict = "ok"
        elif diff <= 2.0:
            verdict = "review_offset"
        else:
            verdict = "inconsistent"
        crossing_rows.append(
            {
                "line_pair": f"{line_a}-{line_b}",
                "line_a": line_a,
                "trace_a": trace_a,
                "depth_a_m": "" if not np.isfinite(da) else round(da, 3),
                "line_b": line_b,
                "trace_b": trace_b,
                "depth_b_m": "" if not np.isfinite(db) else round(db, 3),
                "abs_depth_difference_m": "" if not np.isfinite(diff) else round(diff, 3),
                "verdict": verdict,
            }
        )
    return anchor_rows, crossing_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_corrected_v1")
    ap.add_argument("--out", default="data_corrected_v1_4_terrain_direction")
    ap.add_argument("--report-dir", default="reports/v1_4_terrain_direction_labels")
    args = ap.parse_args()
    src = resolve(args.src)
    out = resolve(args.out)
    report = resolve(args.report_dir)
    if out.exists():
        shutil.rmtree(out)
    if report.exists():
        shutil.rmtree(report)
    (out / "lines").mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    all_notes = []
    summary_rows = []
    line_arrays = {}
    for p in sorted((src / "lines").glob("*.npz")):
        line = p.stem
        old_z = np.load(p, allow_pickle=False)
        arrays, old_path, new_path, review_code, notes = apply_line_corrections(line, old_z)
        np.savez_compressed(out / "lines" / p.name, **arrays)
        new_z = np.load(out / "lines" / p.name, allow_pickle=False)
        line_arrays[line] = {k: new_z[k] for k in new_z.files}
        plot_preview(line, old_z, new_z, old_path, new_path, review_code, report / "previews" / f"{line}_v1_vs_v1_4.png")
        all_notes.extend(notes)
        old_depth = sample_to_depth(old_path, float(old_z["dt_ns"]))
        new_depth = sample_to_depth(new_path, float(old_z["dt_ns"]))
        changed = np.abs(new_depth - old_depth) > 0.25
        s_old = old_z["status_code"].astype(np.int16)
        s_new = new_z["status_code"].astype(np.int16)
        summary_rows.append(
            {
                "line": line,
                "trace_count": int(new_depth.size),
                "changed_gt_0p25m_traces": int(changed.sum()),
                "max_abs_depth_change_m": round(float(np.nanmax(np.abs(new_depth - old_depth))), 3),
                "old_depth_min_m": round(float(np.nanmin(old_depth)), 3),
                "old_depth_max_m": round(float(np.nanmax(old_depth)), 3),
                "new_depth_min_m": round(float(np.nanmin(new_depth)), 3),
                "new_depth_max_m": round(float(np.nanmax(new_depth)), 3),
                "v1_strong": int((s_old == 1).sum()),
                "v1_weak": int((s_old == 2).sum()),
                "v1_4_strong": int((s_new == 1).sum()),
                "v1_4_weak": int((s_new == 2).sum()),
                "review_marked_traces": int((review_code > 0).sum()),
            }
        )
    save_windows(out / "lines", out / "windows", out / "window_index.csv")
    anchor_rows, crossing_rows = postcheck(line_arrays)
    write_csv(report / "v1_4_line_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(report / "v1_4_applied_windows.csv", all_notes, ["line", "window", "start_trace", "end_trace", "status", "confidence", "reason"])
    write_csv(report / "v1_4_pdf_anchor_postcheck.csv", anchor_rows, list(anchor_rows[0].keys()))
    write_csv(report / "v1_4_crossing_consistency_postcheck.csv", crossing_rows, list(crossing_rows[0].keys()))
    readme = [
        "# data_corrected_v1_4_terrain_direction",
        "",
        "This dataset is generated from corrected-v1 labels after the full terrain/direction/crossing re-audit.",
        "",
        "Policy:",
        "",
        "- Raw input remains `raw_full_normalized`; gained B-scans are preview-only.",
        "- Redraws are local and anchored by PDF arrows, GPS crossing trace indices, terrain-aware review, and raw B-scan ridge search.",
        "- Line9 is mostly preserved and remains the main held-out anchor line; its documented 21 m anomaly is not labeled as basal/interface.",
        "- LineX1 remains a conservative review line because no separate engineering-profile PDF is available.",
        "",
        "See `reports/v1_4_terrain_direction_labels` for previews and postchecks.",
    ]
    (out / "DATASET_README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (report / "V1_4_LABEL_CORRECTION_REPORT.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(out)
    print(report)


if __name__ == "__main__":
    main()
