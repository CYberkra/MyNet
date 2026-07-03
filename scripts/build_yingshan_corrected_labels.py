from pathlib import Path
import csv
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data"
OUT = ROOT / "data_corrected_v1"
REPORT = ROOT / "reports" / "yingshan_label_correction_v1"
EXTERNAL = ROOT.parent / "external_yingshan_materials"


RULES = {
    "Line3": {
        "depth_min_m": 14.0,
        "depth_max_m": 17.0,
        "source": "UavGPR p29; ZK07 ~16.7m, ZK08 ~14.4m",
        "notes": "Basal should follow profile-constrained 14-17m zone.",
    },
    "Line6": {
        "depth_min_m": 11.0,
        "depth_max_m": 18.0,
        "source": "UavGPR p30; ZK09 ~11.5m",
        "notes": "Old labels were mostly no-pick/weak; PDF says basal is present.",
    },
    "Line7": {
        "depth_min_m": 10.0,
        "depth_max_m": 18.0,
        "source": "UavGPR p31; ZK09 ~11.5m; cross-check Line3",
        "notes": "Use crossing constraint with Line3.",
    },
    "Line9": {
        "depth_min_m": 12.0,
        "depth_max_m": 16.0,
        "reject_depth_ranges_m": [[20.0, 23.0]],
        "source": "UavGPR p32, p39-p40; ZK08 ~14-14.3m; 21m anomaly is not basal",
        "notes": "Deep 21m anomaly must not be labeled as basal.",
    },
    "LineL1": {
        "depth_min_m": 12.0,
        "depth_max_m": 20.0,
        "reject_depth_ranges_m": [[21.0, 24.0]],
        "source": "UavGPR p33; crosses Line3 and Line6; possible 22m anomaly",
        "notes": "Use crossing constraints; keep 22m anomaly out of basal label.",
    },
    "LineX1": {
        "depth_min_m": 12.0,
        "depth_max_m": 15.0,
        "source": "UavGPR p34; crosses LineL1",
        "notes": "Old labels were empty; PDF says basal is visible.",
    },
}


def depth_to_sample(depth_m, dt_ns, velocity_m_per_ns):
    return int(round((2.0 * float(depth_m) / velocity_m_per_ns) / float(dt_ns)))


def moving_average_axis(a, win, axis):
    if win <= 1:
        return a
    pad = win // 2
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad, pad)
    ap = np.pad(a, pad_width, mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis, ap)


def centerline(mask, min_sum=1e-4):
    h, w = mask.shape
    y = np.arange(h, dtype=np.float32)[:, None]
    s = mask.sum(axis=0)
    c = (mask * y).sum(axis=0) / np.maximum(s, 1e-6)
    v = s > min_sum
    c[~v] = np.nan
    return c, v


def dp_pick(score, lo, hi, max_jump=4, smooth_weight=0.035):
    band = score[lo : hi + 1].astype(np.float32)
    h, w = band.shape
    band = band - np.nanmin(band)
    denom = np.nanpercentile(band, 99.0)
    if not np.isfinite(denom) or denom <= 1e-6:
        denom = float(np.nanmax(band) + 1e-6)
    band = np.clip(band / denom, 0.0, 2.0)
    dp = np.empty((h, w), np.float32)
    back = np.zeros((h, w), np.int16)
    dp[:, 0] = band[:, 0]
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
        dp[:, x] = band[:, x] + cand[arg, np.arange(h)]
        back[:, x] = np.clip(np.arange(h, dtype=np.int32) + offsets[arg].astype(np.int32), 0, h - 1).astype(np.int16)
    path = np.zeros(w, np.float32)
    y = int(np.argmax(dp[:, -1]))
    path[-1] = y + lo
    for x in range(w - 1, 0, -1):
        y = int(back[y, x])
        path[x - 1] = y + lo
    strength = band[np.clip(np.round(path - lo).astype(int), 0, h - 1), np.arange(w)]
    return path, strength


def build_score(raw, old_mask, lo, hi):
    bgr = raw - np.median(raw, axis=1, keepdims=True)
    score = np.abs(bgr)
    scale = np.nanpercentile(score, 98.0)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    score = np.clip(score / scale, 0.0, 3.0)
    score = moving_average_axis(score, 5, axis=0)
    score = moving_average_axis(score, 9, axis=1)

    old = old_mask.astype(np.float32)
    if old.max() > 0:
        old_prior = old / max(float(old.max()), 1e-6)
        allowed = np.zeros_like(old_prior)
        allowed[lo : hi + 1] = old_prior[lo : hi + 1]
        score = score + 0.55 * allowed
    return score.astype(np.float32)


def make_mask(path, confidence, h, sigma_samples):
    yy = np.arange(h, dtype=np.float32)[:, None]
    mask = np.exp(-0.5 * ((yy - path[None, :]) / float(sigma_samples)) ** 2)
    return (mask * confidence[None, :]).astype(np.float32)


def status_from_strength(strength, w, endpoint_frac=0.045):
    if w >= 31:
        kernel = np.ones(31, dtype=np.float32) / 31.0
        strength = np.convolve(np.pad(strength, (15, 15), mode="edge"), kernel, mode="valid")
    q10 = float(np.nanpercentile(strength, 10))
    q85 = float(np.nanpercentile(strength, 85))
    denom = max(q85 - q10, 1e-6)
    conf = 0.34 + 0.52 * np.clip((strength - q10) / denom, 0.0, 1.0)
    status = np.where(conf >= 0.58, 1, 2).astype(np.int16)

    edge = max(8, int(round(w * endpoint_frac)))
    if edge * 2 < w:
        status[:edge] = 2
        status[-edge:] = 2
        conf[:edge] = np.minimum(conf[:edge], 0.42)
        conf[-edge:] = np.minimum(conf[-edge:], 0.42)

    very_low = strength < float(np.nanpercentile(strength, 3))
    status[very_low] = 2
    conf[very_low] = np.minimum(conf[very_low], 0.36)
    min_present_run = max(24, int(round(w * 0.035)))
    i = 0
    while i < w:
        j = i + 1
        while j < w and status[j] == status[i]:
            j += 1
        if status[i] == 1 and (j - i) < min_present_run:
            status[i:j] = 2
            conf[i:j] = np.minimum(conf[i:j], 0.46)
        i = j
    return status, conf.astype(np.float32)


def window_starts(w, width=256, stride=128):
    starts = list(range(0, max(1, w - width + 1), stride))
    last = max(0, w - width)
    if starts[-1] != last:
        starts.append(last)
    return starts


def save_windows(lines_dir, windows_dir, index_path):
    rows = []
    windows_dir.mkdir(parents=True, exist_ok=True)
    for line_path in sorted(lines_dir.glob("*.npz")):
        z = np.load(line_path)
        line = line_path.stem
        raw = z["raw_full_normalized"].astype(np.float32)
        mask = z["soft_mask_train"].astype(np.float32)
        status = z["status_code"].astype(np.int16)
        weight = z["label_weight"].astype(np.float32)
        split = str(z["split"])
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
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "line", "start", "end", "split", "present", "weak", "no_pick"])
        writer.writeheader()
        writer.writerows(rows)


def plot_preview(out, line, raw, old_mask, new_mask, old_c, new_c, lo, hi, dt_ns, rule, status):
    h, w = raw.shape
    t = np.arange(h) * float(dt_ns)
    x = np.arange(w)
    vmax = max(0.08, float(np.nanpercentile(np.abs(raw), 98)))
    fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    ax = ax.ravel()
    ax[0].imshow(raw, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=(0, w - 1, t[-1], t[0]))
    ax[0].axhline(lo * dt_ns, color="#2dd4bf", lw=1)
    ax[0].axhline(hi * dt_ns, color="#2dd4bf", lw=1)
    ax[0].set_title(f"{line} raw with allowed basal band")

    ax[1].imshow(raw, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=(0, w - 1, t[-1], t[0]))
    ax[1].imshow(old_mask, aspect="auto", cmap="magma", alpha=np.clip(old_mask * 0.9, 0, 0.65), extent=(0, w - 1, t[-1], t[0]))
    if np.isfinite(old_c).any():
        ax[1].plot(x, old_c * dt_ns, color="#fde047", lw=1.0)
    ax[1].set_title("old labels")

    ax[2].imshow(raw, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=(0, w - 1, t[-1], t[0]))
    ax[2].imshow(new_mask, aspect="auto", cmap="viridis", alpha=np.clip(new_mask * 0.9, 0, 0.65), extent=(0, w - 1, t[-1], t[0]))
    ax[2].plot(x, new_c * dt_ns, color="#38bdf8", lw=1.1)
    ax[2].set_title("corrected labels v1")

    ax[3].plot(x, status, lw=1)
    ax[3].set_ylim(-0.2, 2.2)
    ax[3].set_yticks([0, 1, 2], ["no", "present", "weak"])
    ax[3].set_title("corrected status")
    ax[3].set_xlabel("trace")

    for a in ax[:3]:
        a.set_xlabel("trace")
        a.set_ylabel("time ns")
    fig.suptitle(f"{line}: {rule['source']} | {rule['notes']}")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    velocity = 0.074
    sigma_ns = 8.0
    if OUT.exists():
        shutil.rmtree(OUT)
    if REPORT.exists():
        shutil.rmtree(REPORT)
    (OUT / "lines").mkdir(parents=True, exist_ok=True)
    (OUT / "windows").mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for p in sorted((SRC / "lines").glob("*.npz")):
        line = p.stem
        z = np.load(p)
        raw = z["raw_full_normalized"].astype(np.float32)
        old_mask = z["soft_mask_train"].astype(np.float32)
        dt_ns = float(z["dt_ns"])
        h, w = raw.shape
        rule = RULES[line]
        lo = max(0, depth_to_sample(rule["depth_min_m"], dt_ns, velocity) - 5)
        hi = min(h - 1, depth_to_sample(rule["depth_max_m"], dt_ns, velocity) + 5)
        score = build_score(raw, old_mask, lo, hi)
        path, strength = dp_pick(score, lo, hi)
        status, conf = status_from_strength(strength, w)
        sigma_samples = max(3, int(round(sigma_ns / dt_ns)))
        new_mask = make_mask(path, conf, h, sigma_samples)
        label_weight = np.where(status == 0, 0.0, conf).astype(np.float32)
        new_mask[:, status == 0] = 0.0

        old_c, old_v = centerline(old_mask)
        new_c = path.astype(np.float32)
        old_time = old_c * dt_ns
        new_time = new_c * dt_ns
        old_in_band = old_v & np.isfinite(old_c) & (old_c >= lo) & (old_c <= hi)
        summaries.append(
            {
                "line": line,
                "depth_min_m": rule["depth_min_m"],
                "depth_max_m": rule["depth_max_m"],
                "allowed_time_min_ns": lo * dt_ns,
                "allowed_time_max_ns": hi * dt_ns,
                "old_present": int((z["status_code"] == 1).sum()),
                "old_weak": int((z["status_code"] == 2).sum()),
                "old_no_pick": int((z["status_code"] == 0).sum()),
                "new_present": int((status == 1).sum()),
                "new_weak": int((status == 2).sum()),
                "new_no_pick": int((status == 0).sum()),
                "old_center_median_ns": float(np.nanmedian(old_time)) if np.isfinite(old_time).any() else float("nan"),
                "new_center_median_ns": float(np.nanmedian(new_time)),
                "old_center_in_allowed_band_ratio": float(old_in_band.mean()),
                "source": rule["source"],
                "notes": rule["notes"],
            }
        )

        np.savez_compressed(
            OUT / "lines" / p.name,
            raw_full_normalized=raw,
            soft_mask_train=new_mask.astype(np.float32),
            status_code=status.astype(np.int16),
            label_weight=label_weight.astype(np.float32),
            dt_ns=z["dt_ns"],
            trace_interval_m=z["trace_interval_m"],
            split=z["split"],
            line=z["line"],
            correction_rule=json.dumps(rule, ensure_ascii=False),
            correction_velocity_m_per_ns=np.float32(velocity),
            correction_sigma_ns=np.float32(sigma_ns),
        )
        plot_preview(REPORT / f"{line}_old_vs_corrected.png", line, raw, old_mask, new_mask, old_c, new_c, lo, hi, dt_ns, rule, status)

    save_windows(OUT / "lines", OUT / "windows", OUT / "window_index.csv")
    with open(REPORT / "line_label_correction_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    (REPORT / "rules.json").write_text(json.dumps(RULES, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(EXTERNAL / "YINGSHAN_DATA_REVIEW_INDEX.md", REPORT / "YINGSHAN_DATA_REVIEW_INDEX.md")
    print(f"Wrote {OUT}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
