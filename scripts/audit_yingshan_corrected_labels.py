from pathlib import Path
import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "data"
CORR = ROOT / "data_corrected_v1"
REPORT = ROOT / "reports" / "yingshan_label_correction_v1"
RULES_PATH = REPORT / "rules.json"


def depth_to_sample(depth_m, dt_ns, velocity_m_per_ns):
    return int(round((2.0 * float(depth_m) / velocity_m_per_ns) / float(dt_ns)))


def centerline(mask, min_sum=1e-4):
    h, _ = mask.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    s = mask.sum(axis=0)
    c = (mask * yy).sum(axis=0) / np.maximum(s, 1e-6)
    valid = s > min_sum
    c[~valid] = np.nan
    return c, valid


def moving_average_axis(a, win, axis):
    if win <= 1:
        return a
    pad = win // 2
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(a, pad_width, mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis, padded)


def gained_bscan(raw):
    """Raw-domain visual gain only; this is not used for training."""
    x = raw.astype(np.float32)
    x = x - np.nanmedian(x, axis=1, keepdims=True)
    base = float(np.nanpercentile(np.abs(x), 50))
    env = np.sqrt(moving_average_axis(x * x, 41, axis=0))
    agc = x / (env + max(base * 0.08, 1e-6))
    time_gain = np.linspace(0.75, 1.55, x.shape[0], dtype=np.float32)[:, None]
    y = moving_average_axis(agc * time_gain, 3, axis=1)
    vmax = float(np.nanpercentile(np.abs(y), 99.2))
    if not np.isfinite(vmax) or vmax <= 1e-6:
        vmax = 1.0
    return np.clip(y / vmax, -1.0, 1.0)


def run_lengths(values, target):
    out = []
    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        if values[i] == target:
            out.append(j - i)
        i = j
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_expected(line):
    if line == "Line9":
        return "test"
    if line == "Line6":
        return "review"
    if line == "Line7":
        return "val"
    if line == "LineX1":
        return "exclude"
    return "train"


def audit_line(line_path, rules, failures, warnings):
    line_name = line_path.stem
    z = np.load(line_path)
    orig = np.load(ORIG / "lines" / line_path.name)
    required = {"raw_full_normalized", "soft_mask_train", "status_code", "label_weight", "dt_ns", "trace_interval_m", "split", "line"}
    missing = required - set(z.files)
    if missing:
        failures.append(f"{line_name}: missing keys {sorted(missing)}")
    raw = z["raw_full_normalized"].astype(np.float32)
    mask = z["soft_mask_train"].astype(np.float32)
    status = z["status_code"].astype(np.int16)
    weight = z["label_weight"].astype(np.float32)
    old_mask = orig["soft_mask_train"].astype(np.float32)
    old_status = orig["status_code"].astype(np.int16)
    dt_ns = float(z["dt_ns"])
    velocity = float(z["correction_velocity_m_per_ns"]) if "correction_velocity_m_per_ns" in z.files else 0.074
    rule = rules[line_name]
    lo = max(0, depth_to_sample(rule["depth_min_m"], dt_ns, velocity) - 5)
    hi = min(raw.shape[0] - 1, depth_to_sample(rule["depth_max_m"], dt_ns, velocity) + 5)

    if raw.shape != orig["raw_full_normalized"].shape:
        failures.append(f"{line_name}: corrected raw shape differs from original")
    if not np.allclose(raw, orig["raw_full_normalized"]):
        failures.append(f"{line_name}: corrected raw differs from original")
    if mask.shape != raw.shape:
        failures.append(f"{line_name}: mask shape {mask.shape} does not match raw {raw.shape}")
    if status.shape != (raw.shape[1],):
        failures.append(f"{line_name}: status length does not match trace count")
    if weight.shape != status.shape:
        failures.append(f"{line_name}: label_weight length does not match status")
    if str(z["split"]) != split_expected(line_name):
        failures.append(f"{line_name}: split {str(z['split'])} != expected {split_expected(line_name)}")
    if not np.isfinite(raw).all() or not np.isfinite(mask).all() or not np.isfinite(weight).all():
        failures.append(f"{line_name}: non-finite data found")
    bad_codes = sorted(set(status.tolist()) - {0, 1, 2})
    if bad_codes:
        failures.append(f"{line_name}: bad status codes {bad_codes}")
    if mask.min() < -1e-6 or mask.max() > 1.0001:
        failures.append(f"{line_name}: mask outside [0, 1]")
    if weight.min() < -1e-6 or weight.max() > 1.0001:
        failures.append(f"{line_name}: label_weight outside [0, 1]")
    if np.any((status == 0) & ((mask.sum(axis=0) > 1e-4) | (weight > 1e-4))):
        failures.append(f"{line_name}: no-pick traces contain mask or weight")

    c, valid = centerline(mask)
    valid_centers = c[valid]
    tolerance_samples = 0.25
    peak = mask.argmax(axis=0)
    peak_in_band_ratio = float(((peak >= lo) & (peak <= hi)).mean())
    if valid_centers.size == 0:
        failures.append(f"{line_name}: no valid corrected centerline")
        center_in_band_ratio = 0.0
        max_center_oob_samples = float("nan")
        max_jump = float("nan")
        median_jump = float("nan")
    else:
        below = np.maximum((lo - tolerance_samples) - valid_centers, 0.0)
        above = np.maximum(valid_centers - (hi + tolerance_samples), 0.0)
        max_center_oob_samples = float(np.maximum(below, above).max())
        center_in_band_ratio = float(((valid_centers >= lo - tolerance_samples) & (valid_centers <= hi + tolerance_samples)).mean())
        jumps = np.abs(np.diff(valid_centers))
        max_jump = float(np.max(jumps)) if jumps.size else 0.0
        median_jump = float(np.median(jumps)) if jumps.size else 0.0
        if max_center_oob_samples > 0.5 or peak_in_band_ratio < 1.0:
            failures.append(f"{line_name}: corrected centerline leaves allowed band")
        if max_jump > 6.0:
            warnings.append(f"{line_name}: max centerline jump is {max_jump:.2f} samples")

    old_c, old_v = centerline(old_mask)
    old_centers = old_c[old_v]
    old_in_band_ratio = float(((old_centers >= lo) & (old_centers <= hi)).mean()) if old_centers.size else 0.0

    reject_energy_ratio = 0.0
    for reject in rule.get("reject_depth_ranges_m", []):
        rlo = max(0, depth_to_sample(reject[0], dt_ns, velocity) - 5)
        rhi = min(raw.shape[0] - 1, depth_to_sample(reject[1], dt_ns, velocity) + 5)
        total = float(mask.sum())
        reject_energy_ratio = max(reject_energy_ratio, float(mask[rlo : rhi + 1].sum()) / max(total, 1e-6))
    if reject_energy_ratio > 1e-4:
        failures.append(f"{line_name}: label energy leaks into reject band ({reject_energy_ratio:.6g})")

    present_runs = run_lengths(status, 1)
    min_present_run = min(present_runs) if present_runs else 0
    if present_runs and min_present_run < 24:
        failures.append(f"{line_name}: present status run shorter than 24 traces")
    weak_ratio = float((status == 2).mean())
    if weak_ratio > 0.85:
        warnings.append(f"{line_name}: weak-label ratio is high ({weak_ratio:.3f}); review visually")

    return {
        "line": line_name,
        "trace_count": int(raw.shape[1]),
        "sample_count": int(raw.shape[0]),
        "dt_ns": dt_ns,
        "velocity_m_per_ns": velocity,
        "allowed_sample_min": int(lo),
        "allowed_sample_max": int(hi),
        "allowed_time_min_ns": float(lo * dt_ns),
        "allowed_time_max_ns": float(hi * dt_ns),
        "center_in_allowed_band_ratio": center_in_band_ratio,
        "peak_in_allowed_band_ratio": peak_in_band_ratio,
        "max_center_oob_samples": max_center_oob_samples,
        "old_center_in_allowed_band_ratio": old_in_band_ratio,
        "center_median_sample": float(np.nanmedian(c)),
        "center_median_time_ns": float(np.nanmedian(c) * dt_ns),
        "center_max_jump_samples": max_jump,
        "center_median_jump_samples": median_jump,
        "present": int((status == 1).sum()),
        "weak": int((status == 2).sum()),
        "no_pick": int((status == 0).sum()),
        "old_present": int((old_status == 1).sum()),
        "old_weak": int((old_status == 2).sum()),
        "old_no_pick": int((old_status == 0).sum()),
        "min_present_run": int(min_present_run),
        "weak_ratio": weak_ratio,
        "reject_band_energy_ratio": reject_energy_ratio,
        "mask_sum": float(mask.sum()),
    }


def audit_windows(metrics_by_line, failures):
    rows = list(csv.DictReader(open(CORR / "window_index.csv", encoding="utf-8")))
    seen = set()
    for r in rows:
        sample_id = r["sample_id"]
        line = r["line"]
        start = int(r["start"])
        end = int(r["end"])
        seen.add(sample_id)
        line_z = np.load(CORR / "lines" / f"{line}.npz")
        win_path = CORR / "windows" / f"{sample_id}.npz"
        if not win_path.exists():
            failures.append(f"{sample_id}: missing window file")
            continue
        w = np.load(win_path)
        sl = slice(start, end + 1)
        checks = [
            ("x_raw", w["x_raw"], line_z["raw_full_normalized"][:, sl]),
            ("y_mask", w["y_mask"], line_z["soft_mask_train"][:, sl]),
            ("status_code", w["status_code"], line_z["status_code"][sl]),
            ("label_weight", w["label_weight"], line_z["label_weight"][sl]),
        ]
        for name, got, expected in checks:
            if got.shape != expected.shape or not np.allclose(got, expected):
                failures.append(f"{sample_id}: {name} does not match line slice")
        st = w["status_code"].astype(np.int16)
        if int(r["present"]) != int((st == 1).sum()) or int(r["weak"]) != int((st == 2).sum()) or int(r["no_pick"]) != int((st == 0).sum()):
            failures.append(f"{sample_id}: index status counts do not match window")
        if r["split"] != split_expected(line):
            failures.append(f"{sample_id}: split {r['split']} != expected {split_expected(line)}")
    actual = {p.stem for p in (CORR / "windows").glob("*.npz")}
    extra = actual - seen
    missing_from_files = seen - actual
    if extra:
        failures.append(f"extra window files not in index: {sorted(extra)[:5]}")
    if missing_from_files:
        failures.append(f"index rows missing files: {sorted(missing_from_files)[:5]}")
    for line, metrics in metrics_by_line.items():
        indexed = [r for r in rows if r["line"] == line]
        if not indexed:
            failures.append(f"{line}: no indexed windows")
        if sum(int(r["present"]) for r in indexed) <= 0 and metrics["present"] > 0:
            failures.append(f"{line}: window index lost present labels")
    return rows


def write_manifest(paths):
    out = REPORT / "deep_audit_manifest_sha256.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        for p in sorted(paths):
            writer.writerow({"path": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return out


def plot_gain_preview(line, rules):
    old = np.load(ORIG / "lines" / f"{line}.npz")
    new = np.load(CORR / "lines" / f"{line}.npz")
    raw = new["raw_full_normalized"].astype(np.float32)
    old_mask = old["soft_mask_train"].astype(np.float32)
    new_mask = new["soft_mask_train"].astype(np.float32)
    status = new["status_code"].astype(np.int16)
    dt_ns = float(new["dt_ns"])
    velocity = float(new["correction_velocity_m_per_ns"]) if "correction_velocity_m_per_ns" in new.files else 0.074
    rule = rules[line]
    lo = max(0, depth_to_sample(rule["depth_min_m"], dt_ns, velocity) - 5)
    hi = min(raw.shape[0] - 1, depth_to_sample(rule["depth_max_m"], dt_ns, velocity) + 5)
    old_c, old_v = centerline(old_mask)
    new_c, _ = centerline(new_mask)
    gain = gained_bscan(raw)
    h, w = raw.shape
    t = np.arange(h, dtype=np.float32) * dt_ns
    x = np.arange(w)
    vmax = max(0.08, float(np.nanpercentile(np.abs(raw), 98)))

    fig, ax = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    ax = ax.ravel()
    extent = (0, w - 1, t[-1], t[0])
    ax[0].imshow(raw, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=extent)
    ax[0].axhline(lo * dt_ns, color="#2dd4bf", lw=1)
    ax[0].axhline(hi * dt_ns, color="#2dd4bf", lw=1)
    ax[0].set_title("raw B-scan, clipped")

    ax[1].imshow(gain, aspect="auto", cmap="gray", vmin=-1, vmax=1, extent=extent)
    ax[1].axhline(lo * dt_ns, color="#2dd4bf", lw=1)
    ax[1].axhline(hi * dt_ns, color="#2dd4bf", lw=1)
    ax[1].set_title("raw B-scan with AGC/time gain")

    ax[2].imshow(gain, aspect="auto", cmap="gray", vmin=-1, vmax=1, extent=extent)
    ax[2].imshow(new_mask, aspect="auto", cmap="viridis", alpha=np.clip(new_mask * 0.85, 0, 0.65), extent=extent)
    ax[2].plot(x, new_c * dt_ns, color="#38bdf8", lw=1.0)
    ax[2].set_title("gain view + corrected label")

    ax[3].imshow(raw, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=extent)
    ax[3].imshow(old_mask, aspect="auto", cmap="magma", alpha=np.clip(old_mask * 0.9, 0, 0.65), extent=extent)
    if np.isfinite(old_c).any():
        ax[3].plot(x[old_v], old_c[old_v] * dt_ns, color="#fde047", lw=0.9)
    ax[3].set_title("old labels")

    ax[4].imshow(raw, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=extent)
    ax[4].imshow(new_mask, aspect="auto", cmap="viridis", alpha=np.clip(new_mask * 0.9, 0, 0.65), extent=extent)
    ax[4].plot(x, new_c * dt_ns, color="#38bdf8", lw=1.0)
    ax[4].set_title("corrected labels v1")

    ax[5].plot(x, status, lw=1.0)
    ax[5].set_ylim(-0.2, 2.2)
    ax[5].set_yticks([0, 1, 2], ["no", "present", "weak"])
    ax[5].set_xlabel("trace")
    ax[5].set_title("corrected status")

    for a in ax[:5]:
        a.set_xlabel("trace")
        a.set_ylabel("time ns")
    fig.suptitle(f"{line}: allowed {rule['depth_min_m']}-{rule['depth_max_m']} m | {rule['notes']}")
    out = REPORT / f"{line}_deep_audit_gain_preview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_report(metrics, failures, warnings, manifest_path, preview_paths):
    out = REPORT / "DEEP_AUDIT_REPORT.md"
    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Deep Audit Report",
        "",
        f"Status: **{status}**",
        "",
        "## Scope",
        "",
        "- Dataset: `data_corrected_v1`",
        "- Reference dataset: `data`",
        "- Checks: label depth constraints, reject-band leakage, window/line consistency, split policy, finite values, status continuity, and file hashes.",
        "- Preview addition: each line now has a raw B-scan with AGC/time gain for visual inspection.",
        "",
        "## Summary Metrics",
        "",
        "| Line | Center In Band | Old Center In Band | Present | Weak | No Pick | Max Jump | Reject Energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['line']} | {m['center_in_allowed_band_ratio']:.3f} | {m['old_center_in_allowed_band_ratio']:.3f} | "
            f"{m['present']} | {m['weak']} | {m['no_pick']} | {m['center_max_jump_samples']:.2f} | {m['reject_band_energy_ratio']:.6g} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- Metrics CSV: `reports/yingshan_label_correction_v1/deep_audit_metrics.csv`",
        f"- Hash manifest: `{manifest_path.relative_to(ROOT).as_posix()}`",
        "- Gain previews:",
    ]
    for p in preview_paths:
        lines.append(f"  - `{p.relative_to(ROOT).as_posix()}`")
    lines += ["", "## Failures", ""]
    if failures:
        lines += [f"- {x}" for x in failures]
    else:
        lines.append("- None")
    lines += ["", "## Warnings", ""]
    if warnings:
        lines += [f"- {x}" for x in warnings]
    else:
        lines.append("- None")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    failures = []
    warnings = []
    metrics = []
    for p in sorted((CORR / "lines").glob("*.npz")):
        metrics.append(audit_line(p, rules, failures, warnings))
    metrics_by_line = {m["line"]: m for m in metrics}
    audit_windows(metrics_by_line, failures)

    metrics_path = REPORT / "deep_audit_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    preview_paths = [plot_gain_preview(m["line"], rules) for m in metrics]
    manifest_inputs = list((CORR / "lines").glob("*.npz")) + list((CORR / "windows").glob("*.npz")) + [
        CORR / "window_index.csv",
        metrics_path,
    ]
    manifest_path = write_manifest(manifest_inputs)
    report_path = write_report(metrics, failures, warnings, manifest_path, preview_paths)
    print(f"DEEP_AUDIT_STATUS={'PASS' if not failures else 'FAIL'}")
    print(metrics_path)
    print(manifest_path)
    print(report_path)
    for p in preview_paths:
        print(p)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
