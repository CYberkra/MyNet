from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "PGDA_SYNTH_DATASET_V1"
OUT = Path(__file__).resolve().parent
PREVIEWS = OUT / "previews"
TIME_MAX_NS = 700.0
DISPLAY_SAMPLES = 701
SEARCH_HALF_NS = 35.0
LABEL_HALF_NS = 4.0


@dataclass(frozen=True)
class CaseCopy:
    case_id: str
    source_group: str
    case_dir: Path
    raw_path: Path
    label_dir: Path
    raw_sha256: str
    label_sha256: str

    @property
    def pair_sha256(self) -> str:
        return f"{self.raw_sha256}:{self.label_sha256}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group(path: Path) -> str:
    text = str(path).lower()
    if "05_accepted_dataset" in text:
        return "accepted"
    if "batch_001" in text:
        return "batch1"
    if "batch_003" in text:
        return "batch3"
    return "other"


def discover_copies() -> list[CaseCopy]:
    copies: list[CaseCopy] = []
    raw_paths = list(DATA_ROOT.rglob("bscan.npy"))
    raw_paths.extend(DATA_ROOT.rglob("raw_bscan.npy"))
    for raw_path in sorted(set(raw_paths)):
        case_dir = raw_path.parent.parent
        label_dir = case_dir / ("label" if raw_path.name == "raw_bscan.npy" else "labels")
        label_path = label_dir / "target_visible_phase_time_ns.npy"
        if not label_path.exists():
            continue
        copies.append(
            CaseCopy(
                case_id=case_dir.name,
                source_group=source_group(case_dir),
                case_dir=case_dir,
                raw_path=raw_path,
                label_dir=label_dir,
                raw_sha256=file_sha256(raw_path),
                label_sha256=file_sha256(label_path),
            )
        )
    return copies


def choose_canonical(copies: list[CaseCopy]) -> CaseCopy:
    priority = {"batch1": 0, "batch3": 1, "accepted": 2, "other": 3}
    return sorted(copies, key=lambda item: (priority[item.source_group], str(item.case_dir)))[0]


def load_previous_rows() -> dict[str, dict[str, str]]:
    path = ROOT / "reports" / "AUDIT_20260710" / "SIM_CASE_RECOMMENDATIONS.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


def load_contract_rows() -> dict[str, dict[str, str]]:
    path = ROOT / "data" / "dataset_contract_v2" / "simulation_cases.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


def load_visual_decisions() -> dict[str, dict[str, str]]:
    path = OUT / "SIMULATION_VISUAL_DECISIONS.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


def resample_time(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_time = np.linspace(0.0, TIME_MAX_NS, DISPLAY_SAMPLES)
    source_time = np.linspace(0.0, TIME_MAX_NS, raw.shape[0])
    result = np.empty((DISPLAY_SAMPLES, raw.shape[1]), dtype=np.float32)
    for trace in range(raw.shape[1]):
        result[:, trace] = np.interp(target_time, source_time, raw[:, trace])
    return result, target_time


def moving_rms(values: np.ndarray, window: int = 17) -> np.ndarray:
    pad = window // 2
    squared = np.pad(values.astype(np.float64) ** 2, ((pad, pad), (0, 0)), mode="edge")
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(squared, axis=0)])
    mean_square = (cumulative[window:] - cumulative[:-window]) / float(window)
    return np.sqrt(np.maximum(mean_square, 1e-12)).astype(np.float32)


def analytic_envelope(values: np.ndarray) -> np.ndarray:
    size = values.shape[0]
    spectrum = np.fft.fft(values, axis=0)
    multiplier = np.zeros(size, dtype=np.float64)
    multiplier[0] = 1.0
    if size % 2 == 0:
        multiplier[1 : size // 2] = 2.0
        multiplier[size // 2] = 1.0
    else:
        multiplier[1 : (size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier[:, None], axis=0)).astype(np.float32)


def sample_curve(values: np.ndarray, curve_ns: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
    result = np.empty(curve_ns.size, dtype=np.float32)
    for trace, value in enumerate(curve_ns):
        index = int(np.argmin(np.abs(time_ns - value)))
        result[trace] = values[index, trace]
    return result


def compute_metrics(copy: CaseCopy) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    raw_native = np.load(copy.raw_path).astype(np.float32)
    label = np.load(copy.label_dir / "target_visible_phase_time_ns.npy").astype(np.float64)
    geom_path = copy.label_dir / "target_geom_time_ns.npy"
    geom = np.load(geom_path).astype(np.float64) if geom_path.exists() else label.copy()
    soft_path = copy.label_dir / "y_soft_501x128.npy"
    soft = np.load(soft_path).astype(np.float64)

    raw, time_ns = resample_time(raw_native)
    residual = raw - np.median(raw, axis=1, keepdims=True)
    gain = (0.025 + (time_ns / TIME_MAX_NS) ** 2.2)[:, None]
    gained = residual * gain
    agc = gained / moving_rms(gained)
    envelope = analytic_envelope(gained)

    ratios: list[float] = []
    signed_ratios: list[float] = []
    offsets: list[float] = []
    contrasts: list[float] = []
    peak_curve = np.empty(label.size, dtype=np.float64)
    for trace, expected in enumerate(label):
        search = np.flatnonzero(np.abs(time_ns - expected) <= SEARCH_HALF_NS)
        narrow = np.flatnonzero(np.abs(time_ns - expected) <= LABEL_HALF_NS)
        if search.size == 0 or narrow.size == 0:
            peak_curve[trace] = np.nan
            continue
        search_env = envelope[search, trace]
        peak_index = search[int(np.argmax(search_env))]
        peak_curve[trace] = time_ns[peak_index]
        label_strength = float(np.max(envelope[narrow, trace]))
        nearby_strength = max(float(np.max(search_env)), 1e-12)
        ratios.append(label_strength / nearby_strength)
        offsets.append(float(time_ns[peak_index] - expected))

        label_signed = float(np.max(np.abs(gained[narrow, trace])))
        nearby_signed = max(float(np.max(np.abs(gained[search, trace]))), 1e-12)
        signed_ratios.append(label_signed / nearby_signed)

        outside = search[np.abs(time_ns[search] - expected) >= 12.0]
        background = float(np.median(envelope[outside, trace])) if outside.size else 0.0
        contrasts.append(label_strength / max(background, 1e-12))

    ratio_array = np.asarray(ratios, dtype=np.float64)
    signed_ratio_array = np.asarray(signed_ratios, dtype=np.float64)
    offset_array = np.asarray(offsets, dtype=np.float64)
    contrast_array = np.asarray(contrasts, dtype=np.float64)

    finite_ok = bool(np.isfinite(raw_native).all() and np.isfinite(label).all())
    label_range_ok = bool((label >= 0.0).all() and (label <= TIME_MAX_NS).all())
    median_ratio = float(np.median(ratio_array))
    median_signed_ratio = float(np.median(signed_ratio_array))
    visible_fraction = float(np.mean(ratio_array >= 0.60))
    strong_fraction = float(np.mean(ratio_array >= 0.80))
    median_abs_offset = float(np.median(np.abs(offset_array)))
    p90_abs_offset = float(np.percentile(np.abs(offset_array), 90.0))
    median_contrast = float(np.median(contrast_array))

    soft_time = np.linspace(0.0, TIME_MAX_NS, soft.shape[0], dtype=np.float64)
    soft_mass = np.sum(soft, axis=0)
    soft_center = np.sum(soft * soft_time[:, None], axis=0) / np.maximum(soft_mass, 1e-12)
    soft_center_offset = soft_center - label
    soft_center_offset_median = float(np.median(soft_center_offset))
    soft_center_offset_p90_abs = float(np.percentile(np.abs(soft_center_offset), 90.0))
    curve_training_contract_ok = bool(soft_center_offset_p90_abs <= 1.5)
    if curve_training_contract_ok:
        soft_target_semantics = "VISIBLE_PHASE_CENTERED"
    elif -7.5 <= soft_center_offset_median <= -4.5:
        soft_target_semantics = "GEOMETRY_TO_VISIBLE_BAND_OR_SHIFTED"
    else:
        soft_target_semantics = "INCONSISTENT_WITH_VISIBLE_PHASE"

    visible_mask_path = copy.label_dir / "interface_mask_visible_phase_bscan.npy"
    if not visible_mask_path.exists():
        visible_mask_path = copy.label_dir / "interface_mask_bscan.npy"
    visible_mask = np.load(visible_mask_path).astype(np.float64)
    visible_mask_time = np.linspace(0.0, TIME_MAX_NS, visible_mask.shape[0], dtype=np.float64)
    visible_mask_mass = np.sum(visible_mask, axis=0)
    visible_mask_center = (
        np.sum(visible_mask * visible_mask_time[:, None], axis=0)
        / np.maximum(visible_mask_mass, 1e-12)
    )
    visible_mask_offset = visible_mask_center - label

    if (
        visible_fraction >= 0.75
        and median_ratio >= 0.78
        and median_abs_offset <= 6.0
        and median_contrast >= 1.35
    ):
        signal_grade = "SUPPORTED"
    elif (
        visible_fraction >= 0.45
        and median_ratio >= 0.58
        and median_abs_offset <= 15.0
        and median_contrast >= 1.10
    ):
        signal_grade = "REVIEW"
    else:
        signal_grade = "UNSUPPORTED"

    metrics: dict[str, object] = {
        "case_id": copy.case_id,
        "canonical_source_group": copy.source_group,
        "raw_shape": f"{raw_native.shape[0]}x{raw_native.shape[1]}",
        "finite_ok": finite_ok,
        "label_range_ok": label_range_ok,
        "label_min_ns": float(np.min(label)),
        "label_max_ns": float(np.max(label)),
        "visible_fraction_ge_0_60": visible_fraction,
        "strong_fraction_ge_0_80": strong_fraction,
        "median_envelope_support": median_ratio,
        "median_signed_support": median_signed_ratio,
        "median_abs_peak_offset_ns": median_abs_offset,
        "p90_abs_peak_offset_ns": p90_abs_offset,
        "median_local_contrast": median_contrast,
        "soft_center_offset_median_ns": soft_center_offset_median,
        "soft_center_offset_p90_abs_ns": soft_center_offset_p90_abs,
        "soft_target_semantics": soft_target_semantics,
        "curve_training_contract_ok": curve_training_contract_ok,
        "visible_mask_center_offset_median_ns": float(np.median(visible_mask_offset)),
        "visible_mask_center_offset_p90_abs_ns": float(np.percentile(np.abs(visible_mask_offset), 90.0)),
        "automatic_signal_grade": signal_grade,
        "raw_sha256": copy.raw_sha256,
        "label_sha256": copy.label_sha256,
        "canonical_case_path": str(copy.case_dir.relative_to(ROOT)),
    }
    arrays = {
        "time_ns": time_ns,
        "raw": raw,
        "gained": gained,
        "agc": agc,
        "envelope": envelope,
        "label": label,
        "geom": geom,
        "peak_curve": peak_curve,
    }
    return metrics, arrays


def image_limit(values: np.ndarray, percentile: float = 99.2) -> float:
    return max(float(np.percentile(np.abs(values), percentile)), 1e-9)


def render_case(copy: CaseCopy, metrics: dict[str, object], arrays: dict[str, np.ndarray]) -> None:
    time_ns = arrays["time_ns"]
    gained = arrays["gained"]
    agc = arrays["agc"]
    envelope = arrays["envelope"]
    label = arrays["label"]
    geom = arrays["geom"]
    peak_curve = arrays["peak_curve"]
    traces = np.arange(label.size)
    zoom_lo = max(0.0, float(np.min(label)) - 70.0)
    zoom_hi = min(TIME_MAX_NS, float(np.max(label)) + 70.0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    extent = [0, label.size - 1, TIME_MAX_NS, 0]
    limit = image_limit(gained)
    axes[0, 0].imshow(gained, cmap="gray", aspect="auto", extent=extent, vmin=-limit, vmax=limit)
    axes[0, 0].plot(traces, geom, color="#ffb300", linewidth=1.0, linestyle="--", label="geometry")
    axes[0, 0].plot(traces, label, color="#e53935", linewidth=1.6, label="visible label")
    axes[0, 0].set_title("Background removed + time gain (full 0-700 ns)")
    axes[0, 0].legend(loc="lower right", fontsize=8)

    agc_limit = image_limit(agc, 99.0)
    axes[0, 1].imshow(agc, cmap="gray", aspect="auto", extent=extent, vmin=-agc_limit, vmax=agc_limit)
    axes[0, 1].plot(traces, label, color="#e53935", linewidth=1.7, label="visible label")
    axes[0, 1].plot(traces, peak_curve, color="#00acc1", linewidth=1.0, linestyle=":", label="nearest envelope peak")
    axes[0, 1].set_ylim(zoom_hi, zoom_lo)
    axes[0, 1].set_title(f"AGC target zoom ({zoom_lo:.0f}-{zoom_hi:.0f} ns)")
    axes[0, 1].legend(loc="lower right", fontsize=8)

    env_limit = max(float(np.percentile(envelope, 99.5)), 1e-9)
    axes[1, 0].imshow(
        envelope,
        cmap="magma",
        aspect="auto",
        extent=extent,
        vmin=0.0,
        vmax=env_limit,
    )
    axes[1, 0].plot(traces, label, color="#29b6f6", linewidth=1.7, label="visible label")
    axes[1, 0].plot(traces, peak_curve, color="#76ff03", linewidth=1.0, linestyle=":", label="nearest envelope peak")
    axes[1, 0].set_ylim(zoom_hi, zoom_lo)
    axes[1, 0].set_title("Envelope target zoom")
    axes[1, 0].legend(loc="lower right", fontsize=8)

    axes[1, 1].axis("off")
    summary = [
        f"Case: {copy.case_id}",
        f"Source: {copy.source_group}",
        f"Automatic signal grade: {metrics['automatic_signal_grade']}",
        "",
        f"Visible fraction (support >= 0.60): {float(metrics['visible_fraction_ge_0_60']):.1%}",
        f"Strong fraction (support >= 0.80): {float(metrics['strong_fraction_ge_0_80']):.1%}",
        f"Median envelope support: {float(metrics['median_envelope_support']):.3f}",
        f"Median signed support: {float(metrics['median_signed_support']):.3f}",
        f"Median abs peak offset: {float(metrics['median_abs_peak_offset_ns']):.2f} ns",
        f"P90 abs peak offset: {float(metrics['p90_abs_peak_offset_ns']):.2f} ns",
        f"Median local contrast: {float(metrics['median_local_contrast']):.2f}x",
        "",
        "Red/cyan solid: current visible-phase label",
        "Cyan/green dotted: strongest envelope peak within +/-35 ns",
        "Automatic grade is a triage aid, not a human ground-truth decision.",
    ]
    axes[1, 1].text(0.03, 0.97, "\n".join(summary), va="top", family="monospace", fontsize=12)

    for axis in axes.ravel()[:3]:
        axis.set_xlabel("trace")
        axis.set_ylabel("time (ns)")
    fig.suptitle(copy.case_id, fontsize=16)
    fig.savefig(PREVIEWS / f"{copy.case_id}.png", dpi=160)
    plt.close(fig)


def render_contact_sheet(
    name: str,
    entries: list[tuple[CaseCopy, dict[str, object], dict[str, np.ndarray]]],
    columns: int = 4,
) -> None:
    rows = (len(entries) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 3.7 * rows), squeeze=False, constrained_layout=True)
    color = {
        "VISUAL_PASS": "#2e7d32",
        "VISUAL_PASS_LOW_CONTRAST": "#ef6c00",
        "VISUAL_REVIEW_LOCAL_ARTIFACT": "#c62828",
        "SUPPORTED": "#2e7d32",
        "REVIEW": "#f9a825",
        "UNSUPPORTED": "#c62828",
    }
    for axis, (copy, metrics, arrays) in zip(axes.ravel(), entries):
        label = arrays["label"]
        agc = arrays["agc"]
        peak_curve = arrays["peak_curve"]
        extent = [0, label.size - 1, TIME_MAX_NS, 0]
        zoom_lo = max(0.0, float(np.min(label)) - 65.0)
        zoom_hi = min(TIME_MAX_NS, float(np.max(label)) + 65.0)
        limit = image_limit(agc, 99.0)
        axis.imshow(agc, cmap="gray", aspect="auto", extent=extent, vmin=-limit, vmax=limit)
        axis.plot(np.arange(label.size), label, color="#e53935", linewidth=1.5)
        axis.plot(np.arange(label.size), peak_curve, color="#00bcd4", linewidth=0.9, linestyle=":")
        axis.set_ylim(zoom_hi, zoom_lo)
        grade = str(metrics.get("visual_decision") or metrics["automatic_signal_grade"])
        if grade == "PENDING":
            grade = str(metrics["automatic_signal_grade"])
        axis.set_title(
            f"{copy.case_id}\n{grade} | offset={float(metrics['median_abs_peak_offset_ns']):.1f}ns",
            fontsize=9,
            color=color.get(grade, "#333333"),
        )
        axis.set_xlabel("trace", fontsize=8)
        axis.set_ylabel("time (ns)", fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in axes.ravel()[len(entries) :]:
        axis.axis("off")
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def correlation(value_a: np.ndarray, value_b: np.ndarray) -> float:
    flat_a = value_a.astype(np.float64, copy=False).ravel()
    flat_b = value_b.astype(np.float64, copy=False).ravel()
    flat_a = flat_a - np.mean(flat_a)
    flat_b = flat_b - np.mean(flat_b)
    denominator = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(flat_a, flat_b) / denominator)


def aligned_curve_patch(arrays: dict[str, np.ndarray]) -> np.ndarray:
    offsets = np.linspace(-30.0, 30.0, 61)
    time_ns = arrays["time_ns"]
    label = arrays["label"]
    agc = arrays["agc"]
    patch = np.empty((offsets.size, label.size), dtype=np.float32)
    for trace, center in enumerate(label):
        patch[:, trace] = np.interp(center + offsets, time_ns, agc[:, trace])
    return patch


def diversity_rows(
    entries: list[tuple[CaseCopy, dict[str, object], dict[str, np.ndarray]]]
) -> list[dict[str, object]]:
    families = {
        "batch1_line9_style": [entry for entry in entries if entry[0].source_group == "batch1"],
        "batch3_meddepth": [entry for entry in entries if entry[0].case_id.startswith("B003_MEDDEPTH")],
        "batch3_shallow_distractor": [
            entry for entry in entries if entry[0].case_id.startswith("B003_SHALLOW_DISTRACTOR")
        ],
    }
    rows: list[dict[str, object]] = []
    for family, members in families.items():
        full_raw_correlations: list[float] = []
        aligned_patch_correlations: list[float] = []
        label_correlations: list[float] = []
        for index, (_, _, arrays_a) in enumerate(members):
            for _, _, arrays_b in members[index + 1 :]:
                full_raw_correlations.append(correlation(arrays_a["raw"][::4, ::2], arrays_b["raw"][::4, ::2]))
                aligned_patch_correlations.append(
                    correlation(aligned_curve_patch(arrays_a), aligned_curve_patch(arrays_b))
                )
                label_correlations.append(correlation(arrays_a["label"], arrays_b["label"]))
        rows.append(
            {
                "family": family,
                "case_count": len(members),
                "pair_count": len(full_raw_correlations),
                "full_raw_correlation_median": float(np.median(full_raw_correlations)),
                "full_raw_correlation_min": float(np.min(full_raw_correlations)),
                "full_raw_correlation_max": float(np.max(full_raw_correlations)),
                "aligned_target_patch_correlation_median": float(np.median(aligned_patch_correlations)),
                "aligned_target_patch_correlation_min": float(np.min(aligned_patch_correlations)),
                "aligned_target_patch_correlation_max": float(np.max(aligned_patch_correlations)),
                "label_curve_correlation_median": float(np.median(label_correlations)),
                "label_curve_correlation_min": float(np.min(label_correlations)),
                "label_curve_correlation_max": float(np.max(label_correlations)),
            }
        )
    return rows


def metadata_summary(copies: list[CaseCopy]) -> dict[str, object]:
    accepted = [copy for copy in copies if copy.source_group == "accepted"]
    run_copies = [copy for copy in copies if copy.source_group in {"batch1", "batch3"}]
    scene_paths = [copy.case_dir / "metadata" / "scene_world.json" for copy in accepted]
    design_paths = [copy.case_dir / "metadata" / "design_metrics.csv" for copy in accepted]
    return {
        "run_case_copies": len(run_copies),
        "run_cases_with_scene_metadata": sum((copy.case_dir / "scene_world.json").exists() for copy in run_copies),
        "run_cases_with_raw_input": sum(any(copy.case_dir.rglob("*.in")) for copy in run_copies),
        "accepted_case_copies": len(accepted),
        "accepted_scene_metadata_files": sum(path.exists() for path in scene_paths),
        "accepted_unique_scene_metadata_hashes": len({file_sha256(path) for path in scene_paths if path.exists()}),
        "accepted_design_metadata_files": sum(path.exists() for path in design_paths),
        "accepted_unique_design_metadata_hashes": len({file_sha256(path) for path in design_paths if path.exists()}),
        "paired_component_cases": sum(
            any(
                token in path.name.lower()
                for token in ("background_only", "basal_only", "target_only", "air_only")
                for path in copy.case_dir.rglob("*.npy")
            )
            for copy in copies
        ),
    }


def file_integrity_summary() -> dict[str, int]:
    npy_paths = list(DATA_ROOT.rglob("*.npy"))
    npy_errors = 0
    nan_files = 0
    inf_files = 0
    for path in npy_paths:
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            if array.dtype.kind in "fc":
                nan_files += int(np.isnan(array).any())
                inf_files += int(np.isinf(array).any())
        except Exception:
            npy_errors += 1

    json_paths = list(DATA_ROOT.rglob("*.json"))
    json_errors = 0
    for path in json_paths:
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            json_errors += 1

    csv_paths = list(DATA_ROOT.rglob("*.csv"))
    csv_errors = 0
    for path in csv_paths:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                list(csv.reader(stream))
        except Exception:
            csv_errors += 1
    return {
        "npy_count": len(npy_paths),
        "npy_load_errors": npy_errors,
        "npy_nan_files": nan_files,
        "npy_inf_files": inf_files,
        "json_count": len(json_paths),
        "json_parse_errors": json_errors,
        "csv_count": len(csv_paths),
        "csv_parse_errors": csv_errors,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    copies = discover_copies()
    grouped: dict[str, list[CaseCopy]] = {}
    for copy in copies:
        grouped.setdefault(copy.pair_sha256, []).append(copy)

    previous = load_previous_rows()
    contract = load_contract_rows()
    visual_decisions = load_visual_decisions()
    audit_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    entries: list[tuple[CaseCopy, dict[str, object], dict[str, np.ndarray]]] = []

    for pair_hash, group in sorted(grouped.items(), key=lambda item: choose_canonical(item[1]).case_id):
        canonical = choose_canonical(group)
        metrics, arrays = compute_metrics(canonical)
        prior = previous.get(canonical.case_id, {})
        contract_row = contract.get(canonical.case_id, {})
        visual_row = visual_decisions.get(canonical.case_id, {})
        duplicate_paths = " | ".join(str(item.case_dir.relative_to(ROOT)) for item in sorted(group, key=lambda value: str(value.case_dir)))
        metrics.update(
            {
                "copy_count": len(group),
                "duplicate_locations": duplicate_paths,
                "previous_automatic_qc_grade": prior.get("automatic_qc_grade", "UNLISTED"),
                "previous_audit_recommendation": prior.get("audit_recommendation", "UNLISTED"),
                "line9_conditioned": contract_row.get("line9_conditioned", "true"),
                "train_allowed": contract_row.get("train_allowed", "false"),
                "visual_decision": visual_row.get("visual_decision", "PENDING"),
                "visual_note": visual_row.get("visual_note", ""),
                "local_review_trace_range": visual_row.get("local_review_trace_range", ""),
                "development_use": visual_row.get("development_use", ""),
                "formal_training_decision": visual_row.get("formal_training_decision", ""),
            }
        )
        audit_rows.append(metrics)
        entries.append((canonical, metrics, arrays))
        render_case(canonical, metrics, arrays)
        for item in group:
            duplicate_rows.append(
                {
                    "case_id": item.case_id,
                    "source_group": item.source_group,
                    "case_path": str(item.case_dir.relative_to(ROOT)),
                    "canonical_case_id": canonical.case_id,
                    "canonical_case_path": str(canonical.case_dir.relative_to(ROOT)),
                    "is_canonical": item.case_dir == canonical.case_dir,
                    "raw_sha256": item.raw_sha256,
                    "label_sha256": item.label_sha256,
                    "pair_sha256": pair_hash,
                }
            )

    write_csv(OUT / "SIMULATION_CASE_AUDIT_AUTO.csv", audit_rows)
    write_csv(OUT / "SIMULATION_DUPLICATE_MAP.csv", duplicate_rows)

    batch1 = [entry for entry in entries if entry[0].source_group == "batch1"]
    batch3 = [entry for entry in entries if entry[0].source_group == "batch3"]
    accepted_unique = [entry for entry in entries if entry[0].source_group == "accepted"]
    render_contact_sheet("BATCH1_UNIQUE_LABEL_ZOOM.png", batch1)
    render_contact_sheet("BATCH3_UNIQUE_LABEL_ZOOM.png", batch3)
    render_contact_sheet("ACCEPTED_ONLY_UNIQUE_LABEL_ZOOM.png", accepted_unique, columns=1)
    family_diversity = diversity_rows(entries)
    write_csv(OUT / "SIMULATION_FAMILY_DIVERSITY.csv", family_diversity)

    grade_counts = {
        grade: sum(row["automatic_signal_grade"] == grade for row in audit_rows)
        for grade in ("SUPPORTED", "REVIEW", "UNSUPPORTED")
    }
    visual_counts = {
        decision: sum(row["visual_decision"] == decision for row in audit_rows)
        for decision in (
            "VISUAL_PASS",
            "VISUAL_PASS_LOW_CONTRAST",
            "VISUAL_REVIEW_LOCAL_ARTIFACT",
            "PENDING",
        )
    }
    summary = {
        "physical_case_copies": len(copies),
        "unique_raw_label_pairs": len(grouped),
        "exact_duplicate_copies": len(copies) - len(grouped),
        "source_copy_counts": {
            group: sum(copy.source_group == group for copy in copies)
            for group in ("batch1", "batch3", "accepted", "other")
        },
        "automatic_signal_grade_counts": grade_counts,
        "visual_decision_counts": visual_counts,
        "all_unique_cases_line9_conditioned": all(str(row["line9_conditioned"]).lower() == "true" for row in audit_rows),
        "formal_train_allowed_count": sum(str(row["train_allowed"]).lower() == "true" for row in audit_rows),
        "curve_training_contract_ok_count": sum(bool(row["curve_training_contract_ok"]) for row in audit_rows),
        "curve_training_contract_blocked_count": sum(not bool(row["curve_training_contract_ok"]) for row in audit_rows),
        "file_integrity": file_integrity_summary(),
        "metadata_audit": metadata_summary(copies),
        "family_diversity": family_diversity,
        "note": "Visual decisions are complete; signal support does not override provenance or label-contract blockers.",
    }
    (OUT / "AUDIT_AUTO_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
