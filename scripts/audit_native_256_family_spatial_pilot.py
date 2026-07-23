#!/usr/bin/env python3
"""Audit distributed native-256 family pilots without requiring air runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import extract_visible_phase, resample_time_axis  # noqa: E402
from scripts.audit_native_256_spatial_pilot import (  # noqa: E402
    _contract_summary,
    _position_tolerance_m,
)
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _full_scene_detectability_gate(
    metrics: dict[str, float],
    manifest: dict[str, object],
) -> dict[str, object]:
    """Reject target energy that is only locally prominent or difference-visible."""

    contract = manifest.get("visibility_gate", {})
    if not isinstance(contract, dict):
        contract = {}
    thresholds = {
        "full_scene_target_to_local_background_rms": float(
            contract.get("full_scene_target_to_local_background_rms_min", 1.0)
        ),
        "full_scene_target_to_background_rms": float(
            contract.get("full_scene_target_to_background_rms_min", 0.35)
        ),
    }
    if "raw_target_to_early_rms_min" in contract:
        thresholds["raw_target_to_early_rms"] = float(
            contract["raw_target_to_early_rms_min"]
        )
    checks = {
        name: bool(float(metrics[name]) >= threshold)
        for name, threshold in thresholds.items()
    }
    local_max = contract.get("full_scene_target_to_local_background_rms_max")
    if local_max is not None:
        checks["full_scene_target_to_local_background_rms_max"] = bool(
            float(metrics["full_scene_target_to_local_background_rms"]) <= float(local_max)
        )
    review_flags: dict[str, bool] = {}
    local_review_above = contract.get(
        "full_scene_target_to_local_background_rms_review_above"
    )
    if local_review_above is not None:
        review_flags["full_scene_target_to_local_background_rms_above_review"] = bool(
            float(metrics["full_scene_target_to_local_background_rms"])
            > float(local_review_above)
        )
    return {
        "passed": bool(all(checks.values())),
        "automatic_metrics_passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": thresholds,
        "review_thresholds": {
            "full_scene_target_to_local_background_rms_review_above": float(
                local_review_above
            )
        }
        if local_review_above is not None
        else {},
        "review_flags": review_flags,
        "human_morphology_review": "pending",
        "human_blind_review_required": bool(
            contract.get("requires_human_visible_multicycle_interface", True)
        ),
        "difference_only_visibility_is_sufficient": bool(
            contract.get("difference_only_visibility_is_sufficient", False)
        ),
        "raw_target_to_early_is_cross_domain_diagnostic_only": bool(
            "raw_target_to_early_rms_min" not in contract
        ),
        "note": (
            "Automatic metrics cannot promote a case. The unlabelled full-scene preview "
            "must expose a distinct, laterally traceable multi-cycle interface."
        ),
    }


def _panel(values: np.ndarray, scale: float, size: tuple[int, int]) -> Image.Image:
    clipped = np.clip(values / max(scale, 1e-30), -1.0, 1.0)
    gray = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize((image.width, size[1]), Image.Resampling.BILINEAR)
    return image.resize(size, Image.Resampling.NEAREST)


def _draw_path(
    draw: ImageDraw.ImageDraw,
    path_ns: np.ndarray,
    time_range: tuple[float, float],
    box: tuple[int, int, int, int],
    colour: str,
) -> None:
    left, top, right, bottom = box
    x = np.linspace(left, right, path_ns.size)
    y = top + (path_ns - time_range[0]) / (time_range[1] - time_range[0]) * (bottom - top)
    draw.line(list(zip(x.tolist(), y.tolist())), fill=colour, width=3)


def _audit_mode(manifest: dict[str, object]) -> str:
    return "positive_pair" if bool(manifest.get("target_presence")) else "true_negative_full"


def _required_output_stems(manifest: dict[str, object]) -> tuple[str, ...]:
    return ("full_scene", "no_basal_contrast_control") if _audit_mode(manifest) == "positive_pair" else ("full_scene",)


def _continuity_step_limit_ns(trace_stride: int, per_canonical_trace_ns: float = 5.6) -> float:
    """Scale the path-step contract to the physical spacing of a sparse pilot."""
    if trace_stride < 1:
        raise ValueError("trace stride must be positive")
    return float(trace_stride) * float(per_canonical_trace_ns)


def _path_step_statistics(values: np.ndarray, trace_stride: int) -> dict[str, float]:
    """Return defined zero step statistics for a one-trace causal smoke."""

    steps = np.abs(np.diff(np.asarray(values, dtype=np.float64)))
    if steps.size == 0:
        p95 = maximum = 0.0
    else:
        p95 = float(np.percentile(steps, 95))
        maximum = float(np.max(steps))
    return {
        "p95_ns": p95,
        "max_ns": maximum,
        "per_canonical_trace_p95_ns": p95 / trace_stride,
        "per_canonical_trace_max_ns": maximum / trace_stride,
    }


def _common_output_end_ns(
    loaded: dict[str, tuple[float, np.ndarray, dict[str, object]]],
) -> float:
    """Return the real common HDF5 end time; never pad a shorter run."""

    ends = [float((item[1].shape[0] - 1) * item[0] * 1e9) for item in loaded.values()]
    if not ends or not np.isfinite(ends).all():
        raise RuntimeError("family pilot has no finite output end time")
    tolerance_ns = max(1e-6, 0.5 * max(item[0] for item in loaded.values()) * 1e9)
    if max(ends) - min(ends) > tolerance_ns:
        raise RuntimeError(f"family pilot output end times are not aligned: {ends}")
    return float(min(ends))


def _solver_output_path(case_dir: Path, stem: str, trace_count: int) -> Path:
    """Resolve the runner's merged multi-trace or bare single-trace output."""

    merged = case_dir / f"{stem}_merged.out"
    if merged.is_file():
        return merged
    single = case_dir / f"{stem}.out"
    if trace_count == 1 and single.is_file():
        return single
    raise FileNotFoundError(
        f"Missing {stem} output: expected {merged.name}"
        + (f" or {single.name}" if trace_count == 1 else "")
    )


def _load_common(case_dir: Path) -> dict[str, object]:
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    selected = np.asarray(run_manifest["selected_trace_indices_zero_based"], dtype=np.int64)
    declared = int(manifest["grid"]["trace_count"])
    if selected.ndim != 1 or selected.size == 0 or np.any(selected < 0) or np.any(selected >= declared):
        raise RuntimeError("run manifest contains invalid trace selection")

    loaded: dict[str, tuple[float, np.ndarray, dict[str, object]]] = {}
    for stem in _required_output_stems(manifest):
        loaded[stem] = read_merged_bscan(_solver_output_path(case_dir, stem, selected.size))
    shapes = {item[1].shape for item in loaded.values()}
    dts = [item[0] for item in loaded.values()]
    if len(shapes) != 1 or not np.allclose(dts, dts[0]):
        raise RuntimeError("family pilot outputs are not aligned")

    analysis_end_ns = _common_output_end_ns(loaded)
    time_ns, full = resample_time_axis(
        loaded["full_scene"][1],
        dts[0],
        time_window_ns=analysis_end_ns,
        output_samples=501,
    )
    if full.shape[1] != selected.size:
        raise RuntimeError("merged output trace count does not match run manifest")
    canonical = {"full_scene": full}
    if "no_basal_contrast_control" in loaded:
        _, canonical["no_basal_contrast_control"] = resample_time_axis(
            loaded["no_basal_contrast_control"][1],
            loaded["no_basal_contrast_control"][0],
            time_window_ns=analysis_end_ns,
            output_samples=501,
        )

    labels = case_dir / "labels"
    source_x = np.load(labels / "source_x_m.npy", allow_pickle=False)[selected]
    receiver_x = np.load(labels / "receiver_x_m.npy", allow_pickle=False)[selected]
    tolerance = _position_tolerance_m(float(manifest["grid"]["dl_m"]))
    contracts = {
        stem: _contract_summary(
            case_dir / "run_logs" / f"{stem}_trace_contract.json",
            selected.size,
            source_x,
            receiver_x,
            tolerance,
        )
        for stem in loaded
    }
    static_audits = {}
    for stem in loaded:
        audit = json.loads((case_dir / "preflight" / f"{stem}_static_audit.json").read_text(encoding="utf-8"))
        static_audits[stem] = {
            "ok": bool(audit.get("ok")),
            "errors": audit.get("errors", []),
            "warnings": audit.get("warnings", []),
        }

    spacing = float(manifest["grid"]["trace_spacing_m"])
    declared_span = spacing * max(declared - 1, 0)
    covered_span = spacing * int(selected[-1] - selected[0]) if selected.size > 1 else 0.0
    finite = all(np.isfinite(value).all() for value in canonical.values())
    versions = sorted({str(item[2].get("gprMax")) for item in loaded.values()})
    return {
        "manifest": manifest,
        "run_manifest": run_manifest,
        "selected": selected,
        "time_ns": time_ns,
        "canonical": canonical,
        "raw_shape": list(next(iter(loaded.values()))[1].shape),
        "finite": finite,
        "versions": versions,
        "contracts": contracts,
        "static_audits": static_audits,
        "declared_span_m": declared_span,
        "covered_span_m": covered_span,
        "covered_fraction": covered_span / max(declared_span, 1e-30),
        "analysis_end_ns": analysis_end_ns,
    }


def _base_result(data: dict[str, object]) -> dict[str, object]:
    manifest = data["manifest"]
    run_manifest = data["run_manifest"]
    selected = np.asarray(data["selected"])
    return {
        "schema": "native_256_family_spatial_pilot_audit_v1",
        "case_id": manifest["case_id"],
        "scene_family_id": manifest.get("scene_family_id", manifest.get("family_id")),
        "audit_mode": _audit_mode(manifest),
        "formal_training_allowed": False,
        "trace_count": int(selected.size),
        "trace_stride": int(run_manifest.get("trace_stride", 1)),
        "selected_trace_indices_zero_based": selected.tolist(),
        "declared_trace_count": int(manifest["grid"]["trace_count"]),
        "covered_scan_span_m": data["covered_span_m"],
        "declared_scan_span_m": data["declared_span_m"],
        "covered_scan_fraction": data["covered_fraction"],
        "distributed_span_coverage_ok": bool(
            int(run_manifest.get("trace_stride", 1)) > 1 and float(data["covered_fraction"]) >= 0.95
        ),
        "raw_shape": data["raw_shape"],
        "canonical_shape": list(np.asarray(data["canonical"]["full_scene"]).shape),
        "finite_ok": data["finite"],
        "gprmax_versions": data["versions"],
        "analysis_window_end_ns": data["analysis_end_ns"],
        "trace_contracts": data["contracts"],
        "static_audits": data["static_audits"],
    }


def _positive_audit(data: dict[str, object], result: dict[str, object], output_dir: Path) -> bool:
    manifest = data["manifest"]
    selected = np.asarray(data["selected"], dtype=np.int64)
    time_ns = np.asarray(data["time_ns"])
    full = np.asarray(data["canonical"]["full_scene"])
    control = np.asarray(data["canonical"]["no_basal_contrast_control"])
    case_dir = Path(result["case_dir"])
    label_dir = case_dir / "labels"
    reference_candidates = (
        ("source_referenced_arrival_time_ns.npy", "geometric_interface_plus_explicit_source_reference_delay"),
        ("reference_arrival_time_ns.npy", "legacy_case_reference"),
        ("geometric_reference_arrival_time_ns.npy", "geometric_interface_only"),
    )
    reference_path = None
    reference_semantics = None
    for filename, semantics in reference_candidates:
        candidate = label_dir / filename
        if candidate.is_file():
            reference_path = candidate
            reference_semantics = semantics
            break
    if reference_path is None or reference_semantics is None:
        raise FileNotFoundError(f"no supported arrival reference exists below {label_dir}")
    reference = np.load(reference_path, allow_pickle=False)[selected]
    result["reference_path"] = _portable_path(reference_path)
    result["reference_semantics"] = reference_semantics
    label_contract = manifest.get("labels", {})
    trace_stride = int(data["run_manifest"].get("trace_stride", 1))
    step_limit_ns = _continuity_step_limit_ns(trace_stride)
    visible, support, contrast = extract_visible_phase(
        full,
        control,
        time_ns,
        reference,
        search_half_width_ns=float(label_contract.get("visible_phase_search_half_width_ns", 35.0)),
        phase_half_width_ns=float(label_contract.get("visible_phase_phase_half_width_ns", 8.0)),
        enforce_continuity=True,
        max_trace_step_ns=step_limit_ns,
    )
    distance = np.abs(time_ns[:, None] - visible[None, :])
    target = distance <= 25.0
    background = (time_ns[:, None] >= 220.0) & (time_ns[:, None] <= 500.0) & (distance >= 70.0)
    local_background = (distance >= 35.0) & (distance <= 70.0) & (time_ns[:, None] <= 500.0)
    target_per_trace = np.sqrt(np.nanmean(np.where(target, contrast**2, np.nan), axis=0))
    target_rms = _rms(contrast[target])
    background_rms = _rms(contrast[background])
    single_trace = full.shape[1] == 1
    full_background_removed = (
        full if single_trace else full - np.median(full, axis=1, keepdims=True)
    )
    control_background_removed = (
        control if single_trace else control - np.median(control, axis=1, keepdims=True)
    )
    full_target_rms = _rms(full_background_removed[target])
    full_background_rms = _rms(full_background_removed[background])
    full_local_background_rms = _rms(full_background_removed[local_background])
    control_target_rms = _rms(control_background_removed[target])
    control_background_rms = _rms(control_background_removed[background])
    control_local_background_rms = _rms(control_background_removed[local_background])
    raw_target_rms = _rms(full[target])
    raw_early_rms = _rms(full[time_ns <= 150.0])
    residual = visible - reference
    visible_steps = _path_step_statistics(visible, trace_stride)
    residual_steps = _path_step_statistics(residual, trace_stride)
    metrics = {
        "reference_range_ns": [float(np.min(reference)), float(np.max(reference))],
        "visible_range_ns": [float(np.min(visible)), float(np.max(visible))],
        "median_visible_minus_reference_ns": float(np.median(residual)),
        "visible_residual_p95_ns": float(np.percentile(np.abs(residual - np.median(residual)), 95)),
        "visible_step_p95_ns": visible_steps["p95_ns"],
        "visible_step_max_ns": visible_steps["max_ns"],
        "visible_step_per_canonical_trace_p95_ns": visible_steps[
            "per_canonical_trace_p95_ns"
        ],
        "visible_step_per_canonical_trace_max_ns": visible_steps[
            "per_canonical_trace_max_ns"
        ],
        "visible_residual_step_p95_ns": residual_steps["p95_ns"],
        "visible_residual_step_max_ns": residual_steps["max_ns"],
        "visible_residual_step_per_canonical_trace_p95_ns": residual_steps[
            "per_canonical_trace_p95_ns"
        ],
        "visible_residual_step_per_canonical_trace_max_ns": residual_steps[
            "per_canonical_trace_max_ns"
        ],
        "sparse_residual_step_limit_ns": step_limit_ns,
        "support_median": float(np.median(support)),
        "support_min": float(np.min(support)),
        "signed_difference_target_rms": target_rms,
        "signed_difference_background_rms": background_rms,
        "target_to_background_rms": target_rms / max(background_rms, 1e-30),
        "full_scene_target_rms": full_target_rms,
        "full_scene_background_rms": full_background_rms,
        "full_scene_target_to_background_rms": full_target_rms / max(full_background_rms, 1e-30),
        "full_scene_local_background_rms": full_local_background_rms,
        "full_scene_target_to_local_background_rms": full_target_rms / max(full_local_background_rms, 1e-30),
        "raw_target_rms": raw_target_rms,
        "raw_early_rms": raw_early_rms,
        "raw_target_to_early_rms": raw_target_rms / max(raw_early_rms, 1e-30),
        "control_target_to_background_rms": control_target_rms / max(control_background_rms, 1e-30),
        "control_target_to_local_background_rms": control_target_rms / max(control_local_background_rms, 1e-30),
        "causal_contrast_to_full_scene_target_rms": target_rms / max(full_target_rms, 1e-30),
        "target_amplitude_cv": float(np.std(target_per_trace) / max(np.mean(target_per_trace), 1e-30)),
        "target_dropout_fraction_below_25pct_median": float(
            np.mean(target_per_trace < 0.25 * np.median(target_per_trace))
        ),
        "early_full_control_relative_difference": _rms((full - control)[time_ns <= 150.0])
        / max(_rms(full[time_ns <= 150.0]), 1e-30),
    }
    result["visible_phase_semantics"] = "continuous signed lobe from solved full-minus-control"
    result["signal_metrics"] = metrics
    if single_trace:
        detectability = {
            "passed": True,
            "evaluated": False,
            "reason": (
                "one trace can prove causal attribution but cannot evaluate "
                "horizontal-background removal or blind B-scan morphology"
            ),
            "human_blind_review_required": True,
        }
    else:
        detectability = _full_scene_detectability_gate(metrics, manifest)
    passed = bool(
        data["finite"]
        and (single_trace or result["distributed_span_coverage_ok"])
        and all(item["ok"] for item in data["contracts"].values())
        and all(item["ok"] and not item["errors"] for item in data["static_audits"].values())
        and metrics["target_to_background_rms"] >= 1.5
        and detectability["passed"]
        and metrics["visible_residual_step_max_ns"] <= step_limit_ns + 1e-6
        and metrics["target_dropout_fraction_below_25pct_median"] <= 0.10
    )
    result["family_spatial_gate"] = {
        "passed": passed,
        "formal_promotion": False,
        "human_morphology_review": "not_evaluated_single_trace" if single_trace else "pending",
        "scope": "one_trace_causal_smoke" if single_trace else "distributed full-span causal pair",
    }
    result["full_scene_detectability_gate"] = detectability
    np.save(output_dir / "visible_phase_time_ns.npy", visible.astype(np.float32))
    np.save(output_dir / "visible_phase_support_score.npy", support.astype(np.float32))
    _render_positive(
        output_dir / "positive_pair_spatial_preview.png",
        manifest,
        time_ns,
        full,
        control,
        contrast,
        reference,
        visible,
        metrics,
        passed,
    )
    return passed


def _negative_audit(data: dict[str, object], result: dict[str, object], output_dir: Path) -> bool:
    manifest = data["manifest"]
    time_ns = np.asarray(data["time_ns"])
    full = np.asarray(data["canonical"]["full_scene"])
    background_removed = full - np.median(full, axis=1, keepdims=True)
    windows = {
        "early_0_150_ns": time_ns <= 150.0,
        "upper_clutter_150_320_ns": (time_ns > 150.0) & (time_ns <= 320.0),
        "deep_protected_320_500_ns": (time_ns > 320.0) & (time_ns <= 500.0),
        "late_diagnostic_after_500_ns": time_ns > 500.0,
    }
    metrics = {
        f"raw_rms_{name}": _rms(full[rows]) for name, rows in windows.items()
    }
    metrics.update({
        f"background_removed_rms_{name}": _rms(background_removed[rows]) for name, rows in windows.items()
    })
    metrics["deep_to_upper_background_removed_rms"] = metrics[
        "background_removed_rms_deep_protected_320_500_ns"
    ] / max(metrics["background_removed_rms_upper_clutter_150_320_ns"], 1e-30)
    metrics["late_to_protected_background_removed_rms"] = metrics[
        "background_removed_rms_late_diagnostic_after_500_ns"
    ] / max(metrics["background_removed_rms_deep_protected_320_500_ns"], 1e-30)
    result["signal_metrics"] = metrics
    solver_passed = bool(
        not bool(manifest.get("target_presence"))
        and data["finite"]
        and result["distributed_span_coverage_ok"]
        and tuple(data["contracts"].keys()) == ("full_scene",)
        and all(item["ok"] for item in data["contracts"].values())
        and all(item["ok"] and not item["errors"] for item in data["static_audits"].values())
    )
    result["family_spatial_gate"] = {
        "passed": solver_passed,
        "formal_promotion": False,
        "scope": "distributed full-span target-absent full scene",
    }
    result["hard_negative_semantics_gate"] = {
        "passed": False,
        "status": "pending_human_review",
        "note": "Solver validity does not prove that every visible reflector is a valid target-negative example.",
    }
    _render_negative(
        output_dir / "true_negative_spatial_preview.png",
        manifest,
        time_ns,
        full,
        background_removed,
        metrics,
        solver_passed,
    )
    return solver_passed


def _render_positive(
    path: Path,
    manifest: dict[str, object],
    time_ns: np.ndarray,
    full: np.ndarray,
    control: np.ndarray,
    contrast: np.ndarray,
    reference: np.ndarray,
    visible: np.ndarray,
    metrics: dict[str, float | list[float]],
    passed: bool,
) -> None:
    time_range = (220.0, 500.0)
    rows = (time_ns >= time_range[0]) & (time_ns <= time_range[1])
    full_b = full - np.median(full, axis=1, keepdims=True)
    control_b = control - np.median(control, axis=1, keepdims=True)
    shared = float(np.percentile(np.abs(np.concatenate((full_b[rows], control_b[rows]), axis=1)), 99.5)) or 1.0
    contrast_scale = float(np.percentile(np.abs(contrast[rows]), 99.5)) or 1.0
    canvas = Image.new("RGB", (2100, 850), "white")
    draw = ImageDraw.Draw(canvas)
    boxes = ((60, 135, 690, 720), (735, 135, 1365, 720), (1410, 135, 2040, 720))
    for values, box, title, scale in (
        (full_b, boxes[0], "Full scene", shared),
        (control_b, boxes[1], "No-basal control (shared gain)", shared),
        (contrast, boxes[2], "Signed full - control", contrast_scale),
    ):
        canvas.paste(_panel(values[rows], scale, (box[2] - box[0], box[3] - box[1])), (box[0], box[1]))
        draw.rectangle(box, outline="#222222", width=2)
        draw.text((box[0], 95), title, fill="#111111", font=_font(22, True))
        _draw_path(draw, reference, time_range, box, "#ffd92f")
        _draw_path(draw, visible, time_range, box, "#d730f0")
    draw.text((60, 25), f"{manifest['case_id']} - {full.shape[1]}-trace full-span pilot", fill="#111111", font=_font(32, True))
    draw.text(
        (60, 58),
        "Yellow: geometry reference. Magenta: solved visible phase. Air reference is intentionally not required.",
        fill="#222222",
        font=_font(18),
    )
    metric_text = (
        f"Pair target/background={metrics['target_to_background_rms']:.2f}; "
        f"full-scene local target/background={metrics['full_scene_target_to_local_background_rms']:.2f}; "
        f"dropout={metrics['target_dropout_fraction_below_25pct_median']:.1%}; "
        f"gate={'PASS' if passed else 'HOLD'}"
    )
    draw.text(
        (60, 765),
        metric_text,
        fill="#0b5d1e" if passed else "#991b1b",
        font=_font(21, True),
    )
    draw.text(
        (60, 802),
        "Sparse full-span validation only; full 256 and human morphology review remain separate gates.",
        fill="#222222",
        font=_font(18),
    )
    canvas.save(path)


def _render_negative(
    path: Path,
    manifest: dict[str, object],
    time_ns: np.ndarray,
    full: np.ndarray,
    background_removed: np.ndarray,
    metrics: dict[str, float],
    passed: bool,
) -> None:
    protected = (time_ns >= 120.0) & (time_ns <= 500.0)
    late = time_ns >= 500.0
    raw_scale = float(np.percentile(np.abs(full[protected]), 99.5)) or 1.0
    removed_scale = float(np.percentile(np.abs(background_removed[protected]), 99.5)) or 1.0
    late_scale = float(np.percentile(np.abs(background_removed[late]), 99.5)) or 1.0
    canvas = Image.new("RGB", (2100, 850), "white")
    draw = ImageDraw.Draw(canvas)
    boxes = ((60, 135, 690, 720), (735, 135, 1365, 720), (1410, 135, 2040, 720))
    panels = (
        (full[protected], boxes[0], "Full scene 120-500 ns", raw_scale),
        (background_removed[protected], boxes[1], "Background removed 120-500 ns", removed_scale),
        (
            background_removed[late],
            boxes[2],
            f"Diagnostic late window 500-{float(time_ns[-1]):.1f} ns",
            late_scale,
        ),
    )
    for values, box, title, scale in panels:
        canvas.paste(_panel(values, scale, (box[2] - box[0], box[3] - box[1])), (box[0], box[1]))
        draw.rectangle(box, outline="#222222", width=2)
        draw.text((box[0], 95), title, fill="#111111", font=_font(22, True))
    draw.text(
        (60, 25),
        f"{manifest['case_id']} - target-absent 32-trace full-span pilot",
        fill="#111111",
        font=_font(31, True),
    )
    draw.text(
        (60, 58),
        "No target path is overlaid: true-negative semantics must not be manufactured from an amplitude minimum.",
        fill="#222222",
        font=_font(18),
    )
    metric_text = (
        f"Deep/upper clutter RMS={metrics['deep_to_upper_background_removed_rms']:.2f}; "
        f"solver gate={'PASS' if passed else 'HOLD'}; human negative gate=PENDING"
    )
    draw.text(
        (60, 765),
        metric_text,
        fill="#0b5d1e" if passed else "#991b1b",
        font=_font(20, True),
    )
    draw.text(
        (60, 802),
        f"500-{float(time_ns[-1]):.1f} ns is diagnostic only; the audit uses the real HDF5 end time.",
        fill="#222222",
        font=_font(18),
    )
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_common(case_dir)
    result = _base_result(data)
    result["case_dir"] = str(case_dir)
    if result["audit_mode"] == "positive_pair":
        passed = _positive_audit(data, result, output_dir)
    else:
        passed = _negative_audit(data, result, output_dir)
    result.pop("case_dir", None)
    (output_dir / "family_spatial_pilot_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["family_spatial_gate"], indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
