#!/usr/bin/env python3
"""Generate the SHAPE02 geometry-only basal morphology bank.

This stage deliberately does not write solver decks or execute FDTD. It varies
only the basal depth profile while keeping every non-shape factor in one shared
contract. Solver packages are generated only after the geometry bank passes
human and automatic review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import h5py

try:
    from .generate_independent_v2_family01 import (
        ROOT,
        Spec,
        _quadratic_metrics,
        _write_case,
        array_sha256,
        build_indices,
        correlated_cover_bins,
        material_rows,
        sha256,
        write_checksums,
    )
except ImportError:  # Direct execution keeps the scripts directory on sys.path.
    from generate_independent_v2_family01 import (
        ROOT,
        Spec,
        _quadratic_metrics,
        _write_case,
        array_sha256,
        build_indices,
        correlated_cover_bins,
        material_rows,
        sha256,
        write_checksums,
    )


DEFAULT_CONTRACT = ROOT / "data" / "contracts" / "simulation_v2" / "basal_shape_batch_v2.json"
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls" / "SHAPE02_BASAL_GEOMETRY_BANK"
DEFAULT_REPORT = ROOT / "reports" / "basal_shape_batch_v2_20260720" / "SHAPE02_GEOMETRY_GENERATION.json"
DEFAULT_PROBE_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls" / "SHAPE02_BASAL_CAUSAL_PROBES"
BASE_DEPTH_M = 15.2
TRANSITION_THICKNESS_M = 1.35


def batch_tag(contract_id: str) -> str:
    if contract_id == "BASAL_SHAPE_BATCH_V2_SHAPE02":
        return "SHAPE02"
    if contract_id == "BASAL_SHAPE_BATCH_V3_SHAPE03_NONFOCUSING":
        return "SHAPE03"
    if contract_id == "BASAL_SHAPE_BATCH_V4_SHAPE04_LOW_CURVATURE_MEANDER":
        return "SHAPE04"
    if contract_id == "BASAL_SHAPE_BATCH_V5_SHAPE05_BROAD_LOW_RELIEF":
        return "SHAPE05"
    raise ValueError(f"unsupported basal shape contract: {contract_id}")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def gaussian(u: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-((u - center) / width) ** 2)


def flat_reference(u: np.ndarray) -> np.ndarray:
    return np.zeros_like(u)


def gentle_dip(u: np.ndarray) -> np.ndarray:
    return 0.92 * np.tanh(u / 0.82)


def broad_bedrock_high(u: np.ndarray) -> np.ndarray:
    return -0.95 * gaussian(u, 0.02, 0.43)


def broad_bedrock_trough(u: np.ndarray) -> np.ndarray:
    return 0.92 * gaussian(u, -0.05, 0.46)


def asymmetric_flexure(u: np.ndarray) -> np.ndarray:
    return 0.62 * np.tanh((u + 0.12) / 0.30) + 0.30 * gaussian(u, 0.48, 0.30)


def double_relief(u: np.ndarray) -> np.ndarray:
    return -0.55 * gaussian(u, -0.34, 0.40) + 0.62 * gaussian(u, 0.31, 0.46)


def aperiodic_multiscale_low(u: np.ndarray) -> np.ndarray:
    return (
        -0.33 * gaussian(u, -0.58, 0.22)
        + 0.29 * gaussian(u, -0.10, 0.30)
        - 0.25 * gaussian(u, 0.43, 0.20)
        + 0.16 * gaussian(u, 0.76, 0.18)
    )


def aperiodic_multiscale_medium(u: np.ndarray) -> np.ndarray:
    return (
        0.25 * gaussian(u, -0.78, 0.23)
        - 0.39 * gaussian(u, -0.47, 0.28)
        + 0.33 * gaussian(u, -0.08, 0.27)
        - 0.36 * gaussian(u, 0.30, 0.28)
        + 0.28 * gaussian(u, 0.66, 0.24)
    )


def smooth_terrace_ramp(u: np.ndarray) -> np.ndarray:
    return 0.88 * (sigmoid((u + 0.48) / 0.10) - sigmoid((u - 0.24) / 0.12))


def broad_incised_low(u: np.ndarray) -> np.ndarray:
    return 0.78 * gaussian(u, -0.10, 0.33) + 0.34 * gaussian(u, 0.30, 0.52)


def distributed_fault_flexure(u: np.ndarray) -> np.ndarray:
    return 0.66 * np.tanh((u - 0.04) / 0.34)


def compound_shoulder(u: np.ndarray) -> np.ndarray:
    return (
        0.36 * np.tanh((u + 0.18) / 0.72)
        - 0.47 * gaussian(u, -0.24, 0.35)
        + 0.25 * gaussian(u, 0.52, 0.43)
    )


def regional_monotonic_dip(u: np.ndarray) -> np.ndarray:
    """Long regional dip with no interior low or high across the aperture."""
    return 0.50 * u + 0.10 * np.tanh((u + 0.25) / 0.70)


def asymmetric_longwave_trend(u: np.ndarray) -> np.ndarray:
    """Monotonic trend with one broad, non-focusing change in slope."""
    return 0.35 * u + 0.20 * np.tanh((u - 0.22) / 0.75)


def gentle_shelf_monotonic(u: np.ndarray) -> np.ndarray:
    """Distributed shelf transition that remains monotonic through the scan."""
    return 0.28 * u + 0.22 * sigmoid((u + 0.25) / 0.80)


def distributed_gradient_meander(u: np.ndarray) -> np.ndarray:
    """Monotonic regional trend with several broad, low-amplitude slope changes."""
    return (
        0.45 * u
        + 0.08 * np.tanh((u + 0.56) / 0.22)
        - 0.06 * np.tanh((u + 0.05) / 0.36)
        + 0.07 * np.tanh((u - 0.58) / 0.28)
    )


def asymmetric_gradient_meander(u: np.ndarray) -> np.ndarray:
    """Long dip with non-periodic derivative changes, but no interior extremum."""
    return (
        0.40 * u
        + 0.06 * np.tanh((u + 0.66) / 0.16)
        - 0.05 * np.tanh((u - 0.02) / 0.28)
        + 0.05 * np.tanh((u - 0.67) / 0.23)
    )


def broad_shallow_sag(u: np.ndarray) -> np.ndarray:
    """Regional dip plus one aperture-scale, deliberately shallow arch.

    The feature spans roughly 15 m of the acquisition aperture. Its relief is
    intentionally low, but unlike SHAPE04 it includes one true broad turning
    point so the FDTD test can distinguish acceptable curvature from focusing.
    """
    return 0.16 * u + 0.20 * gaussian(u, -0.08, 0.65)


def broad_asymmetric_shoulder(u: np.ndarray) -> np.ndarray:
    """Regional dip with a broad asymmetric shoulder, not a closed basin."""
    return 0.40 * u + 0.10 * gaussian(u, -0.62, 1.25) - 0.045 * gaussian(u, 0.72, 1.45)


SHAPE_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "CAL00_FLAT_REFERENCE": flat_reference,
    "CAL01_GENTLE_DIP": gentle_dip,
    "GEO01_BROAD_BEDROCK_HIGH": broad_bedrock_high,
    "GEO02_BROAD_BEDROCK_TROUGH": broad_bedrock_trough,
    "GEO03_ASYMMETRIC_FLEXURE": asymmetric_flexure,
    "GEO04_DOUBLE_RELIEF": double_relief,
    "GEO05_APERIODIC_MULTISCALE_LOW": aperiodic_multiscale_low,
    "GEO06_APERIODIC_MULTISCALE_MEDIUM": aperiodic_multiscale_medium,
    "GEO07_SMOOTH_TERRACE_RAMP": smooth_terrace_ramp,
    "GEO08_BROAD_INCISED_LOW": broad_incised_low,
    "GEO09_DISTRIBUTED_FAULT_FLEXURE": distributed_fault_flexure,
    "GEO10_COMPOUND_SHOULDER": compound_shoulder,
}

# Kept separate so the frozen SHAPE02 implementation/order remains auditable.
NONFOCUSING_SHAPE_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "GEO11_REGIONAL_MONOTONIC_DIP": regional_monotonic_dip,
    "GEO12_ASYMMETRIC_LONGWAVE_TREND": asymmetric_longwave_trend,
    "GEO13_GENTLE_SHELF_MONOTONIC": gentle_shelf_monotonic,
}

LOW_CURVATURE_MEANDER_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "GEO14_DISTRIBUTED_GRADIENT_MEANDER": distributed_gradient_meander,
    "GEO15_ASYMMETRIC_GRADIENT_MEANDER": asymmetric_gradient_meander,
}

BROAD_LOW_RELIEF_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "GEO16_BROAD_SHALLOW_SAG": broad_shallow_sag,
    "GEO17_BROAD_ASYMMETRIC_SHOULDER": broad_asymmetric_shoulder,
}


def shape_function(shape_id: str) -> Callable[[np.ndarray], np.ndarray]:
    try:
        return {
            **SHAPE_FUNCTIONS,
            **NONFOCUSING_SHAPE_FUNCTIONS,
            **LOW_CURVATURE_MEANDER_FUNCTIONS,
            **BROAD_LOW_RELIEF_FUNCTIONS,
        }[shape_id]
    except KeyError as exc:
        raise ValueError(f"unsupported basal morphology: {shape_id}") from exc


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")


def geometry_metrics(
    shape_id: str,
    role: str,
    scan_x_m: np.ndarray,
    scan_depth_m: np.ndarray,
    gates: dict[str, Any],
) -> dict[str, Any]:
    r2, extrema, slope_p95 = _quadratic_metrics(scan_x_m, scan_depth_m)
    relief = float(np.ptp(scan_depth_m))
    depth_min = float(np.min(scan_depth_m))
    depth_max = float(np.max(scan_depth_m))
    calibration = role.startswith("calibration")
    nonfocusing = role.startswith("nonfocusing")
    range_ok = calibration or gates["candidate_relief_m"][0] <= relief <= gates["candidate_relief_m"][1]
    quadratic_ok = calibration or nonfocusing or not (r2 > 0.985 and extrema <= 1)
    no_internal_extremum_ok = not bool(gates.get("require_no_internal_extremum", False)) or extrema == 0
    max_internal_extrema_ok = extrema <= int(gates.get("max_internal_extrema", 1_000_000))
    result = {
        "shape_id": shape_id,
        "role": role,
        "depth_min_m": depth_min,
        "depth_median_m": float(np.median(scan_depth_m)),
        "depth_max_m": depth_max,
        "relief_m": relief,
        "smoothed_extrema_count": int(extrema),
        "quadratic_fit_r2": float(r2),
        "abs_slope_p95": float(slope_p95),
        "depth_gate_ok": bool(gates["depth_m"][0] <= depth_min and depth_max <= gates["depth_m"][1]),
        "relief_gate_ok": bool(range_ok),
        "slope_gate_ok": bool(slope_p95 <= gates["abs_slope_p95_max"]),
        "quadratic_gate_ok": bool(quadratic_ok),
        "no_internal_extremum_gate_ok": bool(no_internal_extremum_ok),
        "max_internal_extrema_gate_ok": bool(max_internal_extrema_ok),
    }
    result["geometry_gate_ok"] = bool(
        result["depth_gate_ok"]
        and result["relief_gate_ok"]
        and result["slope_gate_ok"]
        and result["quadratic_gate_ok"]
        and result["no_internal_extremum_gate_ok"]
        and result["max_internal_extrema_gate_ok"]
    )
    return result


def render_shape_preview(
    path: Path,
    shape_id: str,
    description: str,
    x_m: np.ndarray,
    depth_m: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    scale = 2
    width, height = 1400 * scale, 720 * scale
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28 * scale, bold=True)
    body_font = load_font(17 * scale)
    small_font = load_font(14 * scale)

    draw.text((48 * scale, 28 * scale), shape_id, fill="#111827", font=title_font)
    draw.text((48 * scale, 72 * scale), description, fill="#475569", font=body_font)

    left, top, right, bottom = 90 * scale, 130 * scale, 1325 * scale, 590 * scale
    draw.rectangle((left, top, right, bottom), fill="#dbeafe", outline="#64748b", width=2 * scale)
    depth_lo, depth_hi = 12.8, 18.2

    def px_x(value: float) -> int:
        return int(left + (value - float(x_m[0])) / float(x_m[-1] - x_m[0]) * (right - left))

    def px_y(value: float) -> int:
        return int(top + (value - depth_lo) / (depth_hi - depth_lo) * (bottom - top))

    for depth in (13.0, 14.0, 15.0, 16.0, 17.0, 18.0):
        y = px_y(depth)
        draw.line((left, y, right, y), fill="#bfdbfe", width=1 * scale)
        draw.text((18 * scale, y - 10 * scale), f"{depth:.0f} m", fill="#475569", font=small_font)
    for distance in (0, 5, 10, 15, 20):
        x = px_x(float(distance))
        draw.line((x, top, x, bottom), fill="#cbd5e1", width=1 * scale)
        draw.text((x - 18 * scale, bottom + 10 * scale), f"{distance} m", fill="#475569", font=small_font)

    points = [(px_x(float(x)), px_y(float(y))) for x, y in zip(x_m, depth_m)]
    bedrock_polygon = points + [(right, bottom), (left, bottom)]
    draw.polygon(bedrock_polygon, fill="#cbd5a1")
    transition_top = [(px_x(float(x)), px_y(float(y - TRANSITION_THICKNESS_M))) for x, y in zip(x_m, depth_m)]
    transition_polygon = transition_top + list(reversed(points))
    draw.polygon(transition_polygon, fill="#fde68a")
    draw.line(points, fill="#b91c1c", width=4 * scale)
    draw.line(transition_top, fill="#a16207", width=2 * scale)

    status = "PASS" if metrics["geometry_gate_ok"] else "REJECT"
    status_color = "#166534" if metrics["geometry_gate_ok"] else "#b91c1c"
    summary = (
        f"{status}  depth {metrics['depth_min_m']:.2f}-{metrics['depth_max_m']:.2f} m  "
        f"relief {metrics['relief_m']:.2f} m  extrema {metrics['smoothed_extrema_count']}  "
        f"slope P95 {metrics['abs_slope_p95']:.3f}  quadratic R2 {metrics['quadratic_fit_r2']:.3f}"
    )
    draw.text((90 * scale, 635 * scale), summary, fill=status_color, font=body_font)
    image.resize((1400, 720), Image.Resampling.LANCZOS).save(path)


def render_contact_sheet(output_root: Path, shape_rows: list[dict[str, Any]], *, tag: str) -> Path:
    previews = [Image.open(output_root / row["shape_id"] / row["preview"]).convert("RGB") for row in shape_rows]
    thumb_w, thumb_h = 700, 360
    sheet = Image.new("RGB", (thumb_w * 2, thumb_h * max(1, (len(previews) + 1) // 2)), "white")
    for index, preview in enumerate(previews):
        sheet.paste(preview.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), ((index % 2) * thumb_w, (index // 2) * thumb_h))
    path = output_root / f"{tag}_GEOMETRY_CONTACT_SHEET.png"
    sheet.save(path)
    return path


def _legacy_shape_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Adapt SHAPE02 metrics to the reusable imported-geometry deck writer."""
    return {
        "shape_name": metrics["shape_id"],
        "scan_depth_min_m": metrics["depth_min_m"],
        "scan_depth_median_m": metrics["depth_median_m"],
        "scan_depth_max_m": metrics["depth_max_m"],
        "scan_depth_range_m": metrics["relief_m"],
        "smoothed_extrema_count": metrics["smoothed_extrema_count"],
        "quadratic_fit_r2": metrics["quadratic_fit_r2"],
        "abs_slope_p95": metrics["abs_slope_p95"],
        "broad_shape_gate_ok": metrics["geometry_gate_ok"],
        "flat_ground": True,
        "generated_from_measured_arrays": False,
    }


def write_probe_decks(contract_path: Path, geometry_root: Path, probe_root: Path, *, overwrite: bool) -> dict[str, Any]:
    """Write blocked SHAPE02 source decks for one-trace full/control probes.

    The geometry bank remains the source of morphology truth. Each probe deck
    receives a new deterministic imported index field with the same cover bins,
    constant transition thickness, source, and acquisition contract. No solver
    process is started here.
    """
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tag = batch_tag(str(contract["contract_id"]))
    bank_manifest_path = geometry_root / f"{tag.lower()}_geometry_bank_manifest.json"
    if not bank_manifest_path.is_file():
        raise FileNotFoundError(f"generate the {tag} geometry bank first: {bank_manifest_path}")
    bank = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if bank["contract_id"] != contract["contract_id"]:
        raise ValueError(f"geometry bank does not match the {tag} contract")
    if probe_root.exists():
        if not overwrite:
            raise FileExistsError(f"probe root exists; pass --overwrite: {probe_root}")
        shutil.rmtree(probe_root)
    probe_root.mkdir(parents=True)

    spec = Spec()
    shared_cover_seed = 2026072201
    bins, field_stats = correlated_cover_bins(spec, shared_cover_seed)
    full_rows = material_rows(control=False)
    control_rows = material_rows(control=True)
    shared_factor_sha256 = str(bank["shared_factor_sha256"])
    family_rows: list[dict[str, Any]] = []

    for bank_row in bank["families"]:
        if not bank_row["metrics"]["geometry_gate_ok"]:
            continue
        shape_id = str(bank_row["shape_id"])
        geometry_dir = geometry_root / shape_id
        full_x_m = np.load(geometry_dir / "full_domain_x_m.npy")
        basal_depth_m = np.load(geometry_dir / "basal_depth_full_domain.npy")
        if full_x_m.shape != (spec.nx,) or basal_depth_m.shape != (spec.nx,):
            raise ValueError(f"{shape_id}: profile does not match the probe grid")
        profiles = {
            "x_m": full_x_m.astype(np.float32),
            "ground_y_m": np.full(spec.nx, spec.ground_y_m, dtype=np.float32),
            "basal_depth_m": basal_depth_m.astype(np.float32),
            "transition_thickness_m": np.full(spec.nx, TRANSITION_THICKNESS_M, dtype=np.float32),
            "basal_y_m": (spec.ground_y_m - basal_depth_m).astype(np.float32),
            "transition_top_y_m": (spec.ground_y_m - basal_depth_m + TRANSITION_THICKNESS_M).astype(np.float32),
        }
        data = build_indices(spec, bins, profiles)
        family_dir = probe_root / shape_id
        family_dir.mkdir()
        geometry_source = family_dir / "_shared_geology_indices.h5"
        with h5py.File(geometry_source, "w") as handle:
            handle.attrs["dx_dy_dz"] = (spec.dl_m, spec.dl_m, spec.dl_m)
            handle.attrs["generator"] = "scripts/generate_basal_shape_batch_v2.py"
            handle.attrs["scene_family_id"] = shape_id
            handle.attrs["shared_factor_sha256"] = shared_factor_sha256
            handle.create_dataset(
                "data",
                data=data,
                dtype=np.int16,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=(min(256, spec.nx), min(256, spec.ny), 1),
            )
        source_contract = {
            "contract_id": contract["contract_id"],
            "scene_family_id": shape_id,
            "source": {
                "model": "ideal_hertzian_line_source",
                "waveform": "55 MHz Ricker pulse proxy",
                "center_frequency_hz": 55_000_000.0,
                "tx_rx_offset_m": 0.18,
                "proxy_only": True,
                "not_sfcw": True,
            },
            "provenance": {
                "line9_conditioned": False,
                "measured_files_read_by_generator": [],
                "shape_prior": f"{tag} analytic basal morphology bank",
                "shared_factor_sha256": shared_factor_sha256,
                "shared_cover_seed": shared_cover_seed,
                "transition_thickness_m": TRANSITION_THICKNESS_M,
            },
        }
        case = {
            "case_id": f"{shape_id}_POS",
            "target_presence": True,
            "negative_semantics": "not_a_negative_sample; one-trace positive full/control causal probe only",
        }
        case_manifest = _write_case(
            family_dir / case["case_id"],
            case,
            source_contract,
            spec,
            data,
            profiles,
            _legacy_shape_metrics(bank_row["metrics"]),
            field_stats,
            full_rows,
            control_rows,
            geometry_source,
            contract_path,
        )
        geometry_source.unlink()
        case_dir = family_dir / case["case_id"]
        scene_path = case_dir / "scene_manifest.json"
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        scene["lifecycle_state"] = f"pre_solver_{tag.lower()}_one_trace_probe"
        scene["generator_path"] = "scripts/generate_basal_shape_batch_v2.py"
        scene["generator_sha256"] = sha256(Path(__file__))
        scene["training_block_reason"] = (
            f"{tag} one-trace causal probe. It is screening-only and cannot become training data "
            "without full-span and native-256 paired evidence."
        )
        scene["geometry"]["shared_factor_sha256"] = shared_factor_sha256
        scene["geometry"]["transition_thickness_fixed_m"] = TRANSITION_THICKNESS_M
        scene["strict_pair"]["positive_control_equals_family_negative_full"] = False
        scene["strict_pair"]["probe_scope"] = "full_scene/no_basal_contrast_control only; target-absent negative deferred"
        scene["next_gate"] = "static + geometry-only audit, then one trace full/control GPU causal probe"
        scene_path.write_text(json.dumps(scene, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        # The writer emitted hashes before this manifest rewrite.
        write_checksums(case_dir)
        family_record = {
            "shape_id": shape_id,
            "role": bank_row["role"],
            "case_dir": str(case_dir.relative_to(ROOT)).replace("\\", "/"),
            "scene_manifest_sha256": sha256(scene_path),
            "geometry_index_sha256": sha256(case_dir / "geology_indices.h5"),
            "shared_factor_sha256": shared_factor_sha256,
            "geometry_metrics": bank_row["metrics"],
            "solver_executed": False,
        }
        write_json(family_dir / "probe_family_manifest.json", family_record)
        family_rows.append(family_record)

    report = {
        "contract_id": contract["contract_id"],
        "stage": "B_causal_probe",
        "status": "pre_solver_probe_decks_written",
        "formal_training_allowed": False,
        "probe_grid_dl_m": spec.dl_m,
        "formal_release_grid_dl_m": contract["canonical_release"]["dl_m"],
        "shared_factor_sha256": shared_factor_sha256,
        "shared_cover_seed": shared_cover_seed,
        "probe_case_count": len(family_rows),
        "families": family_rows,
        "next_gate": "static and geometry-only input audit before GPU execution",
    }
    write_json(probe_root / f"{tag.lower()}_probe_deck_manifest.json", report)
    return report


def generate(contract_path: Path, output_root: Path, report_path: Path, *, overwrite: bool) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["contract_id"] not in {
        "BASAL_SHAPE_BATCH_V2_SHAPE02",
        "BASAL_SHAPE_BATCH_V3_SHAPE03_NONFOCUSING",
        "BASAL_SHAPE_BATCH_V4_SHAPE04_LOW_CURVATURE_MEANDER",
        "BASAL_SHAPE_BATCH_V5_SHAPE05_BROAD_LOW_RELIEF",
    }:
        raise ValueError("unexpected contract id")
    shape_contracts = contract["shape_bank"]
    tag = batch_tag(str(contract["contract_id"]))
    shape_ids = [row["id"] for row in shape_contracts]
    if len(shape_ids) != len(set(shape_ids)):
        raise ValueError("shape bank contains duplicate identifiers")
    if contract["contract_id"] == "BASAL_SHAPE_BATCH_V2_SHAPE02" and shape_ids != list(SHAPE_FUNCTIONS):
        raise ValueError("shape bank and implementation order differ")
    unknown = [
        shape_id
        for shape_id in shape_ids
        if shape_id not in SHAPE_FUNCTIONS
        and shape_id not in NONFOCUSING_SHAPE_FUNCTIONS
        and shape_id not in LOW_CURVATURE_MEANDER_FUNCTIONS
        and shape_id not in BROAD_LOW_RELIEF_FUNCTIONS
    ]
    if unknown:
        raise ValueError(f"shape bank contains unsupported identifiers: {unknown}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    spec = Spec()
    scan_x_m = np.arange(spec.trace_count, dtype=np.float64) * spec.trace_spacing_m
    full_x_m = (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dl_m
    scan_center_global_m = spec.scan_start_x_m + spec.tx_rx_offset_m / 2.0 + spec.scan_span_m / 2.0
    half_span_m = spec.scan_span_m / 2.0
    full_u = (full_x_m - scan_center_global_m) / half_span_m
    scan_u = (scan_x_m - spec.scan_span_m / 2.0) / half_span_m

    locked = contract["factor_isolation"]["locked"]
    shared_factor_record = {
        "ground_surface": locked["ground_surface"],
        "flight_height_agl_m": locked["flight_height_agl_m"],
        "waveform": locked["waveform"],
        "center_frequency_hz": locked["center_frequency_hz"],
        "tx_rx_offset_m": locked["tx_rx_offset_m"],
        "cover_index_field": locked["cover_index_field"],
        "weathered_transition_thickness_m": TRANSITION_THICKNESS_M,
        "material_table": locked["material_table"],
        "screening_grid_dl_m": spec.dl_m,
        "scan_span_m": spec.scan_span_m,
        "trace_count": spec.trace_count,
        "trace_spacing_m": spec.trace_spacing_m,
    }
    shared_factor_sha256 = canonical_json_sha256(shared_factor_record)
    write_json(output_root / "shared_factor_contract.json", {**shared_factor_record, "sha256": shared_factor_sha256})

    shape_rows: list[dict[str, Any]] = []
    for shape_contract in shape_contracts:
        shape_id = shape_contract["id"]
        shape_dir = output_root / shape_id
        shape_dir.mkdir()
        function = shape_function(shape_id)
        full_depth_m = BASE_DEPTH_M + function(full_u)
        scan_depth_m = BASE_DEPTH_M + function(scan_u)
        metrics = geometry_metrics(shape_id, shape_contract["role"], scan_x_m, scan_depth_m, contract["geometry_gates"])

        np.save(shape_dir / "scan_x_m.npy", scan_x_m.astype(np.float32))
        np.save(shape_dir / "basal_depth_scan_256.npy", scan_depth_m.astype(np.float32))
        np.save(shape_dir / "full_domain_x_m.npy", full_x_m.astype(np.float32))
        np.save(shape_dir / "basal_depth_full_domain.npy", full_depth_m.astype(np.float32))
        np.save(shape_dir / "transition_thickness_full_domain.npy", np.full(spec.nx, TRANSITION_THICKNESS_M, dtype=np.float32))
        preview_name = f"{shape_id}_GEOMETRY_PREVIEW.png"
        render_shape_preview(
            shape_dir / preview_name,
            shape_id,
            shape_contract["description"],
            scan_x_m,
            scan_depth_m,
            metrics,
        )
        manifest = {
            "contract_id": contract["contract_id"],
            "shape_id": shape_id,
            "role": shape_contract["role"],
            "description": shape_contract["description"],
            "stage": "A_geometry_bank",
            "solver_executed": False,
            "formal_training_allowed": False,
            "promotion_allowed": False,
            "line9_conditioned": False,
            "shared_factor_sha256": shared_factor_sha256,
            "basal_depth_scan_sha256": array_sha256(scan_depth_m.astype(np.float32)),
            "basal_depth_full_domain_sha256": array_sha256(full_depth_m.astype(np.float32)),
            "metrics": metrics,
            "preview": preview_name,
            "next_gate": "one central full/control causal probe" if metrics["geometry_gate_ok"] else "geometry redesign",
        }
        write_json(shape_dir / "shape_manifest.json", manifest)
        shape_rows.append(manifest)

    contact_sheet = render_contact_sheet(output_root, shape_rows, tag=tag)
    gate_pass = [row["shape_id"] for row in shape_rows if row["metrics"]["geometry_gate_ok"]]
    report = {
        "contract_id": contract["contract_id"],
        "stage": "A_geometry_bank",
        "generator": "scripts/generate_basal_shape_batch_v2.py",
        "generator_status": "geometry_only_no_solver",
        "formal_training_allowed": False,
        "promotion_allowed": False,
        "spec": asdict(spec),
        "shared_factor_sha256": shared_factor_sha256,
        "shape_count": len(shape_rows),
        "geometry_gate_pass_count": len(gate_pass),
        "geometry_gate_pass_shapes": gate_pass,
        "families": shape_rows,
        "contact_sheet": contact_sheet.name,
        "next_stage": "human geometry review before one-trace causal probes",
    }
    write_json(output_root / f"{tag.lower()}_geometry_bank_manifest.json", report)
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write-probe-decks", action="store_true")
    parser.add_argument("--probe-output-root", type=Path, default=DEFAULT_PROBE_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = generate(args.contract.resolve(), args.output_root.resolve(), args.report.resolve(), overwrite=args.overwrite)
    if args.write_probe_decks:
        report["probe_decks"] = write_probe_decks(
            args.contract.resolve(),
            args.output_root.resolve(),
            args.probe_output_root.resolve(),
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
