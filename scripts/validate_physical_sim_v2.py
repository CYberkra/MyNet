#!/usr/bin/env python3
"""Static and optional runtime validation for simulation contract V2 controls."""
from __future__ import annotations

import argparse
import csv
import json
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import GridSpec, SourceSpec, sha256_file  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "00_controls"
BOX_RE = re.compile(
    r"^#box:\s+([\deE+\-.]+)\s+([\deE+\-.]+)\s+([\deE+\-.]+)\s+"
    r"([\deE+\-.]+)\s+([\deE+\-.]+)\s+([\deE+\-.]+)\s+(\S+)"
)


def _load_npy(case_dir: Path, name: str) -> np.ndarray:
    path = case_dir / "labels" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _parse_input(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"lines": path.read_text(encoding="utf-8").splitlines()}
    for line in result["lines"]:
        if line.startswith("#domain:"):
            result["domain"] = tuple(float(v) for v in line.split(":", 1)[1].split())
        elif line.startswith("#dx_dy_dz:"):
            result["dl"] = tuple(float(v) for v in line.split(":", 1)[1].split())
        elif line.startswith("#time_window:"):
            result["time_window_s"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("#pml_cells:"):
            result["pml_cells"] = tuple(int(v) for v in line.split(":", 1)[1].split())
        elif line.startswith("#hertzian_dipole:"):
            v = line.split(":", 1)[1].split()
            result["source"] = (v[0], float(v[1]), float(v[2]), float(v[3]), v[4])
        elif line.startswith("#rx:"):
            v = line.split(":", 1)[1].split()
            result["receiver"] = (float(v[0]), float(v[1]), float(v[2]))
        elif line.startswith("#src_steps:"):
            result["src_steps"] = tuple(float(v) for v in line.split(":", 1)[1].split())
        elif line.startswith("#rx_steps:"):
            result["rx_steps"] = tuple(float(v) for v in line.split(":", 1)[1].split())
    return result


def _check_geometry(path: Path, domain: tuple[float, float, float]) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = BOX_RE.match(line)
        if not match:
            if line.strip() and not line.startswith("#box:"):
                errors.append(f"{path.name}:{lineno}: unexpected geometry command")
            continue
        count += 1
        x1, y1, z1, x2, y2, z2 = (float(match.group(i)) for i in range(1, 7))
        if not (x2 > x1 and y2 > y1 and z2 > z1):
            errors.append(f"{path.name}:{lineno}: zero/negative thickness box")
        if x1 < -1e-9 or y1 < -1e-9 or z1 < -1e-9:
            errors.append(f"{path.name}:{lineno}: coordinate below domain origin")
        if x2 > domain[0] + 1e-9 or y2 > domain[1] + 1e-9 or z2 > domain[2] + 1e-9:
            errors.append(f"{path.name}:{lineno}: box exceeds domain")
    if count == 0:
        errors.append(f"{path.name}: no boxes")
    return count, errors


def validate_case(case_dir: Path, *, run_geometry_only: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = case_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        return {"case_id": case_dir.name, "ok": False, "errors": ["missing scene_manifest.json"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_id = str(manifest.get("case_id", case_dir.name))
    target_presence = bool(manifest.get("target_presence"))

    if manifest.get("contract_id") != "PGDA_SIMULATION_CONTRACT_V2":
        errors.append("wrong contract_id")
    if manifest.get("formal_training_allowed") is not False:
        errors.append("control scene must not be training-approved")
    if manifest.get("line9_conditioned") is not False:
        errors.append("control scene must be explicitly non-Line9-conditioned")

    input_text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in case_dir.glob("*.in")
    )
    if "line9" in input_text.lower():
        errors.append("Line9 token found in generated gprMax input")
    forbidden_manifest_values = [
        manifest.get("reference_line"),
        manifest.get("label_origin"),
        manifest.get("geometry", {}).get("source_template"),
    ]
    if any(isinstance(value, str) and "line9" in value.lower() for value in forbidden_manifest_values):
        errors.append("Line9-derived provenance found in control manifest")

    full_input = _parse_input(case_dir / "full_scene.in")
    required = ["domain", "dl", "time_window_s", "pml_cells", "source", "receiver", "src_steps", "rx_steps"]
    for key in required:
        if key not in full_input:
            errors.append(f"missing input command: {key}")
    if errors:
        return {"case_id": case_id, "ok": False, "errors": errors, "warnings": warnings}

    domain = full_input["domain"]
    dl = full_input["dl"]
    if not (math.isclose(dl[0], dl[1], abs_tol=1e-12) and math.isclose(dl[0], dl[2], abs_tol=1e-12)):
        errors.append("spatial steps must be cubic")
    if not math.isclose(domain[2], dl[2], abs_tol=1e-12):
        errors.append("2-D invariant z domain must be exactly one cell")
    if full_input["pml_cells"] != (20, 20, 0, 20, 20, 0):
        errors.append(f"unexpected PML order/value: {full_input['pml_cells']}")
    solver_window_ns = float(manifest["grid"].get("solver_time_window_ns", 701.0))
    if not math.isclose(full_input["time_window_s"], solver_window_ns * 1e-9, rel_tol=0, abs_tol=1e-15):
        errors.append(f"gprMax solver time window must be {solver_window_ns} ns")
    if solver_window_ns <= float(manifest["grid"].get("canonical_time_window_ns", 700.0)):
        errors.append("solver time window must extend beyond canonical endpoint")
    if full_input["source"][0] != "z":
        errors.append("2-D control source must be z-polarized")
    if full_input["src_steps"] != full_input["rx_steps"]:
        errors.append("source and receiver steps differ")
    if not math.isclose(full_input["src_steps"][0], 0.09, abs_tol=1e-12):
        errors.append("trace step must be 0.09 m")
    if any(abs(v) > 1e-12 for v in full_input["src_steps"][1:]):
        errors.append("flat controls must move only in x")

    source_x = _load_npy(case_dir, "source_x_m.npy")
    receiver_x = _load_npy(case_dir, "receiver_x_m.npy")
    midpoint_x = _load_npy(case_dir, "trace_midpoint_x_m.npy")
    ground = _load_npy(case_dir, "ground_y_m.npy")
    antenna = _load_npy(case_dir, "antenna_y_m.npy")
    agl = _load_npy(case_dir, "flight_height_agl_m.npy")
    basal_depth = _load_npy(case_dir, "basal_interface_depth_m.npy")
    cover = _load_npy(case_dir, "cover_thickness_m.npy")
    weathered = _load_npy(case_dir, "weathered_thickness_m.npy")
    geometric = _load_npy(case_dir, "geometric_arrival_time_ns.npy")
    reference = _load_npy(case_dir, "reference_arrival_time_ns.npy")
    visible = _load_npy(case_dir, "visible_phase_time_ns.npy")
    prior = _load_npy(case_dir, "geometric_prior_not_for_training_501x256.npy")
    pending_mask = _load_npy(case_dir, "target_mask_pending_postprocess_501x256.npy")
    time_ns = _load_npy(case_dir, "time_501_ns.npy")

    expected_trace_shape = (256,)
    for name, arr in {
        "source_x": source_x,
        "receiver_x": receiver_x,
        "midpoint_x": midpoint_x,
        "ground": ground,
        "antenna": antenna,
        "agl": agl,
        "basal_depth": basal_depth,
        "cover": cover,
        "weathered": weathered,
        "geometric": geometric,
        "reference": reference,
        "visible": visible,
    }.items():
        if arr.shape != expected_trace_shape:
            errors.append(f"{name} shape {arr.shape} != {expected_trace_shape}")
    if prior.shape != (501, 256) or pending_mask.shape != (501, 256):
        errors.append("mask/prior shape must be 501x256")
    if time_ns.shape != (501,) or not np.allclose(time_ns, np.linspace(0, 700, 501), atol=1e-5):
        errors.append("canonical time grid is not 0..700 ns with 501 samples")
    if not np.allclose(np.diff(midpoint_x), 0.09, atol=1e-6):
        errors.append("midpoint spacing is not 0.09 m")
    if not np.allclose(midpoint_x, 0.5 * (source_x + receiver_x), atol=1e-6):
        errors.append("trace midpoint does not equal Tx/Rx midpoint")
    if not np.allclose(receiver_x - source_x, 0.18, atol=1e-6):
        errors.append("Tx/Rx offset is not 0.18 m")
    if not np.allclose(antenna - ground, agl, atol=1e-6):
        errors.append("antenna_y != ground_y + AGL")
    if np.any(antenna <= ground):
        errors.append("antenna intersects ground")
    if not np.allclose(cover + weathered, basal_depth, atol=3e-5):
        errors.append("cover + weathered != basal depth")
    if np.any(cover < 0.30) or np.any(weathered < 0.30):
        errors.append("layer thickness below 0.30 m")
    pml = full_input["pml_cells"]
    guard_cells = int(manifest["grid"].get("guard_cells", 0))
    if guard_cells < 20:
        errors.append("source/receiver guard must be at least 20 cells beyond the inner PML boundary")
    left_inner = pml[0] * dl[0]
    right_inner = domain[0] - pml[3] * dl[0]
    top_inner = domain[1] - pml[4] * dl[1]
    min_clearance = 20 * dl[0]
    if np.min(source_x) < left_inner + min_clearance - 1e-9:
        errors.append("first source lacks 20-cell clearance from inner left PML")
    if np.max(receiver_x) > right_inner - min_clearance + 1e-9:
        errors.append("last receiver lacks 20-cell clearance from inner right PML")
    if np.max(antenna) > top_inner - min_clearance + 1e-9:
        errors.append("antenna lacks 20-cell air clearance below inner top PML")
    if not np.allclose(reference, geometric, equal_nan=True):
        errors.append("reference_arrival and compatibility geometric_arrival arrays differ")
    arrival_model = str(manifest.get("geometry", {}).get("arrival_model", ""))
    interface_kind = str(manifest.get("geometry", {}).get("nominal_interface", {}).get("kind", ""))
    if interface_kind == "flat" and arrival_model != "horizontal_layered_bistatic_exact":
        errors.append("flat interface must use exact horizontal layered bistatic arrival model")
    if interface_kind != "flat" and arrival_model != "columnar_layered_reference_not_specular_exact":
        errors.append("non-flat interface must identify arrival as a non-exact columnar reference")
    if not np.isnan(visible).all():
        errors.append("visible phase must remain pending before solver run")
    if not np.allclose(pending_mask, 0):
        errors.append("training mask must be zero before postprocess")
    if target_presence:
        if not np.isfinite(geometric).all():
            errors.append("positive case geometric arrivals must be finite")
        elif np.any((geometric <= 0) | (geometric >= 700)):
            errors.append("geometric arrivals outside 0..700 ns")
        if float(prior.max()) <= 0:
            errors.append("positive case missing geometric prior")
        if not (case_dir / "no_basal_contrast_control.in").is_file():
            errors.append("positive case missing matched no-basal control")
    else:
        if not np.isnan(geometric).all():
            errors.append("negative case must not expose target geometric arrival")
        if not np.allclose(prior, 0):
            errors.append("negative case geometric prior must be zero")

    box_counts: dict[str, int] = {}
    for name in ["full_scene_geometry.inc", "no_basal_contrast_geometry.inc"]:
        path = case_dir / name
        if path.is_file():
            count, geometry_errors = _check_geometry(path, domain)
            box_counts[name] = count
            errors.extend(geometry_errors)

    # Validate file hashes created by the generator.
    hash_file = case_dir / "FILE_SHA256.csv"
    if not hash_file.is_file():
        errors.append("missing FILE_SHA256.csv")
    else:
        with hash_file.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                path = case_dir / row["relative_path"]
                if not path.is_file():
                    errors.append(f"hash manifest missing file: {row['relative_path']}")
                elif sha256_file(path) != row["sha256"]:
                    errors.append(f"hash mismatch: {row['relative_path']}")

    geometry_runtime: dict[str, Any] | None = None
    if run_geometry_only:
        if importlib.util.find_spec("gprMax") is None:
            warnings.append("gprMax is not installed; runtime geometry check skipped")
        else:
            command = [sys.executable, "-m", "gprMax", "geometry_check_full.in", "--geometry-only"]
            try:
                proc = subprocess.run(
                    command,
                    cwd=case_dir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=False,
                )
                geometry_runtime = {"returncode": proc.returncode, "tail": proc.stdout[-4000:]}
                if proc.returncode != 0:
                    errors.append("gprMax geometry-only run failed")
            except Exception as exc:  # pragma: no cover - environment-specific
                errors.append(f"geometry-only run error: {exc}")

    return {
        "case_id": case_id,
        "ok": not errors,
        "target_presence": target_presence,
        "errors": errors,
        "warnings": warnings,
        "box_counts": box_counts,
        "arrival_min_ns": float(np.nanmin(geometric)) if target_presence else None,
        "arrival_max_ns": float(np.nanmax(geometric)) if target_presence else None,
        "domain_x_m": domain[0],
        "domain_y_m": domain[1],
        "runtime_geometry": geometry_runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--run-geometry-only", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    root = Path(args.root)
    cases = sorted(p for p in root.iterdir() if p.is_dir() and (p / "scene_manifest.json").is_file())
    results = [validate_case(case, run_geometry_only=args.run_geometry_only) for case in cases]
    by_id = {p.name: p for p in cases}
    result_by_id = {r["case_id"]: r for r in results}
    for case in cases:
        manifest = json.loads((case / "scene_manifest.json").read_text(encoding="utf-8"))
        matched = manifest.get("geometry", {}).get("matched_positive_case_id")
        if not matched:
            continue
        target = by_id.get(str(matched))
        if target is None:
            result_by_id[case.name]["errors"].append(f"matched positive case missing: {matched}")
            result_by_id[case.name]["ok"] = False
            continue
        peer = json.loads((target / "scene_manifest.json").read_text(encoding="utf-8"))
        comparable_fields = [
            ("material set", manifest["materials"]["set"], peer["materials"]["set"]),
            ("nominal interface", manifest["geometry"]["nominal_interface"], peer["geometry"]["nominal_interface"]),
            ("cover fraction", manifest["geometry"]["cover_fraction"], peer["geometry"]["cover_fraction"]),
            ("trace spacing", manifest["grid"]["trace_spacing_m"], peer["grid"]["trace_spacing_m"]),
        ]
        for label, value, expected in comparable_fields:
            if value != expected:
                result_by_id[case.name]["errors"].append(
                    f"matched negative differs from {matched} in {label}"
                )
                result_by_id[case.name]["ok"] = False
        for array_name in ("ground_y_m.npy", "flight_height_agl_m.npy", "cover_thickness_m.npy", "weathered_thickness_m.npy"):
            a = np.load(case / "labels" / array_name, allow_pickle=False)
            b = np.load(target / "labels" / array_name, allow_pickle=False)
            if not np.array_equal(a, b):
                result_by_id[case.name]["errors"].append(
                    f"matched negative differs from {matched}: {array_name}"
                )
                result_by_id[case.name]["ok"] = False
    report = {
        "contract_id": "PGDA_SIMULATION_CONTRACT_V2",
        "root": str(root),
        "ok": bool(results) and all(r["ok"] for r in results),
        "formal_training_allowed": False,
        "case_count": len(results),
        "results": results,
    }
    report_path = Path(args.report) if args.report else root / "preflight_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
