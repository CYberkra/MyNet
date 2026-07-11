#!/usr/bin/env python3
"""Build canonical YingShan full-line NPZ files directly from the audited raw CSV archive.

Official CSV schema (confirmed by the project source-data specification):
    column 1: longitude [deg]
    column 2: latitude [deg]
    column 3: ground elevation [m]
    column 4: radar reflection amplitude
    column 5: flight height AGL [m]

Rows are trace-major.  Each trace contributes ``Number of Samples`` consecutive rows.
The existing overlapping window cache remains the authoritative source for labels and
status vectors until the labels are globally re-audited.  Raw amplitudes and spatial
metadata are taken only from the original CSV archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.spatial_orientation import orientation_metadata

DEFAULT_DATASET_ROOT = ROOT / "data_corrected_v1_4_terrain_direction"
LINE_NAME_RE = re.compile(r"(?P<line>Line(?:3|6|7|9|L1|X1))origin\(36\)\.csv$", re.IGNORECASE)
HEADER_PATTERNS = {
    "samples": re.compile(r"Number of Samples\s*=\s*(\d+)", re.IGNORECASE),
    "time_window_ns": re.compile(r"Time windows?\s*\(ns\)\s*=\s*([0-9.eE+-]+)", re.IGNORECASE),
    "traces": re.compile(r"Number of Traces\s*=\s*(\d+)", re.IGNORECASE),
    "trace_interval_m": re.compile(r"Trace interval\s*\(m\)\s*=\s*([0-9.eE+-]+)", re.IGNORECASE),
}


class RawImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawLine:
    line: str
    source_member: str
    source_csv_sha256: str
    samples: int
    traces: int
    time_window_ns: float
    trace_interval_m: float
    longitude: np.ndarray
    latitude: np.ndarray
    ground_elevation_m: np.ndarray
    raw_amplitude: np.ndarray
    flight_height_agl_m: np.ndarray


@dataclass(frozen=True)
class LineResult:
    line: str
    line_path: Path
    trace_count: int
    sample_count: int
    source_member: str
    source_csv_sha256: str
    line_npz_sha256: str
    normalization_scale_p99_abs: float
    reconstruction_correlation: float
    reconstruction_max_abs_diff: float
    gnss_distance_m: float
    declared_distance_m: float
    flight_height_min_m: float
    flight_height_median_m: float
    flight_height_max_m: float
    flight_height_outside_2_20_count: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_stream(handle: io.BufferedReader) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_header(lines: list[str], *, member: str) -> dict[str, float | int]:
    if len(lines) != 4:
        raise RawImportError(f"{member}: expected four metadata lines")
    values: dict[str, float | int] = {}
    for key, pattern in HEADER_PATTERNS.items():
        match = next((pattern.search(text) for text in lines if pattern.search(text)), None)
        if match is None:
            raise RawImportError(f"{member}: missing header field {key}")
        values[key] = int(match.group(1)) if key in {"samples", "traces"} else float(match.group(1))
    if int(values["samples"]) < 2 or int(values["traces"]) <= 0:
        raise RawImportError(f"{member}: invalid sample/trace count {values}")
    if float(values["time_window_ns"]) <= 0 or float(values["trace_interval_m"]) <= 0:
        raise RawImportError(f"{member}: invalid time window/trace interval {values}")
    return values


def _constant_per_trace(values: np.ndarray, *, name: str, member: str, atol: float) -> np.ndarray:
    # Input shape: traces x samples.
    first = values[:, :1]
    delta = np.max(np.abs(values - first), axis=1)
    if np.any(delta > atol):
        bad = np.flatnonzero(delta > atol)[:10].tolist()
        raise RawImportError(f"{member}: {name} is not constant within trace(s) {bad}; max_delta={float(delta.max())}")
    return first[:, 0].astype(np.float64, copy=False)


def load_raw_line_from_zip(archive: zipfile.ZipFile, member: str) -> RawLine:
    match = LINE_NAME_RE.search(Path(member).name)
    if not match:
        raise RawImportError(f"unrecognised raw CSV member: {member}")
    line = match.group("line")

    with archive.open(member, "r") as src, tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        h = hashlib.sha256()
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            temp.write(chunk)
        temp_path = Path(temp.name)
    try:
        with temp_path.open("r", encoding="utf-8-sig", errors="strict") as handle:
            header_lines = [handle.readline().strip() for _ in range(4)]
        header = parse_header(header_lines, member=member)
        data = np.loadtxt(temp_path, delimiter=",", skiprows=4, dtype=np.float64)
    finally:
        temp_path.unlink(missing_ok=True)

    samples = int(header["samples"])
    traces = int(header["traces"])
    expected_rows = samples * traces
    if data.shape != (expected_rows, 5):
        raise RawImportError(f"{member}: data shape {data.shape}, expected {(expected_rows, 5)}")
    if not np.isfinite(data).all():
        raise RawImportError(f"{member}: raw CSV contains NaN/Inf")

    cube = data.reshape(traces, samples, 5)
    longitude = _constant_per_trace(cube[:, :, 0], name="longitude", member=member, atol=1e-10)
    latitude = _constant_per_trace(cube[:, :, 1], name="latitude", member=member, atol=1e-10)
    ground = _constant_per_trace(cube[:, :, 2], name="ground_elevation_m", member=member, atol=1e-7)
    flight = _constant_per_trace(cube[:, :, 4], name="flight_height_agl_m", member=member, atol=1e-7)
    raw = cube[:, :, 3].T.astype(np.float32)
    if np.any(flight <= 0):
        raise RawImportError(f"{member}: flight height contains non-positive values")

    return RawLine(
        line=line,
        source_member=member,
        source_csv_sha256=h.hexdigest(),
        samples=samples,
        traces=traces,
        time_window_ns=float(header["time_window_ns"]),
        trace_interval_m=float(header["trace_interval_m"]),
        longitude=longitude,
        latitude=latitude,
        ground_elevation_m=ground,
        raw_amplitude=raw,
        flight_height_agl_m=flight,
    )


def haversine_cumulative_m(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    if longitude.shape != latitude.shape or longitude.ndim != 1:
        raise RawImportError("longitude/latitude must be matching one-dimensional vectors")
    if longitude.size == 0:
        return np.empty(0, dtype=np.float64)
    lon1 = np.deg2rad(longitude[:-1])
    lon2 = np.deg2rad(longitude[1:])
    lat1 = np.deg2rad(latitude[:-1])
    lat2 = np.deg2rad(latitude[1:])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    step = 2.0 * 6371008.8 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(step)])


def load_label_source(path: Path, *, line: str, samples: int, traces: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RawImportError(f"{line}: label source line NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = ("raw_full_normalized", "soft_mask_train", "status_code", "label_weight")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise RawImportError(f"{line}: label source lacks {missing}: {path}")
        result = {key: np.asarray(data[key]).copy() for key in required}
    if result["raw_full_normalized"].shape != (samples, traces):
        raise RawImportError(
            f"{line}: label-source raw shape {result['raw_full_normalized'].shape}, expected {(samples, traces)}"
        )
    if result["soft_mask_train"].shape != (samples, traces):
        raise RawImportError(f"{line}: soft-mask shape mismatch")
    if result["status_code"].shape != (traces,) or result["label_weight"].shape != (traces,):
        raise RawImportError(f"{line}: status/weight shape mismatch")
    return result


def load_label_source_from_windows(
    dataset_root: Path, *, line: str, samples: int, traces: int
) -> dict[str, np.ndarray]:
    """Rebuild labels/raw check arrays from the authoritative overlapping windows.

    This avoids a circular re-import where an already-generated canonical line
    archive is compared with itself on subsequent runs. Every overlap must agree
    before a canonical line may be rewritten.
    """
    index_path = dataset_root / "window_index.csv"
    if not index_path.is_file():
        raise RawImportError(f"missing window index: {index_path}")
    with index_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("line") == line]
    if not rows:
        raise RawImportError(f"{line}: no window-index rows")
    specs = {
        "raw_full_normalized": ("x_raw", np.float32, (samples, traces)),
        "soft_mask_train": ("y_mask", np.float32, (samples, traces)),
        "status_code": ("status_code", np.int16, (traces,)),
        "label_weight": ("label_weight", np.float32, (traces,)),
    }
    outputs = {name: np.zeros(shape, dtype=dtype) for name, (_, dtype, shape) in specs.items()}
    seen = {name: np.zeros(shape[-1], dtype=bool) for name, (_, _, shape) in specs.items()}
    for row in sorted(rows, key=lambda item: (int(item["start"]), int(item["end"]))):
        start, end = int(row["start"]), int(row["end"])
        if start < 0 or end < start or end >= traces:
            raise RawImportError(f"{line}: invalid window range {start}-{end}")
        path = dataset_root / "windows" / f"{row['sample_id']}.npz"
        if not path.is_file():
            raise RawImportError(f"{line}: missing window {path}")
        width = end - start + 1
        with np.load(path, allow_pickle=False) as data:
            for out_name, (source_key, dtype, shape) in specs.items():
                if source_key not in data.files:
                    raise RawImportError(f"{path}: missing {source_key}")
                value = np.asarray(data[source_key], dtype=dtype)
                expected = (samples, width) if len(shape) == 2 else (width,)
                if value.shape != expected:
                    raise RawImportError(f"{path}: {source_key} shape {value.shape}, expected {expected}")
                target = outputs[out_name][..., start:end + 1]
                overlap = seen[out_name][start:end + 1]
                if overlap.any():
                    existing = target[..., overlap]
                    incoming = value[..., overlap]
                    if not np.allclose(existing, incoming, rtol=0.0, atol=1e-7):
                        raise RawImportError(f"{line}: conflicting overlap in {out_name} at {start}-{end}")
                target[..., ~overlap] = value[..., ~overlap]
                seen[out_name][start:end + 1] = True
    for name, mask in seen.items():
        if not mask.all():
            gaps = np.flatnonzero(~mask)[:20].tolist()
            raise RawImportError(f"{line}: uncovered traces in {name}: {gaps}")
    return outputs


def build_line_npz(
    raw: RawLine, *, label_source_path: Path | None = None, labels: dict[str, np.ndarray] | None = None,
    output_path: Path, source_zip_sha256: str
) -> LineResult:
    if labels is None:
        if label_source_path is None:
            raise RawImportError(f"{raw.line}: labels or label_source_path is required")
        labels = load_label_source(label_source_path, line=raw.line, samples=raw.samples, traces=raw.traces)
    scale = float(np.percentile(np.abs(raw.raw_amplitude.astype(np.float64)), 99.0))
    if not np.isfinite(scale) or scale <= 0:
        raise RawImportError(f"{raw.line}: invalid P99 absolute normalization scale {scale}")
    normalised = (raw.raw_amplitude.astype(np.float64) / scale).astype(np.float32)
    old = labels["raw_full_normalized"].astype(np.float64)
    # Avoid np.corrcoef: the Windows NumPy runtime used for project validation
    # can throw a native exception in this otherwise straightforward audit.
    lhs = normalised.astype(np.float64).ravel()
    rhs = old.ravel()
    lhs_centered = lhs - lhs.mean()
    rhs_centered = rhs - rhs.mean()
    corr_denom = np.sqrt(np.dot(lhs_centered, lhs_centered) * np.dot(rhs_centered, rhs_centered))
    corr = float(np.dot(lhs_centered, rhs_centered) / corr_denom) if corr_denom > 0 else float("nan")
    max_diff = float(np.max(np.abs(normalised.astype(np.float64) - old)))
    if corr < 0.99999 or max_diff > 5e-4:
        raise RawImportError(
            f"{raw.line}: original CSV does not match window-cache raw; corr={corr:.9f}, max_abs_diff={max_diff:.9g}"
        )

    time_ns = np.linspace(0.0, raw.time_window_ns, raw.samples, dtype=np.float64)
    dt_ns = float(raw.time_window_ns / (raw.samples - 1))
    gnss_distance = haversine_cumulative_m(raw.longitude, raw.latitude)
    declared_distance = np.arange(raw.traces, dtype=np.float64) * raw.trace_interval_m
    antenna_elevation = raw.ground_elevation_m + raw.flight_height_agl_m
    height_outside = (raw.flight_height_agl_m < 2.0) | (raw.flight_height_agl_m > 20.0)
    split = "test" if raw.line == "Line9" else ("exclude" if raw.line == "LineX1" else "unassigned")
    orient = orientation_metadata(raw.line, raw.longitude, raw.latitude)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        raw_amplitude=raw.raw_amplitude.astype(np.float32),
        raw_full_normalized=normalised,
        soft_mask_train=labels["soft_mask_train"].astype(np.float32),
        status_code=labels["status_code"].astype(np.int16),
        label_weight=labels["label_weight"].astype(np.float32),
        time_ns=time_ns.astype(np.float32),
        dt_ns=np.asarray(dt_ns, dtype=np.float32),
        longitude=raw.longitude.astype(np.float64),
        latitude=raw.latitude.astype(np.float64),
        ground_elevation_m=raw.ground_elevation_m.astype(np.float32),
        flight_height_agl_m=raw.flight_height_agl_m.astype(np.float32),
        antenna_elevation_m=antenna_elevation.astype(np.float32),
        gnss_cumulative_distance_m=gnss_distance.astype(np.float64),
        trace_interval_m=np.asarray(raw.trace_interval_m, dtype=np.float32),
        declared_trace_distance_m=declared_distance,
        profile_chainage_m=declared_distance,
        acquisition_bearing_deg=np.asarray(orient["acquisition_bearing_deg"], dtype=np.float64),
        acquisition_compass=np.asarray(orient["acquisition_compass"]),
        engineering_profile=np.asarray(orient["engineering_profile"]),
        profile_left=np.asarray(orient["profile_left"]),
        profile_right=np.asarray(orient["profile_right"]),
        profile_display_flip=np.asarray(bool(orient["profile_display_flip"])),
        profile_orientation_confidence=np.asarray(orient["confidence"]),
        profile_orientation_evidence=np.asarray(orient["evidence"]),
        orientation_contract=np.asarray("canonical arrays remain acquisition order; profile flip is display-only"),
        normalization_method=np.asarray("per_line_p99_abs"),
        normalization_scale_p99_abs=np.asarray(scale, dtype=np.float64),
        split=np.asarray(split),
        line=np.asarray(raw.line),
        canonical_source=np.asarray("original_yingshan_csv"),
        label_source=np.asarray("audited_overlapping_window_cache_reconstructed_each_import"),
        csv_schema=np.asarray("lon,lat,ground_elevation_m,radar_amplitude,flight_height_agl_m"),
        source_zip_sha256=np.asarray(source_zip_sha256),
        source_csv_member=np.asarray(raw.source_member),
        source_csv_sha256=np.asarray(raw.source_csv_sha256),
        flight_height_outside_planned_2_20_m=height_outside.astype(np.uint8),
    )
    declared_distance_total = float((raw.traces - 1) * raw.trace_interval_m)
    return LineResult(
        line=raw.line,
        line_path=output_path,
        trace_count=raw.traces,
        sample_count=raw.samples,
        source_member=raw.source_member,
        source_csv_sha256=raw.source_csv_sha256,
        line_npz_sha256=sha256_file(output_path),
        normalization_scale_p99_abs=scale,
        reconstruction_correlation=corr,
        reconstruction_max_abs_diff=max_diff,
        gnss_distance_m=float(gnss_distance[-1]),
        declared_distance_m=declared_distance_total,
        flight_height_min_m=float(raw.flight_height_agl_m.min()),
        flight_height_median_m=float(np.median(raw.flight_height_agl_m)),
        flight_height_max_m=float(raw.flight_height_agl_m.max()),
        flight_height_outside_2_20_count=int(height_outside.sum()),
    )


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def enrich_window_index(dataset_root: Path, results: dict[str, LineResult]) -> None:
    index_path = dataset_root / "window_index.csv"
    rows = _read_index(index_path)
    line_cache: dict[str, dict[str, np.ndarray]] = {}
    for line in results:
        with np.load(dataset_root / "lines" / f"{line}.npz", allow_pickle=False) as data:
            line_cache[line] = {
                key: np.asarray(data[key]).copy()
                for key in (
                    "flight_height_agl_m", "ground_elevation_m", "antenna_elevation_m",
                    "longitude", "latitude", "gnss_cumulative_distance_m",
                    "declared_trace_distance_m", "profile_chainage_m",
                    "acquisition_bearing_deg", "profile_display_flip",
                )
            }
    out_rows: list[dict[str, object]] = []
    for row in rows:
        line = row["line"]
        start, end = int(row["start"]), int(row["end"])
        sl = slice(start, end + 1)
        meta = line_cache[line]
        height = meta["flight_height_agl_m"][sl].astype(np.float64)
        ground = meta["ground_elevation_m"][sl].astype(np.float64)
        antenna = meta["antenna_elevation_m"][sl].astype(np.float64)
        dist = meta["gnss_cumulative_distance_m"][sl].astype(np.float64)
        outside = (height < 2.0) | (height > 20.0)
        # Values outside the planned 2-20 m operating range remain measured
        # physical heights, not missing metadata. They are flagged for field/QC
        # review but remain valid for the physics-based arrival prior.
        arrival_valid = bool(np.isfinite(height).all() and np.all(height > 0))
        enriched = dict(row)
        enriched.update(
            {
                "source_csv_member": results[line].source_member,
                "source_csv_sha256": results[line].source_csv_sha256,
                "source_trace_start": start,
                "source_trace_end": end,
                "antenna_height_agl_m": f"{float(np.median(height)):.9g}",
                "antenna_height_agl_valid": "true" if arrival_valid else "false",
                "flight_height_min_m": f"{float(height.min()):.9g}",
                "flight_height_max_m": f"{float(height.max()):.9g}",
                "flight_height_std_m": f"{float(height.std()):.9g}",
                "flight_height_outside_2_20_fraction": f"{float(outside.mean()):.9g}",
                "ground_elevation_median_m": f"{float(np.median(ground)):.9g}",
                "antenna_elevation_median_m": f"{float(np.median(antenna)):.9g}",
                "longitude_start": f"{float(meta['longitude'][start]):.10f}",
                "latitude_start": f"{float(meta['latitude'][start]):.10f}",
                "longitude_end": f"{float(meta['longitude'][end]):.10f}",
                "latitude_end": f"{float(meta['latitude'][end]):.10f}",
                "gnss_distance_start_m": f"{float(dist[0]):.9g}",
                "gnss_distance_end_m": f"{float(dist[-1]):.9g}",
                "gnss_span_m": f"{float(dist[-1] - dist[0]):.9g}",
                "nominal_profile_distance_start_m": f"{float(meta['profile_chainage_m'][start]):.9g}",
                "nominal_profile_distance_end_m": f"{float(meta['profile_chainage_m'][end]):.9g}",
                "acquisition_bearing_deg": f"{float(np.asarray(meta['acquisition_bearing_deg']).item()):.9g}",
                "profile_display_flip": "true" if bool(np.asarray(meta['profile_display_flip']).item()) else "false",
                "canonical_trace_order": "acquisition_csv",
                "height_source": "original_csv_column_5_flight_height",
                "height_quality": "measured_outside_planned_range_review" if outside.any() else "measured_valid",
            }
        )
        out_rows.append(enriched)
    base = ["sample_id", "line", "start", "end", "split", "present", "weak", "no_pick"]
    extra = [
        "source_csv_member", "source_csv_sha256", "source_trace_start", "source_trace_end",
        "antenna_height_agl_m", "antenna_height_agl_valid", "flight_height_min_m",
        "flight_height_max_m", "flight_height_std_m", "flight_height_outside_2_20_fraction",
        "ground_elevation_median_m", "antenna_elevation_median_m", "longitude_start", "latitude_start",
        "longitude_end", "latitude_end", "gnss_distance_start_m", "gnss_distance_end_m", "gnss_span_m",
        "nominal_profile_distance_start_m", "nominal_profile_distance_end_m", "acquisition_bearing_deg",
        "profile_display_flip", "canonical_trace_order", "height_source", "height_quality",
    ]
    # Keep dataset-version fields such as ignore/relabelled/label_version when
    # enriching a newer label index.  Dropping them would silently undo the
    # V15 supervision policy during a raw-CSV reimport.
    inherited = sorted({key for row in rows for key in row} - set(base) - set(extra))
    _write_csv(index_path, base + inherited + extra, out_rows)



def write_orientation_registry(dataset_root: Path, results: list[LineResult]) -> None:
    rows: list[dict[str, object]] = []
    for result in sorted(results, key=lambda item: item.line):
        with np.load(result.line_path, allow_pickle=False) as data:
            rows.append(
                {
                    "line": result.line,
                    "canonical_trace_order": "acquisition_csv",
                    "trace0_longitude": f"{float(data['longitude'][0]):.10f}",
                    "trace0_latitude": f"{float(data['latitude'][0]):.10f}",
                    "last_longitude": f"{float(data['longitude'][-1]):.10f}",
                    "last_latitude": f"{float(data['latitude'][-1]):.10f}",
                    "acquisition_bearing_deg": f"{float(data['acquisition_bearing_deg']):.6f}",
                    "acquisition_compass": str(data['acquisition_compass']),
                    "engineering_profile": str(data['engineering_profile']),
                    "profile_left": str(data['profile_left']),
                    "profile_right": str(data['profile_right']),
                    "profile_display_flip": "true" if bool(data['profile_display_flip']) else "false",
                    "profile_orientation_confidence": str(data['profile_orientation_confidence']),
                    "profile_orientation_evidence": str(data['profile_orientation_evidence']),
                    "gnss_cumulative_length_m": f"{float(data['gnss_cumulative_distance_m'][-1]):.6f}",
                    "nominal_profile_length_m": f"{float(data['profile_chainage_m'][-1]):.6f}",
                }
            )
    fields = [
        "line", "canonical_trace_order", "trace0_longitude", "trace0_latitude",
        "last_longitude", "last_latitude", "acquisition_bearing_deg", "acquisition_compass",
        "engineering_profile", "profile_left", "profile_right", "profile_display_flip",
        "profile_orientation_confidence", "profile_orientation_evidence",
        "gnss_cumulative_length_m", "nominal_profile_length_m",
    ]
    _write_csv(dataset_root / "trace_direction_registry.csv", fields, rows)
    payload = {
        "schema_version": "yingshan_orientation_contract_v1",
        "canonical_order": "All saved arrays and model metrics use original CSV acquisition order.",
        "profile_display": "profile_display_flip is applied only to plots/exports requested in engineering-profile order.",
        "distance_axes": {
            "gnss_cumulative_distance_m": "map/acquisition trajectory distance; may include lateral GNSS jitter",
            "profile_chainage_m": "nominal engineering-profile chainage from declared trace interval",
        },
        "lines": rows,
    }
    (dataset_root / "orientation_contract.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

def update_policy_and_manifest(dataset_root: Path, *, source_zip_path: Path, source_zip_sha256: str, results: list[LineResult]) -> None:
    policy_path = dataset_root / "dataset_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
    policy.update(
        {
            "dataset_id": "data_corrected_v1_4_terrain_direction_canonical_from_original_csv",
            "training_allowed": False,
            "reason": (
                "Canonical full-line arrays now come from the original CSV archive with measured ground elevation and flight height. "
                "Formal training remains blocked by the absence of confirmed true negatives, non-Line9-conditioned approved simulations, "
                "and pending label review."
            ),
            "missing_required_artifacts": [],
            "confirmed_true_negative_traces": 0,
            "canonical_source": {
                "mode": "direct_original_csv_import",
                "raw_zip_path": manifest_path(source_zip_path),
                "raw_zip_sha256": source_zip_sha256,
                "csv_schema": [
                    "longitude_deg", "latitude_deg", "ground_elevation_m",
                    "radar_reflection_amplitude", "flight_height_agl_m",
                ],
                "line_count": len(results),
                "original_full_line_sources_available": True,
                "normalization": "per-line P99 absolute amplitude",
            },
            "height_policy": {
                "field": "flight_height_agl_m",
                "source": "original CSV column 5",
                "arrival_prior_uses_tracewise_measured_height": True,
                "window_median_is_legacy_fallback_only": True,
                "values_outside_planned_2_20_m_are_flagged_but_remain_measured_valid": True,
            },
        }
    )
    policy.pop("reconstruction", None)
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "yingshan_original_csv_canonical_v2",
        "dataset_root": manifest_path(dataset_root),
        "source_zip": manifest_path(source_zip_path),
        "source_zip_sha256": source_zip_sha256,
        "csv_schema": {
            "column_1": "longitude_deg",
            "column_2": "latitude_deg",
            "column_3": "ground_elevation_m",
            "column_4": "radar_reflection_amplitude",
            "column_5": "flight_height_agl_m",
            "row_order": "trace-major; each trace has Number of Samples consecutive rows",
        },
        "canonical_waveform_source": "original CSV column 4",
        "label_source": "audited overlapping window cache",
        "normalization": "per_line_p99_abs",
        "original_full_line_sources_available": True,
        "orientation_contract": {
            "canonical_trace_order": "acquisition_csv",
            "profile_display_is_view_only": True,
            "registry_csv": manifest_path(dataset_root / "trace_direction_registry.csv"),
            "contract_json": manifest_path(dataset_root / "orientation_contract.json"),
        },
        "formal_training_allowed": False,
        "lines": [result.__dict__ | {"line_path": manifest_path(result.line_path)} for result in results],
    }
    (dataset_root / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def copy_source_archive(raw_zip: Path, dataset_root: Path) -> Path:
    source_dir = dataset_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / "ying_shan_measurement_lines_original.zip"
    if raw_zip.resolve() != destination.resolve():
        shutil.copy2(raw_zip, destination)
    return destination


def import_dataset(raw_zip: Path, dataset_root: Path, *, copy_source: bool = True) -> dict[str, object]:
    raw_zip = raw_zip.resolve()
    dataset_root = dataset_root.resolve()
    if not raw_zip.is_file():
        raise RawImportError(f"raw ZIP does not exist: {raw_zip}")
    source_zip_path = copy_source_archive(raw_zip, dataset_root) if copy_source else raw_zip
    source_zip_sha256 = sha256_file(source_zip_path)
    lines_dir = dataset_root / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)

    results: list[LineResult] = []
    with zipfile.ZipFile(source_zip_path) as archive:
        members = sorted(name for name in archive.namelist() if LINE_NAME_RE.search(Path(name).name))
        if len(members) != 6:
            raise RawImportError(f"expected six raw line CSV files, found {len(members)}: {members}")
        for member in members:
            raw = load_raw_line_from_zip(archive, member)
            line_path = lines_dir / f"{raw.line}.npz"
            labels = load_label_source_from_windows(
                dataset_root, line=raw.line, samples=raw.samples, traces=raw.traces
            )
            result = build_line_npz(
                raw,
                labels=labels,
                output_path=line_path,
                source_zip_sha256=source_zip_sha256,
            )
            results.append(result)

    result_map = {item.line: item for item in results}
    enrich_window_index(dataset_root, result_map)
    write_orientation_registry(dataset_root, results)
    update_policy_and_manifest(
        dataset_root,
        source_zip_path=source_zip_path,
        source_zip_sha256=source_zip_sha256,
        results=results,
    )
    return {
        "ok": True,
        "dataset_root": manifest_path(dataset_root),
        "source_zip": manifest_path(source_zip_path),
        "source_zip_sha256": source_zip_sha256,
        "lines": [result.__dict__ | {"line_path": manifest_path(result.line_path)} for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-zip", required=True)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT.relative_to(ROOT)))
    parser.add_argument("--no-copy-source", action="store_true")
    parser.add_argument("--report-json", default="reports/yingshan_original_csv_import.json")
    args = parser.parse_args()
    raw_zip = Path(args.raw_zip)
    if not raw_zip.is_absolute():
        raw_zip = ROOT / raw_zip
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    report_path = Path(args.report_json)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    try:
        report = import_dataset(raw_zip, dataset_root, copy_source=not args.no_copy_source)
    except RawImportError as exc:
        report = {"ok": False, "error": str(exc)}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
