"""Reconstruct canonical full-line artifacts from an audited overlapping window cache.

This tool is intentionally conservative.  It never claims that reconstructed
full-line arrays are the original acquisition files.  It verifies every overlap,
rejects gaps or conflicts, writes explicit provenance, and keeps formal training
blocked until independent label/negative/height contracts are satisfied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOW_RE = re.compile(r"^(?P<line>.+)_tr(?P<start>\d+)_(?P<end>\d+)\.npz$")
REQUIRED_KEYS = ("x_raw", "y_mask", "status_code", "label_weight")
DEFAULT_INVENTORY = ROOT / "uavgpr_simlab" / "docs" / "real_data" / "YINGSHAN_REAL_DATA_AUDIT.json"


class ReconstructionError(RuntimeError):
    """Raised when a window cache cannot be reconstructed without ambiguity."""


@dataclass(frozen=True)
class WindowRecord:
    path: Path
    sample_id: str
    line: str
    start: int
    end: int


@dataclass(frozen=True)
class LineSummary:
    line: str
    trace_count: int
    sample_count: int
    time_samples: int
    strong_traces: int
    weak_traces: int
    negative_traces: int
    line_path: Path
    line_sha256: str
    trace_interval_m: float
    source_member: str



def canonical_line_id(value: object) -> str:
    text = str(value).strip()
    mapping = {"3": "Line3", "6": "Line6", "7": "Line7", "9": "Line9", "L1": "LineL1", "X1": "LineX1"}
    return mapping.get(text, text if text.startswith("Line") else f"Line{text}")


def load_inventory(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise ReconstructionError(f"real-data inventory is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("csv_files")
    if not isinstance(rows, list) or not rows:
        raise ReconstructionError(f"real-data inventory has no csv_files rows: {path}")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        line = canonical_line_id(row.get("line_id"))
        samples = int(row.get("samples", 0))
        traces = int(row.get("traces_read", row.get("traces_declared", 0)))
        time_window_ns = float(row.get("time_window_ns", 0.0))
        trace_interval_m = float(row.get("trace_interval_m", 0.0))
        if samples < 2 or traces <= 0 or time_window_ns <= 0 or trace_interval_m <= 0:
            raise ReconstructionError(f"invalid inventory row for {line}: {row}")
        result[line] = {
            "samples": samples,
            "traces": traces,
            "time_window_ns": time_window_ns,
            "dt_ns": time_window_ns / float(samples - 1),
            "trace_interval_m": trace_interval_m,
            "zip_member": str(row.get("zip_member", "")),
        }
    return result


def dataset_level_split(line: str) -> str:
    if line == "Line9":
        return "test"
    if line in {"LineX1", "X1"}:
        return "exclude"
    return "unassigned"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_windows(windows_dir: Path) -> list[WindowRecord]:
    if not windows_dir.is_dir():
        raise ReconstructionError(f"windows directory does not exist: {windows_dir}")
    records: list[WindowRecord] = []
    for path in sorted(windows_dir.glob("*.npz")):
        match = WINDOW_RE.match(path.name)
        if not match:
            raise ReconstructionError(f"window filename lacks trace mapping: {path.name}")
        start, end = int(match.group("start")), int(match.group("end"))
        if start < 0 or end < start:
            raise ReconstructionError(f"invalid trace range in {path.name}: {start}..{end}")
        records.append(
            WindowRecord(
                path=path,
                sample_id=path.stem,
                line=match.group("line"),
                start=start,
                end=end,
            )
        )
    if not records:
        raise ReconstructionError(f"no NPZ windows found in {windows_dir}")
    return records


def _assign_with_overlap_check(
    target: np.ndarray,
    incoming: np.ndarray,
    selection: tuple[slice, ...],
    *,
    key: str,
    sample_id: str,
    atol: float,
) -> None:
    current = target[selection]
    if np.issubdtype(target.dtype, np.floating):
        occupied = np.isfinite(current)
        if occupied.any():
            delta = np.abs(current[occupied].astype(np.float64) - incoming[occupied].astype(np.float64))
            if delta.size and float(delta.max()) > atol:
                raise ReconstructionError(
                    f"overlap conflict for {sample_id}:{key}; max_abs_diff={float(delta.max()):.9g} > {atol}"
                )
        target[selection] = np.where(occupied, current, incoming)
    else:
        sentinel = np.iinfo(target.dtype).min
        occupied = current != sentinel
        if occupied.any() and not np.array_equal(current[occupied], incoming[occupied]):
            raise ReconstructionError(f"overlap conflict for {sample_id}:{key}")
        target[selection] = np.where(occupied, current, incoming)


def reconstruct_line(records: Iterable[WindowRecord], *, atol: float = 0.0) -> dict[str, np.ndarray]:
    rows = sorted(records, key=lambda item: (item.start, item.end, item.sample_id))
    if not rows:
        raise ReconstructionError("cannot reconstruct an empty line")
    line = rows[0].line
    if any(item.line != line for item in rows):
        raise ReconstructionError("reconstruct_line received multiple line IDs")

    first = np.load(rows[0].path, allow_pickle=False)
    missing = [key for key in REQUIRED_KEYS if key not in first.files]
    if missing:
        raise ReconstructionError(f"{rows[0].path} lacks required keys: {missing}")
    time_samples = int(first["x_raw"].shape[0])
    total_traces = max(item.end for item in rows) + 1

    raw = np.full((time_samples, total_traces), np.nan, dtype=np.float32)
    mask = np.full((time_samples, total_traces), np.nan, dtype=np.float32)
    status = np.full(total_traces, np.iinfo(np.int16).min, dtype=np.int16)
    weight = np.full(total_traces, np.nan, dtype=np.float32)

    for item in rows:
        with np.load(item.path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_KEYS if key not in data.files]
            if missing:
                raise ReconstructionError(f"{item.path} lacks required keys: {missing}")
            width = item.end - item.start + 1
            x_raw = np.asarray(data["x_raw"], dtype=np.float32)
            y_mask = np.asarray(data["y_mask"], dtype=np.float32)
            status_code = np.asarray(data["status_code"], dtype=np.int16)
            label_weight = np.asarray(data["label_weight"], dtype=np.float32)
            expected_2d = (time_samples, width)
            if x_raw.shape != expected_2d or y_mask.shape != expected_2d:
                raise ReconstructionError(
                    f"shape mismatch in {item.path}: x_raw={x_raw.shape}, y_mask={y_mask.shape}, expected={expected_2d}"
                )
            if status_code.shape != (width,) or label_weight.shape != (width,):
                raise ReconstructionError(
                    f"trace-vector shape mismatch in {item.path}: status={status_code.shape}, weight={label_weight.shape}"
                )
            sl2 = (slice(None), slice(item.start, item.end + 1))
            sl1 = (slice(item.start, item.end + 1),)
            _assign_with_overlap_check(raw, x_raw, sl2, key="x_raw", sample_id=item.sample_id, atol=atol)
            _assign_with_overlap_check(mask, y_mask, sl2, key="y_mask", sample_id=item.sample_id, atol=atol)
            _assign_with_overlap_check(status, status_code, sl1, key="status_code", sample_id=item.sample_id, atol=atol)
            _assign_with_overlap_check(weight, label_weight, sl1, key="label_weight", sample_id=item.sample_id, atol=atol)

    if not np.isfinite(raw).all() or not np.isfinite(mask).all() or not np.isfinite(weight).all():
        raise ReconstructionError(f"line {line} contains uncovered trace/time samples")
    if np.any(status == np.iinfo(np.int16).min):
        raise ReconstructionError(f"line {line} contains uncovered status traces")
    return {
        "raw_full_normalized": raw,
        "soft_mask_train": mask,
        "status_code": status,
        "label_weight": weight,
    }



def manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_contract(
    dataset_root: Path,
    records: list[WindowRecord],
    summaries: list[LineSummary],
    *,
    dt_ns: float,
) -> None:
    contract = ROOT / "data" / "dataset_contract_v2"
    if not contract.is_dir():
        raise ReconstructionError(f"dataset_contract_v2 is missing: {contract}")

    real_line_rows = [
        {
            "line_id": item.line,
            "line_npz_path": manifest_path(item.line_path),
            "source_path": manifest_path(dataset_root / "windows"),
            "dt_ns": f"{dt_ns:.9g}",
            "trace_count": item.trace_count,
            "sha256": item.line_sha256,
            "approved": "false",
        }
        for item in summaries
    ]
    window_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for item in records:
        with np.load(item.path, allow_pickle=False) as data:
            status = np.asarray(data["status_code"], dtype=np.int16)
        semantics = "positive_or_weak_positive_only" if not np.any(status == 0) else "contains_confirmed_negative"
        window_rows.append(
            {
                "sample_id": item.sample_id,
                "line_id": item.line,
                "window_npz_path": manifest_path(item.path),
                "trace_start": item.start,
                "trace_end": item.end,
                "status_semantics": semantics,
                "height_source": "missing",
                "sha256": sha256_file(item.path),
                "split_group": item.line,
            }
        )
        split_rows.append(
            {
                "sample_id": item.sample_id,
                "group_id": item.line,
                "source_type": "real_window_cache_reconstructed",
                "split": "review_only" if item.line.lower() in {"x1", "linex1"} else "unassigned",
                "holdout_compatible": "false" if item.line.lower() in {"x1", "linex1"} else "true",
                "exclusion_reason": "X1 is review-only" if item.line.lower() in {"x1", "linex1"} else "formal split not assigned",
            }
        )

    write_csv(
        contract / "real_lines.csv",
        ["line_id", "line_npz_path", "source_path", "dt_ns", "trace_count", "sha256", "approved"],
        real_line_rows,
    )
    write_csv(
        contract / "real_windows.csv",
        [
            "sample_id", "line_id", "window_npz_path", "trace_start", "trace_end",
            "status_semantics", "height_source", "sha256", "split_group",
        ],
        window_rows,
    )
    write_csv(
        contract / "split_manifest.csv",
        ["sample_id", "group_id", "source_type", "split", "holdout_compatible", "exclusion_reason"],
        split_rows,
    )

    dataset_manifest_path = contract / "dataset_manifest.json"
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    blockers = [
        item for item in manifest.get("blockers", [])
        if "real line NPZ files" not in item and "canonical window index" not in item
    ]
    reconstruction_blocker = (
        "real full-line arrays are reconstructed exactly from overlapping window caches; "
        "original acquisition-to-window source mapping remains unavailable"
    )
    if reconstruction_blocker not in blockers:
        blockers.insert(0, reconstruction_blocker)
    manifest["blockers"] = blockers
    manifest["formal_training_allowed"] = False
    manifest["real_data_provenance"] = {
        "mode": "exact_overlap_reconstruction_from_window_cache",
        "dataset_root": manifest_path(dataset_root),
        "dt_ns": dt_ns,
        "original_full_line_sources_available": False,
    }
    counts = dict(manifest.get("counts") or {})
    counts["real_lines_registered"] = len(real_line_rows)
    counts["real_windows_registered"] = len(window_rows)
    counts["confirmed_true_negative_windows"] = sum(
        row["status_semantics"] == "contains_confirmed_negative" for row in window_rows
    )
    manifest["counts"] = counts
    dataset_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_dataset_policy(dataset_root: Path, *, dt_ns: float, summaries: list[LineSummary], inventory_path: Path | None) -> None:
    policy_path = dataset_root / "dataset_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
    policy.update(
        {
            "dataset_id": "data_corrected_v1_4_terrain_direction_reconstructed_from_window_cache",
            "training_allowed": False,
            "reason": (
                "Full-line arrays and window_index.csv were reconstructed exactly from overlapping window caches, "
                "but true-negative supervision, original source provenance, unified label semantics, and height metadata remain incomplete."
            ),
            "missing_required_artifacts": [],
            "reconstruction": {
                "mode": "exact_overlap_reconstruction_from_window_cache",
                "dt_ns": dt_ns,
                "line_count": len(summaries),
                "original_full_line_sources_available": False,
                "inventory_path": manifest_path(inventory_path) if inventory_path else None,
            },
            "confirmed_true_negative_traces": int(sum(item.negative_traces for item in summaries)),
        }
    )
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reconstruct_dataset(
    dataset_root: Path,
    *,
    dt_ns: float,
    dt_source: str,
    atol: float = 0.0,
    update_governance: bool = True,
    inventory: dict[str, dict[str, object]] | None = None,
    inventory_path: Path | None = None,
    allow_overwrite_canonical_source: bool = False,
) -> dict[str, object]:
    if dt_ns <= 0:
        raise ReconstructionError(f"dt_ns must be positive, got {dt_ns}")
    dataset_root = dataset_root.resolve()
    windows_dir = dataset_root / "windows"
    lines_dir = dataset_root / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    if not allow_overwrite_canonical_source:
        canonical = []
        for existing in lines_dir.glob("*.npz"):
            try:
                with np.load(existing, allow_pickle=False) as data:
                    if "canonical_source" in data.files and str(np.asarray(data["canonical_source"]).item()) == "original_yingshan_csv":
                        canonical.append(existing.name)
            except Exception:
                continue
        if canonical:
            raise ReconstructionError(
                "refusing to overwrite canonical original-CSV line archives with window-cache reconstruction: "
                + ", ".join(sorted(canonical))
                + "; pass allow_overwrite_canonical_source=True only for an explicit disaster-recovery operation"
            )
    records = parse_windows(windows_dir)
    grouped: dict[str, list[WindowRecord]] = defaultdict(list)
    for item in records:
        grouped[item.line].append(item)

    summaries: list[LineSummary] = []
    index_rows: list[dict[str, object]] = []
    line_manifest: list[dict[str, object]] = []
    for line, line_records in sorted(grouped.items()):
        arrays = reconstruct_line(line_records, atol=atol)
        status = arrays["status_code"]
        metadata = (inventory or {}).get(line, {})
        line_dt_ns = float(metadata.get("dt_ns", dt_ns))
        trace_interval_m = float(metadata.get("trace_interval_m", float("nan")))
        source_member = str(metadata.get("zip_member", ""))
        expected_samples = int(metadata.get("samples", arrays["raw_full_normalized"].shape[0]))
        expected_traces = int(metadata.get("traces", status.size))
        if expected_samples != arrays["raw_full_normalized"].shape[0] or expected_traces != status.size:
            raise ReconstructionError(
                f"inventory shape mismatch for {line}: inventory={expected_samples}x{expected_traces}, "
                f"reconstructed={arrays['raw_full_normalized'].shape[0]}x{status.size}"
            )
        if abs(line_dt_ns - dt_ns) > 1e-9:
            raise ReconstructionError(f"inventory dt mismatch for {line}: {line_dt_ns} vs requested {dt_ns}")
        line_path = lines_dir / f"{line}.npz"
        np.savez_compressed(
            line_path,
            **arrays,
            dt_ns=np.asarray(line_dt_ns, dtype=np.float32),
            trace_interval_m=np.asarray(trace_interval_m, dtype=np.float32),
            split=np.asarray(dataset_level_split(line)),
            line=np.asarray(line),
            reconstruction_source=np.asarray("overlapping_window_cache"),
            reconstruction_dt_source=np.asarray(dt_source),
            reconstruction_overlap_atol=np.asarray(atol, dtype=np.float64),
        )
        summary = LineSummary(
            line=line,
            trace_count=int(status.size),
            sample_count=len(line_records),
            time_samples=int(arrays["raw_full_normalized"].shape[0]),
            strong_traces=int(np.sum(status == 1)),
            weak_traces=int(np.sum(status == 2)),
            negative_traces=int(np.sum(status == 0)),
            line_path=line_path,
            line_sha256=sha256_file(line_path),
            trace_interval_m=trace_interval_m,
            source_member=source_member,
        )
        summaries.append(summary)
        line_manifest.append(
            {
                "line": line,
                "line_npz": manifest_path(line_path),
                "line_sha256": summary.line_sha256,
                "trace_count": summary.trace_count,
                "time_samples": summary.time_samples,
                "window_count": summary.sample_count,
                "strong_traces": summary.strong_traces,
                "weak_traces": summary.weak_traces,
                "negative_traces": summary.negative_traces,
                "dt_ns": line_dt_ns,
                "trace_interval_m": summary.trace_interval_m,
                "source_member": summary.source_member,
            }
        )
        for item in sorted(line_records, key=lambda row: (row.start, row.end)):
            with np.load(item.path, allow_pickle=False) as data:
                st = np.asarray(data["status_code"], dtype=np.int16)
            index_rows.append(
                {
                    "sample_id": item.sample_id,
                    "line": item.line,
                    "start": item.start,
                    "end": item.end,
                    "split": dataset_level_split(item.line),
                    "present": int(np.sum(st == 1)),
                    "weak": int(np.sum(st == 2)),
                    "no_pick": int(np.sum(st == 0)),
                }
            )

    write_csv(
        dataset_root / "window_index.csv",
        ["sample_id", "line", "start", "end", "split", "present", "weak", "no_pick"],
        index_rows,
    )
    reconstruction_manifest = {
        "schema_version": "real_window_cache_reconstruction_v1",
        "dataset_root": manifest_path(dataset_root),
        "source_windows": manifest_path(windows_dir),
        "dt_ns": dt_ns,
        "dt_source": dt_source,
        "inventory_path": manifest_path(inventory_path) if inventory_path else None,
        "overlap_atol": atol,
        "overlap_policy": "all overlapping values must agree within tolerance",
        "original_full_line_sources_available": False,
        "formal_training_allowed": False,
        "lines": line_manifest,
        "window_count": len(records),
    }
    (dataset_root / "reconstruction_manifest.json").write_text(
        json.dumps(reconstruction_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if update_governance:
        update_dataset_policy(dataset_root, dt_ns=dt_ns, summaries=summaries, inventory_path=inventory_path)
        update_contract(dataset_root, records, summaries, dt_ns=dt_ns)
    return reconstruction_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="data_corrected_v1_4_terrain_direction",
        help="Dataset root containing windows/*.npz",
    )
    parser.add_argument("--dt-ns", type=float, default=None, help="Optional explicit time sample interval in ns")
    parser.add_argument("--dt-source", default="", help="Auditable description of the dt source")
    parser.add_argument("--inventory-json", default=str(DEFAULT_INVENTORY.relative_to(ROOT)))
    parser.add_argument("--overlap-atol", type=float, default=0.0)
    parser.add_argument("--no-update-governance", action="store_true")
    parser.add_argument(
        "--allow-overwrite-canonical-source",
        action="store_true",
        help="Disaster-recovery only: permit window-cache reconstruction to replace original-CSV canonical lines.",
    )
    args = parser.parse_args()
    root = Path(args.dataset_root)
    if not root.is_absolute():
        root = ROOT / root
    inventory_path = Path(args.inventory_json)
    if not inventory_path.is_absolute():
        inventory_path = ROOT / inventory_path
    try:
        inventory = load_inventory(inventory_path)
        inventory_dts = {round(float(row["dt_ns"]), 12) for row in inventory.values()}
        if len(inventory_dts) != 1:
            raise ReconstructionError(f"inventory contains multiple dt values: {sorted(inventory_dts)}")
        inventory_dt = float(next(iter(inventory_dts)))
        dt_ns = inventory_dt if args.dt_ns is None else float(args.dt_ns)
        if abs(dt_ns - inventory_dt) > 1e-9:
            raise ReconstructionError(f"requested dt_ns={dt_ns} conflicts with inventory dt_ns={inventory_dt}")
        dt_source = args.dt_source or f"verified_from_{manifest_path(inventory_path)}"
        report = reconstruct_dataset(
            root,
            dt_ns=dt_ns,
            dt_source=dt_source,
            atol=args.overlap_atol,
            update_governance=not args.no_update_governance,
            inventory=inventory,
            inventory_path=inventory_path,
            allow_overwrite_canonical_source=args.allow_overwrite_canonical_source,
        )
    except ReconstructionError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"ok": True, **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
