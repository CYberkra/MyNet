"""Fail-fast experiment and dataset contracts for PGDA-CSNet.

The contract layer prevents nominal mixed runs from silently becoming real-only,
review-only lines from entering formal splits, and Line9-conditioned simulation
from contaminating a formal Line9 holdout. It validates evidence; it never
fabricates missing data or promotes cases from automatic QC alone.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FORMAL_RUN_TYPES = frozenset({"lolo_eval", "holdout_eval", "baseline_eval", "paper_eval", "paper_train"})
NO_VALIDATION_RUN_TYPES = frozenset({"pretrain_sim_only", "debug"})
DEFAULT_REVIEW_ONLY_LINES = frozenset({"LineX1", "X1"})
REQUIRED_WINDOW_INDEX_COLUMNS = frozenset({"sample_id", "line", "start", "end", "present", "weak", "no_pick"})


class ContractError(RuntimeError):
    """Raised when a declared experiment cannot be executed faithfully."""


@dataclass(frozen=True)
class DatasetIndexSummary:
    root: Path
    index_path: Path
    windows_dir: Path
    row_count: int
    lines: tuple[str, ...]
    sample_ids: tuple[str, ...]
    fieldnames: tuple[str, ...]


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    return value


def _normalise_line(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "").replace("-", "")


def _normalise_lines(values: Iterable[Any] | None) -> set[str]:
    return {str(value).strip() for value in (values or []) if str(value).strip()}


def _normalised_line_set(values: Iterable[Any] | None) -> set[str]:
    return {_normalise_line(value) for value in (values or []) if str(value).strip()}




def load_split_policy(cfg: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    value = str(cfg.get("paper_split_file", "")).strip()
    if not value:
        return None, None
    path = resolve_repo_path(repo_root, value)
    if not path.is_file():
        raise ContractError(f"paper_split_file does not exist: {path}")
    policy = load_json_file(path)
    if "review_only_lines" not in policy:
        raise ContractError(f"paper split policy lacks review_only_lines: {path}")
    return policy, path

def validate_experiment_config(
    cfg: dict[str, Any],
    repo_root: Path,
    *,
    config_path: Path | None = None,
    require_run_type: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    run_type = str(cfg.get("run_type", "")).strip().lower()
    config_status = str(cfg.get("config_status", "development")).strip().lower()
    if config_status in {"blocked", "invalid", "deprecated_invalid"} and not bool(cfg.get("allow_blocked_config", False)):
        errors.append(
            f"config_status={config_status}; blocked_reasons={cfg.get('blocked_reasons', cfg.get('paper_block_reason', ''))}"
        )
    train = _normalise_lines(cfg.get("train_lines"))
    val = _normalise_lines(cfg.get("val_lines"))
    test = _normalise_lines(cfg.get("test_lines"))
    review = _normalise_lines(cfg.get("review_lines"))

    if cfg.get("enabled") is False:
        errors.append("configuration is explicitly disabled by the audit freeze")
    if require_run_type and not run_type:
        errors.append("run_type is required; strict split checks must not depend on an omitted field")

    split_policy = None
    split_policy_path = None
    try:
        split_policy, split_policy_path = load_split_policy(cfg, repo_root)
    except ContractError as exc:
        errors.append(str(exc))
    review_only_values = set(DEFAULT_REVIEW_ONLY_LINES)
    if split_policy:
        review_only_values.update(_normalise_lines(split_policy.get("review_only_lines")))
    review_only_norm = {_normalise_line(value) for value in review_only_values}
    misplaced = [line for line in train | val | test if _normalise_line(line) in review_only_norm]
    if misplaced:
        errors.append(f"review-only lines appear in train/val/test: {sorted(misplaced)}")

    if train & test:
        errors.append(f"train/test line overlap: {sorted(train & test)}")
    if run_type in FORMAL_RUN_TYPES:
        if not val:
            errors.append(f"val_lines is empty for formal run_type={run_type}")
        if val & test:
            errors.append(f"validation/test line overlap: {sorted(val & test)}")
        if train & val:
            errors.append(f"train/validation line overlap: {sorted(train & val)}")
    elif not val and run_type not in NO_VALIDATION_RUN_TYPES and not bool(cfg.get("allow_train_loss_monitor", False)):
        warnings.append("val_lines is empty; best checkpoint would monitor train loss")

    if split_policy:
        allowed_main = _normalise_lines(split_policy.get("main_measured_lines"))
        measured = {line for line in train | val | test if _normalise_line(line).startswith("line")}
        unknown = measured - allowed_main - review_only_values
        if unknown:
            errors.append(f"lines are not declared by paper split policy: {sorted(unknown)}")
        inner = split_policy.get("inner_validation_by_holdout") or {}
        if run_type in FORMAL_RUN_TYPES and len(test) == 1:
            heldout = next(iter(test))
            expected = inner.get(heldout)
            if expected and val != {str(expected)}:
                errors.append(
                    f"formal holdout {heldout} must use inner validation line {expected} according to {split_policy_path}"
                )

    sim_ratio = float(cfg.get("sim_batch_ratio", 0.0) or 0.0)
    if not 0.0 <= sim_ratio <= 1.0:
        errors.append(f"sim_batch_ratio must be in [0,1], got {sim_ratio}")
    if sim_ratio > 0 and not str(cfg.get("sim_data_root", "")).strip():
        errors.append("sim_batch_ratio > 0 requires sim_data_root")

    arch = str(cfg.get("model_arch", "")).strip().lower()
    if arch in {"aeropath_ssd", "aeropath", "v3_aeropath_ssd"}:
        backend = str(cfg.get("ssm_impl", "official_mamba2")).strip().lower()
        if backend not in {"official_mamba2", "ssm_lite"}:
            errors.append(f"AeroPath-SSD requires ssm_impl=official_mamba2 or ssm_lite, got {backend!r}")
        if run_type in FORMAL_RUN_TYPES and backend != "official_mamba2":
            errors.append("formal AeroPath-SSD runs require ssm_impl=official_mamba2; ssm_lite is debug-only")
        if run_type not in NO_VALIDATION_RUN_TYPES and not bool(cfg.get("aeropath_enable_structured_loss", True)):
            errors.append("AeroPath-SSD must enable its structured-path loss outside debug runs")
        if run_type in FORMAL_RUN_TYPES and float((cfg.get("loss", {}) or {}).get("aeropath_path_nll_weight", 1.0) or 0.0) <= 0:
            errors.append("formal AeroPath-SSD runs require loss.aeropath_path_nll_weight > 0")

    loss_cfg = cfg.get("loss", {}) or {}
    arrival_weight = max(
        float(loss_cfg.get("arrival_prior_weight", 0.0) or 0.0),
        float(loss_cfg.get("arrival_time_prior_weight", 0.0) or 0.0),
    )
    if arrival_weight > 0:
        policy = str(cfg.get("arrival_prior_missing_height_policy", "")).strip().lower()
        if policy not in {"error", "skip", "default"}:
            errors.append("arrival prior requires explicit arrival_prior_missing_height_policy")
        if policy == "default":
            try:
                if float(cfg.get("default_altitude_m", 0.0)) <= 0:
                    errors.append("default height policy requires positive default_altitude_m")
            except Exception:
                errors.append("default_altitude_m is invalid")
        elif "default_altitude_m" in cfg:
            warnings.append("default_altitude_m is ignored unless missing-height policy is 'default'")

    audit = {
        "config_path": str(config_path) if config_path else "",
        "run_type": run_type or "unspecified",
        "paper_split_file": str(split_policy_path) if split_policy_path else "",
        "train_lines": sorted(train),
        "val_lines": sorted(val),
        "test_lines": sorted(test),
        "review_lines": sorted(review),
        "warnings": warnings,
        "errors": errors,
    }
    if errors:
        raise ContractError("CONFIG_CONTRACT failed: " + json.dumps(audit, ensure_ascii=False))
    return audit


def resolve_window_npz(root: Path, row: dict[str, str]) -> Path:
    relative = str(row.get("path") or row.get("npz_path") or row.get("file") or "").strip()
    if relative:
        path = Path(relative)
        return path if path.is_absolute() else root / path
    return root / "windows" / f"{row['sample_id']}.npz"


def inspect_window_dataset(
    root: Path,
    *,
    required_lines: Iterable[str] | None = None,
    require_windows: bool = True,
) -> DatasetIndexSummary:
    root = Path(root)
    if not root.is_dir():
        raise ContractError(f"dataset root does not exist or is not a directory: {root}")
    index_path = root / "window_index.csv"
    windows_dir = root / "windows"
    if not index_path.is_file():
        raise ContractError(f"dataset is missing required window_index.csv: {index_path}")
    if require_windows and not windows_dir.is_dir():
        raise ContractError(f"dataset is missing required windows directory: {windows_dir}")

    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = REQUIRED_WINDOW_INDEX_COLUMNS - set(fieldnames)
        if missing:
            raise ContractError(f"window_index.csv is missing required columns {sorted(missing)}: {index_path}")
        rows = list(reader)
    if not rows:
        raise ContractError(f"window_index.csv contains no samples: {index_path}")

    requested = _normalise_lines(required_lines)
    selected = [row for row in rows if not requested or row.get("line") in requested]
    if requested and not selected:
        raise ContractError(f"window_index.csv contains no rows for requested lines {sorted(requested)}: {index_path}")

    missing_npz: list[str] = []
    sample_ids: list[str] = []
    lines: set[str] = set()
    seen: set[str] = set()
    for row in selected:
        sample_id = str(row.get("sample_id", "")).strip()
        line = str(row.get("line", "")).strip()
        if not sample_id or not line:
            raise ContractError(f"window_index.csv contains blank sample_id/line: {index_path}")
        if sample_id in seen:
            raise ContractError(f"window_index.csv contains duplicate sample_id={sample_id}: {index_path}")
        seen.add(sample_id)
        try:
            start, end = int(row["start"]), int(row["end"])
            if start < 0 or end < start:
                raise ValueError
        except Exception as exc:
            raise ContractError(f"invalid trace range for {sample_id}: start={row.get('start')} end={row.get('end')}") from exc
        sample_ids.append(sample_id)
        lines.add(line)
        if require_windows and not resolve_window_npz(root, row).is_file():
            missing_npz.append(sample_id)
            if len(missing_npz) >= 20:
                break
    if missing_npz:
        raise ContractError(f"window_index.csv references missing NPZ files: {missing_npz}")

    return DatasetIndexSummary(
        root=root,
        index_path=index_path,
        windows_dir=windows_dir,
        row_count=len(selected),
        lines=tuple(sorted(lines)),
        sample_ids=tuple(sample_ids),
        fieldnames=fieldnames,
    )


def inspect_full_line_dataset(root: Path, required_lines: Iterable[str]) -> dict[str, Any]:
    """Validate full-line files required for formal validation or testing.

    Window caches are sufficient for training batches, but a formal run also
    needs canonical full lines for stitching and trace-level evaluation.
    """
    root = Path(root)
    requested = sorted(_normalise_lines(required_lines))
    if not requested:
        return {"root": str(root), "lines": {}}

    lines_dir = root / "lines"
    if not lines_dir.is_dir():
        raise ContractError(f"dataset is missing required full-line directory: {lines_dir}")

    required_arrays = {"raw_full_normalized", "soft_mask_train", "status_code", "label_weight", "dt_ns"}
    line_facts: dict[str, dict[str, Any]] = {}
    for line in requested:
        path = lines_dir / f"{line}.npz"
        if not path.is_file():
            raise ContractError(f"dataset is missing required full-line NPZ for {line}: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                missing = required_arrays - set(archive.files)
                if missing:
                    raise ContractError(f"full-line NPZ for {line} is missing arrays {sorted(missing)}: {path}")
                raw_shape = tuple(int(value) for value in archive["raw_full_normalized"].shape)
                mask_shape = tuple(int(value) for value in archive["soft_mask_train"].shape)
                status_shape = tuple(int(value) for value in archive["status_code"].shape)
                weight_shape = tuple(int(value) for value in archive["label_weight"].shape)
                if len(raw_shape) != 2 or mask_shape != raw_shape:
                    raise ContractError(f"full-line raw/mask shapes are incompatible for {line}: {raw_shape} vs {mask_shape}")
                if status_shape != (raw_shape[1],) or weight_shape != (raw_shape[1],):
                    raise ContractError(f"full-line trace metadata shapes are incompatible for {line}: {path}")
                dt_ns = float(np.asarray(archive["dt_ns"]).item())
                if not np.isfinite(dt_ns) or dt_ns <= 0:
                    raise ContractError(f"full-line dt_ns must be positive for {line}: {path}")
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"failed to read full-line NPZ for {line}: {path}: {exc}") from exc
        line_facts[line] = {"path": str(path), "shape": raw_shape, "dt_ns": dt_ns}

    return {"root": str(root), "lines_dir": str(lines_dir), "lines": line_facts}


def load_dataset_usage_policy(root: Path) -> dict[str, Any] | None:
    root = Path(root)
    for name in ("dataset_policy.json", "DATA_USAGE_POLICY.json"):
        path = root / name
        if path.is_file():
            policy = load_json_file(path)
            policy["_path"] = str(path)
            return policy
    return None


def enforce_simulation_holdout_policy(cfg: dict[str, Any], sim_root: Path) -> dict[str, Any] | None:
    policy = load_dataset_usage_policy(sim_root)
    tests = _normalised_line_set(cfg.get("test_lines"))
    run_type = str(cfg.get("run_type", "")).strip().lower()
    conditioned = _normalised_line_set((policy or {}).get("conditioned_on_lines"))
    if (policy or {}).get("line9_conditioned") is True:
        conditioned.add("line9")
    if "05accepteddataset" in _normalise_line(str(sim_root)):
        conditioned.add("line9")
    overlap = conditioned & tests
    if overlap and run_type in FORMAL_RUN_TYPES:
        raise ContractError(
            f"simulation dataset {sim_root} is conditioned on formal test line(s) {sorted(overlap)}"
        )
    allowed = (policy or {}).get("training_allowed", (policy or {}).get("train_allowed", True))
    if allowed is False:
        raise ContractError(f"simulation dataset policy forbids training use: {sim_root}")
    return policy
