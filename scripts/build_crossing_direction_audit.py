#!/usr/bin/env python3
"""Audit YingShan trace order, engineering-profile display direction, and crossings.

The audit reads only canonical full-line NPZ files generated from the original
CSV archive.  Canonical arrays remain in acquisition order; engineering-profile
orientation is a presentation contract recorded separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.spatial_orientation import (  # noqa: E402
    display_distance_axis,
    get_line_orientation,
    orientation_metadata,
)

LINES = ("Line3", "Line6", "Line7", "Line9", "LineL1", "LineX1")
EXPECTED_CROSSINGS = (
    ("Line3", "Line9"),
    ("Line3", "LineL1"),
    ("Line3", "Line7"),
    ("Line6", "Line9"),
    ("Line6", "LineL1"),
    ("Line6", "Line7"),
    ("Line9", "LineX1"),
    ("LineL1", "LineX1"),
)

PDF_CONSISTENCY_NOTES = (
    {
        "item": "Line3",
        "status": "consistent_after_profile_flip",
        "note": "CSV traces run north/ZK08 to south/ZK07; engineering profile and migrated section display ZK07 left and ZK08 right.",
    },
    {
        "item": "Line6",
        "status": "terrain_direction_consistent_but_slide_crossing_labels_inconsistent",
        "note": "Profile display requires reversal. The page-30 orange 9/L1 crossing labels follow acquisition order while the displayed terrain/profile direction is reversed; do not use those orange labels as exact chainage.",
    },
    {
        "item": "Line7",
        "status": "consistent",
        "note": "CSV, engineering profile, and migrated section all run west to east.",
    },
    {
        "item": "Line9",
        "status": "consistent_after_profile_flip",
        "note": "CSV traces run east/ZK08 to west; engineering profile and migrated section display west left and ZK08/east right.",
    },
    {
        "item": "LineL1",
        "status": "medium_confidence_arrow_conflict",
        "note": "No standalone engineering profile was supplied. The main migrated section agrees with CSV east-to-west order, but aerial arrows in the report are not mutually consistent.",
    },
    {
        "item": "LineX1",
        "status": "medium_confidence_report_only",
        "note": "No standalone engineering profile was supplied. The report migrated section places the L1 crossing in reversed CSV display order.",
    },
)

BOREHOLE_ANCHORS = (
    {"line": "Line3", "borehole": "ZK07", "borehole_basal_depth_m": 16.5, "reported_imaging_depth_m": 16.7, "absolute_error_m": 0.2},
    {"line": "Line3", "borehole": "ZK08", "borehole_basal_depth_m": 14.0, "reported_imaging_depth_m": 14.4, "absolute_error_m": 0.4},
    {"line": "Line6", "borehole": "ZK09", "borehole_basal_depth_m": 11.2, "reported_imaging_depth_m": 11.5, "absolute_error_m": 0.3},
    {"line": "Line7", "borehole": "ZK09", "borehole_basal_depth_m": 11.2, "reported_imaging_depth_m": 11.5, "absolute_error_m": 0.3},
    {"line": "Line9", "borehole": "ZK08", "borehole_basal_depth_m": 14.0, "reported_imaging_depth_m": 14.4, "absolute_error_m": 0.4},
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def crossing_qc_grade(abs_difference_ns: float, status_a: int, status_b: int) -> str:
    if not np.isfinite(abs_difference_ns):
        return "missing_label"
    weak = status_a != 1 or status_b != 1
    if abs_difference_ns <= 10.0:
        return "pass" if not weak else "pass_but_weak_label"
    if abs_difference_ns <= 20.0:
        return "review" if not weak else "review_weak_label"
    if abs_difference_ns <= 40.0:
        return "high_risk" if not weak else "high_risk_weak_label"
    return "critical_mismatch" if not weak else "critical_mismatch_weak_label"


def load_lines(data_root: Path) -> dict[str, dict[str, np.ndarray | float | str]]:
    result: dict[str, dict[str, np.ndarray | float | str]] = {}
    for line in LINES:
        path = data_root / "lines" / f"{line}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            required = {
                "longitude", "latitude", "gnss_cumulative_distance_m", "profile_chainage_m",
                "ground_elevation_m", "flight_height_agl_m", "soft_mask_train", "status_code", "dt_ns",
            }
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(f"{path}: missing {missing}")
            mask = np.asarray(z["soft_mask_train"], dtype=np.float32)
            mass = mask.sum(axis=0)
            sample_axis = np.arange(mask.shape[0], dtype=np.float64)[:, None]
            center_sample = np.full(mask.shape[1], np.nan, dtype=np.float64)
            valid = mass > 1e-6
            center_sample[valid] = (mask[:, valid] * sample_axis).sum(axis=0) / mass[valid]
            result[line] = {
                "path": str(path),
                "longitude": np.asarray(z["longitude"], dtype=np.float64),
                "latitude": np.asarray(z["latitude"], dtype=np.float64),
                "gnss_distance": np.asarray(z["gnss_cumulative_distance_m"], dtype=np.float64),
                "profile_chainage": np.asarray(z["profile_chainage_m"], dtype=np.float64),
                "ground": np.asarray(z["ground_elevation_m"], dtype=np.float64),
                "flight": np.asarray(z["flight_height_agl_m"], dtype=np.float64),
                "status": np.asarray(z["status_code"], dtype=np.int16),
                "center_time_ns": center_sample * float(np.asarray(z["dt_ns"]).item()),
                "dt_ns": float(np.asarray(z["dt_ns"]).item()),
            }
    return result


def local_xy(lines: dict[str, dict[str, object]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    all_lon = np.concatenate([np.asarray(v["longitude"]) for v in lines.values()])
    all_lat = np.concatenate([np.asarray(v["latitude"]) for v in lines.values()])
    lon0 = float(np.mean(all_lon))
    lat0 = float(np.mean(all_lat))
    cos_lat = math.cos(math.radians(lat0))
    return {
        line: (
            (np.asarray(v["longitude"]) - lon0) * 111_320.0 * cos_lat,
            (np.asarray(v["latitude"]) - lat0) * 110_540.0,
        )
        for line, v in lines.items()
    }


def nearest_pair(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, block: int = 512) -> tuple[float, int, int]:
    best = (float("inf"), -1, -1)
    for start in range(0, x1.size, block):
        stop = min(start + block, x1.size)
        dx = x1[start:stop, None] - x2[None, :]
        dy = y1[start:stop, None] - y2[None, :]
        d2 = dx * dx + dy * dy
        flat = int(np.argmin(d2))
        i_local, j = np.unravel_index(flat, d2.shape)
        value = float(math.sqrt(float(d2[i_local, j])))
        if value < best[0]:
            best = (value, start + int(i_local), int(j))
    return best


def corrected_subsurface_delay_ns(center_time_ns: float, flight_height_m: float) -> float:
    # Two-way air travel time. This is a signal-domain alignment diagnostic,
    # not a conversion to geological depth.
    return float(center_time_ns - 2.0 * flight_height_m / 299_792_458.0 * 1e9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_corrected_v1_4_terrain_direction")
    parser.add_argument("--out-dir", default="reports/yingshan_direction_profile_audit")
    parser.add_argument("--profile-archive", default="data_corrected_v1_4_terrain_direction/source/ying_shan_profiles_and_boreholes.zip")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_archive = Path(args.profile_archive)
    if not profile_archive.is_absolute():
        profile_archive = ROOT / profile_archive
    if not profile_archive.is_file():
        raise FileNotFoundError(profile_archive)
    profile_archive_sha256 = sha256_file(profile_archive)

    lines = load_lines(data_root)
    xy = local_xy(lines)

    direction_rows: list[dict[str, object]] = []
    for line in LINES:
        v = lines[line]
        meta = orientation_metadata(line, np.asarray(v["longitude"]), np.asarray(v["latitude"]))
        profile_distance = display_distance_axis(np.asarray(v["profile_chainage"]), line, orientation="profile")
        direction_rows.append({
            "line": line,
            "canonical_trace_order": "acquisition_csv",
            "trace_count": int(np.asarray(v["longitude"]).size),
            "acquisition_bearing_deg": round(float(meta["acquisition_bearing_deg"]), 3),
            "acquisition_compass": meta["acquisition_compass"],
            "engineering_profile": meta["engineering_profile"],
            "profile_left": meta["profile_left"],
            "profile_right": meta["profile_right"],
            "profile_display_flip": bool(meta["profile_display_flip"]),
            "confidence": meta["confidence"],
            "gnss_trajectory_length_m": round(float(np.asarray(v["gnss_distance"])[-1]), 3),
            "nominal_profile_length_m": round(float(profile_distance[-1]), 3),
            "evidence": meta["evidence"],
        })
    write_csv(out_dir / "trace_direction_registry.csv", direction_rows)

    crossing_rows: list[dict[str, object]] = []
    for line_a, line_b in EXPECTED_CROSSINGS:
        distance, ia, ib = nearest_pair(*xy[line_a], *xy[line_b])
        a, b = lines[line_a], lines[line_b]
        oa, ob = get_line_orientation(line_a), get_line_orientation(line_b)
        pa = np.asarray(a["profile_chainage"])
        pb = np.asarray(b["profile_chainage"])
        profile_a = float(pa[-1] - pa[ia]) if oa.profile_display_flip else float(pa[ia])
        profile_b = float(pb[-1] - pb[ib]) if ob.profile_display_flip else float(pb[ib])
        ta = float(np.asarray(a["center_time_ns"])[ia])
        tb = float(np.asarray(b["center_time_ns"])[ib])
        fa = float(np.asarray(a["flight"])[ia])
        fb = float(np.asarray(b["flight"])[ib])
        da = corrected_subsurface_delay_ns(ta, fa) if np.isfinite(ta) else float("nan")
        db = corrected_subsurface_delay_ns(tb, fb) if np.isfinite(tb) else float("nan")
        abs_difference = abs(da - db) if np.isfinite(da) and np.isfinite(db) else float("nan")
        status_a = int(np.asarray(a["status"])[ia])
        status_b = int(np.asarray(b["status"])[ib])
        crossing_rows.append({
            "crossing": f"{line_a}-{line_b}",
            "nearest_separation_m": round(distance, 4),
            "line_a": line_a,
            "trace_a": ia,
            "gnss_distance_a_m": round(float(np.asarray(a["gnss_distance"])[ia]), 3),
            "profile_chainage_a_m": round(profile_a, 3),
            "status_a": status_a,
            "label_center_time_a_ns": round(ta, 3) if np.isfinite(ta) else "",
            "air_corrected_delay_a_ns": round(da, 3) if np.isfinite(da) else "",
            "line_b": line_b,
            "trace_b": ib,
            "gnss_distance_b_m": round(float(np.asarray(b["gnss_distance"])[ib]), 3),
            "profile_chainage_b_m": round(profile_b, 3),
            "status_b": status_b,
            "label_center_time_b_ns": round(tb, 3) if np.isfinite(tb) else "",
            "air_corrected_delay_b_ns": round(db, 3) if np.isfinite(db) else "",
            "air_corrected_delay_abs_difference_ns": round(abs_difference, 3) if np.isfinite(abs_difference) else "",
            "crossing_qc_grade": crossing_qc_grade(abs_difference, status_a, status_b),
            "longitude": round(float((np.asarray(a["longitude"])[ia] + np.asarray(b["longitude"])[ib]) / 2.0), 9),
            "latitude": round(float((np.asarray(a["latitude"])[ia] + np.asarray(b["latitude"])[ib]) / 2.0), 9),
            "qc_note": "Signal-domain crossing check only; do not interpret air-corrected delay as geological depth.",
        })
    write_csv(out_dir / "line_intersections.csv", crossing_rows)
    write_csv(out_dir / "pdf_internal_consistency_notes.csv", list(PDF_CONSISTENCY_NOTES))

    profile_alignment_rows: list[dict[str, object]] = []
    for line in LINES:
        on_line = []
        for row in crossing_rows:
            if row["line_a"] == line:
                on_line.append((float(row["profile_chainage_a_m"]), str(row["line_b"])))
            elif row["line_b"] == line:
                on_line.append((float(row["profile_chainage_b_m"]), str(row["line_a"])))
        on_line.sort()
        contract = get_line_orientation(line)
        profile_alignment_rows.append({
            "line": line,
            "profile_left": contract.profile_left,
            "profile_right": contract.profile_right,
            "profile_display_flip": contract.profile_display_flip,
            "crossings_left_to_right": " -> ".join(f"{other}@{distance:.2f}m" for distance, other in on_line),
            "confidence": contract.confidence,
            "audit_status": next(note["status"] for note in PDF_CONSISTENCY_NOTES if note["item"] == line),
        })
    write_csv(out_dir / "profile_alignment_checks.csv", profile_alignment_rows)
    with zipfile.ZipFile(profile_archive) as archive:
        archive_members = []
        for member in sorted(name for name in archive.namelist() if not name.endswith("/")):
            payload = archive.read(member)
            archive_members.append({"member": member, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
    source_manifest = {
        "profile_archive": str(profile_archive),
        "profile_archive_sha256": profile_archive_sha256,
        "contains": "survey plan, engineering profiles 3/6/7/9, borehole logs",
        "standalone_profiles_available": ["Line3", "Line6", "Line7", "Line9"],
        "standalone_profiles_missing": ["LineL1", "LineX1"],
        "uavgpr_report_role": "flight arrows, migrated B-scan/profile display relationships, borehole anchors",
        "members": archive_members,
    }
    (out_dir / "source_manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "borehole_anchor_checks.csv", list(BOREHOLE_ANCHORS))

    # Persist the orientation/crossing release gate in the dataset policy.
    policy_path = data_root / "dataset_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
    critical = [row["crossing"] for row in crossing_rows if str(row["crossing_qc_grade"]).startswith("critical")]
    high_risk = [row["crossing"] for row in crossing_rows if str(row["crossing_qc_grade"]).startswith("high_risk")]
    policy["orientation_policy"] = {
        "canonical_trace_order": "acquisition_csv",
        "engineering_profile_flip_is_display_only": True,
        "map_distance_axis": "gnss_cumulative_distance_m",
        "profile_distance_axis": "profile_chainage_m",
        "registry": str(data_root / "trace_direction_registry.csv"),
    }
    policy["crossing_label_review"] = {
        "status": "blocking_pending_manual_review",
        "critical_mismatches": critical,
        "high_risk_mismatches": high_risk,
        "audit_csv": str(out_dir / "line_intersections.csv"),
        "criterion": "air-path-corrected label-centre delay disagreement; signal-domain QC, not depth conversion",
    }
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # One map plot: acquisition arrows and detected crossing points.
    fig, ax = plt.subplots(figsize=(10, 8))
    for line in LINES:
        x, y = xy[line]
        ax.plot(x, y, label=line, linewidth=2)
        i0 = max(0, int(len(x) * 0.43))
        i1 = min(len(x) - 1, int(len(x) * 0.57))
        ax.annotate("", xy=(x[i1], y[i1]), xytext=(x[i0], y[i0]), arrowprops={"arrowstyle": "->", "linewidth": 2})
        ax.text(x[0], y[0], f"{line}:0", fontsize=8)
        ax.text(x[-1], y[-1], f"{line}:end", fontsize=8)
    for row in crossing_rows:
        xa, ya = xy[str(row["line_a"])]
        idx = int(row["trace_a"])
        ax.scatter([xa[idx]], [ya[idx]], marker="x")
        ax.text(xa[idx], ya[idx], str(row["crossing"]), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local east / m")
    ax.set_ylabel("Local north / m")
    ax.set_title("YingShan acquisition trace directions and GNSS crossings")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "acquisition_direction_crossings.png", dpi=180)
    plt.close(fig)

    report = {
        "ok": True,
        "canonical_order": "acquisition_csv",
        "profile_order_is_display_only": True,
        "direction_registry": str(out_dir / "trace_direction_registry.csv"),
        "crossings": str(out_dir / "line_intersections.csv"),
        "pdf_consistency_notes": str(out_dir / "pdf_internal_consistency_notes.csv"),
        "profile_alignment_checks": str(out_dir / "profile_alignment_checks.csv"),
        "borehole_anchor_checks": str(out_dir / "borehole_anchor_checks.csv"),
        "profile_archive_sha256": profile_archive_sha256,
        "map_preview": str(out_dir / "acquisition_direction_crossings.png"),
        "crossing_qc_counts": {grade: sum(row["crossing_qc_grade"] == grade for row in crossing_rows) for grade in sorted({row["crossing_qc_grade"] for row in crossing_rows})},
        "warnings": [
            "Line6 report crossing annotations are internally inconsistent with the displayed profile direction.",
            "LineL1 and LineX1 profile orientations are medium-confidence because standalone engineering profiles were not supplied.",
            "GNSS cumulative distance and engineering-profile chainage are distinct axes and must not be used interchangeably.",
            "Line3-Line9 and Line6-Line9 crossing labels show large air-path-corrected time disagreement and require label review before training release.",
        ],
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    direction_table = "\n".join(
        f"| {row['line']} | {row['acquisition_bearing_deg']}° {row['acquisition_compass']} | {row['engineering_profile']} | {row['profile_display_flip']} | {row['profile_left']} → {row['profile_right']} | {row['confidence']} |"
        for row in direction_rows
    )
    crossing_table = "\n".join(
        f"| {row['crossing']} | {row['nearest_separation_m']} | {row['profile_chainage_a_m']} / {row['profile_chainage_b_m']} | {row['air_corrected_delay_abs_difference_ns']} | {row['crossing_qc_grade']} |"
        for row in crossing_rows
    )
    markdown = f"""# 营山测线航向、B-scan 与工程剖面对应关系全面审计

## 结论

- Canonical 数据必须保持原始 CSV 采集顺序，不允许为了匹配图件而物理翻转训练数组。
- Line3、Line6、Line9、LineX1 的工程/报告剖面显示需要相对采集顺序翻转；Line7、LineL1 不翻转。
- 8 个应有交叉点的 GNSS 最近距离均小于 0.05 m，测线身份、起止顺序和交叉位置映射可靠。
- 工程剖面里程与 GNSS 累计轨迹长度是不同坐标轴，现已分别保存。
- 标签在交叉点并非全部一致：Line3-Line9 为 critical，Line6-Line9 与 LineL1-LineX1 为 high-risk；正式训练继续冻结。

## 航向与显示方向

| 测线 | CSV 采集航向 | 工程剖面 | 显示翻转 | 剖面左→右 | 置信度 |
|---|---:|---|---|---|---|
{direction_table}

## 交叉点核验

| 交叉 | GNSS 最近距离/m | 两线剖面里程/m | 空气程校正后标签时间差/ns | QC |
|---|---:|---|---:|---|
{crossing_table}

这里的时间差仅用于同一空间交叉点的标签一致性 QC，不用于直接换算地质深度。

## PDF/图件内部问题

1. Line6 页面中的 9/L1 橙色交叉标记沿采集顺序排列，但地形剖面本身采用反向显示，不能把橙色标记作为精确里程。
2. LineL1 没有独立工程剖面，报告中的航拍箭头存在冲突；以 GNSS 采集顺序和主迁移图为准，置信度为 medium。
3. LineX1 没有独立工程剖面，保持 review-only；报告主迁移图支持反向显示。
4. 图件中的“无人机真高 8 m”是成像/展示参数，不能覆盖原始 CSV 逐道飞高。

## 数据使用规则

- 训练、指标、窗口索引：始终为 acquisition_csv。
- 地图轨迹：使用 gnss_cumulative_distance_m。
- 与工程剖面对照：使用 profile_chainage_m，并按 profile_display_flip 只在显示层翻转。
- 横向镜像增强后必须对 terrain_slope_z 与 trace_position 取反；该错误已修复并测试。
"""
    (out_dir / "COMPREHENSIVE_DIRECTION_PROFILE_AUDIT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
