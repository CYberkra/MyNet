from __future__ import annotations

import argparse
import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class StreamingStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def to_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        var = self.m2 / max(1, self.count - 1)
        return {"count": self.count, "min": self.min_value, "max": self.max_value, "mean": self.mean, "std": math.sqrt(var)}


def _parse_meta_line(line: str) -> tuple[str, float] | None:
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip().lower().replace(" ", "_").replace("(ns)", "ns").replace("(m)", "m")
    value = value.split(",", 1)[0].strip()
    try:
        return key, float(value)
    except Exception:
        return None


def _line_id_from_name(name: str) -> str:
    stem = Path(name).stem
    for token in ["Line", "origin", "(36)"]:
        stem = stem.replace(token, "")
    return stem.strip() or Path(name).stem


def _read_text_lines_from_zip(zf: zipfile.ZipFile, name: str) -> Iterable[str]:
    with zf.open(name) as fp:
        for raw in fp:
            yield raw.decode("utf-8", errors="replace").strip()


def audit_csv_member(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    meta: dict[str, float] = {}
    data_started = False
    numeric_rows = 0
    invalid_rows = 0
    stats = StreamingStats()
    first_trace: dict[str, float] | None = None
    last_trace: dict[str, float] | None = None
    line_id = _line_id_from_name(name)
    samples = 501
    trace_interval = 0.0
    for line in _read_text_lines_from_zip(zf, name):
        if not data_started:
            parsed = _parse_meta_line(line)
            if parsed is not None:
                meta[parsed[0]] = parsed[1]
                samples = int(meta.get("number_of_samples", samples))
                trace_interval = float(meta.get("trace_interval_m", trace_interval))
                continue
            data_started = True
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            invalid_rows += 1
            continue
        try:
            lon = float(parts[0]); lat = float(parts[1]); elev = float(parts[2]); amp = float(parts[3])
            extra = float(parts[4]) if len(parts) > 4 and parts[4] != "" else float("nan")
        except Exception:
            invalid_rows += 1
            continue
        if samples > 0 and numeric_rows % samples == 0:
            trace_payload = {"longitude": lon, "latitude": lat, "elevation_m": elev, "extra_m": extra}
            if first_trace is None:
                first_trace = trace_payload
            last_trace = trace_payload
        stats.add(amp)
        numeric_rows += 1
    samples = int(meta.get("number_of_samples", samples))
    traces_declared = int(meta.get("number_of_traces", 0))
    traces_read = numeric_rows // samples if samples else 0
    expected_rows = traces_declared * samples if traces_declared and samples else None
    time_window = float(meta.get("time_windows_ns", meta.get("time_window_ns", 0.0)))
    trace_interval = float(meta.get("trace_interval_m", trace_interval))
    return {
        "line_id": line_id,
        "zip_member": name,
        "file_size_bytes": zf.getinfo(name).file_size,
        "samples": samples,
        "time_window_ns": time_window,
        "traces_declared": traces_declared,
        "trace_interval_m": trace_interval,
        "expected_numeric_rows": expected_rows,
        "numeric_rows": numeric_rows,
        "invalid_rows": invalid_rows,
        "traces_read": traces_read,
        "shape_samples_x_traces": [samples, traces_read],
        "distance_m": traces_read * trace_interval if trace_interval else None,
        "row_count_matches_declared": expected_rows == numeric_rows if expected_rows is not None else None,
        "amplitude": stats.to_dict(),
        "first_trace": first_trace,
        "last_trace": last_trace,
        "parser_contract": "metadata lines followed by numeric columns: longitude, latitude, elevation_m, amplitude, extra/flight_height",
    }


def list_zip(path: Path, suffixes: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if suffixes and not info.filename.lower().endswith(suffixes):
                continue
            out.append({"name": info.filename, "size_bytes": info.file_size})
    return out


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 营山真实测线数据包格式兼容预审")
    lines.append("")
    lines.append("## 结论")
    csvs = report.get("csv_files", [])
    all_match = all(item.get("row_count_matches_declared") for item in csvs)
    lines.append(f"- 已识别真实测线 CSV：{len(csvs)} 条。")
    lines.append(f"- CSV 基本格式：{'通过' if all_match else '存在行数或声明不一致，需要复核'}。")
    lines.append("- 当前软件的 `core.real_data.read_uavgpr_csv()` 读取合同与这些 CSV 的结构一致：前置元数据 + 5 列数值数据。")
    lines.append("- 本审计不把原始大文件打入软件发布包，只记录数据清单与格式合同；正式接入时应由用户在本机指定数据目录。")
    lines.append("")
    lines.append("## 测线 CSV 清单")
    lines.append("")
    lines.append("| 测线 | samples | traces | time window(ns) | trace interval(m) | 长度估计(m) | 行数匹配 | 幅值范围 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for item in csvs:
        amp = item.get("amplitude", {})
        rng = "--" if amp.get("min") is None else f"{amp.get('min'):.6g} ~ {amp.get('max'):.6g}"
        dist = item.get("distance_m")
        dist_s = "--" if dist is None else f"{dist:.2f}"
        lines.append(
            f"| {item.get('line_id')} | {item.get('samples')} | {item.get('traces_read')} | {item.get('time_window_ns')} | {item.get('trace_interval_m')} | {dist_s} | {item.get('row_count_matches_declared')} | {rng} |"
        )
    lines.append("")
    lines.append("## 剖面 / 钻孔 / DWG 附件清单")
    lines.append("")
    pdfs = report.get("profile_pdfs", [])
    dwgs = report.get("dwg_files", [])
    lines.append(f"- PDF 附件：{len(pdfs)} 个。")
    for item in pdfs:
        lines.append(f"  - `{item['name']}` ({item['size_bytes']} bytes)")
    lines.append(f"- DWG 附件：{len(dwgs)} 个。")
    for item in dwgs:
        lines.append(f"  - `{item['name']}` ({item['size_bytes']} bytes)")
    lines.append("")
    lines.append("## 需要继续人工确认的合同")
    lines.append("")
    lines.append("1. 钻孔柱状图 PDF 与 ZK07-ZK10 的一一对应关系，需要在后续人工/半自动解析后固化成 `configs/boreholes_yingshan.csv`。")
    lines.append("2. 当前附件中显式存在 3、6、7、9 号测线剖面 PDF；L1、X1 的剖面来源需要确认是否在 DWG 或其他图件中。")
    lines.append("3. 真实数据进入 GUI 前，应增加一个项目级 `real_data/` 导入向导，而不是把绝对路径写死到源码或默认工作区。")
    lines.append("4. 基覆界面误差验证需要把成像界面深度与钻孔揭示深度纳入同一坐标/高程基准后再计算，不能只凭截图判断。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Yingshan real UAV-GPR data zip packages without extracting them into the release tree.")
    ap.add_argument("--data-zip", default="/mnt/data/营山测线数据.zip")
    ap.add_argument("--profile-zip", default="/mnt/data/营山测线剖面图、钻孔柱状图.zip")
    ap.add_argument("--dwg-zip", default="/mnt/data/滑坡剖面图等.zip")
    ap.add_argument("--out-json", default="docs/real_data/YINGSHAN_REAL_DATA_AUDIT.json")
    ap.add_argument("--out-md", default="docs/real_data/YINGSHAN_REAL_DATA_AUDIT.md")
    ap.add_argument("--inventory-yaml", default="configs/yingshan_real_data_inventory.yaml")
    ns = ap.parse_args()

    data_zip = Path(ns.data_zip)
    profile_zip = Path(ns.profile_zip)
    dwg_zip = Path(ns.dwg_zip)
    csv_reports: list[dict[str, Any]] = []
    if data_zip.exists():
        with zipfile.ZipFile(data_zip) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".csv"):
                    csv_reports.append(audit_csv_member(zf, name))

    report = {
        "ok": bool(csv_reports) and all(item.get("row_count_matches_declared") for item in csv_reports),
        "data_zip": str(data_zip),
        "profile_zip": str(profile_zip),
        "dwg_zip": str(dwg_zip),
        "csv_files": csv_reports,
        "profile_pdfs": list_zip(profile_zip, (".pdf",)),
        "dwg_files": list_zip(dwg_zip, (".dwg",)),
        "notes": [
            "This audit reads package metadata and streamed CSV values; it does not permanently extract raw data into the release package.",
            "Shape convention is samples x traces, consistent with core.real_data.",
        ],
    }

    out_json = Path(ns.out_json)
    out_md = Path(ns.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")

    inventory = {
        "name": "yingshan_real_uavgpr_inventory",
        "schema_version": "1",
        "raw_data_package": data_zip.name,
        "profile_package": profile_zip.name,
        "dwg_package": dwg_zip.name,
        "lines": [
            {
                "line_id": item["line_id"],
                "zip_member": item["zip_member"],
                "samples": item["samples"],
                "traces": item["traces_read"],
                "time_window_ns": item["time_window_ns"],
                "trace_interval_m": item["trace_interval_m"],
                "distance_m": item["distance_m"],
            }
            for item in csv_reports
        ],
        "borehole_contract_status": "pending_manual_mapping_ZK07_ZK10",
    }
    inv_path = Path(ns.inventory_yaml)
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        inv_path.write_text(yaml.safe_dump(inventory, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except Exception:
        inv_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
