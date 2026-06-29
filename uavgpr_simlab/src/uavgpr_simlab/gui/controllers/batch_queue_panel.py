from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem

from uavgpr_simlab.core.run_dashboard import DatasetRunDashboard, DatasetRunItem, format_duration_seconds
from uavgpr_simlab.gui.easy_ui import _friendly_variant_name


def _status_rank(status: str) -> int:
    return {
        "running": 0,
        "stale_running": 1,
        "failed": 2,
        "pending": 3,
        "done": 4,
        "skipped": 5,
    }.get(status, 9)


def _status_label(status: str) -> str:
    return {
        "running": "运行中",
        "stale_running": "可能残留运行中",
        "failed": "失败",
        "pending": "等待",
        "done": "完成",
        "skipped": "已跳过",
    }.get(status, status or "未知")


def _item_payload(item: DatasetRunItem) -> dict[str, str]:
    return {
        "case_id": item.case_id,
        "variant": item.variant,
        "status": item.status,
        "bscan_path": item.bscan_path,
        "input_file": item.input_file,
        "error": item.error,
    }


def _add_variant(parent: QTreeWidgetItem, item: DatasetRunItem) -> None:
    node = QTreeWidgetItem([
        _friendly_variant_name(item.variant),
        _status_label(item.status),
        item.variant,
        (item.error or item.bscan_path or item.input_file or "")[:180],
    ])
    node.setData(0, Qt.UserRole, _item_payload(item))
    parent.addChild(node)


def _short_path(value: str, *, keep: int = 64) -> str:
    text = str(value or "").strip()
    if not text:
        return "未设置"
    if len(text) <= keep:
        return text
    return "..." + text[-keep:]


def _runtime_summary_line(dashboard: DatasetRunDashboard) -> str:
    profile = dashboard.runtime_profile or {}
    use_gpu = str(profile.get("use_gpu") or "0").lower() in {"1", "true", "yes", "on"}
    eta = format_duration_seconds(dashboard.estimated_remaining_seconds) if dashboard.estimated_remaining_seconds else "待估算"
    avg = format_duration_seconds(dashboard.average_variant_seconds) if dashboard.average_variant_seconds else "—"
    bits = [
        f"profile={profile.get('machine_profile') or '未设置'}",
        f"GPU={'开' if use_gpu else '关'}({profile.get('gpu_ids') or '0'})",
        f"GPU设备={profile.get('gpu_name') or '未检测'}",
        f"OMP={profile.get('omp_threads') or '未设置'}",
        f"平均/variant={avg}",
        f"预计剩余={eta}",
        f"Python={_short_path(profile.get('python_exe', ''))}",
        f"gprMax={_short_path(profile.get('gprmax_root', ''))}",
    ]
    return "当前运行环境：" + " | ".join(bits)


def _failure_summary_lines(failed_items: list[DatasetRunItem], *, max_cases_per_reason: int = 8) -> list[str]:
    if not failed_items:
        return ["暂无 failed 记录。"]
    grouped: dict[str, list[DatasetRunItem]] = defaultdict(list)
    for item in failed_items:
        reason = (item.error or "未记录失败原因").strip()
        # Keep aggregation stable even when traceback is long.
        key = reason.splitlines()[0][:160] if reason else "未记录失败原因"
        grouped[key].append(item)
    lines = [f"failed 总数：{len(failed_items)}"]
    for reason, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sample = ", ".join(f"{x.case_id}/{x.variant}" for x in items[:max_cases_per_reason])
        if len(items) > max_cases_per_reason:
            sample += f" ... +{len(items) - max_cases_per_reason}"
        lines.append(f"\n[{len(items)}] {reason}\n  {sample}")
    return lines


def refresh_batch_queue_panel(window: Any, dashboard: DatasetRunDashboard) -> None:
    """Render operator queue and failure aggregation without adding GUI state to services."""

    tree = getattr(window, "batch_queue_tree", None)
    if tree is None:
        return
    tree.clear()
    items = dashboard.running_items + dashboard.failed_items + dashboard.next_items + dashboard.latest_done_items
    by_case: dict[str, list[DatasetRunItem]] = defaultdict(list)
    for item in items:
        by_case[item.case_id or "unknown_case"].append(item)
    for case_id, variants in sorted(by_case.items()):
        counts = Counter(v.status for v in variants)
        if counts.get("failed"):
            case_status = "failed"
        elif counts.get("running") or counts.get("stale_running"):
            case_status = "running"
        elif counts.get("pending"):
            case_status = "pending"
        elif counts.get("done"):
            case_status = "done"
        else:
            case_status = "mixed"
        root = QTreeWidgetItem([
            case_id,
            _status_label(case_status),
            ", ".join(sorted({v.family for v in variants if v.family})) or "case",
            f"pending={counts.get('pending', 0)} running={counts.get('running', 0)+counts.get('stale_running', 0)} done={counts.get('done', 0)} failed={counts.get('failed', 0)}",
        ])
        root.setData(0, Qt.UserRole, {"case_id": case_id, "variant": "", "status": case_status})
        tree.addTopLevelItem(root)
        for item in sorted(variants, key=lambda x: (_status_rank(x.status), x.variant)):
            _add_variant(root, item)
    tree.expandToDepth(1)
    tree.resizeColumnToContents(0)
    failure_box = getattr(window, "batch_failure_box", None)
    if failure_box is not None:
        failure_box.setPlainText("\n".join(_failure_summary_lines(dashboard.failed_items)))
    runtime_label = getattr(window, "batch_runtime_summary_label", None)
    if runtime_label is not None:
        runtime_label.setText(_runtime_summary_line(dashboard))


def activate_batch_queue_item(window: Any, item: object, column: int = 0) -> None:
    """Double-click handler: jump from batch queue to history when possible."""

    if not isinstance(item, QTreeWidgetItem):
        return
    payload = item.data(0, Qt.UserRole) or {}
    if not isinstance(payload, dict):
        return
    case_id = str(payload.get("case_id") or "")
    variant = str(payload.get("variant") or "")
    status = str(payload.get("status") or "")
    if not case_id:
        return
    if hasattr(window, "show_page"):
        window.show_page(3)
    if hasattr(window, "refresh_history"):
        window.refresh_history()
    selected = False
    if hasattr(window, "select_history_entry_by_case_variant"):
        selected = bool(window.select_history_entry_by_case_variant(case_id, variant))
    if not selected and hasattr(window, "history_failure_box"):
        window.history_failure_box.setPlainText(
            f"尚未在历史页找到记录。\ncase_id={case_id}\nvariant={variant or '全部'}\nstatus={status}\n\n"
            "pending 任务通常还没有历史 marker；运行完成或失败后会自动进入历史树。"
        )
