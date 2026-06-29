from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GuardResult:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def check_easy_window_size() -> GuardResult:
    path = ROOT / "src" / "uavgpr_simlab" / "gui" / "easy_window.py"
    lines = count_lines(path)
    limit = 350
    return GuardResult(
        "easy_window_size",
        lines <= limit,
        f"{path.relative_to(ROOT)} has {lines} lines; limit={limit}",
    )


def check_controller_sizes() -> GuardResult:
    controller_dir = ROOT / "src" / "uavgpr_simlab" / "gui" / "controllers"
    limit = 450
    offenders: list[str] = []
    details: list[str] = []
    for path in sorted(controller_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        lines = count_lines(path)
        details.append(f"{path.name}={lines}")
        if lines > limit:
            offenders.append(f"{path.name}:{lines}>{limit}")
    return GuardResult(
        "controller_sizes",
        not offenders,
        "; ".join(details) if not offenders else "; ".join(offenders),
    )


def check_docs_root_clean() -> GuardResult:
    docs = ROOT / "docs"
    root_md = sorted(p.name for p in docs.glob("*.md"))
    root_png = sorted(p.name for p in docs.glob("*.png"))
    offenders: list[str] = []
    versioned_doc_patterns = ("_v080a", "_v0_", "CURRENT_ARCHITECTURE_v", "MULTI_MACHINE_GPU_RUNTIME_v", "RUNTIME_ROOT_4090_SETUP_v")
    historical_doc_patterns = ("AUDIT", "V0_", "OPTIMIZATION", "REPORT")
    stable_exceptions = {"YINGSHAN_REAL_DATA_AUDIT.md"}
    for name in root_md:
        if name not in stable_exceptions and (any(pat in name for pat in historical_doc_patterns) or any(pat in name for pat in versioned_doc_patterns)):
            offenders.append(name)
    for name in root_png:
        if "UavGPR-SimLab_v" in name or "GUI" in name:
            offenders.append(name)
    detail = "root markdown=" + ", ".join(root_md)
    if root_png:
        detail += "; root png=" + ", ".join(root_png)
    if offenders:
        detail += "; offenders=" + ", ".join(offenders)
    return GuardResult("docs_root_clean", not offenders, detail)


def check_history_archive_present() -> GuardResult:
    history = ROOT / "docs" / "history"
    count = len(list(history.glob("*.md"))) if history.exists() else 0
    readme = history / "README.md"
    return GuardResult(
        "history_archive_present",
        history.exists() and readme.exists() and count >= 10,
        f"docs/history exists={history.exists()} README={readme.exists()} markdown_count={count}",
    )


def main() -> int:
    results = [
        check_easy_window_size(),
        check_controller_sizes(),
        check_docs_root_clean(),
        check_history_archive_present(),
    ]
    payload = {"ok": all(r.ok for r in results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
