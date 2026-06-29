#!/usr/bin/env python3
"""Hook: Python 文件编辑后运行 ruff lint（仅 error 级别）。
Exit 0 = 正常, Exit 2 = 阻断并显示 lint 错误。
触发条件: PostToolUse on Edit|Write for .py files
"""
import json
import subprocess
import sys
from pathlib import Path

payload = json.load(sys.stdin)
path = payload.get("tool_input", {}).get("file_path", "")

if not path or not path.endswith(".py"):
    sys.exit(0)

p = Path(path)
parts = {part.lower() for part in p.parts}
if not ({"scripts", "pgdacsnet", "src", "hooks"} & parts):
    sys.exit(0)

# Run ruff if available, otherwise skip gracefully
try:
    result = subprocess.run(
        ["ruff", "check", "--no-fix", str(p)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
    )
    if result.returncode != 0 and result.stdout.strip():
        print("RUFF_ERRORS:", file=sys.stderr)
        print(result.stdout.strip(), file=sys.stderr)
        sys.exit(2)
except FileNotFoundError:
    # ruff not installed — skip silently
    pass
except Exception:
    pass

sys.exit(0)
