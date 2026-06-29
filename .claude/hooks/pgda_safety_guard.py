#!/usr/bin/env python3
import json
import re
import sys

payload = json.load(sys.stdin)
command = payload.get("tool_input", {}).get("command", "")

patterns = [
    r"\brm\s+-rf\b",
    r"\brm\s+-f\b.*(_merged\.out|\.zip|checkpoint.*\.pt|outputs[/\\])",
    r"\bdel\b.*(_merged\.out|\.zip|checkpoint.*\.pt)",
    r"Remove-Item\b.*(-Recurse|-Force)",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-",
    r"taskkill\b",
]

if any(re.search(p, command, re.IGNORECASE) for p in patterns):
    print(
        "BLOCKED_PGDA_SAFETY_HOOK: 该命令可能删除/中断高成本仿真或训练产物。"
        "请先向用户说明将删除或终止什么，并获得明确授权。",
        file=sys.stderr,
    )
    sys.exit(2)

sys.exit(0)
