#!/usr/bin/env python3
"""Hook: 评估完成后提示是否生成报告。
Exit 0 = 正常。
触发条件: PostToolUse on Bash containing eval_full_line.py
"""
import json
import re
import sys

payload = json.load(sys.stdin)
command = payload.get("tool_input", {}).get("command", "")

if "eval_full_line.py" not in command:
    sys.exit(0)

# Extract line name from command
line_match = re.search(r"--line\s+(\S+)", command)
line_name = line_match.group(1) if line_match else "unknown"

msg = (
    f"📊 评估完成: {line_name}\n"
    f"提示: 可使用 /pgda-paper-report 生成中文报告，"
    f"或使用 /exp-compare 对比 baseline。"
)
print(msg, file=sys.stderr)
sys.exit(0)
