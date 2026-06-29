#!/usr/bin/env python3
"""Hook: 检查是否有正在运行的训练进程，防止重复启动占用 GPU。
Exit 0 = 允许, Exit 2 = 拦截并警告。
"""
import json
import re
import subprocess
import sys

payload = json.load(sys.stdin)
command = payload.get("tool_input", {}).get("command", "")

# Only trigger on training commands
train_patterns = [
    r"train_raw_only\.py",
    r"resume_train\.py",
    r"train_gprmax_adapter\.py",
    r"train_final_from_cv\.py",
    r"_keep_training\.py",
    r"_retry_train\.py",
]
if not any(re.search(p, command, re.IGNORECASE) for p in train_patterns):
    sys.exit(0)

# Check for existing training processes
try:
    result = subprocess.run(
        ["wmic", "process", "where",
         "name like '%python%'",
         "get", "ProcessId,CommandLine"],
        capture_output=True, text=True, timeout=5
    )
    lines = result.stdout
except Exception:
    sys.exit(0)  # If we can't check, don't block

existing_pids = []
for line in lines.split("\n"):
    for pat in train_patterns:
        if re.search(pat, line, re.IGNORECASE):
            m = re.search(r"(\d+)\s*$", line.strip())
            if m:
                existing_pids.append(m.group(1))

# Extract the config being launched
config_match = re.search(r"configs/(\S+\.json)", command)
config_name = config_match.group(1) if config_match else "unknown"

if existing_pids:
    msg = (
        f"BLOCKED: 检测到 {len(existing_pids)} 个正在运行的训练进程 (PID: {', '.join(existing_pids)})。\n"
        f"本次尝试启动: {config_name}\n"
        "同时运行多个训练进程会导致:\n"
        "  1. GPU 显存不足 (OOM)\n"
        "  2. 训练速度减半\n"
        "  3. Checkpoint 写入冲突\n"
        "请先终止现有训练进程，或确认你确实想并行运行。"
    )
    print(msg, file=sys.stderr)
    sys.exit(2)

sys.exit(0)
