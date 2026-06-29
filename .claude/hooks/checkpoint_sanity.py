#!/usr/bin/env python3
"""Hook: 训练完成后自动检查 checkpoint 完整性。
Exit 0 = 正常, Exit 1 = 打印警告但不阻断。
触发条件: PostToolUse on Bash containing train_raw_only.py or resume_train.py
"""
import json
import re
import sys
import subprocess
import os

payload = json.load(sys.stdin)
command = payload.get("tool_input", {}).get("command", "")
output = payload.get("tool_output", {}).get("output", "")

# Only trigger after training commands finish
train_patterns = [r"train_raw_only\.py", r"resume_train\.py"]
if not any(re.search(p, command) for p in train_patterns):
    sys.exit(0)

# Extract run_dir from command
run_dir_match = re.search(r"configs/(\S+\.json)", command)
if not run_dir_match:
    sys.exit(0)

config_name = run_dir_match.group(1)

# Try to find the checkpoint via Python
check_script = '''
import torch, json, sys
from pathlib import Path

config = sys.argv[1]
# Derive run_dir from config name
base = config.replace("gpu_train_", "").replace(".json", "")
run_dir = Path("outputs") / f"run_{base}"
last = run_dir / "checkpoint_last.pt"
best = run_dir / "checkpoint_best.pt"

if not last.exists():
    print(f"WARNING: No checkpoint_last.pt at {run_dir}")
    sys.exit(1)

c = torch.load(last, map_location="cpu", weights_only=False)
epoch = c.get("epoch", 0)
hist = c.get("history", [])
train_loss = hist[-1].get("train_loss", float("nan")) if hist else float("nan")

import math
has_nan = math.isnan(train_loss) if isinstance(train_loss, float) else False

issues = []
if epoch < 10:
    issues.append(f"训练仅完成 {epoch} epoch，可能提前终止")
if has_nan:
    issues.append("train_loss 为 NaN")
if not best.exists():
    issues.append("无 checkpoint_best.pt")

if issues:
    print(f"⚠️ CHECKPOINT_SANITY [{config}]: epoch={epoch}, train_loss={train_loss:.4f}")
    for i in issues:
        print(f"  - {i}")
else:
    print(f"✅ CHECKPOINT_SANITY [{config}]: epoch={epoch}, train_loss={train_loss:.4f} OK")
'''

try:
    result = subprocess.run(
        ["python", "-c", check_script, config_name],
        capture_output=True, text=True, timeout=10,
        cwd=os.environ.get("CLAUDE_PROJECT_DIR", ".")
    )
    if result.stdout.strip():
        print(result.stdout.strip(), file=sys.stderr)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
except Exception:
    pass

sys.exit(0)
