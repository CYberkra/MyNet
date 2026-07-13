#!/usr/bin/env python3
"""Hook: 检查训练命令是否使用了正确的 Python 解释器。
Exit 0 = 允许, Exit 2 = 警告但允许继续。
"""
import json
import re
import sys
from pathlib import Path

payload = json.load(sys.stdin)
command = payload.get("tool_input", {}).get("command", "")

# Only trigger on training/sim commands
if not any(kw in command for kw in ["train_raw_only", "resume_train", "train_gprmax",
                                      "gprMax", "uavgpr_simlab"]):
    sys.exit(0)

# Check only generic, historically unreliable interpreter locations. The
# preferred executable is machine-local and is loaded from the runtime profile.
bad_pythons = [
    (r"(?:^|[\"'\s])(?:python|python\.exe)(?:[\"'\s]|$)", "unqualified system Python"),
]

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from pgdacsnet.runtime import load_runtime
    preferred_python = str(load_runtime().project_python)
except Exception:
    preferred_python = "environment/project_runtime.local.json -> project_python"

for bad_pat, bad_name in bad_pythons:
    if re.search(bad_pat, command, flags=re.IGNORECASE):
        msg = (
            f"⚠️ VENV_PYTHON_GUARD: 检测到使用 {bad_name}\n"
            f"建议使用本机配置的项目 Python: {preferred_python}\n"
            "请先运行 scripts/validate_machine_runtime.py 检查本机环境。"
        )
        print(msg, file=sys.stderr)
        sys.exit(2)

sys.exit(0)
