#!/usr/bin/env python3
"""Hook: 检查训练命令是否使用了正确的 Python 解释器。
Exit 0 = 允许, Exit 2 = 警告但允许继续。
"""
import json
import re
import sys

payload = json.load(sys.stdin)
command = payload.get("tool_input", {}).get("command", "")

# Only trigger on training/sim commands
if not any(kw in command for kw in ["train_raw_only", "resume_train", "train_gprmax",
                                      "gprMax", "uavgpr_simlab"]):
    sys.exit(0)

# Check for problematic Python interpreters
bad_pythons = [
    (r"E:\\python\\python\.exe", "E:\\python\\python.exe (系统 Python, torch.cuda=False)"),
    (r"E:/python/python\.exe", "E:/python/python (系统 Python, torch.cuda=False)"),
    (r"D:\\Miniconda3\\python\.exe", "D:\\Miniconda3\\python.exe (Miniconda)"),
]
good_python = r"E:\\gprMax\\gprMax-v\.3\.1\.7\\.venv\\Scripts\\python\.exe"
good_python_alt = r"E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python\.exe"

for bad_pat, bad_name in bad_pythons:
    if re.search(bad_pat, command) and not re.search(good_python, command) and not re.search(good_python_alt, command):
        msg = (
            f"⚠️ VENV_PYTHON_GUARD: 检测到使用 {bad_name}\n"
            f"建议使用 gprMax venv Python: E:\\gprMax\\gprMax-v.3.1.7\\.venv\\Scripts\\python.exe\n"
            "该解释器已确认 torch.cuda.is_available()=True 且版本兼容。"
        )
        print(msg, file=sys.stderr)
        sys.exit(2)

sys.exit(0)
