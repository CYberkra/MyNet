#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

payload = json.load(sys.stdin)
tool_input = payload.get("tool_input", {})
path = tool_input.get("file_path")

if not path:
    sys.exit(0)

p = Path(path)
if p.suffix != ".py":
    sys.exit(0)

parts = {part.lower() for part in p.parts}
if not ({"scripts", "pgdacsnet", "src"} & parts):
    sys.exit(0)

result = subprocess.run(
    [sys.executable, "-m", "py_compile", str(p)],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("PY_COMPILE_FAILED:", file=sys.stderr)
    if result.stdout:
        print(result.stdout, file=sys.stderr)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.exit(2)

print(f"PY_COMPILE_OK {p}")
sys.exit(0)
