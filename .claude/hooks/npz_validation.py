#!/usr/bin/env python3
"""Hook: NPZ 转换后自动验证训练数据完整性。
Exit 2 = 阻断（数据有问题）, Exit 0 = 正常。
触发条件: PostToolUse on Bash containing convert_pilot_to_training.py
"""
import json
import subprocess
import sys
from pathlib import Path

payload = json.load(sys.stdin)
command = payload.get("tool_input", {}).get("command", "")

if "convert_pilot_to_training" not in command:
    sys.exit(0)

train_dir = Path("D:/Claude/PGDA-CSNet/data/simulation_pretrain_v3")
npz_dir = train_dir / "windows"
npzs = sorted(npz_dir.glob("*.npz"))

if not npzs:
    print("NPZ_VALIDATION: No NPZ files found to validate.", file=sys.stderr)
    sys.exit(2)

errors = []
for p in npzs:
    try:
        import numpy as np
        d = np.load(p)

        # 1. Keys
        required = ["x_raw", "y_mask", "y_soft", "status_code", "label_weight", "label_weight_2d"]
        for k in required:
            if k not in d:
                errors.append(f"{p.name}: missing key {k}")

        # 2. Shapes
        if d["x_raw"].shape != (501, 256):
            errors.append(f"{p.name}: x_raw shape {d['x_raw'].shape}")
        if d["y_mask"].shape != (501, 256):
            errors.append(f"{p.name}: y_mask shape {d['y_mask'].shape}")
        if d["y_soft"].shape != (501, 256):
            errors.append(f"{p.name}: y_soft shape {d['y_soft'].shape}")

        # 3. Padding zeros (y_mask)
        if d["y_mask"][:, :64].sum() > 0:
            errors.append(f"{p.name}: y_mask left pad not zero")
        if d["y_mask"][:, 192:].sum() > 0:
            errors.append(f"{p.name}: y_mask right pad not zero")

        # 4. Padding zeros (label_weight)
        if d["label_weight"][:64].sum() > 0:
            errors.append(f"{p.name}: weight left pad not zero")
        if d["label_weight"][192:].sum() > 0:
            errors.append(f"{p.name}: weight right pad not zero")

        # 5. y_soft peak
        if d["y_soft"].max() < 0.9:
            errors.append(f"{p.name}: y_soft peak {d['y_soft'].max():.3f} < 0.9")

        # 6. x_raw finite
        if not np.isfinite(d["x_raw"]).all():
            errors.append(f"{p.name}: x_raw has NaN/inf")

        # 7. status_code range
        sc = d["status_code"]
        if set(np.unique(sc)) - {0, 1, 2}:
            errors.append(f"{p.name}: status_code invalid values")

    except Exception as e:
        errors.append(f"{p.name}: read error: {e}")

if errors:
    print("NPZ_VALIDATION_ERRORS:", file=sys.stderr)
    for e in errors[:10]:
        print(f"  {e}", file=sys.stderr)
    sys.exit(2)

print(f"NPZ_VALIDATION_OK: {len(npzs)} files passed")
sys.exit(0)
