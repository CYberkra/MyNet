#!/usr/bin/env python3
"""PostToolUse hook: after gprMax simulation completes, auto-generate three-piece output + clean .vti."""
import json
import re
import sys
from pathlib import Path

payload = json.load(sys.stdin)
tool = payload.get("tool", "")
tool_input = payload.get("tool_input", {})

if tool != "Bash":
    sys.exit(0)

command = tool_input.get("command", "")
result = payload.get("result", {})
stdout = ""
if isinstance(result, dict):
    stdout = result.get("stdout", "") or ""

full_text = command + "\n" + stdout

# Fast exit: check for simulation indicators
cmd_has_gpr = bool(re.search(r'gprMax\s+.+?\.in', command, re.IGNORECASE))
out_has_sim = bool(re.search(r'(files=\d+/\d+|Simulation completed)', stdout, re.IGNORECASE))
# Also check for SafeGprMaxRunner patterns in command
wrapper_has_sim = bool(re.search(r'(n_traces|EXPECTED|N_TRACES|flat_layered|v3_\d)', command, re.IGNORECASE))

if not (cmd_has_gpr or out_has_sim or wrapper_has_sim):
    sys.exit(0)

# Find output directory
out_dir = None
stem = None

# Method 1: direct gprMax command
if cmd_has_gpr:
    m = re.search(r'gprMax\s+(.+?\.in)', command)
    if m:
        p = Path(m.group(1))
        if not p.is_absolute():
            cwd = tool_input.get("cwd", ".")
            p = Path(cwd) / p
        if p.exists():
            out_dir = p.parent
            stem = p.stem

# Method 2: detect from stdout for wrapper scripts
if out_dir is None and out_has_sim:
    m = re.search(r'Output file:\s*(.+?)(\d+)\.out', stdout)
    if m:
        stem_base = m.group(1)
        p = Path(stem_base)
        if p.exists():
            out_dir = p.parent
            stem = p.stem.rstrip("0123456789")
    else:
        # Try cwd
        cwd = tool_input.get("cwd", ".")
        cand = Path(cwd)
        for prefix in ["raw", "flat_layered", "v3_", "continuous"]:
            files = list(cand.glob(f"{prefix}*.out"))
            if len(files) >= 64:
                out_dir = cand
                stem = prefix
                break

# Method 3: wrapper script with known paths
if out_dir is None and wrapper_has_sim:
    # Check common output directories for large .out sets
    base = Path(r"D:\Claude\PGDA-CSNet\data\gprmax_experiments")
    for subdir in sorted(base.iterdir(), reverse=True):
        if not subdir.is_dir():
            continue
        for prefix in ["raw", "flat_layered", "continuous"]:
            files = list(subdir.glob(f"{prefix}*.out"))
            if len(files) >= 64:
                out_dir = subdir
                stem = prefix
                break
        if out_dir:
            break

if out_dir is None or stem is None:
    sys.exit(0)

out_files = list(out_dir.glob(f"{stem}*.out"))
if len(out_files) < 64:
    sys.exit(0)

# Clean .vti
vti_files = list(out_dir.glob("*.vti"))
for v in vti_files:
    try:
        v.unlink()
    except Exception:
        pass

# Generate bscan.npy
bscan_path = out_dir / "bscan.npy"
if not bscan_path.exists() and out_files:
    try:
        import h5py
        import numpy as np

        files = sorted(out_dir.glob(f"{stem}*.out"))
        files = sorted(
            [f for f in files if f.stem.replace(stem, "").isdigit()],
            key=lambda x: int(x.stem.replace(stem, "")),
        )
        if files:
            tr = []
            for f in files:
                with h5py.File(f, "r") as h:
                    tr.append(np.asarray(h["rxs"]["rx1"]["Ez"]))
            arr = np.stack(tr, axis=1)
            np.save(bscan_path, arr)
    except Exception:
        pass

# Generate diagnostic figure
if bscan_path.exists():
    try:
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = np.load(bscan_path)
        t_arr = np.linspace(0, 700, arr.shape[0])
        new = np.linspace(0, 700, 501)
        bi = np.empty((501, arr.shape[1]))
        for i in range(arr.shape[1]):
            bi[:, i] = np.interp(new, t_arr, arr[:, i])
        t = new
        clip = np.percentile(np.abs(bi), 99.9)
        gain_t2 = (t / t[-1]) ** 2
        gain_t4 = (t / t[-1]) ** 4
        t2b = bi * (gain_t2[:, np.newaxis] + 0.01)
        t4b = bi * (gain_t4[:, np.newaxis] + 0.01)
        diffs = [np.max(np.abs(arr[:, i] - arr[:, 0])) for i in range(arr.shape[1])]

        def safe_agc(x, w=31, f=0.02):
            p = w // 2
            y = np.pad(x * x, ((p, p), (0, 0)), "reflect")
            r = np.empty_like(x)
            for i in range(x.shape[0]):
                r[i] = np.sqrt(np.mean(y[i : i + w], axis=0) + 1e-12)
            return x / np.maximum(r, np.percentile(r, 10) * f)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        axes[0, 0].imshow(bi, cmap="gray", aspect="auto", vmin=-clip, vmax=clip,
                          extent=[0, bi.shape[1] - 1, 700, 0])
        axes[0, 0].set_title(f"Raw (trace_var={max(diffs):.2f})", fontweight="bold")
        axes[0, 1].imshow(t2b, cmap="gray", aspect="auto",
                          vmin=-np.percentile(np.abs(t2b), 99.5), vmax=np.percentile(np.abs(t2b), 99.5),
                          extent=[0, bi.shape[1] - 1, 700, 0])
        axes[0, 1].set_title("t^2 gain", fontweight="bold")
        axes[0, 2].imshow(t4b, cmap="gray", aspect="auto",
                          vmin=-np.percentile(np.abs(t4b), 99.5), vmax=np.percentile(np.abs(t4b), 99.5),
                          extent=[0, bi.shape[1] - 1, 700, 0])
        axes[0, 2].set_title("t^4 gain", fontweight="bold")
        axes[1, 0].imshow(safe_agc(bi), cmap="gray", aspect="auto", vmin=-3, vmax=3,
                          extent=[0, bi.shape[1] - 1, 700, 0])
        axes[1, 0].set_title("Safe AGC 2%", fontweight="bold")
        lo, hi = int(390 / 700 * 501), int(450 / 700 * 501)
        axes[1, 1].imshow(t4b[lo:hi], cmap="gray", aspect="auto",
                          vmin=-np.percentile(np.abs(t4b[lo:hi]), 99.5), vmax=np.percentile(np.abs(t4b[lo:hi]), 99.5),
                          extent=[0, bi.shape[1] - 1, t[hi], t[lo]])
        axes[1, 1].set_title("t^4 390-450ns (target)", fontweight="bold")
        axes[1, 2].plot(t, np.mean(bi, axis=1), label="raw", alpha=0.7)
        axes[1, 2].plot(t, np.mean(t2b, axis=1), label="t^2", alpha=0.7)
        axes[1, 2].plot(t, np.mean(t4b, axis=1), label="t^4", alpha=0.7)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].set_title("Mean traces")
        axes[1, 2].set_xlim(0, 700)

        plt.tight_layout()
        fig.savefig(out_dir / "diagnostic.png", dpi=150)
        plt.close()
    except Exception:
        pass

n_files = len(out_files)
report = "\n=== Simulation Report (auto) ==="
report += f"\n  Output: {out_dir.name}"
report += f"\n  Files: {n_files}/128"
report += f"\n  .vti cleaned: {len(vti_files)}"
report += f"\n  Diagnostic: {'generated' if bscan_path.exists() else 'skipped'}"
report += "\n  ================================="
print(report)

sys.exit(0)
