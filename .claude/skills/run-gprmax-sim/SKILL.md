---
name: run-gprmax-sim
description: Run gprMax GPU simulation on any .in file with auto-merge and .vti cleanup. Use when the user asks to run gprMax, "跑N道", "GPU仿真", or provide an .in file path.
---

# Run gprMax GPU Simulation

Given an `.in` file path and number of traces, run the simulation with process isolation, auto-merge output files, and clean up .vti files.

## Implementation

Use `SafeGprMaxRunner` from SimLab with the environment below. After successful run, merge with `tools.outputfiles_merge <stem> --remove-files` and delete any `.vti` files.

### Fixed Environment
```python
PYTHON_EXE = r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
SIMLAB_SRC = r"D:\Claude\PGDA-CSNet\uavgpr_simlab\src"
GPU_TIMEOUT = 14400  # 4h default
```

### Execution Pattern
```python
import sys, os, subprocess, glob
sys.path.insert(0, SIMLAB_SRC)
from uavgpr_simlab.core.runner import SafeGprMaxRunner

def rm(path): 
    try: os.remove(path) if os.path.exists(path) else None
    except: pass

run_dir = os.path.dirname(<infile>)
stem = os.path.basename(<infile>).replace('.in', '')

runner = SafeGprMaxRunner(
    cmd=[PYTHON_EXE, '-m', 'gprMax', <infile>, '-gpu', '-n', str(<n_traces>), '--geometry-fixed'],
    cwd=run_dir, timeout=GPU_TIMEOUT,
)
result = runner.run()
if result.status == 'success':
    subprocess.run([PYTHON_EXE, '-m', 'tools.outputfiles_merge', stem, '--remove-files'],
                   cwd=run_dir, capture_output=True, text=True, timeout=120)
    for f in glob.glob(os.path.join(run_dir, '*.vti')): rm(f)
    merged = os.path.join(run_dir, f'{stem}_merged.out')
    print(f'OK: {merged} ({os.path.getsize(merged)/1e6:.0f}MB, {result.elapsed:.0f}s)')
else:
    print(f'FAIL: {result.stderr_tail[-500:]}')
```

### Geometry-only variant
When user asks "几何检查" or "geometry check", use `--geometry-only` without `-gpu` flag and skip merge.

### Batch variant
When user asks to run multiple variants (e.g. "raw和background各50道"), run each sequentially, showing progress [N/total]. Skip if `_merged.out` already exists.

### Rules
- Always use `--geometry-fixed` for multi-trace runs (geometry unchanged between traces)
- Delete individual `.out` files after merge (merge uses `--remove-files`)
- Delete `.vti` files after each run
- Timeouts: 3600s for ≤50 traces, 7200s for >50, 14400s for >200
