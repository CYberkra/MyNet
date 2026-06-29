"""Quick geometry-only check for any gprMax .in file"""
import sys, os
sys.path.insert(0, r'D:\Claude\PGDA-CSNet\uavgpr_simlab\src')
from uavgpr_simlab.core.runner import run_geometry_dry_run

if len(sys.argv) < 2:
    print("Usage: python run_geometry.py <path/to/scene.in>")
    sys.exit(1)

infile = sys.argv[1]
result = run_geometry_dry_run(
    infile,
    python_exe=r'E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe',
    timeout=120,
)
print(f'Status: {result.status}')
print(f'Elapsed: {result.elapsed:.1f}s')
if result.status != 'success':
    for line in result.stderr_tail.split('\n'):
        if any(kw in line for kw in ['Error', 'error', 'physical', 'CmdInput', 'not exist']):
            print(line.strip()[-200:])
