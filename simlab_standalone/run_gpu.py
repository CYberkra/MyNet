"""Run gprMax GPU simulation on any .in file with process isolation"""
import sys, os
sys.path.insert(0, r'D:\Claude\PGDA-CSNet\uavgpr_simlab\src')
from uavgpr_simlab.core.runner import SafeGprMaxRunner

if len(sys.argv) < 2:
    print("Usage: python run_gpu.py <path/to/scene.in> [n_traces] [output_dir]")
    print("  n_traces: number of traces (default: 10)")
    print("  output_dir: where to save logs and .out files (default: same as .in file)")
    sys.exit(1)

infile = sys.argv[1]
n_traces = int(sys.argv[2]) if len(sys.argv) > 2 else 10
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(infile)

runner = SafeGprMaxRunner(
    cmd=[
        r'E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe',
        '-m', 'gprMax', infile,
        '-gpu', '-n', str(n_traces), '--geometry-fixed',
    ],
    cwd=out_dir,
    timeout=14400,
)
result = runner.run()
print(f'Status: {result.status}')
print(f'Elapsed: {result.elapsed:.0f}s ({result.elapsed/n_traces:.1f}s/trace)')
print(f'Logs: {result.stdout_log}')
