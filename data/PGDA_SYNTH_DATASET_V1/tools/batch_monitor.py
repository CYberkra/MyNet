#!/usr/bin/env python3
"""Batch monitoring script — run periodically to check progress."""
import os, json, time, sys
from datetime import datetime

RUNS = r'D:\Claude\PGDA-CSNet\data\PGDA_SYNTH_DATASET_V1\03_runs\batch_001_line9_style_12cases'
POOL = r'D:\Claude\PGDA-CSNet\data\PGDA_SYNTH_DATASET_V1\02_case_pool\batch_001_line9_style_12cases\cases'

total_cases = len([d for d in os.listdir(POOL) if os.path.isdir(os.path.join(POOL, d))])

while True:
    print(f'\n=== {datetime.now().strftime("%H:%M:%S")} ===')

    # Per-case status
    for cid in sorted(os.listdir(RUNS)):
        rdir = os.path.join(RUNS, cid)
        if not os.path.isdir(rdir) or cid.startswith('.'):
            continue
        raw = os.path.join(rdir, 'raw')
        if os.path.exists(raw):
            outs = [f for f in os.listdir(raw) if f.startswith('raw') and f.endswith('.out')]
            n_out = len(outs)
            # Check if completed
            if n_out >= 128:
                print(f'  ✅ {cid}: {n_out}/128 complete')
            elif n_out > 0:
                newest = max(os.path.getmtime(os.path.join(raw, f)) for f in outs)
                age = (datetime.now() - datetime.fromtimestamp(newest)).total_seconds()
                print(f'  ▶ {cid}: {n_out}/128 ({age:.0f}s since last output)')
            else:
                print(f'  ⏳ {cid}: setup...')
        else:
            print(f'  ⏳ {cid}: waiting...')

    # GPU health check
    try:
        import subprocess
        r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/V'],
                          capture_output=True, text=True, timeout=5)
        # Find gprMax processes by high memory
        gpu_procs = 0
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and 'python' in line.lower():
                try:
                    mem = int(parts[4].replace(',', ''))
                    if mem > 100000:  # >100MB
                        gpu_procs += 1
                        # Show if it's been running long
                        cpu_time = parts[6] if len(parts) > 6 else '?'
                        print(f'  GPU process: PID {parts[1]}, {mem//1024}MB, CPU {cpu_time}')
                except:
                    pass
        if gpu_procs == 0:
            print(f'  ⚠ No GPU processes found! Batch may have died.')
    except:
        pass

    time.sleep(180)  # Check every 3 minutes
