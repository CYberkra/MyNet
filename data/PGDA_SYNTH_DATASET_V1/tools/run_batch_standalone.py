#!/usr/bin/env python3
"""
PGDA_SYNTH_DATASET_V1 — Batch Runner (Standalone)

Run this in a SEPARATE terminal (not from Claude Code).
Stays alive regardless of Claude Code session.

Usage (from any terminal):
    cd D:\Claude\PGDA-CSNet
    E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe data/PGDA_SYNTH_DATASET_V1/tools/run_batch_standalone.py

Features:
    - Runs cases sequentially, 128 traces each
    - GPU watchdog (nvidia-smi every 30s to prevent TDR)
    - Journal-based resume (safe to Ctrl+C and restart)
    - One case at a time, update journal after each
"""
import subprocess, os, time, json, datetime, sys, shutil
from pathlib import Path

# ── Config ──
POOL = Path(r'D:\Claude\PGDA-CSNet\data\PGDA_SYNTH_DATASET_V1\02_case_pool\batch_001_line9_style_12cases\cases')
RUNS = Path(r'D:\Claude\PGDA-CSNet\data\PGDA_SYNTH_DATASET_V1\03_runs\batch_001_line9_style_12cases')
PY = r'E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe'
JOURNAL = RUNS / '.journal.json'

# ── MSVC + CUDA env ──
env = os.environ.copy()
msvc = r'E:\sisual stdio 2022'; ver = r'14.39.33519'
kits = r'C:\Program Files (x86)\Windows Kits\10'; sdk = r'10.0.22621.0'
cuda = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8'
env['PATH'] = os.path.join(msvc,'VC','Tools','MSVC',ver,'bin','Hostx64','x64')+';'+\
              os.path.join(cuda,'bin')+';'+env.get('PATH','')
env['INCLUDE'] = os.path.join(msvc,'VC','Tools','MSVC',ver,'include')+';'+\
                 os.path.join(kits,'Include',sdk,'ucrt')
env['LIB'] = os.path.join(msvc,'VC','Tools','MSVC',ver,'lib','x64')+';'+\
             os.path.join(kits,'Lib',sdk,'ucrt','x64')

def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def load_journal():
    if JOURNAL.exists():
        return json.loads(JOURNAL.read_text())
    return {'cases': {}, 'started_at': datetime.datetime.now().isoformat()}

def save_journal(j):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL.write_text(json.dumps(j, indent=2, ensure_ascii=False))

def gpu_watchdog():
    """Periodically ping nvidia-smi to prevent Windows TDR."""
    while True:
        try:
            subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, timeout=15)
        except:
            pass
        time.sleep(25)

def run_case(cid):
    """Run one case. Returns True on success."""
    raw_in = POOL / cid / 'geometry' / 'raw.in'
    if not raw_in.exists():
        log(f'{cid}: no raw.in, skip')
        return False

    run_dir = RUNS / cid / 'raw'
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(raw_in), str(run_dir / 'raw.in'))

    existing = len(list(run_dir.glob('raw*.out')))
    log(f'{cid}: {existing}/128 existing traces')

    log_file = run_dir / 'gprmax.log'
    t0 = time.time()

    proc = subprocess.Popen(
        [PY, '-m', 'gprMax', str(run_dir / 'raw.in'), '-gpu', '-n', '128', '--geometry-fixed'],
        env=env, cwd=str(run_dir),
        stdout=open(log_file, 'w'), stderr=open(str(log_file)+'.err', 'w')
    )

    # Monitor while running
    while proc.poll() is None:
        # Check last .out timestamp
        outs = list(run_dir.glob('raw*.out'))
        n = len(outs)
        if n > 0 and n % 10 == 0 and n > existing:
            newest = max(os.path.getmtime(f) for f in outs)
            age = (time.time() - newest) / 60
            if age < 2:
                pass  # still running fine
        time.sleep(30)

    elapsed = (time.time() - t0) / 60
    outs = len(list(run_dir.glob('raw*.out')))
    success = proc.returncode == 0 and outs >= 128

    ok_icon = 'OK' if success else 'FAIL'
    log(f'{cid}: {ok_icon} {outs}/128, {elapsed:.0f}min')
    return success

# ── Main ──
def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    journal = load_journal()

    # Start GPU watchdog thread
    from threading import Thread
    wd = Thread(target=gpu_watchdog, daemon=True)
    wd.start()
    log('GPU watchdog started')

    case_dirs = sorted([d.name for d in POOL.iterdir() if d.is_dir()])
    N = len(case_dirs)
    log(f'Batch: {N} cases to run')

    for i, cid in enumerate(case_dirs):
        if journal.get(cid, {}).get('status') == 'completed':
            log(f'[{i+1}/{N}] {cid}: already done, skip')
            continue

        log(f'[{i+1}/{N}] {cid}: starting')

        journal[cid] = {'status': 'running', 'started_at': datetime.datetime.now().isoformat()}
        save_journal(journal)

        ok = run_case(cid)

        if ok:
            journal[cid] = {'status': 'completed', 'traces': 128}
        else:
            journal[cid] = {'status': 'failed'}
            log(f'STOP: {cid} failed')
            save_journal(journal)
            break

        save_journal(journal)

    done = sum(1 for c in journal.values() if isinstance(c, dict) and c.get('status') == 'completed')
    log(f'Done: {done}/{N}')

if __name__ == '__main__':
    main()
