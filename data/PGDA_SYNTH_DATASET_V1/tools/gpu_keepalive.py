#!/usr/bin/env python3
"""
GPU Keepalive — TDR Prevention Sidecar

Usage:
    python tools/gpu_keepalive.py [--interval 15]

Purpose:
  Windows TDR (Timeout Detection and Recovery) resets the GPU driver if it's
  unresponsive for >2 seconds. CUDA long kernels trigger this.

  This script runs in a separate terminal/thread and queries nvidia-smi
  every N seconds, which forces driver IO and resets the TDR timer.

  Run BEFORE starting gprMax, keep running in background until done.

Mechanism:
  nvidia-smi uses NVML (NVIDIA Management Library), a separate channel from
  CUDA. Querying it is low-overhead and doesn't interfere with running CUDA
  kernels, but it does count as driver activity → Windows won't reset it.

References:
  https://docs.nvidia.com/gameworks/content/developertools/desktop/timeout_detection_recovery.htm
  TdrDelay default: 2 seconds (HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers)
"""

import subprocess, time, sys, argparse, datetime

def keepalive(interval=15, max_temp=90):
    print(f"GPU Keepalive started (interval={interval}s, max_temp={max_temp}°C)")
    print(f"Run this alongside gprMax. Press Ctrl+C to stop.\n")

    last_temp = 0
    warnings = 0

    try:
        while True:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(",")
                    temp = int(parts[0].strip())
                    util = int(parts[1].strip()) if len(parts) > 1 else 0
                    mem_used = int(parts[2].strip()) if len(parts) > 2 else 0
                    mem_total = int(parts[3].strip()) if len(parts) > 3 else 0

                    if temp > last_temp + 5 or temp > max_temp - 5:
                        print(f"  [{ts}] GPU {temp}°C, util {util}%, mem {mem_used}/{mem_total}MB")

                    if temp > max_temp:
                        warnings += 1
                        print(f"  ⚠ [{ts}] HIGH TEMPERATURE: {temp}°C (>={max_temp}°C, warning #{warnings})")

                    last_temp = temp
                else:
                    print(f"  [{ts}] nvidia-smi returned empty (GPU may be idle)")
            except subprocess.TimeoutExpired:
                print(f"  [{ts}] nvidia-smi timeout (GPU busy?)")
            except FileNotFoundError:
                print(f"  ❌ nvidia-smi not found! Is NVIDIA driver installed?")
                sys.exit(1)
            except Exception as e:
                print(f"  [{ts}] nvidia-smi error: {e}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nGPU Keepalive stopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPU Keepalive — TDR Prevention Sidecar")
    ap.add_argument("--interval", type=int, default=15, help="nvidia-smi polling interval in seconds (default: 15)")
    ap.add_argument("--max-temp", type=int, default=90, help="High temperature warning threshold in °C (default: 90)")
    args = ap.parse_args()
    keepalive(args.interval, args.max_temp)
