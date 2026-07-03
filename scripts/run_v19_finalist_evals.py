from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
VALID_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
CENTER_WEIGHTS = [0.0, 0.5, 1.0]
SEEDS = [1901, 1902, 1903]


def run(cmd):
    print(" ".join(str(x) for x in cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode:
        print(f"WARNING nonzero eval exit: {result.returncode}", flush=True)


def ensure_final_checkpoint(run_dir):
    run_path = ROOT / run_dir
    final_path = run_path / "checkpoint_final.pt"
    best_path = run_path / "checkpoint_best.pt"
    if final_path.exists():
        return
    if not best_path.exists():
        raise FileNotFoundError(best_path)
    final_path.write_bytes(best_path.read_bytes())
    print(f"WROTE {final_path}", flush=True)


def eval_if_missing(out_dir, eval_name, cmd):
    metric_path = ROOT / out_dir / f"{eval_name}_full_metrics.csv"
    if metric_path.exists():
        print(f"SKIP existing {metric_path}", flush=True)
        return
    run(cmd)


def eval_group(arch, label, run_dirs):
    for run_dir in run_dirs:
        ensure_final_checkpoint(run_dir)
    for weight in CENTER_WEIGHTS:
        suffix = f"w{int(weight * 100):03d}"
        out_dir = f"outputs/eval_paper_{arch}_final_{label}_{suffix}_p065"
        base = [
            PY,
            "scripts/eval_full_line.py",
            "--run-dirs",
            *run_dirs,
            "--checkpoint",
            "final",
            "--presence-thr",
            "0.45",
            "--path-prob-thr",
            "0.65",
            "--search-min-ns",
            "240",
            "--search-max-ns",
            "500",
            "--dp-max-jump",
            "6",
            "--dp-smooth-weight",
            "0.16",
            "--center-fusion-weight",
            str(weight),
            "--out-dir",
            out_dir,
        ]
        for line in VALID_LINES:
            eval_if_missing(
                out_dir,
                line,
                [*base, "--line", line],
            )
        eval_if_missing(
            out_dir,
            "Line9_holdout_tr1664_2377",
            [*base, "--line", "Line9", "--trace-start", "1664", "--trace-end", "2377"],
        )


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "v1_9d_mambavision_hybrid"
    for seed in SEEDS:
        run_dir = f"outputs/run_gpu_paper_{arch}_final_seed{seed}_line9holdout"
        if (ROOT / run_dir).exists():
            eval_group(arch, f"seed{seed}", [run_dir])
    ensemble_dirs = [
        f"outputs/run_gpu_paper_{arch}_final_seed{seed}_line9holdout"
        for seed in SEEDS
        if (ROOT / f"outputs/run_gpu_paper_{arch}_final_seed{seed}_line9holdout").exists()
    ]
    if len(ensemble_dirs) >= 2:
        eval_group(arch, f"ensemble{len(ensemble_dirs)}seed", ensemble_dirs)


if __name__ == "__main__":
    main()
