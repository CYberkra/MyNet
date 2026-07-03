from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
VALID_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
PATH_THRESHOLDS = [0.20, 0.40, 0.50, 0.65]
CENTER_WEIGHT = 0.5


def run(cmd):
    print(" ".join(str(x) for x in cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode:
        print(f"WARNING nonzero eval exit: {result.returncode}", flush=True)


def eval_if_missing(out_dir, eval_name, cmd):
    metric_path = ROOT / out_dir / f"{eval_name}_full_metrics.csv"
    if metric_path.exists():
        print(f"SKIP existing {metric_path}", flush=True)
        return
    run(cmd)


def eval_group(arch, label, run_dirs):
    for pthr in PATH_THRESHOLDS:
        ptag = f"p{int(pthr * 100):03d}"
        out_dir = f"outputs/eval_paper_{arch}_final_{label}_w050_{ptag}"
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
            str(pthr),
            "--search-min-ns",
            "240",
            "--search-max-ns",
            "500",
            "--dp-max-jump",
            "6",
            "--dp-smooth-weight",
            "0.16",
            "--center-fusion-weight",
            str(CENTER_WEIGHT),
            "--out-dir",
            out_dir,
        ]
        for line in VALID_LINES:
            eval_if_missing(out_dir, line, [*base, "--line", line])
        eval_if_missing(
            out_dir,
            "Line9_holdout_tr1664_2377",
            [*base, "--line", "Line9", "--trace-start", "1664", "--trace-end", "2377"],
        )


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "v1_9d_mambavision_hybrid"
    seed1902 = f"outputs/run_gpu_paper_{arch}_final_seed1902_line9holdout"
    if (ROOT / seed1902).exists():
        eval_group(arch, "seed1902", [seed1902])
    ensemble_dirs = [
        f"outputs/run_gpu_paper_{arch}_final_seed{seed}_line9holdout"
        for seed in (1901, 1902, 1903)
        if (ROOT / f"outputs/run_gpu_paper_{arch}_final_seed{seed}_line9holdout").exists()
    ]
    if len(ensemble_dirs) == 3:
        eval_group(arch, "ensemble3seed", ensemble_dirs)


if __name__ == "__main__":
    main()
