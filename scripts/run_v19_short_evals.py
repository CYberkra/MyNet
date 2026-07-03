from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
VALID_LINES = ["Line3", "Line6", "Line7", "Line9", "LineL1"]
CENTER_WEIGHTS = [0.0, 0.5, 1.0]


def run(cmd):
    print(" ".join(str(x) for x in cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode:
        print(f"WARNING nonzero eval exit: {result.returncode}", flush=True)


def eval_if_missing(out_dir, line, eval_name, cmd):
    metric_path = ROOT / out_dir / f"{eval_name}_full_metrics.csv"
    if metric_path.exists():
        print(f"SKIP existing {metric_path}", flush=True)
        return
    run(cmd)


def main():
    manifest = json.loads((ROOT / "reports" / "paper_v1_9_candidate_manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        arch = item["model_arch"]
        run_dir = f"outputs/run_gpu_paper_{arch}_short_line9holdout"
        for weight in CENTER_WEIGHTS:
            suffix = f"w{int(weight * 100):03d}"
            out_dir = f"outputs/eval_paper_{arch}_short_{suffix}_p065"
            for line in VALID_LINES:
                eval_if_missing(out_dir, line, line, [
                    PY, "scripts/eval_full_line.py",
                    "--line", line,
                    "--run-dirs", run_dir,
                    "--checkpoint", "final",
                    "--presence-thr", "0.45",
                    "--path-prob-thr", "0.65",
                    "--search-min-ns", "240",
                    "--search-max-ns", "500",
                    "--dp-max-jump", "6",
                    "--dp-smooth-weight", "0.16",
                    "--center-fusion-weight", str(weight),
                    "--out-dir", out_dir,
                ])
            eval_if_missing(out_dir, "Line9", "Line9_holdout_tr1664_2377", [
                PY, "scripts/eval_full_line.py",
                "--line", "Line9",
                "--run-dirs", run_dir,
                "--checkpoint", "final",
                "--presence-thr", "0.45",
                "--path-prob-thr", "0.65",
                "--search-min-ns", "240",
                "--search-max-ns", "500",
                "--dp-max-jump", "6",
                "--dp-smooth-weight", "0.16",
                "--trace-start", "1664",
                "--trace-end", "2377",
                "--center-fusion-weight", str(weight),
                "--out-dir", out_dir,
            ])


if __name__ == "__main__":
    main()
