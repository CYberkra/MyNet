from pathlib import Path
import shutil
import torch


ROOT = Path(__file__).resolve().parents[1]


def main():
    for run_dir in sorted((ROOT / "outputs").glob("run_gpu_paper_v1_9*_short_line9holdout")):
        best = run_dir / "checkpoint_best.pt"
        final = run_dir / "checkpoint_final.pt"
        if not best.exists():
            print(run_dir.name, "missing checkpoint_best.pt")
            continue
        shutil.copy2(best, final)
        ckpt = torch.load(final, map_location="cpu", weights_only=False)
        print(run_dir.name, "epoch", ckpt.get("epoch"), "history", len(ckpt.get("history", [])), "->", final)


if __name__ == "__main__":
    main()
