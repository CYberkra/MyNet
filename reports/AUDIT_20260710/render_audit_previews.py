from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def render_batch(batch: Path, output_name: str, ncols: int = 4) -> None:
    cases = sorted(path.parent.parent for path in batch.rglob("raw/bscan.npy"))
    nrows = (len(cases) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    target_time = np.linspace(0.0, 700.0, 501)
    for axis, case in zip(axes.ravel(), cases):
        raw = np.load(case / "raw" / "bscan.npy").astype(np.float32)
        native_time = np.linspace(0.0, 700.0, raw.shape[0])
        resampled = np.empty((501, raw.shape[1]), dtype=np.float32)
        for trace in range(raw.shape[1]):
            resampled[:, trace] = np.interp(target_time, native_time, raw[:, trace])
        resampled -= np.median(resampled, axis=1, keepdims=True)
        resampled *= (0.02 + (target_time / 700.0) ** 2.2)[:, None]
        limit = max(float(np.percentile(np.abs(resampled), 99.2)), 1e-9)
        axis.imshow(
            resampled,
            aspect="auto",
            origin="upper",
            extent=[0, resampled.shape[1] - 1, 700, 0],
            cmap="gray",
            vmin=-limit,
            vmax=limit,
        )
        visible_time = np.load(case / "labels" / "target_visible_phase_time_ns.npy")
        axis.plot(np.arange(visible_time.size), visible_time, color="#e53935", linewidth=1.0)
        axis.set_title(case.name, fontsize=8)
        axis.set_xlabel("trace", fontsize=7)
        axis.set_ylabel("time (ns)", fontsize=7)
        axis.tick_params(labelsize=6)
    for axis in axes.ravel()[len(cases) :]:
        axis.axis("off")
    fig.savefig(OUT / output_name, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    runs = ROOT / "data" / "PGDA_SYNTH_DATASET_V1" / "03_runs"
    render_batch(runs / "batch_001_line9_style_12cases", "batch1_12case_visual_audit.png")
    render_batch(
        runs / "BATCH_003_SHALLOW_GENERALIZATION_24CASES_V3_AUDITED_20260704",
        "batch3_20case_visual_audit.png",
    )
