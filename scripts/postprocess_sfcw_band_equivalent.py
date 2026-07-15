#!/usr/bin/env python3
"""Create a band-limited SFCW-equivalent view of a broadband FDTD pair.

This is not a tone-by-tone stepped-frequency FDTD simulation. It applies a
documented raised-cosine 20-170 MHz passband to the solved receiver response,
then resamples it to the canonical 0-700 ns grid. The 80 m guarded pilots are
only interpreted inside their 0-500 ns protected window.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgdacsnet.simulation_v2 import resample_time_axis  # noqa: E402
from scripts.postprocess_physical_sim_v2 import read_merged_bscan  # noqa: E402


def raised_cosine_sfcw_window(
    frequency_hz: np.ndarray,
    *,
    low_hz: float = 20e6,
    high_hz: float = 170e6,
    taper_hz: float = 10e6,
) -> np.ndarray:
    """Return a real symmetric passband with in-band cosine endpoint tapers."""
    if not (0 < low_hz < high_hz):
        raise ValueError("frequency band must satisfy 0 < low < high")
    if taper_hz < 0 or taper_hz * 2 >= high_hz - low_hz:
        raise ValueError("taper must be non-negative and narrower than half the band")
    frequency = np.abs(np.asarray(frequency_hz, dtype=np.float64))
    window = np.zeros_like(frequency)
    inside = (frequency >= low_hz) & (frequency <= high_hz)
    window[inside] = 1.0
    if taper_hz > 0:
        lower = inside & (frequency < low_hz + taper_hz)
        upper = inside & (frequency > high_hz - taper_hz)
        window[lower] = 0.5 - 0.5 * np.cos(np.pi * (frequency[lower] - low_hz) / taper_hz)
        window[upper] = 0.5 - 0.5 * np.cos(np.pi * (high_hz - frequency[upper]) / taper_hz)
    return window


def apply_sfcw_band_equivalent(
    data: np.ndarray,
    dt_s: float,
    *,
    low_hz: float = 20e6,
    high_hz: float = 170e6,
    taper_hz: float = 10e6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Band-limit [time, trace] data without source deconvolution.

    Zero padding improves plotting interpolation but does not claim additional
    independent frequency resolution; both values are recorded in the report.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 4:
        raise ValueError("data must be a nontrivial [time, trace] array")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    n_time = values.shape[0]
    nfft = 1 << math.ceil(math.log2(n_time))
    frequency_hz = np.fft.rfftfreq(nfft, d=dt_s)
    nyquist_hz = float(frequency_hz[-1])
    if high_hz > nyquist_hz:
        raise ValueError(f"requested high frequency {high_hz:g} exceeds Nyquist {nyquist_hz:g}")
    window = raised_cosine_sfcw_window(frequency_hz, low_hz=low_hz, high_hz=high_hz, taper_hz=taper_hz)
    transformed = np.fft.rfft(values, n=nfft, axis=0)
    filtered = np.fft.irfft(transformed * window[:, None], n=nfft, axis=0)[:n_time]
    metadata = {
        "source_sample_count": float(n_time),
        "zero_padded_fft_sample_count": float(nfft),
        "native_frequency_resolution_mhz": float(1.0 / (n_time * dt_s) / 1e6),
        "fft_bin_spacing_mhz": float(1.0 / (nfft * dt_s) / 1e6),
        "nyquist_mhz": nyquist_hz / 1e6,
    }
    return filtered.astype(np.float32), frequency_hz.astype(np.float64), window.astype(np.float64), metadata


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _panel(values: np.ndarray, scale: float, size: tuple[int, int]) -> Image.Image:
    clipped = np.clip(values / max(scale, 1e-30), -1.0, 1.0)
    gray = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _write_preview(
    output_path: Path,
    time_ns: np.ndarray,
    full: np.ndarray,
    control: np.ndarray,
    contrast: np.ndarray,
    *,
    low_mhz: float,
    high_mhz: float,
) -> None:
    protected = time_ns <= 500.0
    data = (full[protected], control[protected], contrast[protected])
    scale = float(np.quantile(np.abs(np.concatenate([item.ravel() for item in data])), 0.995))
    width, height = 1680, 760
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, bold=True)
    text_font = _font(20)
    panel_width, panel_height = 500, 510
    lefts = (60, 590, 1120)
    titles = ("Full scene", "No-basal control", "Signed full - control")
    draw.text((60, 30), "100 MHz Ricker FDTD with SFCW-band-equivalent postprocessing", fill="black", font=title_font)
    draw.text(
        (60, 72),
        f"Raised-cosine {low_mhz:.0f}-{high_mhz:.0f} MHz band; 0-500 ns protected preview; not tone-by-tone SFCW.",
        fill="black",
        font=text_font,
    )
    for left, title, values in zip(lefts, titles, data):
        draw.text((left, 120), title, fill="black", font=text_font)
        canvas.paste(_panel(values, scale, (panel_width, panel_height)), (left, 155))
        draw.rectangle((left, 155, left + panel_width, 155 + panel_height), outline="black", width=2)
    draw.text((60, 700), "Common robust gain across all panels. Frequency grid remains provisional until exported SFCW tone metadata is available.", fill="black", font=text_font)
    canvas.save(output_path)


def _write_difference_diagnostic(
    output_path: Path,
    time_ns: np.ndarray,
    contrast: np.ndarray,
    *,
    selected_trace_indices: list[int] | None,
) -> None:
    """Render sparse full-minus-control traces at their own robust gain.

    A common B-scan gain is the honest way to compare direct wave and target
    response, but it makes a small causal difference visually disappear.
    This diagnostic intentionally uses a separate gain and says so on-image.
    """
    protected = time_ns <= 500.0
    time = time_ns[protected]
    values = contrast[protected]
    trace_count = values.shape[1]
    labels = selected_trace_indices or list(range(trace_count))
    if len(labels) != trace_count:
        labels = list(range(trace_count))
    width, height = 1680, 860
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, bold=True)
    text_font = _font(19)
    draw.text((60, 28), "Causal basal response diagnostic: signed full - no-basal", fill="black", font=title_font)
    draw.text(
        (60, 70),
        "Each trace uses the same difference-only robust gain. This exposes a weak causal response; it is not comparable to the full-scene amplitude scale.",
        fill="black",
        font=text_font,
    )
    scale = float(np.quantile(np.abs(values), 0.995))
    columns = min(trace_count, 4)
    panel_width = 360
    panel_height = 650
    gap = 40
    start_x = 70
    top = 145
    for index in range(trace_count):
        row, column = divmod(index, columns)
        left = start_x + column * (panel_width + gap)
        panel_top = top + row * (panel_height + 45)
        draw.rectangle((left, panel_top, left + panel_width, panel_top + panel_height), outline="black", width=2)
        draw.text((left, panel_top - 28), f"trace {labels[index]}", fill="black", font=text_font)
        center_x = left + panel_width // 2
        draw.line((center_x, panel_top, center_x, panel_top + panel_height), fill=(170, 170, 170), width=1)
        draw.text((left + 5, panel_top + 5), "0 ns", fill="black", font=text_font)
        draw.text((left + 5, panel_top + panel_height - 25), "500 ns", fill="black", font=text_font)
        x = center_x + (values[:, index] / max(scale, 1e-30)) * (panel_width * 0.43)
        y = panel_top + (time / 500.0) * panel_height
        draw.line([(float(px), float(py)) for px, py in zip(x, y)], fill=(24, 76, 160), width=2)
    footer_y = min(height - 45, top + (math.ceil(trace_count / columns)) * (panel_height + 45))
    draw.text((60, footer_y), f"Difference-only P99.5 gain = {scale:.3e}; protected window 0-500 ns.", fill="black", font=text_font)
    canvas.save(output_path)


def process_pair(
    case_dir: Path,
    output_dir: Path,
    *,
    low_mhz: float = 20.0,
    high_mhz: float = 170.0,
    taper_mhz: float = 10.0,
    component: str = "Ez",
) -> dict[str, object]:
    case_dir = case_dir.resolve()
    output_dir = output_dir.resolve()
    manifest = json.loads((case_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    if not bool(manifest.get("target_presence")):
        raise ValueError("SFCW band-equivalent pair processing requires a positive full/control case")
    source = manifest.get("source", {})
    if str(source.get("waveform")) != "ricker" or not np.isclose(float(source.get("center_frequency_hz", 0.0)), 100e6):
        raise ValueError("case is not the required 100 MHz Ricker ablation deck")
    full_dt, full_raw, full_attrs = read_merged_bscan(
        _resolve_gprmax_output(case_dir, "full_scene"), component=component
    )
    control_dt, control_raw, control_attrs = read_merged_bscan(
        _resolve_gprmax_output(case_dir, "no_basal_contrast_control"), component=component
    )
    if full_raw.shape != control_raw.shape or not np.isclose(full_dt, control_dt, rtol=0, atol=1e-18):
        raise RuntimeError("full/control pair is not time-aligned")
    full, frequency_hz, window, spectral_metadata = apply_sfcw_band_equivalent(
        full_raw, full_dt, low_hz=low_mhz * 1e6, high_hz=high_mhz * 1e6, taper_hz=taper_mhz * 1e6
    )
    control, _, _, _ = apply_sfcw_band_equivalent(
        control_raw, control_dt, low_hz=low_mhz * 1e6, high_hz=high_mhz * 1e6, taper_hz=taper_mhz * 1e6
    )
    time_ns, canonical_full = resample_time_axis(full, full_dt, time_window_ns=700.0, output_samples=501)
    _, canonical_control = resample_time_axis(control, control_dt, time_window_ns=700.0, output_samples=501)
    contrast = canonical_full - canonical_control
    run_manifest_path = case_dir / "run_manifest.json"
    selected_indices: list[int] | None = None
    if run_manifest_path.is_file():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        candidate = run_manifest.get("selected_trace_indices_zero_based")
        if isinstance(candidate, list) and all(isinstance(item, int) for item in candidate):
            selected_indices = candidate
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "sfcw_band_equivalent_canonical.npz",
        time_ns=time_ns,
        full_scene=canonical_full,
        no_basal_contrast_control=canonical_control,
        signed_full_minus_control=contrast,
        frequency_hz=frequency_hz,
        spectral_window=window,
    )
    _write_preview(
        output_dir / "sfcw_band_equivalent_preview.png",
        time_ns,
        canonical_full,
        canonical_control,
        contrast,
        low_mhz=low_mhz,
        high_mhz=high_mhz,
    )
    _write_difference_diagnostic(
        output_dir / "sfcw_band_equivalent_difference_diagnostic.png",
        time_ns,
        contrast,
        selected_trace_indices=selected_indices,
    )
    protected = time_ns <= 500.0
    report: dict[str, object] = {
        "schema": "pgda_sfcw_band_equivalent_v1",
        "case_id": manifest["case_id"],
        "formal_training_allowed": False,
        "method": "post-solver raised-cosine frequency-band limitation",
        "is_direct_tone_by_tone_sfcw_forward_simulation": False,
        "source_deconvolution_applied": False,
        "frequency_grid_status": "provisional: the project PDF reports 20-170 MHz, 1 MHz steps, and 501 points, which need hardware-export confirmation",
        "band_mhz": {"low": low_mhz, "high": high_mhz, "endpoint_taper": taper_mhz},
        "valid_time_window_ns": [0.0, 500.0],
        "diagnostic_only_time_window_ns": [500.0, 700.0],
        "raw_shape": list(full_raw.shape),
        "canonical_shape": list(canonical_full.shape),
        "solver_dt_s": full_dt,
        "spectral_metadata": spectral_metadata,
        "gprmax_versions": {"full": full_attrs.get("gprMax"), "control": control_attrs.get("gprMax")},
        "protected_rms": {
            "full": float(np.sqrt(np.mean(np.square(canonical_full[protected])))),
            "control": float(np.sqrt(np.mean(np.square(canonical_control[protected])))),
            "signed_full_minus_control": float(np.sqrt(np.mean(np.square(contrast[protected])))),
        },
        "artifacts": {
            "canonical_npz": "sfcw_band_equivalent_canonical.npz",
            "preview": "sfcw_band_equivalent_preview.png",
            "difference_diagnostic": "sfcw_band_equivalent_difference_diagnostic.png",
        },
    }
    (output_dir / "sfcw_band_equivalent_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _resolve_gprmax_output(case_dir: Path, stem: str) -> Path:
    """Resolve a merged acquisition or gprMax's normal one-trace smoke output."""
    candidates = (case_dir / f"{stem}_merged.out", case_dir / f"{stem}.out")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    names = ", ".join(candidate.name for candidate in candidates)
    raise FileNotFoundError(f"missing gprMax output for {stem}; expected one of: {names}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="Completed staged full/control gprMax run directory.")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory (default: <case>/sfcw_band_equivalent).")
    parser.add_argument("--low-mhz", type=float, default=20.0)
    parser.add_argument("--high-mhz", type=float, default=170.0)
    parser.add_argument("--taper-mhz", type=float, default=10.0)
    parser.add_argument("--component", default="Ez")
    args = parser.parse_args()
    output = args.output_dir or args.case_dir / "sfcw_band_equivalent"
    report = process_pair(
        args.case_dir,
        output,
        low_mhz=args.low_mhz,
        high_mhz=args.high_mhz,
        taper_mhz=args.taper_mhz,
        component=args.component,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
