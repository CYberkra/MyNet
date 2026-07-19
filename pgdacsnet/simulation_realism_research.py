"""Shared fold-safe metrics for measured-domain simulation realism research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, hilbert


FIT_LINES = ("Line3", "Line7", "LineL1")
VALIDATION_LINES = ("Line6",)
HELDOUT_LINES = ("Line9",)
REVIEW_ONLY_LINES = ("LineX1",)


@dataclass(frozen=True)
class SpectrumEstimate:
    frequency_hz: np.ndarray
    log_amplitude: np.ndarray
    trace_count: int


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0


def valid_measured_traces(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Return the strong, non-ignored, nominal-height trace mask."""

    valid = (
        (np.asarray(data["status_code"]) == 1)
        & (np.asarray(data["label_weight"]) > 0)
        & ~np.asarray(data["v15_final_ignore_trace"], dtype=bool)
    )
    if "flight_height_outside_planned_2_20_m" in data.files:
        valid &= ~np.asarray(
            data["flight_height_outside_planned_2_20_m"], dtype=bool
        )
    return valid


def aligned_packets(
    values: np.ndarray,
    time_ns: np.ndarray,
    path_ns: np.ndarray,
    *,
    half_width_ns: float = 56.0,
    sample_step_ns: float = 1.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate traces into a shared target-relative time window."""

    values = np.asarray(values, dtype=np.float64)
    time_ns = np.asarray(time_ns, dtype=np.float64)
    path_ns = np.asarray(path_ns, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != path_ns.size:
        raise ValueError("values must be time-by-trace and match path_ns")
    relative = np.arange(
        -half_width_ns,
        half_width_ns + 0.5 * sample_step_ns,
        sample_step_ns,
        dtype=np.float64,
    )
    packets = np.column_stack(
        [
            np.interp(center + relative, time_ns, values[:, trace])
            for trace, center in enumerate(path_ns)
        ]
    )
    return relative, packets


def packet_spectrum(
    packets: np.ndarray, sample_step_ns: float = 1.4
) -> SpectrumEstimate:
    """Estimate a robust shape-only packet amplitude spectrum."""

    packets = np.asarray(packets, dtype=np.float64)
    packets = packets - np.mean(packets, axis=0, keepdims=True)
    packets *= np.hanning(packets.shape[0])[:, None]
    norm = np.sqrt(np.sum(np.square(packets), axis=0, keepdims=True))
    packets = packets / np.maximum(norm, np.finfo(np.float64).tiny)
    amplitude = np.abs(np.fft.rfft(packets, axis=0))
    log_amplitude = np.median(
        np.log(np.maximum(amplitude, np.finfo(np.float64).tiny)), axis=1
    )
    frequency = np.fft.rfftfreq(packets.shape[0], d=sample_step_ns * 1e-9)
    return SpectrumEstimate(frequency, log_amplitude, packets.shape[1])


def measured_line_spectrum(path: Path) -> SpectrumEstimate:
    with np.load(path, allow_pickle=False) as data:
        valid = valid_measured_traces(data)
        values = np.asarray(data["raw_amplitude"], dtype=np.float64)[:, valid]
        time_ns = np.asarray(data["time_ns"], dtype=np.float64)
        path_ns = np.asarray(data["v15_final_center_time_ns"], dtype=np.float64)[
            valid
        ]
    values -= np.median(values, axis=1, keepdims=True)
    _, packets = aligned_packets(values, time_ns, path_ns)
    return packet_spectrum(packets)


def pool_spectra(estimates: list[SpectrumEstimate]) -> SpectrumEstimate:
    if not estimates:
        raise ValueError("at least one spectrum estimate is required")
    reference_frequency = estimates[0].frequency_hz
    if any(
        estimate.frequency_hz.shape != reference_frequency.shape
        or not np.allclose(estimate.frequency_hz, reference_frequency)
        for estimate in estimates[1:]
    ):
        raise ValueError("spectrum frequency axes do not match")
    return SpectrumEstimate(
        reference_frequency.copy(),
        np.mean(np.stack([item.log_amplitude for item in estimates]), axis=0),
        sum(item.trace_count for item in estimates),
    )


def bounded_system_response_db(
    simulated: SpectrumEstimate,
    measured: SpectrumEstimate,
    *,
    max_abs_db: float = 6.0,
    smooth_sigma_bins: float = 1.2,
    passband_hz: tuple[float, float] = (20e6, 180e6),
    taper_hz: float = 12e6,
) -> np.ndarray:
    """Build a smooth shape-only response without importing measured phase."""

    if not np.allclose(simulated.frequency_hz, measured.frequency_hz):
        raise ValueError("simulated and measured frequency axes must match")
    response = (measured.log_amplitude - simulated.log_amplitude) * (20.0 / np.log(10.0))
    core = (simulated.frequency_hz >= passband_hz[0]) & (
        simulated.frequency_hz <= passband_hz[1]
    )
    response -= np.median(response[core])
    response = gaussian_filter1d(response, smooth_sigma_bins, mode="nearest")
    response = np.clip(response, -max_abs_db, max_abs_db)
    frequency = simulated.frequency_hz
    left = np.clip((frequency - (passband_hz[0] - taper_hz)) / taper_hz, 0.0, 1.0)
    right = np.clip(((passband_hz[1] + taper_hz) - frequency) / taper_hz, 0.0, 1.0)
    weight = np.minimum(left, right)
    response *= 0.5 - 0.5 * np.cos(np.pi * weight)
    response[0] = 0.0
    return response


def apply_zero_phase_response(
    values: np.ndarray,
    sample_step_ns: float,
    response_frequency_hz: np.ndarray,
    response_db: np.ndarray,
    *,
    strength: float,
    preserve_global_rms: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    frequency = np.fft.rfftfreq(values.shape[0], d=sample_step_ns * 1e-9)
    interpolated_db = np.interp(
        frequency, response_frequency_hz, response_db, left=0.0, right=0.0
    )
    gain = np.power(10.0, strength * interpolated_db / 20.0)
    filtered = np.fft.irfft(
        np.fft.rfft(values, axis=0) * gain[:, None], n=values.shape[0], axis=0
    )
    if preserve_global_rms:
        filtered *= rms(values) / max(rms(filtered), np.finfo(np.float64).tiny)
    return filtered


def smooth_unit_noise(
    count: int,
    correlation_sigma_traces: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a reproducible zero-mean, unit-variance smooth 1-D field."""

    if count < 2:
        raise ValueError("count must be at least two")
    if correlation_sigma_traces <= 0:
        raise ValueError("correlation sigma must be positive")
    values = gaussian_filter1d(
        rng.standard_normal(count), correlation_sigma_traces, mode="reflect"
    )
    values -= np.mean(values)
    return values / max(float(np.std(values)), np.finfo(np.float64).tiny)


def apply_trace_gain_delay(
    values: np.ndarray,
    time_ns: np.ndarray,
    log_gain: np.ndarray,
    delay_ns: np.ndarray,
) -> np.ndarray:
    """Apply per-trace multiplicative gain and continuous time delay."""

    values = np.asarray(values, dtype=np.float64)
    time_ns = np.asarray(time_ns, dtype=np.float64)
    log_gain = np.asarray(log_gain, dtype=np.float64)
    delay_ns = np.asarray(delay_ns, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != log_gain.size:
        raise ValueError("gain must match the trace dimension")
    if delay_ns.shape != log_gain.shape:
        raise ValueError("delay and gain must have the same shape")
    output = np.empty_like(values)
    for trace in range(values.shape[1]):
        output[:, trace] = np.interp(
            time_ns - delay_ns[trace],
            time_ns,
            values[:, trace],
            left=0.0,
            right=0.0,
        ) * np.exp(log_gain[trace])
    return output


def target_background_ratio(
    values: np.ndarray, time_ns: np.ndarray, path_ns: np.ndarray
) -> float:
    suppressed = values - np.median(values, axis=1, keepdims=True)
    target: list[np.ndarray] = []
    background: list[np.ndarray] = []
    for trace, center in enumerate(path_ns):
        target.append(suppressed[np.abs(time_ns - center) <= 14.0, trace])
        background.append(
            suppressed[
                ((time_ns >= center - 98.0) & (time_ns <= center - 42.0))
                | ((time_ns >= center + 42.0) & (time_ns <= center + 98.0)),
                trace,
            ]
        )
    return rms(np.concatenate(target)) / max(
        rms(np.concatenate(background)), np.finfo(np.float64).tiny
    )


def packet_metrics(
    values: np.ndarray,
    time_ns: np.ndarray,
    path_ns: np.ndarray,
    *,
    spectrum_reference: SpectrumEstimate | None = None,
) -> dict[str, float]:
    suppressed = values - np.median(values, axis=1, keepdims=True)
    relative, packets = aligned_packets(suppressed, time_ns, path_ns)
    template = np.median(packets, axis=1)
    envelope = np.abs(hilbert(packets, axis=0))
    amplitudes = np.max(envelope[np.abs(relative) <= 14.0], axis=0)
    centered_template = template - np.mean(template)
    correlations = []
    for trace in range(packets.shape[1]):
        centered = packets[:, trace] - np.mean(packets[:, trace])
        denominator = np.sqrt(
            np.sum(np.square(centered)) * np.sum(np.square(centered_template))
        )
        correlations.append(
            float(np.sum(centered * centered_template) / denominator)
            if denominator > 0
            else 0.0
        )
    spectrum = packet_spectrum(packets)
    amplitude = np.exp(spectrum.log_amplitude)
    positive = spectrum.frequency_hz > 0
    signed_peaks, _ = find_peaks(
        np.abs(centered_template),
        prominence=max(0.08 * float(np.max(np.abs(centered_template))), 1e-30),
    )
    metrics = {
        "target_to_background": target_background_ratio(values, time_ns, path_ns),
        "target_envelope_cv": float(
            np.std(amplitudes)
            / max(np.mean(amplitudes), np.finfo(np.float64).tiny)
        ),
        "target_dropout_fraction": float(
            np.mean(amplitudes < 0.25 * np.median(amplitudes))
        ),
        "aligned_template_correlation_median": float(np.median(correlations)),
        "significant_lobe_count": int(signed_peaks.size),
        "aligned_peak_frequency_mhz": float(
            spectrum.frequency_hz[positive][np.argmax(amplitude[positive])] / 1e6
        ),
        "aligned_spectral_centroid_mhz": float(
            np.sum(spectrum.frequency_hz[positive] * amplitude[positive])
            / max(np.sum(amplitude[positive]), np.finfo(np.float64).tiny)
            / 1e6
        ),
    }
    if spectrum_reference is not None:
        band = (spectrum.frequency_hz >= 20e6) & (spectrum.frequency_hz <= 180e6)
        delta = (
            spectrum.log_amplitude[band] - spectrum_reference.log_amplitude[band]
        ) * (20.0 / np.log(10.0))
        delta -= np.median(delta)
        metrics["shape_spectrum_rmse_db"] = float(np.sqrt(np.mean(np.square(delta))))
    return metrics


def display_process(
    values: np.ndarray, time_ns: np.ndarray, power: float = 1.5
) -> np.ndarray:
    suppressed = values - np.median(values, axis=1, keepdims=True)
    gain = np.power(np.clip(time_ns / 500.0, 0.02, 1.0), power)
    return suppressed * gain[:, None]
