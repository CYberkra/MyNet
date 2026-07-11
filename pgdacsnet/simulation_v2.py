"""Physics and governance helpers for PGDA-CSNet simulation contract V2.

This module is intentionally independent of measured Line9 labels. It builds
2-D gprMax point-source control scenes whose horizontal sampling matches the
canonical 256-trace real-data window. gprMax itself runs at the CFL time step;
outputs are resampled to the 501-sample, 0--700 ns training grid only after the
solver has finished.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

C0_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class Material:
    """Non-dispersive gprMax material definition."""

    name: str
    epsilon_r: float
    conductivity_s_per_m: float
    mu_r: float = 1.0
    magnetic_loss_ohm_per_m: float = 0.0

    def validate(self) -> None:
        if not self.name or any(ch.isspace() for ch in self.name):
            raise ValueError(f"Invalid material identifier: {self.name!r}")
        if self.epsilon_r < 1.0:
            raise ValueError(f"epsilon_r must be >= 1 for {self.name}")
        if self.conductivity_s_per_m < 0.0:
            raise ValueError(f"conductivity must be non-negative for {self.name}")
        if self.mu_r <= 0.0:
            raise ValueError(f"mu_r must be positive for {self.name}")
        if self.magnetic_loss_ohm_per_m < 0.0:
            raise ValueError(f"magnetic loss must be non-negative for {self.name}")

    @property
    def velocity_m_per_s(self) -> float:
        return C0_M_PER_S / math.sqrt(self.epsilon_r * self.mu_r)

    def gprmax_command(self) -> str:
        self.validate()
        return (
            f"#material: {self.epsilon_r:.9g} {self.conductivity_s_per_m:.9g} "
            f"{self.mu_r:.9g} {self.magnetic_loss_ohm_per_m:.9g} {self.name}"
        )


@dataclass(frozen=True)
class GridSpec:
    """Numerical and canonical-output sampling."""

    dl_m: float = 0.0225
    trace_count: int = 256
    trace_spacing_m: float = 0.09
    canonical_time_window_ns: float = 700.0
    # Run the solver slightly beyond the canonical endpoint. gprMax records
    # samples at its CFL step, so an exact 700 ns solver request does not
    # guarantee that the last stored sample reaches the inclusive 700 ns
    # canonical endpoint used by the network.
    solver_time_window_ns: float = 701.0
    output_samples: int = 501
    pml_cells: tuple[int, int, int, int, int, int] = (20, 20, 0, 20, 20, 0)
    # Official guidance recommends sources/targets at least 15 cells from the
    # inner PML boundary and roughly 15--20 cells of air above the source.
    guard_cells: int = 20
    max_significant_frequency_multiplier: float = 3.0

    @property
    def time_window_ns(self) -> float:
        """Backward-compatible alias for the canonical network endpoint."""
        return self.canonical_time_window_ns

    @property
    def output_dt_ns(self) -> float:
        return self.canonical_time_window_ns / (self.output_samples - 1)

    @property
    def scan_span_m(self) -> float:
        return self.trace_spacing_m * (self.trace_count - 1)

    @property
    def pml_guard_m(self) -> float:
        return (self.pml_cells[0] + self.guard_cells) * self.dl_m

    def validate(self, max_epsilon_r: float, center_frequency_hz: float) -> None:
        if self.dl_m <= 0:
            raise ValueError("dl_m must be positive")
        if self.trace_count < 2:
            raise ValueError("trace_count must be >= 2")
        if self.output_samples != 501:
            raise ValueError("contract V2 requires 501 canonical output samples")
        if self.solver_time_window_ns <= self.canonical_time_window_ns:
            raise ValueError("solver time window must extend beyond the canonical endpoint")
        if self.guard_cells < 20:
            raise ValueError(
                "guard_cells must be >=20 to preserve official source/PML and top-air clearance"
            )
        if not math.isclose(self.output_dt_ns, 1.4, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"canonical output dt must be 1.4 ns, got {self.output_dt_ns}")
        assert_grid_multiple(self.trace_spacing_m, self.dl_m, "trace_spacing_m")
        if len(self.pml_cells) != 6 or any(int(v) < 0 for v in self.pml_cells):
            raise ValueError("pml_cells must have six non-negative integers")
        # 2-D model uses z as the invariant one-cell direction.
        if self.pml_cells[2] != 0 or self.pml_cells[5] != 0:
            raise ValueError("2-D z-invariant controls require z0/zmax PML = 0")
        f_max = center_frequency_hz * self.max_significant_frequency_multiplier
        wavelength_min = C0_M_PER_S / (f_max * math.sqrt(max_epsilon_r))
        cells_per_wavelength = wavelength_min / self.dl_m
        if cells_per_wavelength < 10.0:
            raise ValueError(
                "grid violates the gprMax lambda/10 rule: "
                f"{cells_per_wavelength:.3f} cells/wavelength"
            )


@dataclass(frozen=True)
class SourceSpec:
    model: str = "ideal_hertzian_line_source"
    polarization: str = "z"
    waveform: str = "ricker"
    center_frequency_hz: float = 100e6
    amplitude: float = 1.0
    tx_rx_offset_m: float = 0.18
    assumption_status: str = "provisional_until_hardware_geometry_confirmed"

    def validate(self, grid: GridSpec) -> None:
        if self.model != "ideal_hertzian_line_source":
            raise ValueError("control V2 currently supports ideal_hertzian_line_source only")
        if self.polarization != "z":
            raise ValueError("official 2-D x-y example uses z-polarized Hertzian dipole")
        if self.waveform != "ricker":
            raise ValueError("control V2 currently fixes a Ricker waveform")
        if self.center_frequency_hz <= 0 or self.amplitude <= 0:
            raise ValueError("source frequency and amplitude must be positive")
        if self.tx_rx_offset_m < 0:
            raise ValueError("tx_rx_offset_m must be non-negative")
        assert_grid_multiple(self.tx_rx_offset_m, grid.dl_m, "tx_rx_offset_m")


@dataclass(frozen=True)
class SceneArrays:
    trace_midpoint_x_m: np.ndarray
    source_x_m: np.ndarray
    receiver_x_m: np.ndarray
    ground_y_m: np.ndarray
    antenna_y_m: np.ndarray
    flight_height_agl_m: np.ndarray
    basal_depth_m: np.ndarray
    cover_thickness_m: np.ndarray
    weathered_thickness_m: np.ndarray
    cover_bottom_y_m: np.ndarray
    basal_interface_y_m: np.ndarray
    reference_arrival_time_ns: np.ndarray
    arrival_model: str

    @property
    def geometric_arrival_time_ns(self) -> np.ndarray:
        """Compatibility alias; see ``arrival_model`` before interpreting it."""
        return self.reference_arrival_time_ns


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_grid_multiple(value: float, dl_m: float, name: str, tol: float = 1e-9) -> int:
    cells = value / dl_m
    rounded = int(round(cells))
    if not math.isclose(cells, rounded, rel_tol=0.0, abs_tol=tol):
        raise ValueError(f"{name}={value} m is not an integer multiple of dl={dl_m} m")
    return rounded


def snap_to_grid(value: float | np.ndarray, dl_m: float) -> float | np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    snapped = np.rint(arr / dl_m) * dl_m
    if np.ndim(value) == 0:
        return float(snapped)
    return snapped


def layered_bistatic_twt_ns(
    thicknesses_m: Sequence[float],
    epsilon_r: Sequence[float],
    tx_rx_offset_m: float = 0.0,
) -> float:
    """Two-way specular reflection time for horizontal isotropic layers.

    A symmetric ray is solved using Snell's law. ``thicknesses_m`` contains the
    one-way thickness above the reflector, including air if present. For zero
    offset this reduces to ``2 * sum(d_i / v_i)``.
    """

    d = np.asarray(thicknesses_m, dtype=np.float64)
    er = np.asarray(epsilon_r, dtype=np.float64)
    if d.ndim != 1 or er.ndim != 1 or d.shape != er.shape or d.size == 0:
        raise ValueError("thicknesses_m and epsilon_r must be non-empty matching vectors")
    if np.any(d < 0) or np.any(er < 1):
        raise ValueError("layer thicknesses must be >=0 and epsilon_r >=1")
    velocities = C0_M_PER_S / np.sqrt(er)
    offset = float(tx_rx_offset_m)
    if offset < 0:
        raise ValueError("tx_rx_offset_m must be non-negative")
    if offset == 0.0:
        return float(2e9 * np.sum(d / velocities))

    # p has units s/m and must satisfy p*v_i < 1 in every layer.
    p_lo = 0.0
    p_hi = (1.0 - 1e-12) / float(np.max(velocities))

    def horizontal_distance(p: float) -> float:
        pv = p * velocities
        cos_theta = np.sqrt(np.maximum(1.0 - pv * pv, 1e-30))
        return float(2.0 * np.sum(d * pv / cos_theta))

    if horizontal_distance(p_hi) < offset:
        raise ValueError("requested offset cannot be represented by the layer stack")
    for _ in range(100):
        p_mid = 0.5 * (p_lo + p_hi)
        if horizontal_distance(p_mid) < offset:
            p_lo = p_mid
        else:
            p_hi = p_mid
    p = 0.5 * (p_lo + p_hi)
    pv = p * velocities
    cos_theta = np.sqrt(np.maximum(1.0 - pv * pv, 1e-30))
    twt_s = 2.0 * np.sum(d / (velocities * cos_theta))
    return float(twt_s * 1e9)


def smooth_interface_depth(
    x_local_m: np.ndarray,
    *,
    base_depth_m: float,
    slope: float = 0.0,
    sinusoid_amplitude_m: float = 0.0,
    sinusoid_wavelength_m: float | None = None,
) -> np.ndarray:
    x = np.asarray(x_local_m, dtype=np.float64)
    centered = x - 0.5 * (float(x.min()) + float(x.max()))
    out = np.full_like(x, float(base_depth_m)) + float(slope) * centered
    if sinusoid_amplitude_m:
        wavelength = float(sinusoid_wavelength_m or (x.max() - x.min()))
        if wavelength <= 0:
            raise ValueError("sinusoid_wavelength_m must be positive")
        out = out + float(sinusoid_amplitude_m) * np.sin(2.0 * np.pi * centered / wavelength)
    return out


def make_scene_arrays(
    *,
    grid: GridSpec,
    source: SourceSpec,
    basal_depth_m: np.ndarray,
    flight_height_agl_m: np.ndarray,
    cover_fraction: float,
    ground_y_m: float | np.ndarray,
    cover_material: Material,
    weathered_material: Material,
    arrival_model: str = "horizontal_layered_bistatic_exact",
) -> SceneArrays:
    source.validate(grid)
    n = grid.trace_count
    basal = np.asarray(basal_depth_m, dtype=np.float64)
    agl = np.asarray(flight_height_agl_m, dtype=np.float64)
    if basal.shape != (n,) or agl.shape != (n,):
        raise ValueError(f"basal_depth_m and flight_height_agl_m must have shape ({n},)")
    if np.any(basal <= 0) or np.any(agl <= 0):
        raise ValueError("basal depth and flight height must be positive")
    if not 0.05 <= cover_fraction <= 0.95:
        raise ValueError("cover_fraction must be in [0.05, 0.95]")

    left_margin = grid.pml_guard_m
    half_offset = source.tx_rx_offset_m / 2.0
    first_mid = snap_to_grid(left_margin + half_offset, grid.dl_m)
    trace_mid = first_mid + np.arange(n, dtype=np.float64) * grid.trace_spacing_m
    src_x = trace_mid - half_offset
    rx_x = trace_mid + half_offset

    ground = np.broadcast_to(np.asarray(ground_y_m, dtype=np.float64), (n,)).copy()
    ground = np.asarray(snap_to_grid(ground, grid.dl_m), dtype=np.float64)
    basal = np.asarray(snap_to_grid(basal, grid.dl_m), dtype=np.float64)
    agl = np.asarray(snap_to_grid(agl, grid.dl_m), dtype=np.float64)
    cover = np.asarray(snap_to_grid(basal * cover_fraction, grid.dl_m), dtype=np.float64)
    weathered = basal - cover
    if np.any(cover < 0.30) or np.any(weathered < 0.30):
        raise ValueError("cover and weathered layers must each be at least 0.30 m")

    antenna_y = ground + agl
    cover_bottom = ground - cover
    basal_y = ground - basal
    geometric = np.empty(n, dtype=np.float64)
    for i in range(n):
        geometric[i] = layered_bistatic_twt_ns(
            [agl[i], cover[i], weathered[i]],
            [1.0, cover_material.epsilon_r, weathered_material.epsilon_r],
            source.tx_rx_offset_m,
        )
    return SceneArrays(
        trace_midpoint_x_m=trace_mid,
        source_x_m=src_x,
        receiver_x_m=rx_x,
        ground_y_m=ground,
        antenna_y_m=antenna_y,
        flight_height_agl_m=agl,
        basal_depth_m=basal,
        cover_thickness_m=cover,
        weathered_thickness_m=weathered,
        cover_bottom_y_m=cover_bottom,
        basal_interface_y_m=basal_y,
        reference_arrival_time_ns=geometric,
        arrival_model=arrival_model,
    )


def compress_column_boxes(
    *,
    x0_m: float,
    dl_m: float,
    lower_y_m: np.ndarray,
    upper_y_m: np.ndarray,
    material_name: str,
    z_size_m: float,
    smoothing: str = "y",
) -> list[str]:
    """Run-length encode cell-aligned 2-D column boxes."""

    lower = np.rint(np.asarray(lower_y_m, dtype=np.float64) / dl_m).astype(np.int64)
    upper = np.rint(np.asarray(upper_y_m, dtype=np.float64) / dl_m).astype(np.int64)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower/upper arrays must be matching one-dimensional arrays")
    if np.any(upper <= lower):
        raise ValueError(f"zero/negative-thickness {material_name} geometry")
    commands: list[str] = []
    start = 0
    for i in range(1, lower.size + 1):
        boundary = i == lower.size or lower[i] != lower[start] or upper[i] != upper[start]
        if not boundary:
            continue
        x1 = x0_m + start * dl_m
        x2 = x0_m + i * dl_m
        y1 = lower[start] * dl_m
        y2 = upper[start] * dl_m
        commands.append(
            f"#box: {x1:.9g} {y1:.9g} 0 {x2:.9g} {y2:.9g} {z_size_m:.9g} "
            f"{material_name} {smoothing}"
        )
        start = i
    return commands


def resample_time_axis(
    data: np.ndarray,
    source_dt_s: float,
    *,
    time_window_ns: float = 700.0,
    output_samples: int = 501,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample gprMax CFL-step traces to the canonical 501-sample grid."""

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError("data must be [time] or [time, trace]")
    if source_dt_s <= 0:
        raise ValueError("source_dt_s must be positive")
    source_t_ns = np.arange(arr.shape[0], dtype=np.float64) * source_dt_s * 1e9
    target_t_ns = np.linspace(0.0, time_window_ns, output_samples, dtype=np.float64)
    if source_t_ns[-1] + 1e-9 < target_t_ns[-1]:
        raise ValueError(
            f"source output ends at {source_t_ns[-1]:.3f} ns before target {target_t_ns[-1]:.3f} ns"
        )
    if arr.ndim == 1:
        return target_t_ns, np.interp(target_t_ns, source_t_ns, arr).astype(np.float32)
    out = np.empty((output_samples, arr.shape[1]), dtype=np.float32)
    for trace in range(arr.shape[1]):
        out[:, trace] = np.interp(target_t_ns, source_t_ns, arr[:, trace])
    return target_t_ns, out


def extract_visible_phase(
    full_bscan: np.ndarray,
    control_bscan: np.ndarray,
    time_ns: np.ndarray,
    geometric_arrival_time_ns: np.ndarray,
    *,
    search_half_width_ns: float = 35.0,
    phase_half_width_ns: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a visible target phase from a matched-contrast difference.

    The control scene must be identical except that basal material contrast is
    removed. The envelope locates the target wavelet; the returned phase is the
    strongest signed sample near the envelope maximum. This is a post-solver
    observable and is kept separate from the geometric arrival.
    """

    from scipy.signal import hilbert

    full = np.asarray(full_bscan, dtype=np.float64)
    control = np.asarray(control_bscan, dtype=np.float64)
    t = np.asarray(time_ns, dtype=np.float64)
    geom = np.asarray(geometric_arrival_time_ns, dtype=np.float64)
    if full.shape != control.shape or full.ndim != 2:
        raise ValueError("full/control B-scans must be matching [time, trace] arrays")
    if t.shape != (full.shape[0],) or geom.shape != (full.shape[1],):
        raise ValueError("time/geometric-arrival shapes do not match B-scan")
    contrast = full - control
    envelope = np.abs(hilbert(contrast, axis=0))
    visible = np.full(full.shape[1], np.nan, dtype=np.float64)
    support = np.zeros(full.shape[1], dtype=np.float64)
    for j, center in enumerate(geom):
        if not np.isfinite(center):
            continue
        search = np.flatnonzero(np.abs(t - center) <= search_half_width_ns)
        if search.size == 0:
            continue
        env_idx = int(search[np.argmax(envelope[search, j])])
        phase = np.flatnonzero(np.abs(t - t[env_idx]) <= phase_half_width_ns)
        phase_idx = int(phase[np.argmax(np.abs(contrast[phase, j]))])
        visible[j] = t[phase_idx]
        local_peak = float(np.max(envelope[search, j]))
        baseline = float(np.median(envelope[:, j])) + 1e-12
        support[j] = local_peak / baseline
    return visible, support, contrast.astype(np.float32)


def gaussian_curve_mask(
    time_ns: np.ndarray,
    center_time_ns: np.ndarray,
    sigma_ns: float = 8.4,
) -> np.ndarray:
    t = np.asarray(time_ns, dtype=np.float64)[:, None]
    c = np.asarray(center_time_ns, dtype=np.float64)[None, :]
    mask = np.exp(-0.5 * ((t - c) / sigma_ns) ** 2)
    mask[:, ~np.isfinite(c[0])] = 0.0
    return mask.astype(np.float32)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
