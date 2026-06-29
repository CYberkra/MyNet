"""Domain randomization for gprMax SceneWorld generation.

Adds configurable subsurface heterogeneity to reduce sim-to-real gap:
- Random boulders (high-eps_r scatterers in overburden)
- Fractures (dry or water-filled narrow zones)
- Moisture variation (spatially correlated eps_r/sigma perturbation)
- Bedrock steps (discontinuous interface offsets)

All parameters are config-driven, tiered, and recorded in manifest.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uavgpr_simlab.core.scene_world import SceneObject


@dataclass
class BouldersConfig:
    enabled: bool = True
    probability: float = 0.60                       # per-scene generation probability
    count_per_10m: tuple[int, int] = (5, 15)        # density along profile
    radius_range_m: tuple[float, float] = (0.10, 0.35)  # min 3dx at 0.05m
    eps_r_range: tuple[float, float] = (12.0, 30.0)
    sigma_range: tuple[float, float] = (0.001, 0.01)
    min_clearance_to_interface_m: float = 0.5
    min_clearance_to_ground_m: float = 0.5


@dataclass
class FracturesConfig:
    enabled: bool = True
    count_per_20m: tuple[int, int] = (2, 8)
    width_range_m: tuple[float, float] = (0.05, 0.25)
    length_range_m: tuple[float, float] = (0.5, 4.0)
    dry_eps_r_range: tuple[float, float] = (2.0, 6.0)
    water_eps_r_range: tuple[float, float] = (20.0, 35.0)   # capped for dx=0.05m wavelength check
    dry_sigma_range: tuple[float, float] = (0.0001, 0.005)
    water_sigma_range: tuple[float, float] = (0.01, 0.08)
    dry_probability: float = 0.6                   # 60% dry, 40% water-filled
    min_clearance_to_interface_m: float = 0.5
    max_depth_from_ground_m: float = 10.0
    probability: float = 0.35                      # per-fracture generation probability


@dataclass
class MoistureVariationConfig:
    enabled: bool = True
    correlation_length_m: tuple[float, float] = (1.0, 5.0)
    eps_r_delta_range: tuple[float, float] = (-3.0, 5.0)
    sigma_multiplier_range: tuple[float, float] = (0.5, 3.0)


@dataclass
class BedrockStepsConfig:
    enabled: bool = True
    probability: float = 0.35
    step_count_per_50m: tuple[int, int] = (1, 3)
    offset_range_m: tuple[float, float] = (-1.0, 1.0)
    min_step_width_m: float = 2.0
    smooth_transition_cells: int = 5                 # linear taper between steps


@dataclass
class DomainRandomizationConfig:
    """Master config for all domain randomization options."""
    enabled: bool = True
    profile: str = "moderate"                        # none, basic, moderate, aggressive
    boulders: BouldersConfig = field(default_factory=BouldersConfig)
    fractures: FracturesConfig = field(default_factory=FracturesConfig)
    moisture: MoistureVariationConfig = field(default_factory=MoistureVariationConfig)
    bedrock_steps: BedrockStepsConfig = field(default_factory=BedrockStepsConfig)
    random_seed: int = 0
    save_to_manifest: bool = True

    @classmethod
    def preset(cls, profile: str, random_seed: int = 0) -> "DomainRandomizationConfig":
        """Create a preset configuration by tier."""
        if profile == "none":
            return cls(enabled=False, profile="none", random_seed=random_seed)
        elif profile == "basic":
            return cls(
                enabled=True, profile="basic", random_seed=random_seed,
                boulders=BouldersConfig(enabled=False),
                fractures=FracturesConfig(enabled=False),
                moisture=MoistureVariationConfig(
                    enabled=True,
                    eps_r_delta_range=(-1.5, 3.0),
                    sigma_multiplier_range=(0.7, 1.5),
                ),
                bedrock_steps=BedrockStepsConfig(enabled=False),
            )
        elif profile == "aggressive":
            return cls(
                enabled=True, profile="aggressive", random_seed=random_seed,
                boulders=BouldersConfig(
                    enabled=True,
                    count_per_10m=(15, 30),
                    radius_range_m=(0.10, 0.60),
                ),
                fractures=FracturesConfig(
                    enabled=True,
                    count_per_20m=(5, 15),
                    dry_probability=0.4,
                ),
                moisture=MoistureVariationConfig(
                    enabled=True,
                    eps_r_delta_range=(-5.0, 8.0),
                    sigma_multiplier_range=(0.3, 5.0),
                ),
                bedrock_steps=BedrockStepsConfig(
                    enabled=True,
                    step_count_per_50m=(2, 6),
                    offset_range_m=(-2.5, 2.5),
                ),
            )
        else:  # moderate (default)
            return cls(
                enabled=True, profile="moderate", random_seed=random_seed,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "profile": self.profile,
            "boulders": {k: v for k, v in self.boulders.__dict__.items()},
            "fractures": {k: v for k, v in self.fractures.__dict__.items()},
            "moisture": {k: v for k, v in self.moisture.__dict__.items()},
            "bedrock_steps": {k: v for k, v in self.bedrock_steps.__dict__.items()},
            "random_seed": self.random_seed,
        }


def _smooth_noise(rng: random.Random, n: int, scale: float, correlation: float) -> np.ndarray:
    """Generate spatially correlated smooth noise for moisture variation."""
    raw = np.array([rng.gauss(0, scale) for _ in range(n)], dtype=float)
    window = max(1, int(correlation))
    kernel = np.ones(window) / window
    # Apply multiple passes for smoother variation
    result = raw.copy()
    for _ in range(3):
        result = np.convolve(result, kernel, mode="same")
    # Normalize to preserve approximate scale
    result *= scale / (result.std() + 1e-8) * 0.7
    return result


def generate_boulders(
    cfg: BouldersConfig,
    rng: random.Random,
    *,
    ground_x: np.ndarray,
    ground_y: np.ndarray,
    interface_y: np.ndarray,
    column_materials: list[str],
    domain_z_m: float,
    min_cell_m: float,
) -> list[SceneObject]:
    """Generate random boulder scatterers in the overburden."""
    if not cfg.enabled:
        return []

    profile_length = float(ground_x[-1] - ground_x[0])
    count_per_10m = rng.randint(*cfg.count_per_10m)
    n_boulders = max(1, int(count_per_10m * profile_length / 10.0))

    # Empirically verified safe eps_r limit for dx=0.05m, 100 MHz Ricker
    max_safe_eps_r = 35.0

    objects: list[SceneObject] = []
    for i in range(n_boulders):
        # Random position within overburden
        x = rng.uniform(float(ground_x[0]) + 1.0, float(ground_x[-1]) - 1.0)
        gnd = float(np.interp(x, ground_x, ground_y))
        iface = float(np.interp(x, ground_x, interface_y))

        y_min = iface + cfg.min_clearance_to_interface_m
        y_max = gnd - cfg.min_clearance_to_ground_m
        if y_max <= y_min:
            continue

        y = rng.uniform(y_min, y_max)
        r = rng.uniform(*cfg.radius_range_m)

        # Minimum size constraint: radius >= 3 * cell_size
        if r < 3.0 * min_cell_m:
            r = 3.0 * min_cell_m

        eps_r = min(round(rng.uniform(*cfg.eps_r_range), 2), max_safe_eps_r)
        sigma = round(rng.uniform(*cfg.sigma_range), 6)

        material_name = f"boulder_{i+1:02d}"
        objects.append(SceneObject(
            object_id=f"boulder_{i+1:03d}",
            kind="boulder",
            material=material_name,
            x0_m=float(x - r), x1_m=float(x + r),
            y0_m=float(y - r), y1_m=float(y + r),
            z0_m=0.0, z1_m=domain_z_m,
            radius_m=r,
            center_depth_m=float(gnd - y),
            include_in=["raw", "target_only", "background_only"],
            note=f"eps_r={eps_r}, sigma={sigma} S/m",
        ))

    return objects


def generate_fractures(
    cfg: FracturesConfig,
    rng: random.Random,
    *,
    ground_x: np.ndarray,
    ground_y: np.ndarray,
    interface_y: np.ndarray,
    domain_z_m: float,
    min_cell_m: float,
) -> list[SceneObject]:
    """Generate random dry or water-filled fractures."""
    if not cfg.enabled:
        return []

    profile_length = float(ground_x[-1] - ground_x[0])
    count_per_20m = rng.randint(*cfg.count_per_20m)
    n_fractures = max(0, int(count_per_20m * profile_length / 20.0))

    objects: list[SceneObject] = []
    # Empirically verified safe eps_r limit for dx=0.05m, 100 MHz Ricker
    max_safe_eps_r = 35.0

    for i in range(n_fractures):
        x = rng.uniform(float(ground_x[0]) + 0.5, float(ground_x[-1]) - 0.5)
        gnd = float(np.interp(x, ground_x, ground_y))
        iface = float(np.interp(x, ground_x, interface_y))

        # Fracture starts from bedrock interface upward
        max_depth = min(gnd - iface, cfg.max_depth_from_ground_m)
        y_bottom = iface + cfg.min_clearance_to_interface_m
        length = rng.uniform(*cfg.length_range_m)
        y_top = min(gnd - 0.2, y_bottom + length)

        if y_top <= y_bottom + 0.1:
            continue

        width = rng.uniform(*cfg.width_range_m)

        is_dry = rng.random() < cfg.dry_probability
        if is_dry:
            eps_r = round(rng.uniform(*cfg.dry_eps_r_range), 2)
            sigma = round(rng.uniform(*cfg.dry_sigma_range), 6)
            kind = "dry_fracture"
        else:
            eps_r = round(rng.uniform(*cfg.water_eps_r_range), 2)
            sigma = round(rng.uniform(*cfg.water_sigma_range), 6)
            kind = "water_filled_fracture"
        # Clamp to grid-safe range
        eps_r = min(eps_r, max_safe_eps_r)

        material_name = f"fracture_{i+1:02d}"
        objects.append(SceneObject(
            object_id=f"fracture_{i+1:03d}",
            kind=kind,
            material=material_name,
            x0_m=float(x - width/2), x1_m=float(x + width/2),
            y0_m=float(y_bottom), y1_m=float(y_top),
            z0_m=0.0, z1_m=domain_z_m,
            note=f"eps_r={eps_r}, sigma={sigma} S/m, width={width:.3f}m",
            include_in=["raw", "target_only", "background_only"],
        ))

    return objects


def apply_moisture_variation(
    cfg: MoistureVariationConfig,
    rng: random.Random,
    *,
    ground_x: np.ndarray,
    cover_materials: list[str],
    cover_eps_r: list[float],
    cover_sigma: list[float],
) -> tuple[list[float], list[float]]:
    """Perturb cover material properties with spatially correlated noise."""
    if not cfg.enabled:
        return cover_eps_r, cover_sigma

    n = len(ground_x)
    if n < 3:
        return cover_eps_r, cover_sigma

    corr_length = rng.uniform(*cfg.correlation_length_m)
    # Convert correlation length to number of columns
    col_width = float(ground_x[1] - ground_x[0]) if n > 1 else 1.0
    corr_cells = max(1, int(corr_length / col_width))

    eps_delta = _smooth_noise(rng, n, rng.uniform(*cfg.eps_r_delta_range), corr_cells)
    sigma_mult = _smooth_noise(rng, n, rng.uniform(*cfg.sigma_multiplier_range), corr_cells)

    # Apply perturbations (clipped to physically valid ranges)
    new_eps = [max(3.0, min(35.0, e + d)) for e, d in zip(cover_eps_r, eps_delta)]
    new_sigma = [max(0.0001, min(0.15, s * max(0.1, m))) for s, m in zip(cover_sigma, sigma_mult)]

    return new_eps, new_sigma


def apply_bedrock_steps(
    cfg: BedrockStepsConfig,
    rng: random.Random,
    *,
    ground_x: np.ndarray,
    ground_y: np.ndarray,
    interface_y: np.ndarray,
    min_cell_m: float,
) -> np.ndarray:
    """Add discontinuous step offsets to bedrock interface."""
    if not cfg.enabled:
        return interface_y

    profile_length = float(ground_x[-1] - ground_x[0])
    count_per_50m = rng.randint(*cfg.step_count_per_50m)
    n_steps = max(0, int(count_per_50m * profile_length / 50.0))

    result = interface_y.copy()
    n = len(ground_x)

    for _ in range(n_steps):
        # Random step location
        step_x = rng.uniform(float(ground_x[0]) + cfg.min_step_width_m,
                             float(ground_x[-1]) - cfg.min_step_width_m)
        offset = rng.uniform(*cfg.offset_range_m)

        # Find the nearest column index
        step_idx = int(np.searchsorted(ground_x, step_x))

        # Apply offset to interface to the right of the step
        for i in range(step_idx, n):
            result[i] += offset

        # Smooth transition zone
        smooth_cells = cfg.smooth_transition_cells
        for i in range(max(0, step_idx - smooth_cells), min(n, step_idx + smooth_cells)):
            frac = (i - (step_idx - smooth_cells)) / (2 * smooth_cells)
            frac = max(0.0, min(1.0, frac))
            # Override: linear interpolation across the transition
            left_val = result[max(0, step_idx - smooth_cells - 1)]
            right_val = result[min(n - 1, step_idx + smooth_cells)]
            result[i] = left_val + (right_val - left_val) * frac

    # Topology constraints: interface must stay below ground
    for i in range(n):
        result[i] = min(result[i], ground_y[i] - 0.3)
        result[i] = max(result[i], 0.5)

    return result
