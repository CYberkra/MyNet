from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Any

import numpy as np

from uavgpr_simlab.core.config import AppConfig
from uavgpr_simlab.core.scene_world import InterbedLayer, SceneObject, SceneWorld, TrajectoryModel
from uavgpr_simlab.sim.materials import default_material_ranges, sample_materials
from uavgpr_simlab.simulation.yingshan_families import family_spec, normalize_family
from uavgpr_simlab.simulation.domain_randomization import (
    DomainRandomizationConfig,
    generate_boulders,
    generate_fractures,
    apply_moisture_variation,
    apply_bedrock_steps,
)


def _smooth_noise(rng: random.Random, n: int, scale: float, passes: int = 4) -> np.ndarray:
    arr = np.array([rng.uniform(-scale, scale) for _ in range(n)], dtype=float)
    kernel = np.array([0.15, 0.25, 0.2, 0.25, 0.15], dtype=float)
    kernel /= kernel.sum()
    for _ in range(max(1, passes)):
        arr = np.convolve(arr, kernel, mode="same")
    return arr


def _pick_material(rng: random.Random, weighted: list[tuple[str, float]]) -> str:
    total = sum(w for _, w in weighted)
    v = rng.random() * total
    acc = 0.0
    for name, weight in weighted:
        acc += weight
        if v <= acc:
            return name
    return weighted[-1][0]


def _interp(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.interp(float(x), xs, ys))


def generate_scene_world(cfg: AppConfig, case_index: int, case_id: str | None = None) -> SceneWorld:
    """Generate a single Yingshan SceneWorld.

    The returned object is the sole randomised source for all variants.  Variant
    writers may mask components but must not call random again for geology,
    material maps, water zones, anomalies or external clutter.
    """

    seed = int(cfg.geology.random_seed) + int(case_index) * 1009
    rng = random.Random(seed)
    cid = case_id or f"case_{case_index + 1:06d}"
    family = normalize_family(getattr(cfg.geology, "scenario_family", None), case_index=case_index)
    spec = family_spec(family, case_index=case_index)

    col = max(float(cfg.geometry.geometry_column_width_m), float(cfg.geometry.dx_m))
    start_x = max(2.0, float(cfg.geometry.dx_m) * 20)
    trace_span = max(0, int(cfg.geometry.trace_count) - 1) * float(cfg.geometry.trace_step_m)
    min_scan_length = start_x + float(cfg.radar.tx_rx_offset_m) + trace_span + 2.0
    model_length_config = float(cfg.geometry.model_length_m)
    model_length_actual = max(model_length_config, min_scan_length)
    xs = np.arange(0.0, model_length_actual, col, dtype=float)
    if xs.size < 4:
        xs = np.linspace(0, model_length_actual, 4)

    depth = float(cfg.geometry.subsurface_depth_m)
    air = float(cfg.radar.nominal_flight_height_m + cfg.geometry.air_margin_m)
    dx_val = max(float(cfg.geometry.dx_m), 1e-6)
    domain_x = math.ceil((model_length_actual + 4.0) / dx_val) * dx_val
    domain_z = max(float(cfg.geometry.y_thickness_m), float(cfg.geometry.dz_m), 0.05)

    relief_target = rng.uniform(*spec.ground_relief_m)
    slope_deg = rng.uniform(float(cfg.geology.slope_deg_min), float(cfg.geology.slope_deg_max))
    # Build a deterministic relief component first, then add smooth local
    # roughness.  For the high-relief family this guarantees ground_relief_m
    # is in the requested 8–30 m range instead of being clipped to <1 m.
    trend = relief_target * (xs / max(model_length_actual, 1.0) - 0.5) * rng.choice([-1.0, 1.0])
    base_ground = depth + trend
    ground = base_ground + _smooth_noise(rng, xs.size, max(0.25, relief_target / 10.0), passes=5)
    if family in {"gentle_interbed", "terrace_paddy"} and rng.random() < 0.75:
        step_count = rng.randint(1, 4 if family == "terrace_paddy" else 2)
        for j in range(step_count):
            center = model_length_actual * (j + 1) / rng.uniform(3.5, 7.0)
            ground += rng.uniform(-0.6, 0.8) * (xs > center)
    if family == "wire_tree_endpoint":
        ground += 0.7 * np.sin(2 * np.pi * xs / max(model_length_actual, 1.0))
    if family == "cross_slope_high_relief":
        ground += 1.2 * np.sin(2 * np.pi * xs / max(model_length_actual, 1.0))
    # Keep the full model inside the FDTD domain while preserving family relief.
    # Domain y is derived after clipping, so high-relief worlds can be taller.
    ground = np.clip(ground, 4.0, max(5.0, depth + relief_target / 2.0 + 4.0))
    domain_y = float(np.max(ground) + air + 2.0)
    domain_y = math.ceil(domain_y / dx_val) * dx_val

    base_depth = rng.uniform(*spec.bedrock_depth_m)
    trend = rng.uniform(-2.0, 2.0) * (xs / max(model_length_actual, 1.0) - 0.5)
    rough = _smooth_noise(rng, xs.size, min(float(cfg.geology.interface_roughness_m), 1.2), passes=6)
    spoon = rng.uniform(0.0, 1.5) * np.sin(np.pi * xs / max(model_length_actual, 1.0))
    interface_depth = np.clip(base_depth + trend + rough + spoon, 2.5, depth - 1.0)
    interface_y = np.clip(ground - interface_depth, 0.8, ground - 0.8)

    # v0.8.0-alpha.14: constant-level flight means TX/RX move on the
    # same fixed platform height, not separate terrain-following heights.
    # Keep the entire trace path above the maximum terrain elevation by the
    # nominal UAV clearance.  If the current domain is not tall enough, expand
    # it rather than clipping the antenna path into a low-clearance hard case.
    rx_x = start_x + float(cfg.radar.tx_rx_offset_m)
    nominal_height = float(cfg.radar.nominal_flight_height_m)
    ground_max = float(np.max(ground))
    platform_y = ground_max + nominal_height
    if platform_y > domain_y - 0.8:
        domain_y = float(platform_y + max(1.0, float(cfg.geometry.air_margin_m)))
        domain_y = math.ceil(domain_y / dx_val) * dx_val
    source_y = float(platform_y)
    rx_y = float(platform_y)
    min_clearance = float(platform_y - ground_max)
    trajectory = TrajectoryModel(
        mode="constant_level",
        nominal_height_m=nominal_height,
        actual_height_profile_available=False,
        scan_start_x_m=float(start_x),
        scan_end_x_m=float(start_x + trace_span),
        source_y_m=float(source_y),
        receiver_y_m=float(rx_y),
        tx_rx_offset_m=float(cfg.radar.tx_rx_offset_m),
        note=(
            "FDTD TX/RX path is a constant-level UAV platform: "
            "source_y_m == receiver_y_m == max_ground_y + nominal_flight_height_m. "
            "This is not true terrain-following flight."
        ),
    )

    materials = sample_materials(rng, default_material_ranges())
    cover_materials = [
        _pick_material(
            rng,
            [
                ("silty_clay", 0.55),
                ("gravelly_silty_clay", 0.35),
                ("saturated_zone", 0.10 if rng.random() < spec.water_zone_probability else 0.02),
            ],
        )
        for _ in xs
    ]
    bedrock_materials = [
        _pick_material(rng, [("sandstone_bedrock", 0.58), ("weathered_mudstone", 0.42)])
        for _ in xs
    ]

    interbeds: list[InterbedLayer] = []
    layer_count = rng.randint(*spec.interbed_count)
    cumulative = 0.6
    for j in range(layer_count):
        thickness = rng.uniform(0.5, 3.0)
        dip = rng.uniform(0.0, 8.0) * rng.choice([-1.0, 1.0])
        line = interface_y - cumulative - math.tan(math.radians(dip)) * (xs - xs.mean()) * 0.03
        line = line + _smooth_noise(rng, xs.size, 0.20, passes=4)
        line = np.clip(line, 0.4, interface_y - 0.15)
        interbeds.append(
            InterbedLayer(
                layer_id=f"interbed_{j+1:02d}",
                material="weathered_mudstone" if j % 2 == 0 else "sandstone_bedrock",
                x_m=[float(v) for v in xs],
                y_m=[float(v) for v in line],
                thickness_m=float(thickness),
                dip_deg=float(dip),
            )
        )
        cumulative += thickness

    water_zones: list[SceneObject] = []
    water_count = 0
    if rng.random() < spec.water_zone_probability:
        water_count = 2 if family == "terrace_paddy" else 1
    for k in range(water_count):
        x0 = rng.uniform(4.0, max(6.0, model_length_actual - 10.0))
        x1 = min(model_length_actual - 1.0, x0 + rng.uniform(4.0, 12.0 if family == "terrace_paddy" else 8.0))
        gy = _interp((x0 + x1) / 2.0, xs, ground)
        water_zones.append(
            SceneObject(
                object_id=f"water_zone_{k+1:02d}",
                kind="saturated_zone",
                material="saturated_zone",
                x0_m=float(x0), x1_m=float(x1),
                y0_m=float(max(0.5, gy - rng.uniform(2.5, 6.5))),
                y1_m=float(max(0.6, gy - 0.20)),
                z1_m=domain_z,
                include_in=["raw", "target_only", "background_only"],
                note="FDTD-safe high-permittivity saturated/paddy cover surrogate.",
            )
        )

    anomaly_objects: list[SceneObject] = []
    if family == "deep_anomaly_21m" or rng.random() < spec.anomaly_probability:
        ax0 = rng.uniform(max(3.0, model_length_actual * 0.35), max(4.0, model_length_actual * 0.70))
        ax1 = min(model_length_actual - 1.0, ax0 + rng.uniform(3.0, 9.0))
        amid = (ax0 + ax1) / 2.0
        gy = _interp(amid, xs, ground)
        anomaly_depth = rng.uniform(18.0, 23.0) if family == "deep_anomaly_21m" else rng.uniform(12.0, 20.0)
        ay = max(0.4, gy - anomaly_depth)
        anomaly_objects.append(
            SceneObject(
                object_id="deep_anomaly_01" if family == "deep_anomaly_21m" else "anomaly_01",
                kind="deep_anomaly",
                material="fracture_zone",
                x0_m=float(ax0), x1_m=float(ax1),
                y0_m=float(max(0.2, ay - 0.5)),
                y1_m=float(ay + 0.5),
                z1_m=domain_z,
                include_in=["raw", "target_only"],
                note=f"Anomaly center depth {anomaly_depth:.2f} m below local ground; deep_anomaly_21m requires 18-23 m.",
                center_depth_m=float(anomaly_depth),
            )
        )

    external: list[SceneObject] = []
    if rng.random() < spec.external_clutter_probability:
        z1 = domain_z
        if rng.random() < spec.wire_probability:
            x0 = rng.uniform(1.0, max(3.0, model_length_actual * 0.15))
            x1 = min(model_length_actual - 0.5, x0 + rng.uniform(10.0, min(35.0, model_length_actual)))
            wy = min(max(_interp((x0 + x1) / 2.0, xs, ground) + rng.uniform(5.0, 12.0), 1.0), domain_y - 0.4)
            external.append(SceneObject("wire_01", "wire", "pec", float(x0), float(x1), float(wy), float(wy), 0.0, z1, radius_m=0.025, include_in=["raw", "clutter_only", "background_only"], note="PEC cylinder overhead wire surrogate."))
        if rng.random() < spec.tree_probability:
            for k in range(2 if family == "wire_tree_endpoint" else 1):
                tx = rng.choice([rng.uniform(1.0, max(2.0, model_length_actual * 0.25)), rng.uniform(max(2.0, model_length_actual * 0.75), model_length_actual - 1.0)])
                gy = _interp(tx, xs, ground)
                external.append(SceneObject(f"tree_{k+1:02d}", "tree", "vegetation", float(tx), float(tx), float(gy), float(min(domain_y - 0.3, gy + rng.uniform(3.0, 9.0))), 0.0, z1, radius_m=float(rng.uniform(0.08, 0.22)), include_in=["raw", "clutter_only", "background_only"], note="Vegetation cylinder surrogate."))
        if rng.random() < spec.building_probability:
            bx0 = rng.choice([rng.uniform(1.0, 6.0), rng.uniform(max(2.0, model_length_actual - 10.0), model_length_actual - 4.0)])
            gy = _interp(bx0, xs, ground)
            external.append(SceneObject("building_01", "building", "building_wall", float(bx0), float(min(model_length_actual - 0.5, bx0 + rng.uniform(1.5, 4.0))), float(gy), float(min(domain_y - 0.3, gy + rng.uniform(1.2, 3.2))), 0.0, z1, include_in=["raw", "clutter_only", "background_only"], note="Endpoint wall/building reflector surrogate."))

    # --- Domain randomization (sim-to-real gap reduction) ---
    dr_cfg_raw = getattr(cfg, "domain_randomization", None) or {}
    if isinstance(dr_cfg_raw, dict) and dr_cfg_raw.get("enabled"):
        profile = dr_cfg_raw.get("profile", "moderate")
        dr_cfg = DomainRandomizationConfig.preset(profile, random_seed=seed)
        # Override sub-configs from YAML
        for key in ["boulders", "fractures", "moisture", "bedrock_steps"]:
            if key in dr_cfg_raw and isinstance(dr_cfg_raw[key], dict):
                sub = dr_cfg_raw[key]
                existing = getattr(dr_cfg, key)
                for k, v in sub.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
    else:
        dr_cfg = DomainRandomizationConfig.preset("moderate", random_seed=seed)

    dr_rng = random.Random(seed + 7777)  # independent RNG for randomization
    dr_objects: list[SceneObject] = []

    if dr_cfg.enabled:
        dx_m = float(cfg.geometry.dx_m)
        # 1. Random boulders in overburden
        if dr_rng.random() < dr_cfg.boulders.probability:
            boulders = generate_boulders(
                dr_cfg.boulders, dr_rng,
                ground_x=xs, ground_y=ground, interface_y=interface_y,
                column_materials=cover_materials, domain_z_m=domain_z,
                min_cell_m=dx_m,
            )
            dr_objects.extend(boulders)

        # 2. Random fractures (dry or water-filled)
        if dr_rng.random() < dr_cfg.fractures.probability:
            fractures = generate_fractures(
                dr_cfg.fractures, dr_rng,
                ground_x=xs, ground_y=ground, interface_y=interface_y,
                domain_z_m=domain_z, min_cell_m=dx_m,
            )
            dr_objects.extend(fractures)

        # 3. Moisture variation — perturb cover eps_r and sigma
        # Extract current cover material properties for perturbation
        cover_eps = [materials.get(m, materials.get("silty_clay", {})).get("eps_r", 12.0)
                     for m in cover_materials]
        cover_sig = [materials.get(m, materials.get("silty_clay", {})).get("sigma", 0.005)
                     for m in cover_materials]
        new_cover_eps, new_cover_sig = apply_moisture_variation(
            dr_cfg.moisture, dr_rng,
            ground_x=xs,
            cover_materials=cover_materials,
            cover_eps_r=cover_eps,
            cover_sigma=cover_sig,
        )
        # Update cover material properties per column
        for i, m_name in enumerate(cover_materials):
            if m_name in materials:
                materials[m_name]["eps_r"] = round(float(new_cover_eps[i]), 4)
                materials[m_name]["sigma"] = round(float(new_cover_sig[i]), 6)

        # 4. Bedrock steps — add discontinuous offsets
        if dr_rng.random() < dr_cfg.bedrock_steps.probability:
            interface_y = apply_bedrock_steps(
                dr_cfg.bedrock_steps, dr_rng,
                ground_x=xs, ground_y=ground, interface_y=interface_y,
                min_cell_m=float(cfg.geometry.dx_m),
            )

    metadata: dict[str, Any] = {
        "source": "v0.8.0-alpha.2 SceneWorld generator",
        "field_context": "Yingshan Dayangping UavGPR landslide scenario; smoke/pilot families cover interbeds, paddy saturation, external clutter, deep anomaly and high relief.",
        "implemented_families": ["gentle_interbed", "terrace_paddy", "wire_tree_endpoint", "deep_anomaly_21m", "cross_slope_high_relief"],
        "material_caution": "surface/saturated water uses FDTD-safe high-permittivity surrogate, not high-fidelity true water modelling.",
        "trajectory_caution": "constant_level means fixed source/rx y in FDTD, not true terrain following.",
        "min_clearance_m": float(min_clearance),
        "family_description": spec.description,
        "radar": asdict(cfg.radar),
        "domain_randomization": dr_cfg.to_dict(),
        "domain_randomization_objects": [
            {"id": o.object_id, "kind": o.kind, "material": o.material, "x0": o.x0_m, "x1": o.x1_m, "y0": o.y0_m, "y1": o.y1_m, "note": o.note}
            for o in dr_objects
        ],
    }

    # Domain randomization: register new materials and merge objects
    for obj in dr_objects:
        if obj.note and "eps_r=" in obj.note:
            import re
            m = re.search(r"eps_r=([\d.]+),\s*sigma=([\d.]+)", obj.note)
            if m:
                materials[obj.material] = {
                    "eps_r": float(m.group(1)),
                    "sigma": float(m.group(2)),
                    "mu_r": 1.0,
                    "magnetic_loss": 0.0,
                    "_source": "domain_randomization",
                }
    # Merge into anomaly_objects (subsurface geological features, not external clutter)
    anomaly_objects = list(anomaly_objects) + dr_objects

    return SceneWorld.from_arrays(
        case_id=cid,
        family=family,
        random_seed=seed,
        ground_x=xs,
        ground_y=ground,
        bedrock_interface_y=interface_y,
        domain_x_m=domain_x,
        domain_y_m=domain_y,
        domain_z_m=domain_z,
        model_length_config_m=model_length_config,
        model_length_actual_m=domain_x,
        dx_m=float(cfg.geometry.dx_m),
        trace_count=int(cfg.geometry.trace_count),
        trace_spacing_m=float(cfg.geometry.trace_step_m),
        time_window_ns=float(cfg.radar.time_window_ns),
        samples=int(cfg.radar.frequency_points),
        trajectory=trajectory,
        materials=materials,
        cover_material_by_column=cover_materials,
        bedrock_material_by_column=bedrock_materials,
        interbed_layers=interbeds,
        anomaly_objects=anomaly_objects,
        water_zones=water_zones,
        external_clutter_objects=external,
        metadata=metadata,
    )
