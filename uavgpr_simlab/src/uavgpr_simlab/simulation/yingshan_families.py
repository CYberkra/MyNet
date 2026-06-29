from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneFamilySpec:
    name: str
    description: str
    bedrock_depth_m: tuple[float, float]
    ground_relief_m: tuple[float, float]
    interbed_count: tuple[int, int]
    external_clutter_probability: float
    water_zone_probability: float
    wire_probability: float = 0.0
    tree_probability: float = 0.0
    building_probability: float = 0.0
    anomaly_probability: float = 0.0
    saturated_zone_probability: float = 0.0


# v0.8.0-alpha.2 smoke families.  These are intentionally deterministic
# enough to satisfy dataset-structure checks while still randomising geometry
# within each family.
YINGSHAN_SCENEWORLD_FAMILIES: dict[str, SceneFamilySpec] = {
    "gentle_interbed": SceneFamilySpec(
        name="gentle_interbed",
        description="常规基覆界面 + 砂岩/泥岩连续互层；营山训练集基础样本。",
        bedrock_depth_m=(5.0, 18.0),
        ground_relief_m=(0.8, 4.5),
        interbed_count=(2, 5),
        external_clutter_probability=0.20,
        water_zone_probability=0.20,
        wire_probability=0.10,
        tree_probability=0.18,
        building_probability=0.04,
    ),
    "terrace_paddy": SceneFamilySpec(
        name="terrace_paddy",
        description="梯田/水田饱和带场景；必须生成 water_zones / saturated_zones。",
        bedrock_depth_m=(6.0, 20.0),
        ground_relief_m=(2.0, 8.0),
        interbed_count=(2, 5),
        external_clutter_probability=0.25,
        water_zone_probability=1.00,
        saturated_zone_probability=1.00,
        wire_probability=0.12,
        tree_probability=0.22,
        building_probability=0.06,
    ),
    "wire_tree_endpoint": SceneFamilySpec(
        name="wire_tree_endpoint",
        description="电线/树木/端点强杂波；必须生成 wires / trees / buildings 等 external_clutter_objects。",
        bedrock_depth_m=(8.0, 18.0),
        ground_relief_m=(1.5, 7.0),
        interbed_count=(2, 4),
        external_clutter_probability=0.0,     # 暂不做外部杂波
        water_zone_probability=0.15,
        wire_probability=0.0,                 # 暂不做
        tree_probability=0.0,                 # 暂不做
        building_probability=0.0,             # 暂不做
    ),
    "deep_anomaly_21m": SceneFamilySpec(
        name="deep_anomaly_21m",
        description="约 21m 深部异常体；必须生成 anomaly_objects，深度范围 18–23m。",
        bedrock_depth_m=(11.0, 20.0),
        ground_relief_m=(1.0, 6.0),
        interbed_count=(2, 5),
        external_clutter_probability=0.20,
        water_zone_probability=0.12,
        wire_probability=0.10,
        tree_probability=0.18,
        building_probability=0.05,
        anomaly_probability=1.00,
    ),
    "cross_slope_high_relief": SceneFamilySpec(
        name="cross_slope_high_relief",
        description="跨坡高起伏地形；ground_relief_m 目标范围 8–30m。",
        bedrock_depth_m=(8.0, 22.0),
        ground_relief_m=(8.0, 30.0),
        interbed_count=(2, 6),
        external_clutter_probability=0.0,     # 暂不做外部杂波
        water_zone_probability=0.20,
        wire_probability=0.0,
        tree_probability=0.0,
        building_probability=0.0,
    ),
}

# Backward compatible name used by v0.8.0-alpha.1.
YINGSHAN_ALPHA_FAMILIES = YINGSHAN_SCENEWORLD_FAMILIES
FAMILY_CYCLE = [
    "gentle_interbed",
    "terrace_paddy",
    "wire_tree_endpoint",
    "deep_anomaly_21m",
    "cross_slope_high_relief",
]

# Pilot family mix: 12-case cycle with one high-relief case (~8.3%).
# This keeps high-relief in the dataset without letting it dominate long pilot runs.
PILOT_FAMILY_CYCLE = [
    "gentle_interbed", "terrace_paddy", "wire_tree_endpoint", "deep_anomaly_21m",
    "gentle_interbed", "terrace_paddy", "wire_tree_endpoint", "deep_anomaly_21m",
    "gentle_interbed", "terrace_paddy", "wire_tree_endpoint", "cross_slope_high_relief",
]


def normalize_family(name: str | None, case_index: int = 0) -> str:
    """Return a supported SceneWorld family.

    Existing v0.7 run plans often use generic names.  SceneWorld smoke/pilot
    plans should cycle all five Yingshan families unless a concrete family is
    selected explicitly.
    """

    text = str(name or "").strip()
    if text in YINGSHAN_SCENEWORLD_FAMILIES:
        return text
    if text == "yingshan_pilot_all_families":
        return PILOT_FAMILY_CYCLE[case_index % len(PILOT_FAMILY_CYCLE)]
    return FAMILY_CYCLE[case_index % len(FAMILY_CYCLE)]


def family_spec(name: str | None, case_index: int = 0) -> SceneFamilySpec:
    return YINGSHAN_SCENEWORLD_FAMILIES[normalize_family(name, case_index=case_index)]
