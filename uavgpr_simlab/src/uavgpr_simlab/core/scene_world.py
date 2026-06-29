from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _arr(values: Any) -> list[float]:
    """Convert numpy/list-like numeric arrays to JSON-safe float lists."""
    return [float(x) for x in np.asarray(values, dtype=float).ravel().tolist()]


@dataclass(frozen=True)
class TrajectoryModel:
    """UavGPR source/receiver trajectory description for one simulated scene."""

    mode: str
    nominal_height_m: float
    actual_height_profile_available: bool
    scan_start_x_m: float
    scan_end_x_m: float
    source_y_m: float
    receiver_y_m: float
    tx_rx_offset_m: float
    note: str = "FDTD source/rx path is constant level, not true terrain following."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterbedLayer:
    """A continuous sandstone/mudstone interbed line below the bedrock interface."""

    layer_id: str
    material: str
    x_m: list[float]
    y_m: list[float]
    thickness_m: float
    dip_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneObject:
    """Serializable object descriptor for anomalies, water zones or clutter."""

    object_id: str
    kind: str
    material: str
    x0_m: float
    x1_m: float
    y0_m: float
    y1_m: float
    z0_m: float = 0.0
    z1_m: float = 0.05
    radius_m: float | None = None
    include_in: list[str] = field(default_factory=list)
    note: str = ""
    center_depth_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneWorld:
    """Single source of truth for all raw/target/background/clutter variants.

    A SceneWorld is generated once per case_id. Variants must only mask or keep
    components from this object; they must not re-randomise geometry, materials,
    water zones, anomalies or external clutter.
    """

    case_id: str
    family: str
    random_seed: int
    ground_x: list[float]
    ground_y: list[float]
    bedrock_interface_y: list[float]
    bedrock_interface_depth_m: list[float]
    domain_x_m: float
    domain_y_m: float
    domain_z_m: float
    model_length_config_m: float
    model_length_actual_m: float
    dx_m: float
    trace_count: int
    trace_spacing_m: float
    time_window_ns: float
    samples: int
    trajectory: TrajectoryModel
    materials: dict[str, dict[str, Any]]
    cover_material_by_column: list[str]
    bedrock_material_by_column: list[str]
    interbed_layers: list[InterbedLayer] = field(default_factory=list)
    anomaly_objects: list[SceneObject] = field(default_factory=list)
    water_zones: list[SceneObject] = field(default_factory=list)
    external_clutter_objects: list[SceneObject] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        *,
        case_id: str,
        family: str,
        random_seed: int,
        ground_x: Any,
        ground_y: Any,
        bedrock_interface_y: Any,
        domain_x_m: float,
        domain_y_m: float,
        domain_z_m: float,
        model_length_config_m: float,
        model_length_actual_m: float,
        dx_m: float,
        trace_count: int,
        trace_spacing_m: float,
        time_window_ns: float,
        samples: int,
        trajectory: TrajectoryModel,
        materials: dict[str, dict[str, Any]],
        cover_material_by_column: list[str],
        bedrock_material_by_column: list[str],
        interbed_layers: list[InterbedLayer] | None = None,
        anomaly_objects: list[SceneObject] | None = None,
        water_zones: list[SceneObject] | None = None,
        external_clutter_objects: list[SceneObject] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SceneWorld":
        x = np.asarray(ground_x, dtype=float)
        gy = np.asarray(ground_y, dtype=float)
        iy = np.asarray(bedrock_interface_y, dtype=float)
        if x.size < 2 or gy.size != x.size or iy.size != x.size:
            raise ValueError("SceneWorld requires matching x/ground/interface arrays with at least two points")
        depth = gy - iy
        return cls(
            case_id=case_id,
            family=family,
            random_seed=int(random_seed),
            ground_x=_arr(x),
            ground_y=_arr(gy),
            bedrock_interface_y=_arr(iy),
            bedrock_interface_depth_m=_arr(depth),
            domain_x_m=float(domain_x_m),
            domain_y_m=float(domain_y_m),
            domain_z_m=float(domain_z_m),
            model_length_config_m=float(model_length_config_m),
            model_length_actual_m=float(model_length_actual_m),
            dx_m=float(dx_m),
            trace_count=int(trace_count),
            trace_spacing_m=float(trace_spacing_m),
            time_window_ns=float(time_window_ns),
            samples=int(samples),
            trajectory=trajectory,
            materials=materials,
            cover_material_by_column=list(cover_material_by_column),
            bedrock_material_by_column=list(bedrock_material_by_column),
            interbed_layers=list(interbed_layers or []),
            anomaly_objects=list(anomaly_objects or []),
            water_zones=list(water_zones or []),
            external_clutter_objects=list(external_clutter_objects or []),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "uavgpr_simlab.scene_world.v1alpha1",
            "case_id": self.case_id,
            "family": self.family,
            "random_seed": self.random_seed,
            "ground_profile": {"x_m": self.ground_x, "y_m": self.ground_y},
            "bedrock_interface": {
                "x_m": self.ground_x,
                "y_m": self.bedrock_interface_y,
                "depth_m": self.bedrock_interface_depth_m,
            },
            "domain": {"x_m": self.domain_x_m, "y_m": self.domain_y_m, "z_m": self.domain_z_m, "dx_m": self.dx_m},
            "model_length_config_m": self.model_length_config_m,
            "model_length_actual_m": self.model_length_actual_m,
            "samples": self.samples,
            "trace_count": self.trace_count,
            "trace_spacing_m": self.trace_spacing_m,
            "time_window_ns": self.time_window_ns,
            "trajectory": self.trajectory.to_dict(),
            "materials": self.materials,
            "material_map": {
                "cover_material_by_column": self.cover_material_by_column,
                "bedrock_material_by_column": self.bedrock_material_by_column,
            },
            "interbed_layers": [x.to_dict() for x in self.interbed_layers],
            "anomaly_objects": [x.to_dict() for x in self.anomaly_objects],
            "water_zones": [x.to_dict() for x in self.water_zones],
            "external_clutter_objects": [x.to_dict() for x in self.external_clutter_objects],
            "metadata": self.metadata,
        }

    def to_label_json(self, radar: dict[str, Any], geometry: dict[str, Any], geology: dict[str, Any]) -> dict[str, Any]:
        """Return the legacy label_json shape expected by existing GUI canvases."""
        return {
            "case_id": self.case_id,
            "family": self.family,
            "random_seed": self.random_seed,
            "coordinate_note": "gprMax x horizontal, y vertical; interface_depth = ground_y - interface_y. Trajectory metadata states whether FDTD is constant-level or terrain-following.",
            "x_m": self.ground_x,
            "ground_y_m": self.ground_y,
            "interface_y_m": self.bedrock_interface_y,
            "interface_depth_m": self.bedrock_interface_depth_m,
            "materials": self.materials,
            "radar": radar,
            "geometry": geometry,
            "geology": geology,
            "trajectory": self.trajectory.to_dict(),
            "interbed_layers": [x.to_dict() for x in self.interbed_layers],
            "anomaly_objects": [x.to_dict() for x in self.anomaly_objects],
            "water_zones": [x.to_dict() for x in self.water_zones],
            "external_clutter_objects": [x.to_dict() for x in self.external_clutter_objects],
            "scene_world_schema": "uavgpr_simlab.scene_world.v1alpha1",
        }

    def metadata_summary(self, variants: list[str]) -> dict[str, Any]:
        depths = np.asarray(self.bedrock_interface_depth_m, dtype=float)
        ground = np.asarray(self.ground_y, dtype=float)
        return {
            "case_id": self.case_id,
            "family": self.family,
            "random_seed": self.random_seed,
            "variant_list": list(variants),
            "model_length_config_m": self.model_length_config_m,
            "model_length_actual_m": self.model_length_actual_m,
            "domain_x_m": self.domain_x_m,
            "domain_y_m": self.domain_y_m,
            "dx_m": self.dx_m,
            "time_window_ns": self.time_window_ns,
            "samples": self.samples,
            "trace_count": self.trace_count,
            "trace_spacing_m": self.trace_spacing_m,
            "scan_start_x_m": self.trajectory.scan_start_x_m,
            "scan_end_x_m": self.trajectory.scan_end_x_m,
            "ground_relief_m": float(np.max(ground) - np.min(ground)),
            "ground_min_y_m": float(np.min(ground)),
            "ground_max_y_m": float(np.max(ground)),
            "bedrock_depth_min_m": float(np.min(depths)),
            "bedrock_depth_max_m": float(np.max(depths)),
            "bedrock_depth_mean_m": float(np.mean(depths)),
            "flight_height_mode": self.trajectory.mode,
            "nominal_flight_height_m": self.trajectory.nominal_height_m,
            "source_y_m": self.trajectory.source_y_m,
            "receiver_y_m": self.trajectory.receiver_y_m,
            "min_clearance_m": float(min(self.trajectory.source_y_m, self.trajectory.receiver_y_m) - np.max(ground)),
            "tx_rx_offset_m": self.trajectory.tx_rx_offset_m,
            "objects": {
                "wires": [o.to_dict() for o in self.external_clutter_objects if o.kind == "wire"],
                "trees": [o.to_dict() for o in self.external_clutter_objects if o.kind == "tree"],
                "buildings": [o.to_dict() for o in self.external_clutter_objects if o.kind == "building"],
                "water_zones": [o.to_dict() for o in self.water_zones],
                "anomalies": [o.to_dict() for o in self.anomaly_objects],
            },
            "quality_flags": {
                "is_pilot_safe": True,
                "is_ml_pair_valid": True,
                "has_external_clutter": bool(self.external_clutter_objects),
                "has_deep_anomaly": bool(self.anomaly_objects),
                "has_high_relief": float(np.max(ground) - np.min(ground)) >= 8.0,
            },
            "trajectory": self.trajectory.to_dict(),
        }
