from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class MaterialRange:
    name: str
    eps_r: Tuple[float, float]
    sigma: Tuple[float, float]
    mu_r: float = 1.0
    magnetic_loss: float = 0.0
    note: str = ""

    def sample(self, rng: random.Random) -> Dict[str, float | str]:
        return {
            "name": self.name,
            "eps_r": rng.uniform(*self.eps_r),
            "sigma": rng.uniform(*self.sigma),
            "mu_r": self.mu_r,
            "magnetic_loss": self.magnetic_loss,
            "note": self.note,
        }


DEFAULT_MATERIAL_RANGES: Dict[str, MaterialRange] = {
    "air": MaterialRange("air", (1.0, 1.0), (0.0, 0.0), note="air"),
    "silty_clay": MaterialRange("silty_clay", (8.0, 22.0), (0.008, 0.120), note="Q4del silty clay"),
    "gravelly_silty_clay": MaterialRange("gravelly_silty_clay", (6.0, 18.0), (0.004, 0.080), note="silty clay with gravel/blocks"),
    "wet_silty_clay": MaterialRange("wet_silty_clay", (18.0, 36.0), (0.050, 0.250), note="wet paddy/lowland clay"),
    "sandstone": MaterialRange("sandstone", (4.0, 10.0), (0.0005, 0.030), note="J3p sandstone/bedrock"),
    "mudstone": MaterialRange("mudstone", (7.0, 18.0), (0.003, 0.100), note="J3p mudstone interbed"),
    "weathered_bedrock": MaterialRange("weathered_bedrock", (5.0, 14.0), (0.002, 0.060), note="weathered sandstone/mudstone fragments"),
    "fresh_water": MaterialRange("fresh_water", (60.0, 82.0), (0.005, 0.080), note="surface water/paddy water"),
    "wood": MaterialRange("wood", (2.0, 6.0), (0.001, 0.050), note="tree/vegetation clutter"),
    "concrete": MaterialRange("concrete", (5.0, 12.0), (0.001, 0.030), note="building/wall clutter"),
    "metal_pec": MaterialRange("pec", (1.0, 1.0), (0.0, 0.0), note="PEC wire/cable approximation"),
}


def sample_material_library(seed: int) -> Dict[str, Dict[str, float | str]]:
    rng = random.Random(seed)
    return {key: val.sample(rng) for key, val in DEFAULT_MATERIAL_RANGES.items()}


def gprmax_material_lines(materials: Dict[str, Dict[str, float | str]]) -> str:
    lines = []
    for key, mat in materials.items():
        if key in {"air", "metal_pec"}:
            continue
        name = str(mat["name"]).replace(" ", "_")
        eps_r = float(mat["eps_r"])
        sigma = float(mat["sigma"])
        mu_r = float(mat.get("mu_r", 1.0))
        magnetic_loss = float(mat.get("magnetic_loss", 0.0))
        lines.append(f"#material: {eps_r:.6g} {sigma:.6g} {mu_r:.6g} {magnetic_loss:.6g} {name}")
    return "\n".join(lines)


def material_ranges_as_dict() -> Dict[str, dict]:
    return {k: asdict(v) for k, v in DEFAULT_MATERIAL_RANGES.items()}
