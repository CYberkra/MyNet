from __future__ import annotations
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple
import yaml

@dataclass
class MaterialRange:
    eps_r: Tuple[float, float]
    sigma: Tuple[float, float]
    mu_r: float = 1.0
    magnetic_loss: float = 0.0
    def sample(self, rng: random.Random) -> tuple[float, float, float, float]:
        return rng.uniform(*self.eps_r), rng.uniform(*self.sigma), self.mu_r, self.magnetic_loss

def default_material_ranges() -> Dict[str, MaterialRange]:
    return {
        'air': MaterialRange((1.0,1.0),(0.0,0.0)),
        'silty_clay': MaterialRange((8.0,28.0),(0.002,0.010)),
        'gravelly_silty_clay': MaterialRange((6.0,22.0),(0.001,0.008)),
        'weathered_mudstone': MaterialRange((6.0,16.0),(0.002,0.015)),
        'sandstone_bedrock': MaterialRange((4.0,10.0),(0.0003,0.008)),
        'saturated_zone': MaterialRange((18.0,32.0),(0.005,0.030)),
        # FDTD-safe surrogate for shallow water at the default 5 cm grid.
        # True fresh water eps_r~80 would violate gprMax's wavelength sampling
        # check at the 100 MHz Ricker wavelet significant-frequency band unless
        # dx is reduced to ~1-2 cm, which is impractical for large paper batches.
        'surface_water': MaterialRange((24.0,32.0),(0.001,0.050)),
        'vegetation': MaterialRange((2.0,9.0),(0.0005,0.040)),
        'building_wall': MaterialRange((4.0,12.0),(0.001,0.030)),
        'fracture_zone': MaterialRange((12.0,30.0),(0.015,0.120)),
    }

def load_material_ranges(path: str | Path | None = None) -> Dict[str, MaterialRange]:
    if not path or not Path(path).exists():
        return default_material_ranges()
    doc = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    ranges: Dict[str, MaterialRange] = {}
    for name, data in doc.get('materials', doc).items():
        ranges[name] = MaterialRange(tuple(data.get('eps_r',[1.0,1.0])), tuple(data.get('sigma',[0.0,0.0])), float(data.get('mu_r',1.0)), float(data.get('magnetic_loss',0.0)))
    return ranges

def sample_materials(rng: random.Random, ranges: Dict[str, MaterialRange] | None = None) -> Dict[str, Dict[str, float | str]]:
    out: Dict[str, Dict[str, float | str]] = {}
    for name, mr in (ranges or default_material_ranges()).items():
        eps, sig, mu, mloss = mr.sample(rng)
        out[name] = {'eps_r': round(eps,4), 'sigma': round(sig,6), 'mu_r': mu, 'magnetic_loss': mloss}
    out['pec'] = {'builtin': 'pec'}
    return out

def material_lines(materials: Dict[str, Dict[str, float | str]]) -> list[str]:
    lines = []
    for name, vals in materials.items():
        if vals.get('builtin') == 'pec':
            continue
        lines.append(f"#material: {vals['eps_r']} {vals['sigma']} {vals.get('mu_r',1.0)} {vals.get('magnetic_loss',0.0)} {name}")
    return lines
