"""Run the formal AeroPath official-Mamba2 forward pass at 501x256.

This is deliberately independent of the data-release gate: it validates CUDA,
the installed ``mamba_ssm`` build, the headdim contract, and peak VRAM without
starting a blocked paper experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pgdacsnet.model_raw_unet import build_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aeropath_ssd_v15_formal_blocked.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backward", action="store_true", help="Also measure one backward pass.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the official-Mamba2 smoke test.")
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if str(cfg.get("model_arch", "")).lower() not in {"aeropath_ssd", "aeropath", "v3_aeropath_ssd"}:
        raise SystemExit("config must select AeroPath-SSD")
    if str(cfg.get("ssm_impl", "")).lower() != "official_mamba2":
        raise SystemExit("config must select ssm_impl=official_mamba2")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg).to(device).train(args.backward)
    b = int(args.batch_size)
    h, w = int(cfg["height_resize"]), int(cfg["width_resize"])
    x = torch.randn(b, int(cfg.get("input_channels", 1)), h, w, device=device)
    altitude = torch.full((b, w), 8.0, device=device)
    chainage = torch.linspace(0.0, float(w - 1), w, device=device)[None].expand(b, -1)
    with torch.autocast(device_type="cuda", enabled=True):
        output = model(x, altitude=altitude, chainage_m=chainage)
        loss = output.path_marginals.mean() + output.null_marginals.mean()
    if args.backward:
        loss.backward()
    torch.cuda.synchronize(device)
    report = {
        "config": str(path),
        "device": torch.cuda.get_device_name(device),
        "shape": [b, int(cfg.get("input_channels", 1)), h, w],
        "backward": bool(args.backward),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2),
        "path_mass_error": float((output.path_marginals.sum(dim=2) + output.null_marginals - 1.0).abs().max().detach().cpu()),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
