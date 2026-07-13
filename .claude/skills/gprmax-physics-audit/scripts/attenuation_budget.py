#!/usr/bin/env python3
"""Estimate field attenuation in a nondispersive lossy dielectric."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
NEPERS_TO_DB = 20.0 / math.log(10.0)


def attenuation(frequency_hz: float, epsilon_r: float, conductivity_s_m: float) -> dict[str, float]:
    omega = 2.0 * math.pi * frequency_hz
    epsilon = EPSILON_0 * epsilon_r
    loss_ratio = conductivity_s_m / (omega * epsilon)
    root = math.sqrt(1.0 + loss_ratio * loss_ratio)
    common = omega * math.sqrt(MU_0 * epsilon / 2.0)
    alpha_np_m = common * math.sqrt(root - 1.0)
    beta_rad_m = common * math.sqrt(root + 1.0)
    return {
        "frequency_hz": frequency_hz,
        "loss_ratio_sigma_over_omega_epsilon": loss_ratio,
        "alpha_np_per_m": alpha_np_m,
        "field_db_per_m": alpha_np_m * NEPERS_TO_DB,
        "beta_rad_per_m": beta_rad_m,
        "wavelength_m": 2.0 * math.pi / beta_rad_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon-r", type=float, required=True)
    parser.add_argument("--conductivity-s-m", type=float, required=True)
    parser.add_argument("--center-frequency-mhz", type=float, required=True)
    parser.add_argument("--upper-multiplier", type=float, default=2.8)
    parser.add_argument("--one-way-depth-m", type=float, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.epsilon_r <= 0 or args.conductivity_s_m < 0:
        raise SystemExit("epsilon-r must be positive and conductivity must be non-negative")
    if args.center_frequency_mhz <= 0 or args.upper_multiplier < 1 or args.one_way_depth_m < 0:
        raise SystemExit("frequency/depth must be valid and upper-multiplier must be >= 1")

    center_hz = args.center_frequency_mhz * 1e6
    path_m = 2.0 * args.one_way_depth_m
    bands = {
        "center": attenuation(center_hz, args.epsilon_r, args.conductivity_s_m),
        "upper_significant": attenuation(
            center_hz * args.upper_multiplier,
            args.epsilon_r,
            args.conductivity_s_m,
        ),
    }
    for values in bands.values():
        alpha = values["alpha_np_per_m"]
        values["two_way_path_m"] = path_m
        values["two_way_field_amplitude_factor"] = math.exp(-alpha * path_m)
        values["two_way_field_loss_db"] = values["field_db_per_m"] * path_m

    report = {
        "model": "exact nondispersive lossy dielectric propagation constant",
        "epsilon_r": args.epsilon_r,
        "conductivity_s_m": args.conductivity_s_m,
        "center_frequency_mhz": args.center_frequency_mhz,
        "upper_significant_multiplier": args.upper_multiplier,
        "one_way_depth_m": args.one_way_depth_m,
        "bands": bands,
        "limitations": [
            "does not include geometric spreading, antenna response, interfaces, or dispersive poles",
            "use as a pre-run plausibility gate, not as a substitute for a paired gprMax smoke run",
        ],
    }

    payload = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
