#!/usr/bin/env python3
"""Generate the FORMAL06C subtle-interface development candidate.

FORMAL06C is a strict one-factor successor to FORMAL06B. It keeps the exact
FORMAL06A/B geometry and acquisition while halving the remaining
cap-to-bedrock constitutive contrast after FORMAL06B remained visually
dominant in an eight-trace blind checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import generate_formal06_interface_conditioned_development as formal06


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "00_controls"
FAMILY_ID = "FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT"
CASE_ID = "FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT"
SOURCE = formal06.SOURCE
DESIGN = formal06.MaterialDesign(
    cover_epsilon_min=12.0,
    cover_epsilon_max=12.8,
    cover_conductivity_min_s_per_m=0.0018,
    cover_conductivity_max_s_per_m=0.0025,
    weathered_cap_epsilon_r=13.0,
    weathered_cap_conductivity_s_per_m=0.0026,
    bedrock_epsilon_r=12.55,
    bedrock_conductivity_s_per_m=0.00225,
    bulk_long_x_scale_m=6.7,
    bulk_long_y_scale_m=3.0,
    bulk_meso_x_scale_m=1.8,
    bulk_meso_y_scale_m=0.96,
    bulk_meso_weight=0.18,
)


def default_spec():
    return formal06.default_spec()


def generate(output_root: Path, spec=None) -> Path:
    spec = spec or default_spec()
    return formal06.generate_case(
        output_root,
        spec,
        design=DESIGN,
        source=SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL06C_SUBTLE_INTERFACE_POLICY.json",
        run_prefix="formal06c",
        purpose="subtle cap-to-bedrock contrast morphology development",
        predecessor_case_id="FORMAL06B_TEMPERED_INTERFACE_DEVELOPMENT",
        changed_factors=[
            "bedrock epsilon and conductivity",
            "cap-to-bedrock reflection proxy",
        ],
        generator_path=Path(__file__),
        preview_title=(
            f"{FAMILY_ID}: FORMAL06A/B geometry with subtle interface contrast; "
            "pre-solver only"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate(args.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
