#!/usr/bin/env python3
"""Generate the FORMAL06B tempered-interface development candidate.

FORMAL06B is a strict one-factor successor to FORMAL06A. It preserves the
source, grid, domain, acquisition, stochastic bulk field, basal profile, and
weathered-cap geometry. Only the cap-to-bedrock constitutive contrast is
reduced after FORMAL06A proved causal but visually overstrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import generate_formal06_interface_conditioned_development as formal06


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "PGDA_SYNTH_DATASET_V2" / "00_controls"
FAMILY_ID = "FORMAL06B_TEMPERED_INTERFACE_DEVELOPMENT"
CASE_ID = "FORMAL06B_TEMPERED_INTERFACE_DEVELOPMENT"
SOURCE = formal06.SOURCE
DESIGN = formal06.MaterialDesign(
    cover_epsilon_min=12.0,
    cover_epsilon_max=12.8,
    cover_conductivity_min_s_per_m=0.0018,
    cover_conductivity_max_s_per_m=0.0025,
    weathered_cap_epsilon_r=13.0,
    weathered_cap_conductivity_s_per_m=0.0026,
    bedrock_epsilon_r=12.1,
    bedrock_conductivity_s_per_m=0.0019,
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
        policy_filename="FORMAL06B_TEMPERED_INTERFACE_POLICY.json",
        run_prefix="formal06b",
        purpose="tempered cap-to-bedrock contrast morphology development",
        predecessor_case_id=formal06.CASE_ID,
        changed_factors=[
            "weathered-cap endpoint epsilon and conductivity",
            "bedrock epsilon and conductivity",
            "cap-to-bedrock reflection proxy",
        ],
        generator_path=Path(__file__),
        preview_title=(
            f"{FAMILY_ID}: FORMAL06A geometry with tempered interface contrast; "
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
