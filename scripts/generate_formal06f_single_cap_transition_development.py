#!/usr/bin/env python3
"""Generate the FORMAL06F single-cap transition development successor.

FORMAL06F is a causal diagnosis of FORMAL06D/06E's native-spacing failure.
It locks the FORMAL06D cover field, source, basal profile, material endpoints,
grid, and acquisition.  The only changed physical mechanism is the weathered
transition discretisation: an eight-stage dielectric staircase becomes one
full-contrast weathered cap of the same variable thickness.

The purpose is not to claim a final geology model.  It isolates whether the
parallel background packet is created chiefly by the eight repeated material
boundaries.  As with its predecessors, this deck is development-only and
blocked from formal training and strict Line9 evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

try:
    import generate_formal03_correlated_cover_source_ablation as formal03
    import generate_formal06_interface_conditioned_development as formal06
    import generate_formal06c_subtle_interface_development as formal06c
    import generate_formal06d_independent_mechanism_development as formal06d
except ModuleNotFoundError:  # Package import used by tests.
    from scripts import generate_formal03_correlated_cover_source_ablation as formal03
    from scripts import generate_formal06_interface_conditioned_development as formal06
    from scripts import generate_formal06c_subtle_interface_development as formal06c
    from scripts import generate_formal06d_independent_mechanism_development as formal06d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL06F_SINGLE_CAP_TRANSITION_DEVELOPMENT"
CASE_ID = FAMILY_ID
SOURCE = formal06c.SOURCE
DESIGN = formal06c.DESIGN


def default_spec() -> formal03.Spec:
    """Retain FORMAL06D exactly except for one weathered-cap material stage."""

    return replace(formal06d.default_spec(), transition_levels=1)


def one_cap_material_rows(
    spec: formal03.Spec,
    control: bool,
    *,
    design: formal06.MaterialDesign = DESIGN,
) -> list[formal03.Material]:
    """Use one full-contrast cap, not a stack of graded reflector layers."""

    if spec.transition_levels != 1:
        raise ValueError("FORMAL06F requires exactly one weathered-cap stage")
    bases = formal06.base_materials(spec, design=design)
    rows = list(bases)
    for base in bases:
        rows.append(
            formal03.Material(
                f"weathered_cap_cover_{base.material_id}",
                base.epsilon_r if control else design.weathered_cap_epsilon_r,
                base.conductivity_s_per_m
                if control
                else design.weathered_cap_conductivity_s_per_m,
            )
        )
    for base in bases:
        rows.append(
            formal03.Material(
                f"bedrock_{base.material_id}",
                base.epsilon_r if control else design.bedrock_epsilon_r,
                base.conductivity_s_per_m
                if control
                else design.bedrock_conductivity_s_per_m,
            )
        )
    return rows


def generate(output_root: Path, spec: formal03.Spec | None = None) -> Path:
    spec = spec or default_spec()
    case_dir = formal06.generate_case(
        output_root,
        spec,
        design=DESIGN,
        source=SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL06F_SINGLE_CAP_TRANSITION_POLICY.json",
        run_prefix="formal06f",
        purpose="single weathered-cap diagnosis of repeated transition reflections",
        predecessor_case_id="FORMAL06D_INDEPENDENT_MECHANISM_DEVELOPMENT",
        changed_factors=[
            "weathered transition material mapping: eight graded stages to one full-contrast cap",
        ],
        generator_path=Path(__file__),
        preview_title=(
            "FORMAL06F: FORMAL06D mechanism with a single variable-thickness "
            "weathered cap; pre-solver only"
        ),
        profile_builder=formal03.build_profiles,
        bulk_field_builder=formal06.build_bulk_field,
        material_rows_builder=one_cap_material_rows,
        locked_factors=[
            "FORMAL06D generic profile and cover-field seeds",
            "FORMAL06D cover covariance and material endpoints",
            "80 MHz zero-mean Gabor waveform and reference delay",
            "0.03 m grid, 256 native traces, 0.09 m trace spacing, PML and domain",
            "flat ground, 8.01 m flight height, and Tx/Rx separation",
            "basal depth family, transition-thickness geometry and cap-to-bedrock endpoint",
            "strict full/no-basal shared indexed geometry and local-cover control mapping",
        ],
        geometry_description=(
            "FORMAL06D cover and basal geometry with one variable-thickness, "
            "full-contrast weathered cap rather than an eight-stage transition stack"
        ),
    )

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visibility_gate = manifest["visibility_gate"]
    visibility_gate.pop("full_scene_target_to_local_background_rms_max", None)
    visibility_gate["full_scene_target_to_local_background_rms_review_above"] = 5.0
    manifest["development_scope"] = {
        "line9_conditioned": True,
        "conditioning_scope": "mechanism diagnosis only",
        "strict_line9_holdout_allowed": False,
        "formal_training_allowed": False,
    }
    manifest["transition_diagnosis"] = {
        "predecessor_transition_levels": 8,
        "candidate_transition_levels": 1,
        "candidate_cap_blend": 1.0,
        "hypothesis": (
            "the predecessor's native-spacing parallel packet is dominated by "
            "repeated dielectric transition boundaries"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    formal03.write_checksums(case_dir)
    return case_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate(args.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
