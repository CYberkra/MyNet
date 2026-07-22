#!/usr/bin/env python3
"""Generate an independently seeded FORMAL06C-mechanism successor.

FORMAL06D deliberately preserves the complete FORMAL06C source, material,
cover-field, transition, acquisition, and domain contract.  It changes only
the deterministic geometry seeds.  This makes it a useful visual-inheritance
check without copying FORMAL06C arrays or pretending that a different cover
field is the same mechanism.

The mechanism was selected during Line9-conditioned development, so this deck
remains development-only even though its geometry generator reads no measured
arrays.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

try:
    import generate_formal06_interface_conditioned_development as formal06
    import generate_formal06c_subtle_interface_development as formal06c
    import generate_formal03_correlated_cover_source_ablation as formal03
except ModuleNotFoundError:  # Package import used by tests.
    from scripts import generate_formal06_interface_conditioned_development as formal06
    from scripts import generate_formal06c_subtle_interface_development as formal06c
    from scripts import generate_formal03_correlated_cover_source_ablation as formal03


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "simulations" / "v2" / "00_controls"
FAMILY_ID = "FORMAL06D_INDEPENDENT_MECHANISM_DEVELOPMENT"
CASE_ID = "FORMAL06D_INDEPENDENT_MECHANISM_DEVELOPMENT"
PROFILE_SEED = 2026072211
FIELD_SEED = 2026072212


def default_spec() -> formal03.Spec:
    """Use the FORMAL06C numerical contract with a new generic geometry draw."""
    return replace(
        formal06c.default_spec(),
        profile_seed=PROFILE_SEED,
        field_seed=FIELD_SEED,
    )


def generate(output_root: Path, spec: formal03.Spec | None = None) -> Path:
    spec = spec or default_spec()
    case_dir = formal06.generate_case(
        output_root,
        spec,
        design=formal06c.DESIGN,
        source=formal06c.SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL06D_INDEPENDENT_MECHANISM_POLICY.json",
        run_prefix="formal06d",
        purpose="independently seeded visual-inheritance check of the FORMAL06C mechanism",
        predecessor_case_id="FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT",
        changed_factors=[
            "generic profile seed",
            "generic correlated cover-field seed",
        ],
        generator_path=Path(__file__),
        preview_title=(
            "FORMAL06D: independently seeded geometry under the complete "
            "FORMAL06C source/material/cover/transition contract; pre-solver only"
        ),
        locked_factors=[
            "80 MHz zero-mean Gabor waveform and reference delay",
            "0.03 m grid, 256 native traces, 0.09 m trace spacing, PML and domain",
            "flat ground, 8.01 m flight height, and Tx/Rx separation",
            "FORMAL06C cover field scales, meso weight, bin count, and material endpoints",
            "FORMAL06C eight-level variable weathered transition mechanism",
            "FORMAL06C bedrock constitutive endpoint and strict full/no-basal mapping",
        ],
        geometry_description=(
            "new generic seeded basal profile and cover field; FORMAL06C "
            "measurement/cover/transition contract otherwise locked"
        ),
    )

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["geometry"]["independent_geometry"] = {
        "arrays_reused_from_formal06c": False,
        "measured_arrays_read": False,
        "profile_seed": spec.profile_seed,
        "cover_field_seed": spec.field_seed,
        "source_case": "new deterministic draw",
    }
    manifest["development_scope"] = {
        "line9_conditioned": True,
        "conditioning_scope": "mechanism selection only",
        "strict_line9_holdout_allowed": False,
        "formal_training_allowed": False,
    }
    # FORMAL06D is deliberately a visible development mechanism check.  A
    # strong full-scene local ratio should prompt blind review, not masquerade
    # as a failed causal simulation.  Legacy families retain their hard cap.
    visibility_gate = manifest.get("visibility_gate", {})
    if isinstance(visibility_gate, dict):
        visibility_gate.pop("full_scene_target_to_local_background_rms_max", None)
        visibility_gate[
            "full_scene_target_to_local_background_rms_review_above"
        ] = 5.0
        manifest["visibility_gate"] = visibility_gate
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
