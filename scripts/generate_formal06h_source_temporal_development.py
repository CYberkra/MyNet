#!/usr/bin/env python3
"""Generate the FORMAL06H 2-D source-temporal diagnostic.

FORMAL06E, FORMAL06F, and FORMAL06G separately excluded cover covariance,
transition staging, and bounded terrain/AGL geometry as the sole cause of the
native-spacing wavelet comb. FORMAL06H keeps the complete FORMAL06D numerical
and geological contract byte-for-byte reproducible and changes only the
temporal excitation: the inherited 80 MHz zero-mean Gabor proxy becomes an
80 MHz Ricker pulse.

This is deliberately not a finite-antenna claim. In 2-D gprMax the Hertzian
dipole is a line source, so this diagnostic can isolate temporal support but
cannot establish antenna directivity or coupling realism.
"""

from __future__ import annotations

import argparse
import json
import math
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
FAMILY_ID = "FORMAL06H_SOURCE_TEMPORAL_DEVELOPMENT"
CASE_ID = FAMILY_ID
SOURCE = formal03.SourceVariant(
    case_id="FORMAL06H_RICKER80",
    kind="ricker",
    center_frequency_hz=80e6,
    waveform_id="formal06h_ricker80",
    reference_delay_ns=math.sqrt(2.0) / 80e6 * 1e9,
)


def default_spec() -> formal03.Spec:
    """Use the FORMAL06D seeds so geometry is independently reproducible."""

    return formal06d.default_spec()


def generate(output_root: Path, spec: formal03.Spec | None = None) -> Path:
    spec = spec or default_spec()
    case_dir = formal06.generate_case(
        output_root,
        spec,
        design=formal06c.DESIGN,
        source=SOURCE,
        family_id=FAMILY_ID,
        case_id=CASE_ID,
        policy_filename="FORMAL06H_SOURCE_TEMPORAL_POLICY.json",
        run_prefix="formal06h",
        purpose="isolated 2-D source-temporal diagnostic of native wavelet coherence",
        predecessor_case_id="FORMAL06D_INDEPENDENT_MECHANISM_DEVELOPMENT",
        changed_factors=[
            "temporal source excitation: 80 MHz zero-mean Gabor to 80 MHz Ricker",
        ],
        generator_path=Path(__file__),
        preview_title=(
            "FORMAL06H: FORMAL06D geometry/material/acquisition contract with only "
            "the 80 MHz Ricker temporal excitation changed; pre-solver only"
        ),
        locked_factors=[
            "FORMAL06D generic basal profile and cover-field seeds",
            "FORMAL06D cover covariance, material endpoints, and eight-level transition",
            "0.03 m grid, 256 native traces, 0.09 m trace spacing, PML, and domain",
            "flat ground, 8.01 m flight height, Tx/Rx geometry, and acquisition positions",
            "strict full/no-basal shared indexed geometry and local-cover control mapping",
            "all processing and blind morphology gates",
        ],
        geometry_description=(
            "exactly reproducible FORMAL06D flat-ground geometry and material mapping; "
            "only the temporal line-source excitation differs"
        ),
    )

    manifest_path = case_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["development_scope"] = {
        "line9_conditioned": True,
        "conditioning_scope": "mechanism diagnosis only",
        "strict_line9_holdout_allowed": False,
        "formal_training_allowed": False,
    }
    manifest["source_abstraction_diagnosis"] = {
        "installed_gprmax_2d_source": "Hertzian dipole line source",
        "tested_scope": "temporal support and phase only",
        "not_tested": ["finite antenna directivity", "feed coupling", "3-D radiation pattern"],
        "predecessor_source": {
            "kind": formal06c.SOURCE.kind,
            "center_frequency_hz": formal06c.SOURCE.center_frequency_hz,
            "custom_sigma_ns": formal06c.SOURCE.custom_sigma_ns,
        },
        "candidate_source": {
            "kind": SOURCE.kind,
            "center_frequency_hz": SOURCE.center_frequency_hz,
            "reference_delay_ns": SOURCE.reference_delay_ns,
        },
        "hypothesis": (
            "the ideal 2-D line source's finite Gabor temporal support contributes "
            "to the full-window repeated lobe train"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (case_dir / "SOURCE_ABLATION_SCOPE.md").write_text(
        """# FORMAL06H Source-Temporal Scope

FORMAL06H changes only the 80 MHz temporal waveform from the FORMAL06D
zero-mean Gabor proxy to the gprMax 80 MHz Ricker waveform. Its geometry,
materials, acquisition, full/no-basal mapping, and display gates are locked.

This is a 2-D line-source diagnostic, not a finite-UAV-antenna model. A pass
can only show that temporal source support changes the native morphology; it
cannot validate antenna directivity or coupling.
""",
        encoding="utf-8",
    )
    formal03.preview_sources(
        case_dir / "preview_source_waveforms.png",
        spec,
        variants=(formal06c.SOURCE, SOURCE),
    )
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
