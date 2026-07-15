# Independent V2 Family 02 Mechanism-Transfer Audit

Date: 2026-07-15

## Decision

`IV2_F02_FORMAL06C_MECHANISM_POS` passes as a **development-only mechanism
transfer**. It is the correct continuation of FORMAL06C for mechanism study:
the geometry is independently regenerated from the Family 01 generic seeds,
while the source and weak-interface constitutive mechanism are inherited from
FORMAL06C. No Line9, FORMAL06, or FORMAL07 arrays are read by the generator.

It is not formal training data. The decision to select this mechanism used
Line9 development diagnostics, so the positive and its exact matched negative
remain `line9_conditioned=true` at the mechanism-selection scope.

## What Changed From Family 01

Geometry, acquisition, grid, PML, time window, seeds, and strict control
construction are held fixed. Family 02 changes only:

- the Family 01 pulse to an 80 MHz zero-mean Gaussian-modulated waveform;
- cover, weathered-cap, transition, and bedrock constitutive values to the
  FORMAL06C weak-interface mechanism.

This makes the F01/F02 comparison a mechanism ablation rather than a geometry
comparison.

## Provenance And Physics Checks

- Shared geometry file SHA256: `2ca8e0df4e11972ec08712c0b215080b4f4435a32d73b475e845f5d16df76801`.
- Shared geometry-array SHA256: `f2c59754d6ea6f56ef9798da133b103ec7f2fb582f6625bea20b875f78249703`.
- Positive-control material SHA256 equals negative-full material SHA256:
  `2de5eb14c9bb5b620076ae9b4db3666268d1f8f8530515d0e6d9e618b4753e91`.
- The one-trace positive control and negative full scene produced byte-identical
  `Ez` arrays; maximum absolute difference was zero.
- Static input, geometry build, finite output, trace timing, and one-trace
  causal-pair checks passed.
- The custom source has near-zero discrete integral and a measured generated
  peak near 79.46 MHz.

## 32-Trace Blind Morphology Result

The full-scene pilot used canonical traces `0, 8, ..., 248`, equivalent to
0.72 m sparse spacing. No horizontal interpolation was used for analysis.

| Metric | F01 | F02 | FORMAL06C | Line9 diagnostic |
|---|---:|---:|---:|---:|
| Path/geometric correlation | 0.9981 | 0.9993 | 0.9999 | n/a |
| Target/adjacent-background RMS | 7.22 | 14.96 | 17.29 | 2.35 |
| Target envelope CV | n/a | 0.445 | 0.334 | 0.465 |
| Median aligned-template correlation | 0.721 | 0.694 | 0.660 | 0.646 |
| Significant signed lobes | 4 | 7 | 7 | n/a |
| Peak frequency | 44.09 MHz | 79.37 MHz | 79.37 MHz | 79.69 MHz |
| Spectral centroid | 57.70 MHz | 79.30 MHz | 92.07 MHz | 84.64 MHz |

Blind visual review reaches the same conclusion as the metrics:

- F02 recovers FORMAL06C's thick, alternating, multi-cycle basal packet.
- The event is continuous and follows broad relief rather than appearing as a
  stitched chain of isolated hyperbolas.
- F02 differs from FORMAL06C mainly in the independently generated basal shape
  and cover field; it does not lose the accepted mechanism.
- The target is still too isolated relative to Line9. That difficulty mismatch
  is a reason to add independent continuous background variation later, not a
  reason to discard the recovered mechanism.

## Claim Limits

The 32-trace run is full-scene-only. It supports morphology selection but does
not provide dense causal labels, formal promotion, or native-resolution local
coherence evidence. The one-trace strict pair proves local causal attribution
only.

The next formal-data step is not to copy Family 02 unchanged. It is to preserve
this audited implementation pattern while deriving the source/material ranges
from independent physical bounds or a predeclared non-held-out factorial, then
repeat the strict paired and blind gates.

## Evidence

- `blind_comparisons/f01_vs_f02_blind.png`
- `blind_comparisons/formal06c_vs_f02_blind.png`
- `distributed32_morphology/full_only_morphology_audit.json`
- `smoke1_pair_audit/family_spatial_pilot_audit.json`
