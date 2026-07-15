# FORMAL06C Development Audit

Date: 2026-07-15

## Scope

FORMAL06C keeps the FORMAL06A/B grid, acquisition, source, stochastic index
geometry, basal path, and transition thickness fixed. It changes only the
weathered-cap-to-bedrock constitutive contrast. The case remains permanently
development-only because Line9 aggregate morphology is used as a diagnostic.

## Physical and runtime checks

- Static full/control input audits: pass, no errors or warnings.
- Geometry-only full/control builds: pass.
- Source: zero-mean 80 MHz Gaussian-modulated waveform.
- Highest significant frequency: 123.05 MHz.
- Minimum wavelength resolution: 22 cells; phase error -0.16%.
- Shared FORMAL06A/B/C index-array SHA256:
  `2c16ea5a8ab3abec11a2e58541bf580a580b5ea75076c3ce86410271f25cf83c`.
- Cap-to-bedrock reflection proxy: -0.00880695.

## One-trace strict pair

The full/no-basal pair passed the causal smoke gate.

- Reference arrival: 427.832 ns.
- Solved visible phase: 429.045 ns.
- Visible-minus-reference: 1.214 ns.
- Signed difference target RMS: 0.0014561.
- Signed difference background RMS: 9.60e-6.
- Early full/control relative difference: 3.89e-7.
- Causal contrast/full target RMS: 0.9145.

This proves that the late packet is caused by the basal contrast and is not an
early-time leakage. One trace does not prove horizontal morphology.

## Blind local eight-trace checkpoint

The eight consecutive traces passed the frozen local visibility interval:

- Target/adjacent-background RMS: 4.009 (required 1 to 5).
- FORMAL06A: 25.23.
- FORMAL06B: 7.44.

FORMAL06C therefore fixes the gross local overexposure seen in A and B.

## Distributed 32-trace morphology

The full-only run used canonical indices 0, 8, ..., 248 and covers about
22.3 m. All 32 source/receiver positions and outputs passed the trace contract.

- Path-to-geometric correlation: 0.99993.
- Median path-minus-geometric time: -0.879 ns.
- Path range: 394.06 to 428.52 ns.
- Target envelope CV: 0.334.
- Aligned template-correlation median: 0.660.
- Peak frequency: 79.37 MHz.
- Spectral centroid: 92.07 MHz.
- Target/adjacent-background RMS: 17.29.

The target is a continuous multi-cycle interface, not a set of joined
hyperbolas. The automatic cross-domain diagnostic finds it more isolated from
its surroundings than Line9. The
development-only Line9 contract has target/background RMS 2.35, envelope CV
0.465, peak frequency 79.69 MHz, and spectral centroid 84.64 MHz. The source
character and phase coherence are now close, while non-target geology is too
clean and the 22.3 m basal time range is larger. These measurements remain
diagnostics because the compared domains and clutter fields are not identical.

Project-owner blind visual review on 2026-07-15 accepted FORMAL06C as the
desired development morphology. The accepted character is a continuous,
gently varying, multi-cycle basal packet visible after background suppression
and restrained time-power gain, without a chain of isolated hyperbolas.

## Decision

`HUMAN_MORPHOLOGY_ACCEPTED_DEVELOPMENT_BASELINE_PENDING_STRICT_PAIR`

FORMAL06C is retained as the accepted development morphology baseline, but it
is not a training release. Its next release gate is a distributed matched
full/no-basal pair plus independent-data governance. Further successors may:

1. reduce long-wave basal relief over a 20-25 m crop;
2. add continuous multiscale stratigraphic/background texture without discrete
   point targets or isolated inclusions;
3. target measured-like background balance and amplitude variability;
4. keep the 80 MHz Gaussian-modulated source and FORMAL06C contrast initially;
5. repeat static, one-trace, blind local 8-trace, and distributed 32-trace
   gates before any full 256-trace pair.
