# Simulation Program V2

## Objective

Run two explicit simulation tracks instead of forcing one dataset to satisfy
two incompatible claims:

1. **Measured-realism calibration.** Use Line9 openly as a development
   reference for wavelet character, continuity, target prominence, and
   non-target texture. These cases are `line9_conditioned=true` and are not
   eligible for a strict unseen-Line9 claim.
2. **Formal generalisation.** Freeze or independently re-origin the accepted
   mechanism, generate strict positive/control/negative families, and exclude
   the held-out evaluation line from every parameter-selection decision.

The immediate visual lineage is the first track. Formal eligibility remains a
separate provenance and solved-evidence decision.

## Fixed Measurement Contract

- Network tensor: 501 time samples by 256 native traces.
- Canonical time axis: 0-700 ns; solver window includes a guard beyond 700 ns.
- Native trace spacing: 0.09 m; no horizontal interpolation for release data.
- Project hardware: nominal 100 MHz antenna, 50-150 MHz nominal bandwidth;
  field acquisition metadata declares 20-170 MHz, 1 MHz stepping, 501 frequency
  points, and a 0-700 ns output window.
- Simulation source remains an explicitly named broadband pulse proxy until a
  measured complex antenna/system transfer function is available.
- 2-D TMz is the production generator. Small 3-D studies are a later domain-gap
  audit and do not replace the 2-D paired-control corpus.

## Evidence Classes

| Class | Purpose | Training eligible |
|---|---|---:|
| Realism-calibration baseline | Match measured morphology and define the useful mechanism | No for strict Line9 holdout |
| Development baseline | Discover and preserve useful mechanisms | No |
| Independent pilot | Test a predeclared generator and strict controls | No |
| Released family | Passed complete numerical, causal, visual, provenance, and human gates | Yes |
| Regression control | Preserve a known failure or comparison | No |

FORMAL06C is the project-owner accepted realism-calibration baseline. Family 02
is a geometry-transfer ablation whose morphology is visibly worse than
FORMAL06C despite retaining its source and materials. Family 03 is an
instrument-band source ablation and is visibly worse again. Neither Family 02
nor Family 03 is the preferred visual successor. Family 01 remains the first
independent pilot.

## Calibration And Claim Contract

- A Line9-calibrated generator may use Line9 raw/processed morphology and
  aggregate diagnostics, but every artifact must declare
  `line9_conditioned=true` and `strict_line9_holdout_allowed=false`.
- Such a generator is scientifically useful for simulator realism, ablations,
  pretraining development, and domain-gap analysis. Only the unseen-Line9 claim
  is unavailable.
- A strict Line9 experiment must use a generator selected without Line9. An
  alternative paper experiment may use Line9 for calibration and evaluate a
  separately held-out line or a leave-one-line-out protocol whose simulator is
  recalibrated without the held-out fold.
- Visual ranking is a required project-owner gate. Numerical physics checks do
  not promote a case that is visibly less realistic than its locked
  predecessor.

## Family Roadmap

### Phase A: FORMAL06C Realism Successor

Lock FORMAL06C's source, weak-interface materials, grid, acquisition, basal
packet, and transition construction. Change one factor group at a time to close
the measured-domain gap, beginning with continuous non-target geology and only
then broad basal relief. Compare every candidate directly with FORMAL06C and
Line9 using identical crops, gain, and colour scales.

FORMAL08A completed the first weak continuous-background ablation. Its exact
32-trace blind comparison preserved the FORMAL06C packet and reduced the
full-span target/adjacent-background RMS from about 17.29 to 14.77, but the
visual background change was too small to justify promotion. Keep it as an
ablation and do not run its matched control or native-256 scene. The next Phase
A candidate must still inherit FORMAL06C directly and must show a materially
visible improvement at the full-span blind gate before further solver work.

FORMAL08B completed the next staged candidate. It changed only the continuous
transition-following deep-cover field and preserves FORMAL06C's accepted basal
packet exactly. Its geometry gate passes with correlation 0.84656,
perturbation RMS 0.51900, changed-bin fraction 0.22021, bin-delta P99 7, and
exact protected bins. Eight consecutive and full-span 32 full-scene runs
completed. The 32-trace result preserved a 0.99995 path correlation, seven
signed lobes, and a 79.37 MHz peak, but target/adjacent-background RMS increased
from about 17.29 to 21.33. The blind image showed no useful background gain.
Stop before matched controls and native 256, retain FORMAL06C as the mother
model, and do not continue by merely increasing transition-following texture.

The next realism factor must decouple non-target structure from the basal
neighbourhood. Prefer broad, oblique or locally discontinuous but non-point
background organisation placed away from the protected interface corridor,
with a predeclared amplitude budget. Do not add isolated bodies or tune another
uniform texture-strength multiplier.

Family 03 remains useful only as evidence that the project-wide amplitude-only
100 MHz band proxy produces a narrower, sharper packet. Do not extend it into a
source sweep until a measured complex system response is available.

### Phase B: Realism-Calibrated Factorial

Hold one geometry and source fixed. Test three predeclared interface classes:

1. subtle negative permittivity contrast;
2. moderate negative contrast;
3. subtle positive or polarity-reversed contrast.

Each positive owns an exact no-basal material control. Reject combinations
that are undetectable, target-dominating, or dominated by transition-layer
combs before changing geometry.

### Phase C: Independent Formal Geometry And Background Matrix

Scale only after Phases A and B pass. The planned 24-family pilot is:

- four independent broad geometry classes: gentle multiscale, tilted relief,
  low-relief undulating, and piecewise-curvature without discontinuities;
- three accepted constitutive classes from Phase B;
- two continuous cover-background strengths.

This yields 24 positives and 24 exact target-absent family controls. All
variants from one latent geometry remain in one indivisible split group. This
phase must not inherit Line9-selected parameter values unless the resulting
paper experiment is explicitly reported as Line9-conditioned.

### Phase D: Acquisition Domain Randomisation

After the flat fixed-height mechanism passes, add independently sampled flight
height, smooth terrain, source amplitude, band taper, and bounded system noise.
Change one factor group at a time. Terrain or flight variation must not be used
to hide a failed subsurface model.

### Phase E: Limited 3-D Domain-Gap Audit

Use short 3-D windows for off-line scatterers, antenna-pattern sensitivity,
and out-of-plane clutter. These cases quantify 2-D limitations and are not mixed
silently with the 2-D training distribution.

## Required Physical Runs

For a positive scene:

- `full_scene`: required;
- exact `no_basal_contrast_control`: required for causal labels;
- `air_reference`: one reusable run per identical source/acquisition/domain
  cohort, required only for decomposition/system diagnostics.

For a designed true negative:

- target-absent `full_scene`: required;
- it must be physically identical to its positive's no-basal state;
- it has no target path, target mask, or visible-phase label;
- it may reference the cohort air run instead of duplicating it.

## Staged Runtime Gate

Run in this order and stop at the first failure:

1. contract and generator tests;
2. static input and wavelength audit;
3. transient geometry build with VTI hash then deletion;
4. one-trace positive full/control plus negative-full equality;
5. 8-16 consecutive traces for local continuity;
6. 32 full-span traces for blind gross morphology;
7. 32 full-span strict pair for causal continuity;
8. native 256 full/control and required negative;
9. canonical export, immutable evidence package, and human release decision.

Full-only pilots never produce training labels. Sparse pilots never prove
native-spacing coherence.

## Mandatory Visual Audit After Every Runtime Stage

Every completed simulation stage must produce and inspect all applicable views:

1. geometry/property preview at fixed physical scale;
2. label-free raw B-scan with a shared robust scale;
3. horizontal-background-suppressed plus restrained `time^1.5` gain;
4. background-suppressed AGC(13), clearly marked as morphology-only;
5. signed full-minus-control at common gain and difference-only gain;
6. reference overlay opened only after the blind decision;
7. target-envelope and adjacent-background profiles.

Accept only when the basal packet is laterally continuous, has plausible
multi-cycle signed structure, follows broad geology, varies in amplitude, and
does not become a chain of isolated hyperbolas. Reject or redesign when the
background is blank, the target is a single overexposed line, regular horizontal
combs dominate, phases switch independently by trace, or an event is explained
by a PML/endpoint/material wall.

Visual similarity supports a decision but never overrides failed causality,
provenance, trace completeness, or split governance.

## Quantitative Screening

Report at minimum:

- path/geometric correlation and residual-step continuity;
- target dropout and envelope coefficient of variation;
- full-scene target/adjacent-background RMS;
- signed-pair target/background RMS and early leakage;
- aligned-template correlation, signed-lobe sequence, peak frequency, and
  spectral centroid;
- cover-field neighbour-change rates and vertical spectral concentration.

These metrics screen failures; none is an automatic realism score.

## Release And Storage

- Commit contracts, deterministic generators, source decks, small geometry,
  hashes, reports, and selected previews.
- Keep raw per-trace `.out` files and VTI outside Git.
- Release only curated merged evidence and canonical training arrays through
  the existing registry/LFS policy.
- Update `simulation_cases.csv`, `human_audit_manifest.csv`,
  `simulation_asset_registry.json`, `dataset_manifest.json`, and
  `docs/current_state.md` together.
- A second computer inherits work by pulling the same commit and LFS objects,
  then filling only its ignored machine runtime profile.

## Formal Exit Gate

Simulation data may enter paper training only when:

- `line9_conditioned=false` at array and decision levels;
- every parameter source is recorded and independent;
- positive/control/negative equivalence is proven;
- all required 256-trace runs and visual audits pass;
- no target label is derived from a full-only or unpaired run;
- human release explicitly sets `train_allowed=true`;
- project formal-ready validation passes.
