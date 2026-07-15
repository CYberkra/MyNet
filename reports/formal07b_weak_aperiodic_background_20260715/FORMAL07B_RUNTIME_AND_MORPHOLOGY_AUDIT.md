# FORMAL07B Runtime and Morphology Audit

Date: 2026-07-15

## Decision

FORMAL07B passes as an **agent-accepted controlled development successor** to
FORMAL06C. It preserves the accepted basal morphology while adding only weak,
aperiodic non-target cover texture. It does not replace FORMAL06C as the
project-owner accepted visual reference, and it is not a training release.

The case remains development-only because its mechanism was selected after
Line9 morphology diagnosis. `formal_training_allowed` remains `false` and
`strict_line9_holdout_allowed` remains `false`.

## Controlled Factor

The generator locks the FORMAL06C source, grid, acquisition, constitutive
materials, basal profile, and transition-thickness profile. The only changed
factor is a weak two-dimensional aperiodic perturbation of the non-target cover
field. It adds no sinusoidal slabs, isolated inclusions, point targets, or
vertical partitions.

Pre-solver gates passed:

| Metric | Result | Gate |
|---|---:|---:|
| Predecessor latent correlation | 0.98903 | >= 0.985 |
| Perturbation RMS | 0.13167 | 0.07-0.16 |
| Cover-bin delta P99 | 2 bins | <= 3 bins |
| Layer-coherent variance ratio | 0.54545 -> 0.51332 | increase <= 0.02 |
| Vertical spectral-peak fraction | 0.94594 -> 0.93887 | increase <= 0.04 |

Approximately 49.8% of cover voxels changed quantized bin, but 99% of changes
were no larger than two 0.025-epsilon bins. The high changed fraction therefore
represents spatially distributed weak variation, not a large material jump.

## Static and Runtime Checks

- Full, no-basal, and air-reference input audits passed without warnings.
- Geometry-only execution passed; generated VTI files were hashed and deleted.
- The 0.03 m grid retained 12.37 cells per minimum wavelength under the project
  `2.8 * fc` guard.
- The 32-trace full-scene CUDA run completed all canonical indices
  `0, 8, ..., 248` in 13 minutes 49 seconds.
- The run produced 9,188 native samples per trace to 650.07 ns and retained the
  protected 0-500 ns analysis window.

## One-Trace Causal Gate

The matched full/no-basal one-trace pair passed:

| Metric | FORMAL07B |
|---|---:|
| Visible minus source-reference time | +1.083 ns |
| Signed target RMS | 0.001518 |
| Signed target/background RMS | 96.48 |
| Full target/local-background RMS | 4.21 |
| Early full/control relative difference | 3.51e-7 |

This proves causal timing at the smoke trace only. It does not license
full-span causal attribution or visible-phase training labels.

## Distributed 32-Trace Morphology

The comparison used exactly the same 32 canonical trace positions as the
released FORMAL06C evidence, a common 0-500 ns crop, common P99.5 scales, and
no horizontal interpolation.

| Metric | FORMAL06C | FORMAL07B |
|---|---:|---:|
| Path/geometric correlation | 0.99993 | 0.99994 |
| Target/adjacent-background RMS | 17.29 | 16.74 |
| Target envelope CV | 0.334 | 0.349 |
| Median aligned-template correlation | 0.660 | 0.668 |
| Significant signed lobes | 7 | 7 |
| Aligned peak frequency | 79.37 MHz | 79.37 MHz |

Blind visual review found that FORMAL07B:

- preserves the same gently varying continuous basal packet;
- retains the multi-cycle black/white wavelet character;
- makes the background slightly less uniform without obscuring the target;
- introduces no regular horizontal comb, hyperbola chain, or new isolated
  target-like event;
- remains visually close to FORMAL06C, as required for a one-factor ablation.

The improvement is deliberately modest. It reduces target dominance by about
3.2% in the distributed metric; it is not evidence that measured-domain
realism is solved.

## Claim Limits and Next Gate

The 32-trace run is full-scene-only. It supports morphology selection but not
full-span causal labels. FORMAL07B may be retained as a controlled development
successor and generator-design lesson. It must not enter formal training.

The next simulation work should use independent physical priors, without
Line9 geometry, timing, labels, or morphology as generator inputs. FORMAL06C
and FORMAL07B remain development references for wavelet and morphology review,
not formal-data templates.
