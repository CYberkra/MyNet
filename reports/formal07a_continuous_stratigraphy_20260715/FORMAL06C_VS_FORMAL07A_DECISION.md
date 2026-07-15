# FORMAL06C Versus FORMAL07A Decision

Date: 2026-07-15

## Question

Does FORMAL07A preserve FORMAL06C's accepted measured-like basal response while
improving the realism of the non-target cover background?

## Comparison Contract

- Common canonical traces: `0,32,64,96,128,160,192,224`
- Common time window: 0-500 ns
- Common temporal output: 358 samples
- Common display scale: joint P99.5 independently locked for raw, time-power,
  and AGC rows
- Identical processing: horizontal median suppression, normalized time^1.5,
  and AGC(13)
- No horizontal interpolation: sparse trace columns use nearest-neighbour
  expansion
- Primary decision image: no label or material-reference overlay
- Reference overlays opened only after the blind decision

## Blind Visual Finding

FORMAL06C retains the better project-specific morphology. Its basal response
is a continuous, visible, multi-cycle band with meaningful long-wave relief.
Its non-target background is quieter than measured data, but it does not create
a competing synthetic layer stack.

FORMAL07A remains physically causal, but its cover contains several strong,
regular, laterally coherent wave groups. The result reads as a constructed
stratigraphic stack rather than weak, irregular measured clutter. Its basal
band remains visible but is flatter and more repetitive.

Opening the material/source references confirmed that the blind comparison
followed the intended basal responses rather than adjacent phases.

## Supporting Metrics

| Metric | FORMAL06C | FORMAL07A | Interpretation |
|---|---:|---:|---|
| Basal path range | 34.460 ns | 10.897 ns | 07A is substantially flatter |
| Path/geometric correlation | 0.999931 | 0.999974 | both track their designed path |
| Target/adjacent-background RMS | 17.291 | 15.305 | both visible; 06C is cleaner |
| Target envelope CV | 0.334 | 0.272 | 07A is more uniform/repetitive |
| Aligned-template correlation median | 0.660 | 0.948 | 07A repeats a more uniform wavelet |
| Significant alternating lobes | 7 | 9 | 07A has a longer ringing character |
| Peak frequency | 79.365 MHz | 79.365 MHz | source contract is preserved |

## Decision

- Retain FORMAL06C as the human-accepted development morphology baseline.
- Reject FORMAL07A as FORMAL06C's successor.
- Keep FORMAL07A only as an unreleased, Line9-conditioned causal regression
  showing the failure mode of over-regular continuous stratigraphy.
- Cancel FORMAL07A 32-trace and 256-trace expansion.
- Do not use either family for formal training.

## Next Design Contract

The next ablation must inherit FORMAL06C's basal path, transition, source,
materials, grid, acquisition, and strict pair. Change only the non-target cover
texture, with the following limits:

1. weak amplitude relative to the basal band;
2. aperiodic and multiscale rather than sinusoidal layer stacks;
3. laterally finite coherence without full-width constant reflectors;
4. no point-target field, vertical walls, or repeated hyperbolic motifs;
5. common-trace blind comparison against FORMAL06C before any wider run;
6. reject if background texture competes with or visually regularises the basal
   event.
