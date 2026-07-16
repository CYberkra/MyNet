# FORMAL09C Visual Audit Decision

Date: 2026-07-16

## Decision

`FORMAL09C` is retained as a mechanism sandbox and rejected as a promoted
realism candidate. None of `light`, `balanced`, or `rich` is eligible for
training export or a gprMax causal claim.

## What worked

- The detector found finite target-excluded coherent components on Line3,
  Line7, and LineL1 without reading Line9.
- The generator sampled new event locations, phases, lengths, slopes, and
  amplitudes; it did not copy measured patches or event coordinates.
- Every generated event respected the declared target-corridor overlap gate.
- The basal packet remained continuous, retained a 79.37 MHz peak, and had no
  dropout in all three candidates.

## What failed the visual gate

- After replacing horizontal bilinear interpolation with nearest-neighbour
  display, the apparent improvement over FORMAL09B-1 became small.
- At the 0.72 m sparse trace spacing, short events occupy only a few observed
  traces. Several therefore look block-like or nearly vertical and cannot
  establish native lateral morphology.
- `light` is visually almost indistinguishable from FORMAL09B-1.
- `balanced` adds finite packets but includes a short isolated block that is
  not yet a credible geological event.
- `rich` adds more upper-window packets, but they remain too few and too
  regular to approach the multi-line measured background.
- All candidates remain much cleaner and more wavelet-regular than the
  measured references. The background envelope and coherence metrics changed
  only slightly after robust peak scaling.

## Ranking

For preservation of the accepted basal packet:

```text
09C-light ~= 09B1-paper > 09C-balanced > 09C-rich
```

For added finite-event visibility:

```text
09C-rich > 09C-balanced > 09C-light
```

No candidate wins both criteria strongly enough to replace FORMAL06C as the
mother morphology.

## Next gate

Translate the finite-event hypothesis into a physical, native-trace gprMax
ablation instead of increasing post-solver weights:

1. Preserve the FORMAL06C source, basal path, transition, grid, acquisition,
   and material baseline.
2. Add only low-contrast, finite, gently dipping mid-cover laminae or lenses;
   forbid point targets, vertical partitions, and periodic slabs.
3. Convert the paper-fold event length and slope ranges to bounded physical
   extent and dip priors. Keep curvature conservative because the measured
   second-derivative estimate is unstable.
4. Run a geometry audit and a native-spacing consecutive 64-trace full-scene
   checkpoint first. Sparse stride-8 output is not a valid topology gate.
5. If the blind morphology improves, run the exact full/no-basal pair and then
   the native 256-trace checkpoint. Otherwise archive the factor.

The next case identifier is
`FORMAL09C_P1_DENSE_PHYSICAL_FINITE_LAMINAE`.
