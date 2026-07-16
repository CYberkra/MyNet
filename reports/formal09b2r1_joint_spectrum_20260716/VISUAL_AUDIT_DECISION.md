# FORMAL09B-2R1 Joint 2D Spectrum Visual Audit

Date: 2026-07-16

## Decision

```text
REJECT_JOINT_GAUSSIAN_NUISANCE_FIELD
DO_NOT_PROCEED_TO_FORMAL09B-3_METADATA_CONDITIONING
RETAIN_FORMAL09B-1_AS_DIFFUSE_SPECTRAL_COMPONENT
NEXT: FORMAL09C_SPARSE_COHERENT_EVENT_FIELD
```

The blind mapping was A=09B-1 paper, B=joint stationary, C=joint
nonstationary, and D=rejected separable nonstationary.

Both joint candidates preserve the accepted basal packet but remain visually
too similar to the controls. They do not reproduce the measured lines'
finite-length sloping events, local packet changes, discontinuities, or mixed
orientations. The pooled joint spectrum is dominated by a broad temporal band
with weak frequency-wavenumber ridges. Random-phase Gaussian sampling preserves
second-order power but destroys sparse event topology.

The adjacent background-envelope correlation also remains low (about
0.043-0.099), and the apparent background is still a field of long regular
ripples. Metadata-conditioned gain would only modulate this wrong morphology.

## What remains valid

- FORMAL06C remains the physical mother model.
- FORMAL09B-1 remains the accepted diffuse temporal-spectrum component.
- The target-exclusion, equal-line pooling, physical cycles/m axes, fold-local
  fitting, and no-copy contracts remain valid infrastructure.
- The separable and joint Gaussian candidates remain negative ablations.

## Next mechanism

FORMAL09C should model a sparse coherent event field rather than another
stationary random field:

1. estimate target-excluded event slope, length, curvature, amplitude, and
   dropout distributions with a structure-tensor/Radon-style detector;
2. synthesise new parametric paths from those distributions without copying
   measured patches or coordinates;
3. render each path with a newly sampled 09B-1 empirical wavelet and a bounded
   local amplitude envelope;
4. mix a small number of finite events with the accepted 09B-1 diffuse field;
5. apply one identical realisation to full/control;
6. preserve Line3/Line7/L1 fit, Line6 validation, Line9 holdout;
7. blind-review event count, orientation, finite support, packet preservation,
   joined-hyperbola risk, and target/background ratio before metadata factors.

Flight-height gain and timing remain deferred.
