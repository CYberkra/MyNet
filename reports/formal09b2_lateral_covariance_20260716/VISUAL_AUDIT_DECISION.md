# FORMAL09B-2 Separable Lateral Covariance Visual Audit

Date: 2026-07-16

## Decision

```text
REJECT_SEPARABLE_LATERAL_COVARIANCE
DO_NOT_PROCEED_TO_METADATA_CONDITIONING
REDESIGN_AS_FORMAL09B-2R1_JOINT_2D_SPECTRUM
```

The blind mapping was A=paper-fold nonstationary, B=09B-1 paper spectrum,
C=all-lines nonstationary, and D=paper-fold covariance-only.

All candidates preserve the FORMAL06C basal packet, but none provides a
material visual improvement over 09B-1. The long, nearly horizontal and
spatially uniform ripple remains. Nonstationary amplitude changes are too weak
to create the measured trace-local structure, and increasing their magnitude
would only make the same wrong texture uneven.

## Numerical warning

The adjacent background-envelope correlation changed from about 0.261 in
09B-1 to 0.069-0.162 in the covariance candidates. The estimated spatial
spectrum is broad and was applied as a separable temporal-filter times
spatial-filter model. That operation matches two marginal spectra but discards
joint temporal-frequency/spatial-frequency coupling. Sloping and oriented
coherent events live in that joint structure.

## Required redesign

FORMAL09B-2R1 must:

1. fit target-excluded two-dimensional residual power on time-by-distance
   patches;
2. represent axes in MHz and cycles/m, not samples and trace index;
3. pool line-normalised 2D spectra with equal line weight;
4. symmetrise acquisition direction so one line orientation cannot dominate;
5. sample new complex coefficients rather than copying measured patches;
6. compare joint-2D stationary and joint-2D nonstationary variants against
   this rejected separable control at the same target/background ratio;
7. keep Line6 validation-only and Line9 held out in the paper-fold candidate.

FORMAL09B-3 metadata conditioning remains blocked until this gate passes.
