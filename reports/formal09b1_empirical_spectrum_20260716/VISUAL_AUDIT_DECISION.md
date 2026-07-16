# FORMAL09B-1 Empirical Spectrum Visual Audit

Date: 2026-07-16

## Decision

```text
PASS_SPECTRAL_MECHANISM_ONLY
DO_NOT_PROMOTE_AS_FINAL_REALISM_CANDIDATE
PROCEED_TO_FORMAL09B-2_LATERAL_COVARIANCE
FORMAL06C_REMAINS_PHYSICAL_MOTHER_MODEL
```

FORMAL09B-1 is a post-solver single-factor experiment. It does not run a new
gprMax scene and is not training data. It replaces only FORMAL09A's hand-shaped
Gaussian temporal filter with an equal-line pooled spectrum fitted outside the
V15 target corridor.

## Blind visual review

The blind mapping was:

```text
A = FORMAL06C baseline
B = 09B-1 paper Line9-holdout fit
C = 09A balanced hand-shaped spectrum
D = 09B-1 development all-lines fit
```

- A remains too clean and target-dominant.
- B and D preserve the accepted continuous multi-cycle basal packet. Their
  visual similarity is evidence that Line9 does not dominate the empirical
  spectrum.
- C adds a broader and less instrument-specific texture. Its solved target
  centroid moves to about 105.3 MHz.
- B and D recover a narrower measured residual band and move the solved target
  centroid to about 97.7 MHz while retaining the 79.37 MHz dominant lobe.
- B and D still contain long, spatially uniform horizontal ripples. They do not
  reproduce the measured lines' trace-local variability, changing coherence,
  or multiple oriented events.

The spectral mechanism therefore passes its narrow intended gate, but the
overall realism candidate does not pass.

## Leakage and fold audit

The spectrum fit removes the stable median trace and excludes a +/-42 ns V15
target corridor before spectral frames are collected. Each line is normalised
independently, then pooled by an equal-line geometric mean. Trace count cannot
give Line9 extra statistical weight.

| Fit | Lines | Frames | Peak | Centroid |
|---|---|---:|---:|---:|
| Development | Line3/6/7/9/L1 | 25,696 | 103.24 MHz | 93.95 MHz |
| Paper fold | Line3/7/L1 | 14,488 | 104.63 MHz | 94.53 MHz |

The pooled 20-250 MHz log-spectrum RMSE between the two fits is about 1.02 dB.
This is small enough to retain the paper-fold version as the default starting
point for the next stage.

## Next factor

FORMAL09B-2 must change only lateral structure:

1. estimate target-excluded cross-trace cross-spectral covariance by line;
2. represent covariance with a bounded low-rank or coherence-length mixture;
3. sample new coefficients, never measured trace blocks;
4. keep the 09B-1 paper-fold spectrum fixed;
5. keep target/background ratio, gain budget, physical scene, and display fixed;
6. reject uniform ripple, hard dropout, joined hyperbolas, packet fracture, or
   target-aligned enhancement.

Metadata-conditioned gain and timing remain deferred to FORMAL09B-3.

## Evidence

- `FORMAL09B1_blind_empirical_spectrum.png`
- `FORMAL09B1_multiline_visual_comparison.png`
- `FORMAL09B1_spectrum_audit.png`
- `FORMAL09B1_fitted_spectra.json`
- `formal09b1_manifest.json`
