# FORMAL09A Multi-Line Acquisition Realism Visual Audit

Date: 2026-07-16

## Decision

```text
REJECT_AS_FINAL_CANDIDATE
RETAIN_BALANCED_AS_NEXT-MECHANISM_START
FORMAL06C_REMAINS_PHYSICAL_MOTHER_MODEL
```

FORMAL09A is a deterministic post-solver development experiment, not a new
gprMax scene and not training data. It tests whether acquisition/system-domain
variation can reduce the common gap between FORMAL06C and five credible
measured lines without changing the physical basal model.

## Blind visual review

The blind mapping was A=FORMAL06C, B=mild, C=balanced, D=strong.

- A preserves the accepted continuous multi-cycle basal packet but remains
  visibly too clean and target-dominant.
- B adds a faint diffuse background and preserves the packet. The change is
  useful but too small to cover the measured reference pool.
- C is the best compromise. It adds visible non-target energy above and below
  the packet without creating joined hyperbolas, combs, hard dropout, or a
  broken interface.
- D reaches a measured-like scalar target/background ratio but produces too
  much spatially uniform wavelet-like texture. It looks like globally added
  colored noise rather than the non-stationary measured domain.

The multi-line panel confirms that the measured lines contain stronger
trace-local variability, multiple coherent events, and spatially changing
background orientation. FORMAL09A does not yet reproduce those properties.

## Metric interpretation

| Data | Target/background | Envelope CV | Template correlation |
|---|---:|---:|---:|
| FORMAL06C | 17.29 | 0.334 | 0.660 |
| 09A mild | 8.00 | 0.337 | 0.617 |
| 09A balanced | 4.50 | 0.339 | 0.558 |
| 09A strong | 2.50 | 0.328 | 0.371 |
| Measured five-line range | 1.22-2.89 | 0.389-1.043 | 0.064-0.458 |

The strong variant is numerically close in two metrics but visually wrong.
The balanced variant is visually safer but still too stationary and too
amplitude-uniform. This is another case where scalar proximity is not a visual
release gate.

## Next design: empirical paired nuisance transfer

FORMAL09B should replace the hand-shaped Gaussian band with a multi-line
empirical acquisition operator:

1. estimate target-excluded amplitude spectra from Line3, Line6, valid Line7,
   Line9, and LineL1 separately;
2. estimate lateral cross-spectral covariance and a smooth spatial amplitude
   envelope instead of assuming stationary independent traces;
3. condition bounded gain and time-zero variation on measured flight height
   and terrain slope;
4. sample new complex coefficients rather than copying measured traces;
5. apply the identical operator realization to each matched full/control pair;
6. keep the canonical gprMax outputs immutable and version the transformed
   arrays as derived evidence;
7. select the development version on all lines, and independently refit the
   paper version on Line3/Line7/LineL1 with Line6 validation before Line9 is
   opened.

The first 09B ablation should be empirical spectrum only. The second adds
lateral covariance/non-stationarity. Time-zero and metadata conditioning come
only after the first two visual gates pass.

## Longer-term physical factors

- Fit bounded multi-Debye material dispersion as a separate one-factor gprMax
  family; do not confound it with acquisition noise.
- Generate several correlated basal/cover families from the multi-line range,
  not one fixed FORMAL06C geometry.
- Use a small number of 3D patches to estimate out-of-plane residual effects
  only after the 2D paired pipeline is stable.
- Add feature-level domain adaptation to the network as a separate ablation;
  do not use full-image style translation to overwrite causal labels.

## Evidence

- `FORMAL09A_blind_variants.png`
- `FORMAL09A_multiline_visual_comparison.png`
- `formal09a_manifest.json`

Primary-method references:

- Stephan, Allroggen, and Tronicke (2024), convolution-based realistic GPR
  noise modelling: https://doi.org/10.1002/nsg.12273
- Koyan and Tronicke (2020), multiscale realistic sedimentary gprMax model:
  https://doi.org/10.1016/j.cageo.2020.104422
- Majchrowska et al. (2021), arbitrary dispersive material approximation in
  gprMax: https://arxiv.org/abs/2109.01928
