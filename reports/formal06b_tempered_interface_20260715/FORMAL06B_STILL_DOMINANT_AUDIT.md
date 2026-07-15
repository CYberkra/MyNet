# FORMAL06B Tempered-Interface Audit

Date: 2026-07-15

## Decision

`CAUSAL_PASS_VISUAL_STILL_DOMINANT_STOP`

FORMAL06B preserves the exact FORMAL06A geometry and reduces the cap-to-bedrock
reflection proxy from -0.08318 to -0.01793. The one-trace strict pair passes:

- signed visible-phase offset from the source-referenced geometric estimate:
  +1.21 ns;
- signed difference target/background RMS: 279.28;
- early full/control relative difference: 4.08e-7;
- signed difference target RMS: 0.002866, 3.84 times smaller than FORMAL06A.

The eight-trace unlabelled full-scene checkpoint still fails the frozen upper
visibility bound. Target-to-adjacent-background RMS is 7.44, versus 25.23 for
FORMAL06A, but remains above the declared maximum of 5.0. The late multicycle
band is still the immediately dominant event in the frozen time-power view.

FORMAL06B is retained as a strong-positive development bound. It is not scaled
to a 32-trace pair or admitted to training.

## Next Controlled Ablation

FORMAL06C keeps the source, grid, domain, acquisition, stochastic field,
profile, cap geometry, and strict control unchanged. It changes only the
cap-to-bedrock constitutive contrast, targeting an epsilon reflection proxy of
approximately -0.009 and an eight-trace target/background ratio near 3-4.
