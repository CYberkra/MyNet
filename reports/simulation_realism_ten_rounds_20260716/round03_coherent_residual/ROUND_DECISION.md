# Round 03 decision: target-excluded low-rank coherent residue

Date: 2026-07-19

## Experiment

- Parent: Round 01 frozen bounded system-response operator.
- Fit: Line3, Line7, LineL1; validation: Line6; Line9 opened only after
  the rank selection was frozen.
- Target corridors were excluded by +/-35 ns before fitting temporal modes.
- Only temporal modes were retained; each candidate sampled new spatial
  coefficients. No measured trace, patch, target path, or coordinate was
  copied into a synthetic panel.

## Result

All non-zero ranks were amplitude-calibrated to the fold target/background
ratio (1.40). The score selected rank 6, but it remained structurally wrong:
its first six singular components contained 99.56% of processed energy, while
Line6 contained 62.69% and frozen Line9 contained 63.34%. The blind image is
dominated by regular horizontal bands. It is neither the local, mixed-orientation
topology of the measured background nor a plausible physical clutter field.

This independently reproduces the repository's prior FORMAL09B-2 and
FORMAL09B-2R1 conclusion: marginal or joint second-order structure does not
preserve finite event topology when it is synthesized with low-rank/random
coefficients.

## Decision

```text
target-excluded low-rank coherent residual: REJECT
low-rank rank-6 selection: diagnostic only, not an augmentation
second-order residual synthesis: closed as a design branch
successor: non-Gaussian finite-event and physically generated mechanisms
```

The failed result remains valuable as a negative ablation. It prevents later
work from trying to compensate for missing finite geometry with stronger
low-rank or random-phase clutter.
