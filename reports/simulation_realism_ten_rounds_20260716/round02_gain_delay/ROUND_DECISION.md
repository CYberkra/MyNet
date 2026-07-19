# Round 02 decision: trace gain and post-hoc delay

Date: 2026-07-16

## Frozen experiment

- Parent: Round 01 `strength_1.00` effective system response.
- Fit lines: Line3, Line7, LineL1.
- Validation line: Line6.
- Held-out diagnostic: Line9, opened only after `gain_0.08` was frozen.
- Candidate gain scales: 0.00, 0.03, 0.05, 0.08 times the fitted local log-gain sigma.
- Separate delay probe: 0.001 times the fitted height/early-wave delay trajectory.

## Result

The fitted measured target/background ratios were 1.22, 1.40, and 1.52 on
the fit lines and 1.37 on Line6. The Round 01 parent ratio was 9.98. Smooth
gain drift lowered this ratio to 2.69, 1.64, and 1.15 for the three non-zero
candidates. The fold score selected `gain_0.08`; its target-envelope CV was
0.319, compared with 0.601 on Line6. The frozen Line9 diagnostic had a ratio
of 2.89 and CV 0.389.

Blind visual review found that the gain candidates expose broad, regular
coherent bands. They add variability but do not reproduce the measured
multi-scale background. The strongest candidate is therefore useful only as
a weak acquisition augmentation, not as a standalone realism solution.

The post-hoc delay probe was rejected. A p95 delay of only 0.000619 ns drove
the target/background ratio to 0.080 because sub-sample shifting destroys the
nearly perfect common-mode cancellation of the synthetic wavefield. This is a
numerical subtraction artefact, not a physical flight-height response.

## Decision

```text
smooth gain drift: PASS_AS_OPTIONAL_LOW_AMPLITUDE_COMPONENT
smooth gain drift as realism solution: REJECT
post-hoc trace delay as height model: REJECT
physical height/pose experiment: DEFER_TO_ROUND_05_FDTD
```

No Round 02 output is approved as formal training data.
