# FORMAL06A over-strong interface audit

Date: 2026-07-15

Case: `FORMAL06_INTERFACE_CONDITIONED_DEVELOPMENT`

Decision: `CAUSAL_PASS_VISUAL_OVERSTRONG_STOP`

## Result

- Static and geometry-only audits passed with zero errors and zero warnings.
- The installed solver reported about 22 cells per wavelength at its estimated
  highest significant source frequency and about -0.16% phase-velocity error.
- The one-trace full/control pair passed exact alignment and causal timing.
- Difference-envelope pick: 429.51 ns; source-referenced arrival: 428.01 ns.
- Signed target difference exceeded the pre-target difference by 54.29 dB.
- The target difference RMS was about 29 times FORMAL05.
- In the unlabelled eight-trace full scene, the basal packet is immediately
  visible after the frozen time-power processing, but it dominates the deep
  field and is unrealistically easy.

The case is retained as a mechanism upper bound. It is not trainable and is
not expanded to 32/256 traces.

## Next ablation

Keep the exact source, grid, domain, bulk-field construction, profile family,
and strict local-cover control. Change only the cap-to-bedrock constitutive
contrast. Target a dielectric reflection proxy near -0.02 rather than -0.083,
then repeat one trace and an eight-trace blind full-scene checkpoint.
