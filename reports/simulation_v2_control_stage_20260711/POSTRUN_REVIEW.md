# Simulation V2 Controls: Post-run Review

## Scope

All four official controls were solved on the local RTX 5070 with gprMax
3.1.7, then resampled from the CFL output to the canonical `501 x 256`
representation. The visual pack is in `postrun_review/`.

## Computational Review

| Control | Result | Decision |
|---|---|---|
| CTRL01 flat shallow positive | matched contrast, phase and exact flat-reference contract pass | review-ready |
| CTRL02 flat deep positive | matched contrast, phase and exact flat-reference contract pass | review-ready |
| CTRL03 smooth-interface positive | continuous-path extraction pass; curved geometry reference is explicitly non-exact | review-ready, retain for physical review |
| CTRL04 matched background negative | confirmed target mask has zero nonzero pixels | review-ready negative |

The positive controls all retain `formal_training_allowed=false`. Review-ready
means the solver, HDF5 contract, matched controls, and extraction contract
passed; it does not mean that a human has approved physical realism or data
promotion.

## Important Correction

The first CTRL03 extraction selected a small number of isolated, stronger
signed lobes independently per trace. Its P95 jump was only 1.4 ns, but its
maximum jump was 47.6 ns. That was a real gate defect: P95 alone hid a
physically implausible phase switch in a smooth-interface control.

The extractor now uses a geometry-anchored continuous envelope path for
non-flat controls, and postprocess checks both P95 and maximum trace-to-trace
step. CTRL03 now has a maximum step of 5.6 ns and passes without rerunning
FDTD. A regression test covers a single-trace stronger-lobe distractor.

## Visual Notes

- CTRL01 and CTRL02 exhibit coherent flat target events at their extracted
  visible phases. CTRL02 has lower support than CTRL01, as expected for a
  deeper, more attenuated target, but remains well above the acceptance floor.
- CTRL03 follows the curved event after continuous extraction. Its absolute
  offset from the columnar geometry reference is not reported as an arrival
  accuracy claim because that reference is deliberately non-specular.
- CTRL04 has no target mask. Edge-dominated full-air residuals are retained as
  a boundary-effect observation, not relabelled as a target.

## Still Required Before Any Promotion

1. Check the four rendered panels against the intended material, depth, and
   antenna assumptions.
2. Confirm or revise the provisional 0.18 m Tx/Rx offset from hardware
   documentation. A revision requires regenerated controls and rerun outputs.
3. Define an explicit edge-trace policy before exporting training windows;
   endpoint transients must not silently become target evidence.
4. Record a human audit decision in the simulation governance manifest.

No control is promoted by this report.
