# FORMAL06H Native-64 Decision

**Case:** `FORMAL06H_SOURCE_TEMPORAL_DEVELOPMENT`
**Date:** 2026-07-22
**Decision:** Reject for further 2D native-spacing promotion.

## Question

Could the regular full-window lobe train in FORMAL06D--G be explained mainly
by the temporal support of its 80 MHz zero-mean Gabor source?

FORMAL06H preserves the FORMAL06D geometry, materials, basal interface,
transition, grid, acquisition, and trace positions. Its only model change is
the source waveform: 80 MHz zero-mean Gabor is replaced by 80 MHz Ricker.
This is a temporal-source ablation only; it does not model finite-antenna
radiation, directional coupling, or a three-dimensional antenna geometry.

## Evidence

1. The generator regression test confirms that FORMAL06D and FORMAL06H share
   the same geometry-index arrays and differ only in the declared source
   waveform. The source-deck static audit reports zero errors and zero
   warnings.
2. A strict one-trace full/control pair completed with coherent source and
   receiver positions. Its signed full-minus-control signal has a target to
   background RMS ratio of 129.01, so the temporal change did not remove the
   intended basal contrast mechanism.
3. A blind native-spacing, consecutive 64-trace full-scene run (canonical
   traces 109--172) captured all requested traces. The raw, horizontal-
   background-removal plus AGC, and restrained time-power views are available
   beside this report.
4. Both label-free post-processed views retain a dense, regular, nearly
   parallel multi-cycle lobe train across almost the entire 0--500 ns window.
   The Ricker source changes local wavelet appearance but does not create the
   sparse, laterally traceable, subtle interface morphology required for this
   project.

## Verdict

FORMAL06H fails the blind morphology gate. Do not run its native-64 control
pair or a 256-trace release pair. Do not promote it to training data.

Together with the independent FORMAL06E (cover covariance), FORMAL06F
(transition staging), and FORMAL06G (terrain/AGL) rejections, this rules out
the tested cover, transition, terrain, and temporal-waveform factors as a
sufficient explanation of the coherent comb. The leading unresolved cause is
the ideal two-dimensional Hertzian line-source / receiver abstraction.

## Next Valid Experiment

Before another geology or generic waveform sweep, obtain a hardware
measurement contract: antenna dimensions, Tx/Rx spacing, polarization,
mounting, AGL, and a measured direct/air pulse or defensible source spectrum.
Then design one bounded, auditable finite-antenna three-dimensional local
window or a documented reduced-order equivalent. It must remain distinct from
the present 2D temporal waveform ablation.

## Artifacts

- `formal06h_static_input_audit.json`
- `formal06h_smoke1_strict_audit/family_spatial_pilot_audit.json`
- `formal06h_native64_agc.png`
- `formal06h_native64_tpower15.png`
