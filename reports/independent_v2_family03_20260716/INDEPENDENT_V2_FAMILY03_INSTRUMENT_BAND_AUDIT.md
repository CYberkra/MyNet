# Independent V2 Family 03 Instrument-Band Audit

## Decision

Family 03 is retained as an independent, physically valid source candidate,
but it is not the preferred visual morphology and is not training-approved.
It is the first pilot in this lineage whose source definition is derived from
project-wide hardware evidence rather than the Line9-conditioned development
selection used by FORMAL06C and Family 02.

## Source Contract

- antenna centre: 100 MHz;
- nominal band: 50-150 MHz;
- supported acquisition band: 20-170 MHz;
- pulse: amplitude-only, zero-phase, band-limited proxy;
- source spectrum peak: 100.10 MHz;
- source spectrum centroid: 98.43 MHz;
- measured arrays read by the generator: none.

This is not an instrument-faithful SFCW reconstruction because measured
complex system phase is unavailable.

## Runtime Evidence

Static audits passed for positive full, positive control, air, and negative
full inputs. Geometry-only gprMax execution loaded the expected 2D TMz domain,
3 cm grid, 750 ns time window, and imported material/index contracts.

The positive one-trace pair passed. The signed full-minus-control target RMS
was 71.17 times the background RMS, while the early full/control relative
difference was 3.16e-7. The matched negative full trace was byte-identical to
the positive no-basal control trace.

The distributed32, stride-8 full-scene run covered canonical traces 0 through
248 without horizontal interpolation. Its path/geometry correlation was
0.9983, target/adjacent-background RMS was 9.65, dropout was zero, and seven
significant signed lobes were retained.

## Visual Review

Blind shared-scale comparisons show a continuous, non-hyperbolic basal packet
that follows the intended broad interface. No isolated target chain, vertical
partition artifact, or missing-interface segment is visible.

Compared with Family 02 and FORMAL06C, however, Family 03 is narrower and
sharper. Its solved aligned spectral centroid is 116.20 MHz rather than Family
02's 79.30 MHz, and its target/adjacent-background ratio is lower (9.65 versus
14.96). The independent source is therefore plausible and useful for domain
diversity, but it is not a visual replacement for the accepted development
packet.

## Claim Limits

- no formal training promotion;
- no visible-phase training label from the full-only distributed run;
- no causal claim beyond the one-trace pair;
- no claim of exact instrument or SFCW phase reproduction;
- no Line9-based source retuning.

The next source experiment must keep geometry and materials fixed and use a
predeclared hardware/literature-constrained source family. Only the selected
candidate should proceed to distributed or native-resolution causal release.
