# Independent V2 Family 01 Pilot Audit

Date: 2026-07-15  
Family: `IV2_F01_GENTLE_APERIODIC_COVER_BEDROCK`  
Decision: `pilot_passed_pending_full256`  
Formal training: **blocked**

## Scope

This is the first V2 family generated without reading Line9, measured arrays,
or FORMAL06/07 development geometry. The family contains one positive case and
one target-absent case that is exactly the positive no-basal physical state.

## Physics And Provenance

- 2D one-cell-z FDTD, 0.03 m grid, 55 MHz Ricker source.
- 256 declared traces at 0.09 m spacing and 8.01 m AGL.
- 108 m physical side guards; earliest side return is after the protected
  0-700 ns window.
- Smooth seeded three-scale cover field; no point targets, lenses, isolated
  inclusions, sinusoidal layers, or vertical partitions.
- Positive full/control share one HDF5 geometry array.
- Positive control materials and negative full materials are byte-identical.
- Geometry VTI was generated only for inspection, hashed, and deleted.

## Runtime Evidence

The one-trace positive pair passed and located a signed visible phase within
6.51 ns of the independent source-referenced geometric arrival. The sampled
negative full scene and positive no-basal control produced byte-identical `Ez`
arrays.

The 8-trace full-span causal pair passed with no dropout. Its signed
full-minus-control target/background RMS ratio was 38.53, while the full-scene
target/local-background ratio was 5.69. This proves causal localisation but
also shows that the target is not especially hard.

The 32-trace full-only run completed all trace contracts in about 21.8 minutes.
It covered canonical indices 0-248 at 0.72 m effective spacing. The blind
morphology path tracked the independent geometry with correlation 0.9981,
retained 100.9% of its dynamic range, showed four significant signed lobes,
and had no target dropout. Target/adjacent-background RMS was 7.22.

## Visual Decision

The blind AGC and time-power views show a continuous, gently varying basal
wave packet around 404-433 ns. It is not a chain of local hyperbolas and does
not disappear between sampled positions. The time-power view also makes clear
that the interface remains easier than the intended measured-data difficulty.

Sparse preview columns are rendered with horizontal nearest-neighbour display.
No horizontal interpolation is allowed to manufacture apparent continuity.

## Decision And Limits

The family is accepted as an independent pilot and retained as the first
positive/exact-negative V2 scene family. It is **not** released for training.
The 32-trace full-only path is an audit diagnostic, not a training label.

Before release, run all native 256 traces for positive full/control/air and
negative full/air, package immutable evidence, obtain independent human
approval, and add harder independent families with stronger continuous
non-target background so Family 01 does not dominate the training domain.
