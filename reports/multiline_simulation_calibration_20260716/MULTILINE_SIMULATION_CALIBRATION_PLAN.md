# Multi-Line Simulation Calibration Plan

Date: 2026-07-16

## Locked visual decision

The project-owner ranking is:

```text
FORMAL06C > Independent Family 02 > Independent Family 03
```

FORMAL06C remains the only mother model for the next physical family. Family 02
and Family 03 are retained as ablations and must not become accidental parents.

## Objective

The simulator should represent the range shared by credible measured data, not
copy a single Line9 profile. Measured references are separated by purpose:

- `signal_style`: frequency band, background energy, lateral continuity,
  dynamic range, gain response, and acquisition-domain variation;
- `interface_morphology`: basal path and multi-cycle packet shape, using only
  strong V15 labels outside ignore and height-exception traces;
- `stress_only`: unusual but measured acquisition conditions, used only after
  a nominal candidate is frozen;
- `review_only`: evidence that cannot tune or promote a simulator.

LineX1 is review-only. Line7 traces 475-708 are a height-exception stress
segment, not nominal calibration evidence.

## Two explicit tracks

### Development all-lines track

Use credible segments from Line3, Line6, Line7, Line9, and LineL1. This is the
fastest route to a simulator that broadly resembles the measured domain. Any
candidate selected on this pool is `line9_conditioned=true` and cannot support
an unseen-Line9 claim.

### Paper Line9-holdout track

Use:

```text
fit:        Line3, Line7 valid segments, LineL1
validation: Line6
test:       Line9
review:     LineX1
```

The generator mechanism may be shared with the development track, but its
parameters and visual selection must be frozen before opening Line9 results.
This track is the primary generalization experiment.

## Next simulator family

The next family inherits FORMAL06C exactly at the physical baseline. It varies
one declared factor at a time across ranges estimated from the multi-line
reference pool:

1. basal relief family and transition thickness;
2. cover attenuation and weak correlated heterogeneity;
3. measured flight-height family;
4. acquisition-domain common-mode residue, smooth gain jitter, and colored
   noise, applied with a shared seed to matched full/control pairs.

Measured arrays are never copied into synthetic cases. The acquisition-domain
layer is a versioned derived artifact; canonical gprMax outputs remain
immutable.

## Visual release gate

Every solved pilot must be reviewed at a fixed display contract before wider
simulation:

1. raw signed B-scan;
2. common-mode suppressed plus time-power gain;
3. target-path crop without label overlay;
4. independently robust-scaled measured references;
5. blind panel containing at least three measured lines, FORMAL06C, and the
   candidate;
6. explicit checks for continuous basal packet, non-periodic background,
   absence of joined hyperbolas, no combing, no unexplained dropout, and no
   target overexposure.

Acceptance is based on a range across lines, not nearest distance to Line9.

## Immediate decision

FORMAL08A and FORMAL08B remain failed ablations. The next run should not add
more transition-following deep texture. First evaluate a deterministic,
bounded acquisition-realism layer on released FORMAL06C traces, then transfer
the accepted range to independently generated FORMAL06C-family geometries.
