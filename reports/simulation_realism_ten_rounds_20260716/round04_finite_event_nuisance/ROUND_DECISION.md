# Round 04 Decision: reject image-domain finite-event nuisance

## Decision

`sparse` was the frozen fold-selected variant numerically, but the entire
round is rejected as a standalone realism mechanism and cannot enter a
training export.

## Evidence

- Fit fold: `Line3`, `Line7`, and `LineL1`; validation: `Line6`; Line9 opened
  only after the candidate was frozen.
- The selected field used one new 35-trace event at native 0.09 m spacing. It
  contained no measured waveform segment, coordinates, or target-path values,
  and did not overlap the synthetic target corridor.
- The selected target/background RMS was `1.3900`, close to the Line6 fold
  target of `1.3734`, with zero target dropout.
- The blind and equal-aperture panels show an isolated, over-clean packet that
  is visually separable from the parent background. Increasing density makes
  more isolated packets, not the coupled multiscale texture seen in the fit
  lines.

## Interpretation

This is an important negative result. A finite-support, non-Gaussian process
is more plausible than a Gaussian covariance field, but generating it after
the FDTD solve still leaves the event unconnected to propagation, attenuation,
and neighbouring material structure. Matching one energy statistic is not a
replacement for a physical geological factor.

## Successor rule

Retain only the **topology constraint** for the physical path: any future
mid-cover lamina/lens must be finite, tapered, low contrast, native-spacing
audited, target-separated, and paired with an exact no-basal control. Do not
use the Round 04 field itself as augmentation or as a simulator component.
