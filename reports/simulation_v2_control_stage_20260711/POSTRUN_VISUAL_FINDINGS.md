# Post-run Visual Findings

## CTRL01: Accept as shallow calibration control

The `80-210 ns` local raw panel contains a coherent, later horizontal bipolar
event at the red visible-phase line. The matched `full - no-basal` panel
isolates the same event cleanly. The cyan geometry reference is earlier by a
stable wavelet phase offset; this is expected and does not indicate a depth
error.

## CTRL02: Accept as deep calibration control

The `279-409 ns` local panel contains the same coherent target wavelet at the
red line. It is weaker than CTRL01, as intended for the deeper and more
attenuated geometry, but remains visible in both raw and matched-contrast
views. This is a useful deep positive control.

## CTRL03: Do not use as a label-template control yet

The curved event is visible, but the target window has a competing branch near
the central traces. The geometry-anchored continuous extraction avoids the
previous 47.6 ns discontinuity, yet the remaining local notch means the phase
identity is not visually unambiguous. Retain it for algorithm stress testing
only; regenerate or redesign its interface/contrast before any label-template
promotion.

## CTRL04: Accept as confirmed-negative control

The target mask is exactly zero. Its full-air residual is concentrated at scan
boundaries, not a basal event. It is valid for negative-control verification,
subject to an explicit future edge-trace policy.

## Display Caveat

The original overview uses cross-trace median subtraction. That deliberately
suppresses flat, laterally continuous reflectors and therefore makes CTRL01
and CTRL02 look as though the basal interface is absent. Use the
`*_target_zoom.png` images for physical review of flat controls; they show raw
local windows without that suppression.
