# FORMAL08B Pre-Solver Audit

Date: 2026-07-16
State: pre-solver pass; gprMax runtime not started

## Decision

FORMAL08B is accepted only for an eight-consecutive-trace `full_scene`
checkpoint. It is not accepted for distributed simulation, dense simulation,
training, visible-phase labels, causal release, or an unseen-Line9 claim.

The candidate is a direct FORMAL06C successor. Source waveform, constitutive
materials, grid, PML, acquisition, solver window, basal path, transition
thickness, surface cover bins, and transition-neighbour cover bins are locked.
Only transition-following multiscale continuous deep-cover texture changes.

## Conditioning And Claim Boundary

Line9 is used openly to calibrate the desired morphology, spectrum, target
prominence, and continuous background character. The generator reads no
measured array, but its design is still informed by Line9. Therefore the case
is `line9_conditioned=true`, `formal_training_allowed=false`, and permanently
blocked from a strict unseen-Line9 claim.

## Geometry Result

- Predecessor latent correlation: 0.846557
- Perturbation RMS: 0.519003
- Changed cover-bin fraction: 0.220213
- Cover-bin delta P99: 7 bins
- Protected surface and transition bins: exact
- Horizontal neighbour-bin change: 0.023906 (FORMAL06C: 0.021547)
- Vertical neighbour-bin change: 0.056732 (FORMAL06C: 0.047454)
- Laterally coherent depth-variance ratio: 0.480563 (FORMAL06C: 0.545453)
- Vertical spectral peak fraction: 0.935717 (FORMAL06C: 0.945942)
- Isolated inclusions, point targets, and vertical partitions: zero

The blind geometry preview shows a materially stronger, smooth, aperiodic deep
background while the FORMAL06C basal packet is unchanged. The enhanced delta
view exposes quantised contours. These are not explicit layers, but runtime
must reject the case if they solve into a regular parallel wavelet comb.

## Physics And Provenance

All full, no-basal, and air input decks pass the static audit with zero errors
and warnings. At 80 MHz, epsilon 12.8, conductivity 0.0025 S/m, and a
conservative 15.8 m one-way depth, the nondispersive attenuation budget is
about 36.12 dB two-way field loss before geometric spreading, antenna response,
interfaces, and dispersion. This is a plausibility gate, not solved evidence.

The generator SHA256 matches the scene manifest. All 27 entries in
`FILE_SHA256.csv` match their current files. Six focused FORMAL08A/08B tests
pass.

## Runtime Gate

1. Solve eight consecutive `full_scene` traces only.
2. Compare exact common traces with FORMAL06C at the same crop, gain, and scale.
3. Review a label-free image before any reference overlay.
4. Continue only if the accepted multicycle packet survives and the background
   change is visually useful without combs, fragments, hyperbolas, or dropout.
5. Run a full-span distributed 32-trace scene only after that pass.
6. Do not run matched controls or native 256 until the full-span morphology
   review passes.
