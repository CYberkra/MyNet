# FORMAL08A Pre-Solver Audit

Date: 2026-07-16
State: pre-solver pass; gprMax runtime not started

## Decision

FORMAL08A is accepted for an eight-consecutive-trace full-scene checkpoint. It
is not accepted for distributed simulation, dense simulation, training, or a
strict unseen-Line9 claim.

The candidate is a direct one-factor successor to the project-owner accepted
FORMAL06C. It locks the source, material deck, grid, PML, acquisition, solver
window, basal path, transition thickness, surface cover bins, and basal-neighbour
cover bins. Only a depth-tapered continuous middle-cover texture changes.

## Conditioning And Claim Boundary

Line9 is used openly as the measured-realism reference for morphology,
spectrum, target prominence, and continuous non-target background. The
generator reads no measured array. The case is still `line9_conditioned=true`
because the design decision is informed by Line9. It is permanently blocked
from an unseen-Line9 or strict-Line9-holdout claim.

## Geometry Result

- Predecessor latent correlation: 0.920372
- Perturbation RMS: 0.339649
- Changed cover-bin fraction: 0.152366
- Cover-bin delta P99: 3 bins
- Protected surface and basal bins: exact
- Horizontal neighbour-bin change: 0.021967 (FORMAL06C: 0.021547)
- Vertical neighbour-bin change: 0.049197 (FORMAL06C: 0.047454)
- Laterally coherent depth-variance ratio: 0.412954 (FORMAL06C: 0.545453)
- Vertical spectral peak fraction: 0.941719 (FORMAL06C: 0.945942)
- Isolated inclusions, point targets, and vertical partitions: zero

The fixed-scale preview shows the FORMAL06C basal interface and transition
unchanged. Added variation is broad, continuous, and confined to the middle
cover. The enhanced delta panel contains quantisation bands by construction;
they are a difference display, not additional geological layers.

## Physics And Static Checks

All three input decks pass the static audit with zero errors and warnings:

- `full_scene`
- `no_basal_contrast_control`
- `air_reference`

The reviewed gprMax source fingerprint matches version 3.1.7 (`Big Smoke`) and
the maintained project baseline. The nondispersive attenuation budget at
80 MHz, epsilon 13, conductivity 0.0026 S/m, and 13.83 m one-way depth predicts
about 32.63 dB two-way field loss before geometric spreading, antennas,
interfaces, and dispersion. This is a plausibility gate, not a solved response.

## Runtime Gate

1. Solve eight consecutive `full_scene` traces only.
2. Render FORMAL06C, FORMAL08A, and Line9 with the same late-time crop, colour
   limits, and three views: raw, restrained time-power gain, and
   background-suppressed.
3. Review without label overlay first.
4. Continue only if FORMAL08A preserves a continuous multicycle basal packet,
   adds plausible continuous background, and does not create hyperbola-like
   fragments, regular slabs, or dropout.
5. Run a distributed 32-trace full scene only after the visual checkpoint.
6. Run the strict signed pair only after the 32-trace morphology pass.

No full gprMax runtime evidence exists at this stage.
