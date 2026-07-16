# Simulation Lineage Recalibration

Date: 2026-07-16

## Corrected Decision

The project-owner visual ranking is:

```text
FORMAL06C > Independent Family 02 > Independent Family 03
```

Family 02 and Family 03 remain useful controlled ablations, but neither is the
preferred realism successor. The unrun Family 04 nominal-band sweep was
withdrawn.

## Why Family 02 Looks Worse

Family 02 preserves the important FORMAL06C mechanism but not the accepted
scene morphology.

| Factor | FORMAL06C | Family 02 | Consequence |
|---|---:|---:|---|
| Source | 80 MHz zero-mean Gaussian-modulated | numerically identical over the shared window | Not the cause |
| Material deck | cover 12.0-12.8, cap 13.0, bedrock 12.55 | same values to rounding precision | Not the cause |
| Basal depth median | 13.273 m | 14.469 m | Later packet; not itself a realism failure |
| Basal depth range | 1.474 m | 1.225 m | Slightly less vertical range |
| Smoothed extrema | 2 | 7 | More local turns and segmented appearance |
| Quadratic fit R2 | 0.975 | 0.872 | Less simple broad relief |
| Transition median | 0.909 m | 1.340 m | Broader transition family |
| Horizontal neighbour-bin change | 0.0215 | 0.0594 | 2.75 times more change-prone |
| Vertical neighbour-bin change | 0.0475 | 0.1074 | 2.26 times more change-prone |
| Local cover component | none below the meso design | 0.75 x 0.45 m component | Finer spatial fragmentation |
| Domain / solver window | 179.73 m / 650 ns | 242.73 m / 750 ns | Higher cost; not the main morphology cause |

The blind image confirms the numerical diagnosis: Family 02 keeps the
multi-cycle wavelet but loses FORMAL06C's gently varying, visually unified
basal band. It is more piecewise and remains too isolated from the measured
background.

## Why Family 03 Looks Worse Again

Family 03 keeps Family 02 geometry but changes both the source and weak
constitutive range. The amplitude-only 100 MHz zero-phase band proxy produced a
narrower, sharper packet and moved the solved spectral centroid to about
116.20 MHz. Its target/adjacent-background ratio fell to about 9.65, but that
numeric change did not improve the project-owner visual judgement. It is a
source-basis diagnostic, not a successor.

## Correct Use Of Line9

Line9 is now an explicit measured-realism calibration reference. It may guide:

- signed-lobe count and packet thickness;
- peak frequency and spectral centroid;
- lateral continuity and amplitude modulation;
- target prominence after identical background suppression and gain;
- continuous non-target texture and absence of joined-hyperbola morphology.

Every simulator selected this way must be marked `line9_conditioned=true`.
This prevents only the claim that Line9 is a completely unseen strict holdout;
it does not make the simulator unusable. A strict experiment must use a
separate independently selected generator, a different held-out line, or a
leave-one-line-out simulator calibration protocol.

## Next Candidate Contract

The next runtime candidate must inherit FORMAL06C directly and lock:

- the 80 MHz zero-mean Gaussian-modulated source;
- cover/cap/bedrock material endpoints;
- grid, PML, acquisition, and flight height;
- basal and transition profiles;
- full/no-basal causal-control construction.

Only continuous non-target geology may change first. It must add multiscale,
aperiodic background energy without isolated bodies, point targets, vertical
partitions, or regular horizontal combs. It will be compared blindly against
FORMAL06C and Line9 at identical physical spans, time windows, gain, and colour
scales before any long run or source change.
