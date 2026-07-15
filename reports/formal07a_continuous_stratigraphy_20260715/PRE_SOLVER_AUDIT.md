# FORMAL07A Pre-Solver Audit

Date: 2026-07-15

## Decision

`PRE_SOLVER_GEOMETRY_ACCEPTED_FOR_STAGED_PAIR_ONLY`

FORMAL07A is a development-only successor to FORMAL06C. It is not a training
case and has no solver evidence yet. The generator reads no measured array,
but the selected mechanism was informed by held-out Line9 morphology, so
`line9_conditioned=true` and `formal_training_allowed=false` are permanent.

## Locked factors

- zero-mean 80 MHz Gaussian-modulated source;
- 0.03 m grid, 60-cell PML, and 179.73 m by 48.0 m domain;
- 256 traces at 0.09 m spacing with 0.18 m Tx/Rx offset;
- flat ground and fixed 8.01 m AGL;
- FORMAL06C cover, cap, and bedrock material values;
- exact shared indexed geometry for full/no-basal material remapping.

## Changed geology factor group

- acquisition-crop basal range reduced from 1.474 m to 0.480 m;
- basal absolute-slope P95 reduced from 0.152 to 0.038;
- continuous warped stratigraphy added at 2.35 m and 0.92 m vertical scales;
- broad and mesoscale two-dimensional property variation retained;
- no isolated inclusion, point target, or vertical material partition.

The source-referenced geometric arrival spans 410.76-421.71 ns, compared with
394.90-429.11 ns in FORMAL06C. This is a geometric reference, not a
visible-phase label.

## Static and physics gates

- `full_scene.in`: pass, zero errors and warnings;
- `no_basal_contrast_control.in`: pass, zero errors and warnings;
- `air_reference.in`: pass, zero errors and warnings;
- minimum-wavelength resolution at `2.8 * fc`: 12.37 cells;
- earliest lateral boundary round trip: 510.35 ns;
- protected analysis window: 0-500 ns;
- basal reflection proxy: -0.00880695, identical to FORMAL06C;
- conservative 14.2 m two-way material attenuation: about 32.46 dB at 80 MHz.

The attenuation estimate excludes spreading, antenna response, interfaces,
and dispersion. It is a plausibility gate, not a predicted B-scan amplitude.

## Visual assessment

The enhanced geometry preview confirms a continuous, gently warped layered
cover above a smooth cap/basal path. The morphology does not contain joined
hyperbolas by construction. The vertical material-neighbour change rate rises
from 0.047 in FORMAL06C to 0.383 in FORMAL07A, so the first solver checkpoint
must explicitly test whether the new background is useful structure or an
overly coherent horizontal comb.

## Next gate

1. one-trace strict full/no-basal causal smoke;
2. blind local full-scene morphology checkpoint;
3. distributed 32-trace full scene only after both pass;
4. matched distributed control only after morphology review;
5. no 256-trace solve and no label extraction before those decisions.
