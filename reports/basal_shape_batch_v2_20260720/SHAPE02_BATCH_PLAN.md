# SHAPE02 basal-interface batch plan

Date: 2026-07-20

## Decision

The next batch isolates basal-interface morphology. Flat ground, fixed flight
height, source, acquisition, materials, transition thickness, and cover field
remain identical across every case. Terrain, height variation, noise, material
diversity, and extra clutter are deferred.

This is the right order because otherwise a visually different B-scan cannot
be attributed to interface shape. Acquisition noise can later be added without
rerunning FDTD, while terrain and physical height variation require separate
paired solver branches.

## What the batch changes

The bank contains two calibration cases and ten candidate/stress morphologies:
flat, gentle dip, broad bedrock high, broad trough, asymmetric flexure, double
relief, two aperiodic multiscale levels, a smooth terrace-ramp, a broad incised
low, a distributed fault-flexure, and a compound shoulder.

All shapes are continuous, non-periodic, and bounded. Sharp vertical offsets,
V-shaped notches, isolated bodies, and repeating sine waves are forbidden
because they tend to create hyperbola chains or constructed-looking packets.

## Why SHAPE01 is not reused

- The contiguous 32-trace run covered only 2.79 m, not the full 22.95 m scan.
- `BROAD_RISE` was actually a centre-deepening trough.
- Transition thickness changed between shape cases, so morphology was not the
  only variable.
- The resulting output is useful diagnostic evidence but not batch data.

## Execution order

1. Generate all full-domain shapes and inspect full-aperture geometry previews.
2. Run one central full/control trace for every geometry-passing candidate.
3. Run a full-span sparse pair at native indices `0,8,...,248`. These 32 traces
   sample the whole 22.95 m aperture at 0.72 m spacing; they must not be
   interpolated or described as a native B-scan.
4. Select four to six winners by causal attribution, path continuity, shape
   readability, and absence of constructed artifacts.
5. Run those winners at native `256 x 0.09 m` spacing and the release grid
   `dl=0.0225 m`, then export `501 x 256` arrays.
6. Expand accepted morphology ranges to at least 24 independent scene
   families before introducing the next factor group.

## Acceptance gates

- The exact same non-shape hashes must appear in every case.
- Full/control inputs may differ only in basal contrast.
- The signed `full - control` response must be localized near the geometric
  arrival and remain laterally continuous.
- Candidate relief is normally 0.45-2.2 m with P95 absolute slope <= 0.20.
- Candidate shapes must not collapse into a single near-quadratic bowl or arch;
  analytic calibration cases are exempt.
- No output becomes trainable without a native-256 pair, visible-phase audit,
  provenance hashes, and explicit human promotion.

## Deferred factors

After shape families stabilize, add factors one group at a time:

1. material/transition diversity;
2. gentle terrain;
3. physically modelled flight-height variation;
4. finite non-target geology/clutter;
5. measurement response and noise.

Noise is last because it should not be used to hide a weak or incorrect basal
mechanism. Terrain and height come later because they alter travel time and
must be tested with exact full/control geometry pairs.

The machine-readable contract is
`data/contracts/simulation_v2/basal_shape_batch_v2.json`.

