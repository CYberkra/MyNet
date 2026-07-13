# PGDA-CSNet Current State (2026-07-11)

## Research Direction

The active research line is **AeroPath-SSD**, an acquisition-conditioned
structured interface-path model. It supersedes A/S/G decomposition as the paper
candidate. Legacy GprMambaSep remains historical/development code only; its
components are not established physical decompositions.

## What Is Implemented

- An anisotropic time/trace stem, not a claimed phase/IQ encoder.
- Per-trace FiLM conditioning from terrain/acquisition metadata.
- Explicit bidirectional axial sequence mixers for AeroPath.
- Official Mamba-2 `headdim` contract; the formal default is `headdim=16`.
- Soft forward/backward path inference with physical, NULL, path-start, and
  path-end states. Its marginals are protected by a brute-force exact test.
- GNSS chainage-aware slope transitions, trace-state supervision, and safe
  weak/negative/ignore handling for no-pick losses.
- Semantically separate mask, unary curve, structured-path, uncertainty, and
  no-pick evaluation outputs.

## Formal Protocol (Locked, Disabled)

`configs/aeropath_ssd_v15_formal_blocked.json` encodes:

| Role | Lines |
|---|---|
| Train | LineL1, Line3, Line7 |
| Validation | Line6 |
| Test | Line9 |
| Review only | LineX1 |

It uses `501x256`, `official_mamba2`, structured loss, and true bidirectional
axial blocks. It must remain disabled until the V2 data-release gate passes.

## Data Status

- YingShan V15 labels are the current label release; high-risk crossing regions
  are explicitly weakened or ignored.
- Line9 remains test-only. Existing V1 legacy simulations are Line9-conditioned
  and development-only; none may enter formal training.
- Real confirmed negative windows and approved independent V2 simulation families
  are still absent. This is the current formal-training blocker.

## Next Work

1. Stage and preflight the native `N256_F01_GENTLE_DEEP_MODERATE_POS` source
   deck: static input audit, geometry-only check, then 1/32/64-trace smoke
   runs before any full GPU execution.
2. Audit the required `full_scene`, `no_basal_contrast_control`, and
   `air_reference` outputs; only then progress through `N256_F02-F04` and
   `N256_N01-N02`.
3. Verify visible-phase extraction and promote only audited independent scene
   families. Source decks remain read-only; raw solver products live only in
   the ignored local solver-run root.
4. Create confirmed real negative windows and pass the V2 data gate.
5. Run CUDA/VRAM validation for official Mamba-2 at 501x256.
6. Only then run the locked multi-seed AeroPath formal protocol and compare it
   with the frozen ConvNeXt curve and Route-2 baselines.
