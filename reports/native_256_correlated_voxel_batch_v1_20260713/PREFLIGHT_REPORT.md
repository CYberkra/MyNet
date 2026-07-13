# Native 256 Correlated-Voxel Batch V1 Preflight

Date: 2026-07-13

Status: source decks accepted for a representative solver preflight; formal
training remains blocked.

## Design

- Canonical acquisition: 501 time samples, 256 traces, 0.09 m spacing.
- Grid: 0.0225 m; domain: 243.135 x 36 m; 110.0025 m lateral margins.
- Source: 55 MHz Ricker, z-polarized ideal line source, 0.18 m Tx/Rx offset.
- Geology: independent full-domain long-, meso-, and local-scale correlated
  fields, sampled by an unscaled 22.95 m native scan window.
- Positive contract: shared HDF5 geometry and upper materials; only transition
  and bedrock constitutive mappings differ in the no-basal control.
- Negative contract: no transition or bedrock region indices are present.

## Morphology

| Case | Target | Depth range | Extrema | Quadratic R2 |
|---|---:|---:|---:|---:|
| CV01 balanced | yes | 0.805 m | 4 | 0.741 |
| CV02 low contrast | yes | 1.648 m | 4 | 0.904 |
| CV03 patchy | yes | 0.849 m | 3 | 0.374 |
| CV04 upper clutter | no | n/a | n/a | n/a |

No positive is dominated by a single quadratic bowl. Finite clutter lenses
taper to zero thickness and do not introduce vertical material walls.

## Verification

- Generator and runner compile: passed.
- Targeted tests: 13 passed.
- gprMax source-aware static audits: 11/11 passed, zero errors, zero warnings.
- Minimum wavelength sampling: 22.64-22.80 cells at the 2.8 x source-frequency
  check used by the maintained audit skill.
- Repository and installed gprMax skills: both valid after maintenance update.

## Remaining Gates

1. Run a 32-trace stride-8 distributed full/control/air preflight for CV01.
2. Audit signed target visibility, continuity, edge energy, and full/control
   alignment from solver output.
3. Run complete 256 traces only after the representative preflight passes.
4. Keep every case blocked until independent visual review and promotion.
