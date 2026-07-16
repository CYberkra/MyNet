# PGDA-CSNet Current State (2026-07-16)

## Active research line

**AeroPath-SSD** is the paper candidate. It combines an anisotropic radar stem,
per-trace acquisition conditioning, bidirectional axial sequence mixing, and a
structured interface-path objective with physical, NULL, start, and end
states. The official Mamba-2 path has an explicit `headdim` contract; the
formal config uses `headdim=16`.

GprMambaSep/Route-2 and the ConvNeXt curve model are frozen comparison
baselines. A/S/G decomposition is not established as a physical separation.

## Locked measured protocol

| Role | Lines |
|---|---|
| Train | LineL1, Line3, Line7 |
| Validation | Line6 |
| Test | Line9 |
| Review only | LineX1 |

The immutable measured release is `data/measured/yingshan_v15/`. It contains
six full lines, 78 exact windows, source archives, GNSS, ground elevation,
flight height, terrain features, V14 rollback fields, V15 labels, and explicit
ignore regions. Canonical arrays remain in acquisition order.

## Simulation state

- FORMAL06A: rejected because the interface response was overstrong.
- FORMAL06B: rejected because the tempered response remained too dominant.
- FORMAL06C: human-accepted development morphology with released evidence.
- FORMAL07A: causal/runtime checks pass, but the strict common-trace blind
  comparison rejects it as FORMAL06C's successor. Its cover is too strongly
  and regularly stratified, while its basal path is too flat and repetitive.
  It remains an unreleased regression ablation only.
- FORMAL07B: agent-accepted controlled development successor. It locks the
  FORMAL06C basal packet and adds only weak aperiodic 2D cover texture. The
  exact 32-trace comparison preserved seven signed lobes and 79.37 MHz peak
  frequency while reducing target/adjacent-background RMS from 17.29 to 16.74.
  It remains unreleased development evidence.
- Independent V2 Family 01: geometry/provenance-independent pilot. It passes
  the staged physics checks but its four-lobe 44 MHz response does not inherit
  FORMAL06C's accepted basal packet and remains somewhat easy.
- Independent V2 Family 02: a controlled mechanism-transfer development case.
  It regenerates Family 01 geometry from the same generic seeds, then applies
  FORMAL06C's 80 MHz zero-mean Gaussian-modulated source and weak constitutive
  mechanism. Blind 32-trace review recovered seven alternating lobes, a
  79.37 MHz peak, 0.445 envelope CV, and a continuous non-hyperbolic path.
  This proves mechanism transfer, but the mechanism-selection decision remains
  Line9-conditioned; Family 02 is development-only and prohibited from formal
  training.
- Independent V2 Family 03: the first source-independent hardware-band pilot.
  It keeps the Family 01 geometry and weak-interface materials but replaces the
  Line9-conditioned 80 MHz source choice with a project-wide 100 MHz,
  20-170 MHz amplitude-only pulse proxy. Static geometry, the one-trace causal
  pair, exact negative equivalence, and distributed32 full-scene morphology
  passed. The path/geometry correlation is 0.9983 with seven signed lobes and
  no dropout. Blind review found a narrower, sharper, less visible packet than
  Family 02/FORMAL06C; the solved spectral centroid rose to 116.20 MHz and the
  target/adjacent-background RMS ratio fell to 9.65. Family 03 is retained as
  an independent valid diversity candidate, not the preferred morphology and
  not training-approved before a full causal release and source-family audit.
- No V2 scene family is training-approved yet.

The source registry is `data/simulations/v2/simulation_asset_registry.json`;
the training-governance view is
`data/contracts/dataset_v2/simulation_cases.csv`.

## Formal training gate

`configs/aeropath_ssd_v15_formal_blocked.json` remains disabled. Only two
dataset blockers remain:

1. confirmed real true-negative windows are absent;
2. approved non-Line9-conditioned simulation families are absent.

The V15 release and the measured split are complete. They are not current
blockers. Run `scripts/validate_project_contracts.py --require-formal-ready`
before changing the formal config.

## Next work

1. Keep FORMAL06C as the project-owner accepted visual baseline and FORMAL07B
   as its controlled weak-background successor. Neither is training data.
2. Use Family 02 to lock the successful physical mechanism. Use Family 03 as
   the independent hardware-band anchor, then run a predeclared same-geometry,
   same-material source-family ablation. Do not tune source parameters against
   Line9 and do not promote Family 02 itself.
3. Solve matched positive controls and promote only cases passing numerical,
   causal, visual, provenance, and human gates.
4. Audit candidate real true-negative intervals; ambiguous/failed-positive
   regions remain weak or ignored.
5. Run official-Mamba2 CUDA and 501x256 VRAM smoke tests.
6. Pass the formal data gate, then run multi-seed AeroPath training and frozen
   baseline comparisons.

## Continuation contract

Milestone continuation follows `docs/HANDOFF_STANDARD.md`. Reviewed records are
stored under `reports/handoffs/<task_id>/` and reference the clean
implementation commit, real validation outcomes, artifacts, risks, and the
exact next-entry condition. Conversation history is background only.
