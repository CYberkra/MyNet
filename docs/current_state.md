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
- Independent V2 Family 02: a controlled mechanism-transfer ablation.
  It regenerates Family 01 geometry from the same generic seeds, then applies
  FORMAL06C's 80 MHz zero-mean Gaussian-modulated source and weak constitutive
  mechanism. It recovers seven alternating lobes and a 79.37 MHz peak, but
  project-owner visual review ranks it below FORMAL06C. Its independently
  regenerated basal relief is more segmented and the cover field is over twice
  as change-prone in both axes. It proves that the wavelet/material mechanism
  transfers, not that the resulting scene is the preferred realism successor.
- Independent V2 Family 03: the first source-independent hardware-band pilot.
  It keeps the Family 01 geometry and weak-interface materials but replaces the
  Line9-conditioned 80 MHz source choice with a project-wide 100 MHz,
  20-170 MHz amplitude-only pulse proxy. Static geometry, the one-trace causal
  pair, exact negative equivalence, and distributed32 full-scene morphology
  passed. The path/geometry correlation is 0.9983 with seven signed lobes and
  no dropout. Blind review found a narrower, sharper, less visible packet than
  Family 02/FORMAL06C; the solved spectral centroid rose to 116.20 MHz and the
  target/adjacent-background RMS ratio fell to 9.65. Project-owner visual
  ranking is `FORMAL06C > Family 02 > Family 03`. Family 03 is retained only as
  a source-basis ablation and must not drive the immediate successor lineage.
- FORMAL08A: pre-solver Line9-realism candidate built directly from FORMAL06C.
  It locks source, materials, grid, acquisition, basal path, transition, and
  protected interface-neighbour bins, and changes only depth-tapered continuous
  middle-cover texture. Static, attenuation, source-fingerprint, and geometry
  gates pass. Runtime has not started; the next gate is eight consecutive
  full-scene traces reviewed against FORMAL06C and Line9. It is explicitly
  `line9_conditioned=true` and cannot support an unseen-Line9 claim.
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

1. Keep FORMAL06C as the sole project-owner accepted visual inheritance
   baseline. Family 02 and Family 03 remain ablations, not successors.
2. Use Line9 explicitly as the measured-realism calibration reference. Mark
   every resulting development case `line9_conditioned=true`; do not call its
   Line9 result a strict unseen holdout.
3. Review FORMAL08A's pre-solver geometry, then run only its eight-consecutive-
   trace full-scene checkpoint. Do not start a distributed solve until the
   project-owner visual gate passes.
4. Maintain a separate independent/formal generator track for strict holdout
   claims or use a held-out-line/leave-one-line-out evaluation contract.
5. Solve matched positive controls and promote only cases passing numerical,
   causal, visual, provenance, and human gates.
6. Audit candidate real true-negative intervals; ambiguous/failed-positive
   regions remain weak or ignored.
7. Run official-Mamba2 CUDA and 501x256 VRAM smoke tests.
8. Pass the formal data gate, then run multi-seed AeroPath training and frozen
   baseline comparisons.

## Continuation contract

Milestone continuation follows `docs/HANDOFF_STANDARD.md`. Reviewed records are
stored under `reports/handoffs/<task_id>/` and reference the clean
implementation commit, real validation outcomes, artifacts, risks, and the
exact next-entry condition. Conversation history is background only.
