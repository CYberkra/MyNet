# PGDA-CSNet Current State (2026-07-22)

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
- FORMAL08A: solved Line9-realism background ablation built directly from
  FORMAL06C. It locks source, materials, grid, acquisition, basal path,
  transition, and protected interface-neighbour bins, and changes only
  depth-tapered continuous middle-cover texture. Eight consecutive traces and
  32 full-span traces completed cleanly. The 32-trace path correlation is
  0.99989 with seven signed lobes, 79.37 MHz peak frequency, and zero dropout.
  Target/adjacent-background RMS decreased from FORMAL06C's 17.29 to 14.77,
  but blind review found only a weak background increase and no material visual
  improvement over FORMAL06C. It is retained as a background ablation, not a
  successor; no matched control, visible-phase label, or training release is
  authorised. It is explicitly `line9_conditioned=true` and cannot support an
  unseen-Line9 claim.
- FORMAL08B: solved direct FORMAL06C stronger-deep-background ablation. It keeps
  the accepted source, materials, grid, acquisition, basal path, transition,
  and protected interface-neighbour bins exact while strengthening only the
  transition-following continuous deep-cover field. The full-domain candidate
  has predecessor latent correlation 0.84656, perturbation RMS 0.51900,
  changed cover-bin fraction 0.22021, and bin-delta P99 of 7. The blind geometry
  preview shows a materially stronger continuous material field without
  isolated bodies or vertical partitions. Eight consecutive and full-span 32
  full-scene traces completed without dropout. The 32-trace path correlation
  is 0.99995 with seven signed lobes and a 79.37 MHz peak, but the
  target/adjacent-background RMS increased from FORMAL06C's 17.29 to 21.33.
  Blind review shows almost no useful background improvement and a cleaner,
  more dominant target than desired. It is retained as a failed factor
  ablation; matched control and native 256 are stopped. It remains
  `line9_conditioned=true`, development-only, and blocked from training.
- No V2 scene family is training-approved yet.

The source registry is `data/simulations/v2/simulation_asset_registry.json`;
the training-governance view is
`data/contracts/dataset_v2/simulation_cases.csv`.

## Formal training gate

`configs/aeropath_ssd_v15_formal_blocked.json` remains disabled. V15 supports
the primary conditional path-picking task, but it contains no confirmed real
negative traces because the survey was designed to follow the basal interface.
This blocks measured rejection/no-pick claims rather than conditional path
picking. NULL/no-pick evidence is limited to approved controlled simulations
until an external measured rejection set exists.

The remaining formal dataset blocker is:

1. approved non-Line9-conditioned simulation families are absent.

The V15 release and the measured split are complete. They are not current
blockers. Run `scripts/validate_project_contracts.py --require-formal-ready`
before changing the formal config.

## Next work

1. Keep FORMAL06C as the sole project-owner accepted visual inheritance
   baseline. Family 02 and Family 03 remain ablations, not successors.
2. Use Line9 explicitly as the measured-realism calibration reference. Mark
   every resulting development case `line9_conditioned=true`; do not call its
   Line9 result a strict unseen holdout.
3. Keep FORMAL08A as a completed weak-background ablation. Do not spend solver
   time on its matched control or native-256 run because the full-span blind
   comparison did not improve on FORMAL06C.
4. Keep FORMAL08B as a completed failed stronger-deep-background ablation. Do
   not run its matched control or native 256: target dominance worsened and the
   full-span blind image did not improve on FORMAL06C.
5. Maintain a separate independent/formal generator track for strict holdout
   claims or use a held-out-line/leave-one-line-out evaluation contract.
6. Solve matched positive controls and promote only cases passing numerical,
   causal, visual, provenance, and human gates.
7. Do not manufacture real negatives from the existing interface-following
   lines. When a separate measured rejection survey is acquired, audit it as
   an external abstention-evaluation set; ambiguous/failed-positive regions
   remain weak or ignored.
8. Run official-Mamba2 CUDA and 501x256 VRAM smoke tests.
9. Pass the formal data gate, then run multi-seed AeroPath training and frozen
   baseline comparisons.

## Continuation contract

Milestone continuation follows `docs/HANDOFF_STANDARD.md`. Reviewed records are
stored under `reports/handoffs/<task_id>/` and reference the clean
implementation commit, real validation outcomes, artifacts, risks, and the
exact next-entry condition. Conversation history is background only.
