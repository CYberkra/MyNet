# PGDA-CSNet Current State (2026-07-15)

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
- FORMAL07A: pre-solver successor with gentler basal relief and continuous
  multiscale stratigraphy; static and attenuation gates pass, but no solver
  evidence exists yet.
- All four are Line9-conditioned and prohibited from formal training.
- No independent V2 scene family is training-approved yet.

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

1. Run the staged FORMAL07A development gates to test whether continuous
   stratigraphy corrects FORMAL06C's overly clean non-target background.
2. Design independent V2 positive and true-negative scene families without
   using Line9 labels, geometry, timing, or morphology as a generator prior.
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
