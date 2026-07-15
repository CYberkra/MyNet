# Repository Operations and Lifecycle

## Active Mainline

`master` is the active **AeroPath-SSD** research line. It owns the portable
runtime layer, V15 data contracts, independent V2 simulations, and the native
`501x256` release standard. It is the only branch intended for normal work on
either computer.

## Archived Research

The complete pre-cleanup mainline is retained as
`archive/pre-master-cleanup-20260715`. The older AeroPath transition is also
retained as `archive/pre-aeropath-master-20260713`.
Earlier audited snapshots remain under `archive/master-1818b25-20260711` and
the existing named archive branches. Historical GprMambaSep/A-S-G work stays in
the tree only as a frozen baseline or compatibility code; it must not be
presented as the paper's active physical-decomposition model.

## Data Tiers

| Tier | Location | Git policy |
|---|---|---|
| Contract and source specifications | `data/contracts/simulation_v2/` | Versioned |
| Source decks and pre-solver priors | `data/simulations/v2/01_native_256_release_pilot/` | Versioned, blocked from training |
| Compact audited canonical arrays | `data/simulations/v2/02_released_canonical/` | Versioned after explicit promotion |
| Raw solver products | `01_solver_runs/`, case directories | Local and ignored |
| Geometry views | `*.vti` | Disposable and ignored |
| Training/evaluation outputs | `outputs/` | Local and ignored |

## Daily Workflow

1. `git pull --ff-only origin master`.
2. Validate the local profile before training or simulation.
3. Generate or run only from committed source decks.
4. Promote completed simulations with `promote_native_256_solver_result.py`.
5. Commit source, manifest, compact canonical array, and release report in one
   focused change. Do not add raw solver outputs.
6. For a milestone, create and validate a handoff record according to
   `docs/HANDOFF_STANDARD.md`.
7. Push `master`; the other computer then pulls the same commit and starts from
   the handoff record's entry document and command.

## Cleanup Rule

Raw solver artifacts can be removed only after their compact promoted release
contains hashes, postprocess validation, trace contracts, and a human decision.
Keep source decks indefinitely. Never delete V15 labels or original measured
data as part of simulation cleanup.
