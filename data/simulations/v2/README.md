# Simulation V2 Storage Contract

This tree separates immutable source decks from machine-local solver products.

- `00_controls/` contains versioned analytic and diagnostic source controls.
- `01_native_256_correlated_voxel_batch_v1/` contains the original native-256
  correlated-voxel source families.
- `02_native_256_domain_equivalence_v1/` contains exact cropped/shifted domain
  equivalence source decks.
- `03_native_256_80m_family_pilot_v1/` contains the current 80 m guard family
  source decks. These remain blocked from training until solved and reviewed.
- `01_solver_runs/` is the default machine-local, Git-ignored execution root
  selected by `environment/project_runtime.local.json`. It may contain staged
  inputs, per-trace contracts, merged `.out` files, and run state, but it is
  never an immutable source registry or a training release.
- `release_specs/` contains reviewed, portable instructions for extracting a
  minimal evidence package from a machine-local solver run.
- `02_released_solver_evidence/` contains immutable, checksum-verified evidence
  packages. Merged `.out` files in this directory use Git LFS. Entry is allowed
  only through `scripts/package_gprmax_release.py`.

Runtime plans, audits, previews, and cleanup evidence belong under `reports/`.
Geometry-only VTI files are removed after validation. Raw per-trace solver
outputs may be removed only after their trace contract is captured and the
official merge succeeds.

Do not commit `01_solver_runs/`, `_staging/`, `_superseded_*`, VTI views, or raw
solver outputs. Do not edit a source deck in place to resume a run; stage a new
run ID instead. A case may enter a training dataset only after numerical,
causal-control, visual/physical, provenance, and governance gates all pass.

The complete storage and two-workstation synchronization rules are defined in
`docs/SIMULATION_ASSET_POLICY.md`. A development evidence package preserves a
useful result but does not make it training eligible.

For distributed full-span pilots, scale continuity checks by the actual trace
stride. For true-negative cases, audit the full target-absent scene without
inventing a target path; solver validity and hard-negative semantic acceptance
are separate decisions.
