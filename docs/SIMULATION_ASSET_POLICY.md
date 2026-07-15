# Simulation Asset and Two-Workstation Sync Policy

This is the normative storage contract for gprMax work in MyNet.

## Five asset layers

| Layer | Location | Versioned | Purpose |
|---|---|---:|---|
| Source | `data/simulations/v2/00_controls/` | Git | Decks, indexed geometry, labels, manifests, checksums, and pre-solver previews |
| Release specification | `data/simulations/v2/release_specs/` | Git | Exact whitelist for building an immutable evidence package |
| Released evidence | `data/simulations/v2/02_released_solver_evidence/` | Git + LFS | Selected merged solver outputs, trace contracts, audits, and human-review evidence |
| Training release | `data/simulations/v2/02_released_canonical/` | Git + LFS where configured | Canonical 501 x N arrays admitted by the training-data contract |
| Runtime cache | `data/simulations/v2/01_solver_runs/` | Never | Staged decks, per-trace outputs, logs, resume state, and temporary merge products |

## Release classes

- `source_only`: reproducible model definition; no solver output is retained.
- `rejected_evidence`: compact report and preview only; no solver output is retained.
- `development_evidence`: a human-accepted development baseline. It may include
  a one-trace strict pair and one selected merged full-scene run. It is not a
  training release and may be conditioned by a held-out measured-line audit.
- `training_candidate`: complete required full/control/air runs, labels, and
  automated audits, still blocked until governance approval.
- `formal_release`: immutable canonical arrays whose manifest passes all data,
  holdout, provenance, and label-semantic gates.

Promotion is monotonic. Human visual acceptance can promote a case to
`development_evidence`; it cannot skip causal, holdout, or training gates.

## Files that may be committed

- deterministic source decks and generators;
- source manifests, geometry hashes, material tables, and compact labels;
- merged `.out` selected by a reviewed release spec;
- trace contracts captured before per-trace deletion;
- audit JSON/CSV, decision reports, and representative PNG previews;
- release manifests and SHA256 lists.

## Files that must remain local

- `*.vti` geometry views after geometry validation;
- runner stdout/stderr and transient logs;
- `full_scene1.out`, `full_scene2.out`, and other unmerged per-trace outputs;
- duplicate staged inputs copied into each solver run;
- PID/watcher state, caches, and temporary archives;
- failed solver products unless a small report is required to explain the failure.

Raw trace files may be deleted only after the trace contract is complete and
the official merge succeeds. Never delete the only copy of an unmerged run.

## Packaging workflow

1. Generate and statically audit a source case under `00_controls/`.
2. Solve under the machine-local `01_solver_runs/` tree.
3. Complete numerical, causal, morphology, and human review gates.
4. Add a release spec under `release_specs/`; list every retained artifact and
   its semantic role. Absolute machine paths are prohibited.
5. Run:

```powershell
python scripts/package_gprmax_release.py data/simulations/v2/release_specs/<spec>.json
python scripts/package_gprmax_release.py data/simulations/v2/release_specs/<spec>.json --verify-only
```

6. Confirm released `.out` files are LFS objects before committing:

```powershell
git check-attr filter -- data/simulations/v2/02_released_solver_evidence/**/*.out
git lfs status
```

7. Commit source/model changes separately from released solver evidence.

## Two-workstation workflow

Each workstation keeps its own ignored
`environment/project_runtime.local.json`. Executable and CUDA paths never enter
Git. After the first clone:

```powershell
git lfs install
git pull --ff-only
git lfs pull
python scripts/package_gprmax_release.py <spec> --verify-only
```

Before switching computers, finish or safely stop active solver jobs, package
accepted evidence, commit, and push. The receiving computer pulls Git and LFS,
then verifies the release manifest. Never synchronize `01_solver_runs/` through
Git, cloud-drive mirroring, or manual folder merging.

## VTI geometry-view lifecycle

VTI files are visualization exports from `#geometry_view`; they are not used
by the FDTD update, merged receiver output, label extraction, or network
training. They are therefore prohibited from Git, Git LFS, release packages,
and long-lived solver caches.

Use a geometry view once when a case introduces a new domain, PML layout,
source/receiver placement, object topology, or voxel-index geometry. The
project runner hashes each generated VTI, records its name and size in
`run_logs/geometry_view_cleanup.json`, and then deletes it. This retained JSON
is the audit evidence.

A repeated run may skip VTI generation when the locked geometry/index-array
SHA256, domain, PML, and acquisition coordinates are unchanged and only
material values, waveform parameters, or trace count changed. Production
`full_scene.in`, `no_basal_contrast_control.in`, and `air_reference.in` files
must not contain `#geometry_view`. Keep a VTI locally only while investigating
a concrete geometry dispute; delete it after the decision is recorded.

Audit and clear legacy views with:

```powershell
python scripts/cleanup_gprmax_geometry_views.py `
  --report reports/gprmax_vti_cleanup_YYYYMMDD.json --delete
```

## FORMAL06 decision

- FORMAL06A/B: retain source definitions and rejection audits only.
- FORMAL06C: retain source definition plus
  `development_evidence_v1`; project-owner visual review accepted its continuous
  multi-cycle basal morphology on 2026-07-15.
- FORMAL06C remains blocked from training until a complete distributed matched
  pair and the independent simulation-data contract pass.
