# Native Simulation Sync And Retention Policy

This policy makes two Windows workstations interchangeable without turning the
Git repository into an unbounded raw FDTD archive.

## What is versioned in Git

1. Simulation standards, material/case catalogs, generators, validators, and
   the repository-local gprMax skill.
2. Native pre-solver decks under
   `data/simulations/v2/01_native_256_release_pilot/`. These contain
   inputs, geometry includes, labels that are explicitly pending, manifests,
   hashes, and small previews.
3. Solver-validated canonical releases under
   `data/simulations/v2/02_released_canonical/<case_id>/`. A release
   contains the `501x256` canonical arrays, postprocess report, pre-merge
   trace contracts, source manifest, and a release hash manifest.

The release copy is still `formal_training_allowed=false`. Git publication
means another workstation can inspect, train-development code against, and
continue audit work; it is not paper-data promotion.

## What is never versioned

- Individual gprMax trace `.out` files and merged `.out` files.
- Raw HDF5 solver intermediates.
- `.vti` geometry views.
- transient logs, GPU scratch data, and partial solver runs.

Those objects are reproducible evidence during an active run but are too large
for normal repository history. The runner must capture trace metadata before a
merge deletes individual trace outputs. Keep raw files locally until the
canonical release is validated and copied; then remove them according to local
storage needs. `.vti` is a disposable geometry-inspection artifact and should
be removed immediately after successful geometry review.

## Two-workstation handoff

1. On either workstation, `git pull --ff-only` before generating or running.
2. Generate a case package from the committed standard. Do not modify the case
   catalog in place during a solver run.
3. Run full/control/air, capture all trace contracts before merge, then run
   postprocess and static validation.
4. Promote the compact validated result with
   `scripts/promote_native_256_solver_result.py`; commit only the release
   directory and associated manifest changes.
5. Push the commit. The second workstation uses `git pull --ff-only` and has
   the exact canonical arrays and provenance JSON without downloading raw FDTD
   outputs.

Never use `git add -f` to publish `.out`, `.h5`, or `.vti` files. A large raw
run that must be retained for an external reviewer belongs in a separately
hashed archive, not the repository's ordinary history.
