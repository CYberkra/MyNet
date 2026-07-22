# Contributing and Merge Rules

`master` is the protected project record. It is the branch both workstations
pull, but it is not a scratch branch.

## Change flow

1. Start from `master` and create `agent/<purpose>-YYYYMMDD` for every
   non-trivial task.
2. Keep one task, one focused change, and one evidence trail per branch.
3. Do not edit immutable measured releases in place. Create a versioned release
   and update its manifest instead.
4. Do not commit local runtime profiles, raw solver products, VTI views, model
   outputs, scratch files, or machine-specific paths.
5. Open a pull request into `master`. Merge only after the CPU CI is green and
   the required local validation has been recorded.

## Required local checks

Run from the repository root using the intended project Python environment:

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check pgdacsnet scripts tests
python scripts\check_configs.py
python -m pytest -q `
  tests\test_p0_training_guards.py `
  tests\test_aeropath_ssd.py `
  tests\test_experiment_contract.py `
  tests\test_runtime_portability.py `
  tests\test_handoff_record.py `
  tests\test_hardware_measurement_contract.py
```

Run `python scripts\validate_project_contracts.py` before a release or a
cross-computer handoff. `--require-formal-ready` is reserved for the formal
training release gate and is expected to fail until the manifest authorizes it.

## GPU and physics changes

For official Mamba2, run the self-hosted `official-mamba2-cuda` workflow and
record the result. For gprMax source-deck or release changes, follow
`docs/SIMULATION_ASSET_POLICY.md`, the gprMax audit skill, and the applicable
hardware/physics contracts. CPU CI does not validate either class of result.

## Environment reproducibility

`requirements.txt` declares compatible project dependencies; it is not a paper
environment lock. Before any formal experiment, capture the exact environment
from the selected CPU or CUDA machine and commit the resulting lock artifact
with the experiment evidence. See `environment/README.md`.
