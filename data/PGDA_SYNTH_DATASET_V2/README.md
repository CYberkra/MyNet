# Simulation V2 Storage Contract

`00_controls/` contains versioned, audited gprMax input controls and their
immutable pre-solver labels. These are source artifacts, not training data.

Solver execution copies, merged `.out` files, VTK geometry, and postprocessed
arrays belong under `outputs/simulation_v2_controls/<run_id>/`. Runtime plans,
validation reports, and cleanup manifests belong under
`reports/simulation_v2_control_stage_YYYYMMDD/`.

Do not restore `01_solver_runs/`, `_staging/`, or `_superseded_*` beneath this
directory. They are temporary lifecycle locations and are intentionally
ignored by Git. A control may only be promoted after postprocess validation,
visual/physical review, and an explicit governance decision.
