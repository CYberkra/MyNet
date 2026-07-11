# Simulation V2 Control Cleanup (2026-07-11)

## Retained

- `data/PGDA_SYNTH_DATASET_V2/00_controls/`: audited, versioned source decks
  from `00_controls_gprMax_official_audited_20260711.zip`.
- `outputs/simulation_v2_controls/official_audited_20260711/`: reproducible
  solver execution copy and CTRL01 runtime evidence. It is intentionally
  ignored by Git because merged gprMax outputs are generated artifacts.
- `reports/simulation_v2_control_stage_20260711/`: import provenance, GPU
  smoke evidence, execution plans, and validation results.

## Removed After Hash Verification

- `data/PGDA_SYNTH_DATASET_V2/_staging/00_controls_gprMax_official_audited_20260711/`
  was an import-only duplicate of the retained canonical source package.
- `data/PGDA_SYNTH_DATASET_V2/_superseded_pre_official_audited_20260711/`
  was replaced by the audited source package.
- `data/PGDA_SYNTH_DATASET_V2/00_controls.zip` was byte-packaging-different
  but its extracted 104 files matched the superseded snapshot exactly by
  relative-path SHA256; it therefore contained no unique evidence.
- All tracked and untracked Python bytecode and pytest cache directories were
  removed. They are now covered by `.gitignore`.

## Lifecycle Rule

Pre-solver control labels are immutable source artifacts. Postprocessed labels
are generated only in an execution copy under `outputs/`. The static validator
records `pre_solver` or `postprocessed` explicitly and never treats a verified
visible-phase label as a failed pre-solver placeholder.

CTRL01 remains a reviewed control candidate only: `formal_training_allowed`
is `false` pending visual/physical review and explicit promotion.
