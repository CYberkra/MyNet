# Agent Entry Point

This repository's active research line is AeroPath-SSD. Read these files before
changing code, data, simulations, or experiment status:

1. `docs/PROJECT_STANDARD.md`
2. `docs/current_state.md`
3. `docs/REPOSITORY_LAYOUT.md`
4. `docs/HANDOFF_STANDARD.md`
5. `docs/AEROPATH_SSD.md`
6. `docs/SIMULATION_ASSET_POLICY.md`
7. `.claude/skills/gprmax-physics-audit/SKILL.md` for gprMax work

## Non-negotiable contracts

- Canonical measured arrays stay in CSV acquisition order. Profile reversal is
  display-only.
- Formal split: LineL1/Line3/Line7 train, Line6 validation, Line9 test,
  LineX1 review-only.
- V15 labels are final, but formal training is blocked by missing true
  negatives and missing approved independent simulations.
- FORMAL06C is Line9-conditioned development evidence, not formal training
  data.
- A positive simulation requires matched `full_scene`,
  `no_basal_contrast_control`, and `air_reference` evidence.
- Failed positives and ambiguous traces are never relabeled as true negatives.
- Machine paths belong only in the ignored
  `environment/project_runtime.local.json`.
- Raw solver caches, VTI files, ordinary outputs, and Python environments are
  not committed.
- Do not claim GprMambaSep A/S/G branches are physically identified.
- A completed milestone must reference a clean implementation commit and pass
  the final handoff-record validator. Chat is not an authoritative project
  state record.

## Required checks

```powershell
python scripts\check_configs.py
python scripts\check_dataset.py --data-root data\measured\yingshan_v15
python scripts\validate_yingshan_v15_final.py
python scripts\validate_project_contracts.py
python scripts\handoff_record.py --help
python -m pytest -q tests
```

The formal gate is expected to fail until the dataset manifest explicitly
permits training:

```powershell
python scripts\validate_project_contracts.py --require-formal-ready
```

For a completed simulation report, provide geometry/model evidence, raw
B-scan, processed B-scan, the matched-control audit, and provenance hashes.
