---
name: gprmax-package-audit
description: Audit PGDA-CSNet gprMax simulation packages for geometry, PML, outputs, merge completeness, QC deliverables, and training suitability.
---

# gprMax Package Audit

Use this skill when the user provides a gprMax package, `.in` files, simulation output folder, or asks whether a simulation bundle is usable.

## Safety rules

- Do not delete `.out`, `_merged.out`, logs, or zip packages.
- Do not start long GPU runs unless the user explicitly asks to run them.
- Prefer geometry dry-runs and file/QC inspection before full simulation.
- If a background simulation is already running, report status before attempting to stop or clean anything.

## Audit checklist

1. Inventory package structure:
   - `models/`, `inputs/`, `outputs_gprmax/`, `logs/`, `qc_previews/`, `labels/`, `scripts/`
   - `.in`, `.out`, `_merged.out`, `.csv`, `.png`, `.zip`, `.log`
2. Inspect `.in` headers:
   - `#domain`
   - `#dx_dy_dz`
   - `#time_window`
   - `#waveform`
   - source/receiver coordinates
   - `#src_steps` / `#rx_steps`
   - `#pml_cells`
3. Check 2D PML order:
   - gprMax expects `#pml_cells: x0 y0 z0 xmax ymax zmax`
   - for 2D models with one z cell, z PML must be zero:
     - valid: `20 20 0 20 20 0`
     - invalid: `20 20 20 20 0 0`
4. Run geometry dry-run when local gprMax is available:
   - use the project `SafeGprMaxRunner` pattern where possible
   - do not use conda/vcvars manually unless needed
5. Check output completeness:
   - every intended model has `_merged.out`
   - logs exist
   - QC metrics/images exist
6. Inspect HDF5 outputs when needed:
   - verify `rxs/rx1/Ez` or expected field component
   - verify `dt`
   - confirm shape and trace count
7. Generate or rerun QC only if cheap and requested:
   - background removal
   - AGC preview
   - early/deep RMS metrics
8. Classify package:
   - usable for training
   - usable only for diagnostic comparison
   - blocked by geometry/runtime/output issue

## Report format

Respond in Chinese with:

- `结论`
- `通过项`
- `阻断问题`
- `非阻断风险`
- `是否建议继续 GPU 全量运行`
- `推荐配置` when comparing variants
