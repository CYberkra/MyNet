---
name: gprmax-physics-audit
description: Design, inspect, run, and maintain physically auditable gprMax FDTD models. Use for gprMax .in/.inc/HDF5 geometry, B-scan simulation design, source/grid/PML checks, heterogeneous media, full/no-target controls, GPU execution, provenance, or diagnosing unrealistic simulated GPR data.
---

# gprMax Physics Audit

Build simulations whose geometry, physics, controls, and provenance can survive a paper audit. Treat the installed source as the executable contract and the official manual as the conceptual contract.

## Start Here

1. Locate the exact gprMax package used by the run and record its version plus a source-tree hash.
2. Read `references/source-and-manual-contract.md` before changing input commands or material models.
3. Use `references/execution-flow.md` when debugging build, stepping, GPU, or output behavior.
4. For MyNet/PGDA work, also read `references/mynet-simulation-contract.md`.
5. Run `scripts/audit_gprmax_input.py MODEL.in --json REPORT.json` before geometry-only or GPU execution.
6. For deep conductive media, run `scripts/attenuation_budget.py` before the paired smoke.
7. Fix every error. Record justified warnings in the case manifest.

## Evidence Precedence

Use this order when sources disagree:

1. Installed source code that will execute the model.
2. Matching-version local documentation and tests.
3. Current official gprMax documentation.
4. Papers and domain references.
5. Project conventions and visual intuition.

Never assume an online command signature matches an older local installation. Record the discrepancy.

## Design Workflow

### 1. Define the Measurement Contract

Lock before geometry work:

- dimensionality and polarization;
- antenna model, center frequency, Tx/Rx offset, trace spacing, and trace count;
- solver and canonical time windows;
- acquisition span, domain margins, and coordinate convention;
- target definition and label semantics;
- formal split restrictions and forbidden conditioning sources.

Mark unknown hardware values as provisional. Do not hide them in defaults.

### 2. Pass the Physics Gate

Check all of the following:

- Grid step resolves the smallest relevant wavelength by at least the official `lambda_min / 10` rule.
- Every source, receiver, and target is at least 15 cells from the PML; include at least 15-20 air cells above the source.
- PML thickness is inside the declared domain.
- The time window captures the event and excludes unexplained late boundary returns.
- For long airborne scans, size each lateral physical guard from `2 * guard / c` and the protected target/search window; do not choose a domain from a scan-to-domain ratio alone. Validate a reduced domain with an exact cropped/shifted equivalence case.
- Geometry coordinates round to intended FDTD cells.
- Material values are frequency-appropriate and traceable.
- Dispersion, conductivity, and relaxation times are compatible with the time step.
- The target-depth two-way attenuation budget leaves a measurable response; estimate it before spending a long GPU run.

Do not use `#soil_peplinski` outside its documented 0.3-1.3 GHz validity range without a separate scientific justification. At tens of MHz, prefer measured/bounded custom materials or another frequency-valid constitutive model.

### 3. Choose Geometry Deliberately

- Use normal gprMax objects for simple analytic controls.
- Use seeded `#fractal_box`/rough surfaces only where their material-model and frequency contracts are valid.
- Use `#geometry_objects_read` for deterministic indexed heterogeneity or paired geometry. Validate HDF5 dtype, `dx_dy_dz`, dimensions, material indices, and local command signature.
- Remember that object commands are a layered canvas: later objects overwrite earlier ones.
- External voxel geometry is not dielectric-smoothed by gprMax 3.1.7. Smooth the spatial field physically before quantization and keep cells comfortably below the relevant wavelength.

Avoid abrupt vertical material partitions unless they are actual geology. They create coherent walls and crossing artifacts.

### 4. Build Strict Paired Controls

For `full` versus `no-target/no-basal`:

- share domain, grid, PML, source, receiver, stepping, time window, geometry indices, stochastic seeds, and upper-medium materials;
- change only the causal target contrast;
- hash the common geometry and both material maps;
- confirm identical trace count and sample timing;
- derive the target response from a signed pair difference before envelope or visible-phase extraction.

Prefer one shared geometry HDF5 with two audited material maps when the control changes constitutive contrast only. An air reference is useful for decomposition but is not a substitute for the strict paired control.

### 5. Validate Before Expensive Runs

Run, in order:

1. static input audit;
2. geometry-only build;
3. geometry-view or indexed-geometry preview;
4. one-trace CPU/GPU smoke;
5. short B-scan subset;
6. full paired run.

Inspect the geometry around all PML boundaries, source/receiver positions, imported-array extents, interfaces, and scan endpoints.

For a long B-scan, add an early checkpoint after roughly 10-25% of traces. Render the physically relevant late-time crop with fixed gain and compare target-band RMS against adjacent background. Stop and archive the attempt when the target is already below the declared detectability gate; do not let sunk GPU time turn a failed material budget into an accepted dataset.

### 6. Audit Outputs

Require:

- expected trace count and samples per trace;
- finite arrays and consistent time axes;
- full/control alignment before subtraction;
- a complete per-trace capture report before any merge that removes source files;
- target-support continuity, amplitude variation, and background/target energy ratio;
- comparison at identical gain, crop, distance axis, and color scale;
- provenance hashes, commands, environment, GPU, and completion state.

Visual similarity is supporting evidence, not a physics proof. A visually strong interface can still be leakage, a boundary artifact, or an unrealistically easy target.

Prefer a synchronous capture-and-validate step after gprMax returns and before the merge command. A resumable background watcher is useful for live progress and early-stop evidence, but it must not be the only barrier protecting per-trace provenance.

## Maintenance

At the start of any substantial gprMax task:

1. compare the installed version and source hash with `references/version-baseline.md`;
2. check official documentation for commands being changed;
3. update the baseline date, differences, and affected guidance;
4. run the skill validator and one representative static audit;
5. keep changes factual and sourced; do not silently rewrite earlier constraints.

Generate a fresh machine-readable fingerprint with `scripts/fingerprint_gprmax.py GPRMAX_ROOT --json fingerprint.json`.

Update this skill whenever gprMax is upgraded, a source/manual discrepancy is found, a simulation failure reveals a missing guard, or the project measurement contract changes.

## References

- `references/source-and-manual-contract.md`: official rules and installed-source behavior.
- `references/execution-flow.md`: reviewed 3.1.7 build, stepping, solve, and output call chain.
- `references/mynet-simulation-contract.md`: project-specific dataset and paired-control rules.
- `references/version-baseline.md`: reviewed version, source fingerprints, and maintenance log.
- `scripts/audit_gprmax_input.py`: reusable static audit utility.
- `scripts/attenuation_budget.py`: exact nondispersive field-attenuation plausibility budget.
- `scripts/capture_gprmax_trace_contract.py`: preserve per-trace positions, attributes, shapes, and hashes before merge removal.
- `scripts/fingerprint_gprmax.py`: version/source/manual fingerprint for maintenance.
