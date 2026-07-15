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
5. Before tuning against measured imagery, read `references/measured-line-reproduction.md` and identify the signal domain.
6. Run `scripts/audit_gprmax_input.py MODEL.in --json REPORT.json` before geometry-only or GPU execution.
7. For deep conductive media, run `scripts/attenuation_budget.py` before the paired smoke.
8. Fix every error. Record justified warnings in the case manifest.

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

For a distributed sparse B-scan pilot, scale lateral continuity limits by the
actual trace stride and report both the sparse-step change and its
per-canonical-trace equivalent. A limit defined for 0.09 m traces cannot be
applied unchanged to traces sampled every 0.72 m.

For a designed true-negative scene, run and audit the target-absent full scene
without inventing a target path or requiring a no-target control. Keep solver
validity separate from human acceptance of the hard-negative semantics.

Visual similarity is supporting evidence, not a physics proof. A visually strong interface can still be leakage, a boundary artifact, or an unrealistically easy target.

Prefer a synchronous capture-and-validate step after gprMax returns and before the merge command. A resumable background watcher is useful for live progress and early-stop evidence, but it must not be the only barrier protecting per-trace provenance.

### Broadband FDTD and SFCW-Band Proxies

- A Ricker or Gaussian FDTD source is a broadband transient source, not a
  stepped-frequency (SFCW) forward simulation.
- A post-solver frequency window may be useful as a diagnostic band proxy,
  provided it is applied identically to a strict full/control pair and is
  labelled as a proxy in every artifact and report.
- Do not call zero-padding a denser measured frequency grid. Record the
  native time-derived spectral resolution separately from plotting-bin
  spacing.
- If the stated band, tone increment, and point count disagree arithmetically,
  freeze the ambiguity in the manifest and obtain the exported tone table or
  hardware-processing metadata before claiming an SFCW-equivalent dataset.
- Compare weak full-minus-control responses with both common gain (honest
  scale context) and difference-only gain (causal visibility). Neither view
  alone is sufficient for promotion.

## Maintenance

At the start of any substantial gprMax task:

1. compare the installed version and source hash with `references/version-baseline.md`;
2. check official documentation for commands being changed;
3. update the baseline date, differences, and affected guidance;
4. run the skill validator and one representative static audit;
5. keep changes factual and sourced; do not silently rewrite earlier constraints.

Generate a fresh machine-readable fingerprint with `scripts/fingerprint_gprmax.py GPRMAX_ROOT --json fingerprint.json`.

Update this skill whenever gprMax is upgraded, a source/manual discrepancy is found, a simulation failure reveals a missing guard, or the project measurement contract changes.

## Dense Cover-Bedrock Release Pattern

For a continuous cover-weathered-bedrock mechanism family, apply the following
additional gates before adding stochastic complexity:

1. Begin with a flat-ground, fixed-height, dense local window and no discrete
   anomaly body. A clean baseline must show that the deep event is caused by
   the basal contrast itself.
2. Audit the Ricker high-frequency content, not merely its nominal centre
   frequency. The project static gate evaluates wavelength resolution at
   `2.8 * fc`; preserve at least ten cells per estimated shortest wavelength.
   If this fails, reduce the grid step or change the source/grid contract
   before any GPU run. Do not waive the warning because a geometry preview
   looks smooth.
3. Use a full/no-basal pair with shared indexed geometry and acquisition.
   Capture all per-trace HDF5 contracts and SHA256 hashes immediately after the
   solve, before any merge or cleanup.
4. Keep the material-interface arrival and visible signed phase separate. A
   finite weathered transition can shift the strongest causal wavelet lobe by
   tens of nanoseconds relative to the geometric interface.
5. Never choose a visible phase independently on every trace. First restrict
   the signed full-control response to a declared geometric search window;
   then select one globally continuous path. Its transition penalty must be
   relative to the expected geometric/acquisition time increment (a
   `delta_chainage`-style transition), not an absolute preference for a flat
   time path. This is an audit candidate, not an automatic training label.
6. Record both unconstrained peak behaviour and constrained-path behaviour.
   A large reduction in side-lobe jumps is evidence of extraction quality only
   when the resulting path remains inside the signed causal response and has
   acceptable adjacent wavelet correlation.
7. Before running a correlated-cover variant, audit its quantised field on
   the actual cover voxels: used levels, horizontal/vertical neighbour-change
   rates, correlation scales, and absence of artificial full-depth walls.
   Do not use a resized preview alone to judge spatial texture.
8. A strict full/control pass proves causal attribution, not measured-line
   realism. Before scaling a family, inspect raw and identically processed
   B-scans for dominant direct/ground wavelets, repeated parallel layer bands,
   ringing combs, and diffraction/X-shaped structures. A multi-step transition
   encoded as several constant material layers can itself create coherent
   reflectors; it is not a physically smooth transition merely because its
   total thickness varies smoothly. Likewise, a quantised stochastic property
   field must not substitute for a geologically plausible multiscale texture.
   Keep such cases as regression controls if they fail the morphology review.

The FORMAL01 F0 baseline (2026-07-15) passed this mechanism gate with a
100 MHz Ricker proxy, 0.025 m grid, 256 traces at 0.10 m spacing, and a
continuous finite transition. The F1 correlated-cover variant subsequently
passed the same strict-pair and continuous-path checks on a 32-trace smoke
run. Visual audit subsequently found that F0--F3 remain overly layered and
wavelet-regular for the intended measured-line morphology. They are retained
as causal-control regressions, not realism candidates, and must not be scaled
to training data without a redesigned subsurface/source/processing contract.

### FORMAL02 Graded-Bedrock Successor

FORMAL02 replaces the failed F-series morphology with a deliberately smaller
claim: a non-periodic cover-to-weathered-bedrock baseline that must pass causal
and spatial gates before any realistic clutter is added.

1. Generate the basal path from seeded generic multiscale priors over the full
   solver domain, then crop the acquisition window. Reject crops that are nearly
   quadratic, lack multiple smooth extrema, exceed the slope budget, or have too
   little/too much vertical range. Do not read a measured line, label, waveform,
   or held-out statistic in the generator.
2. Use one shared indexed HDF5 geometry for full and no-basal models. In the
   control, map every transition and bedrock index back to the cover material;
   do not alter geometry or introduce a replacement interface.
3. Bound the number of transition material levels by the thinnest transition in
   cells. More named materials do not create a smoother model when several bins
   have no voxels. Record the maximum adjacent epsilon and conductivity step.
4. Keep the protected supervision window separate from the solver window. Size
   lateral guards so the earliest lateral boundary round trip occurs after the
   protected window; reserve the remaining samples for boundary diagnostics.
5. A generated source deck must satisfy the shared runner schema (`target_presence`,
   `grid.trace_count`, `grid.trace_spacing_m`, `grid.dl_m`, and
   `geometry.index_file`). Exercise staging in a test before spending GPU time.
6. Do not create a visible-phase label before a successful runtime pair. Store
   only an explicitly geometric reference, then extract a signed visible phase
   from `full - control` inside a declared search window.
7. A one-trace gate may omit air when it records `air_reference_included=false`.
   Air remains a later source/decomposition diagnostic, not a prerequisite for
   proving basal causality.
8. For a distributed sparse pilot, require full/control trace contracts, broad
   span coverage, target/background contrast, no dropout, low control spatial
   residual, correlation with the independent geometric path, and retention of
   its dynamic range. Read the analysis time limit from the output HDF5 rather
   than padding a shorter solver window to a historical constant.
9. Never draw the candidate target path over a no-basal control panel. Such an
   overlay can be mistaken for a physical control response. Report the measured
   control residual numerically and keep the control image unannotated.
10. Passing the sparse pilot does not promote the case. Run the full native trace
    count, inspect raw/common-gain/difference-only views, and obtain a human
    morphology decision before adding heterogeneity or exporting training data.

## References

- `references/source-and-manual-contract.md`: official rules and installed-source behavior.
- `references/execution-flow.md`: reviewed 3.1.7 build, stepping, solve, and output call chain.
- `references/mynet-simulation-contract.md`: project-specific dataset and paired-control rules.
- `references/measured-line-reproduction.md`: raw/processed/migrated domain separation and staged measured-line calibration.
- `references/version-baseline.md`: reviewed version, source fingerprints, and maintenance log.
- `scripts/audit_gprmax_input.py`: reusable static audit utility.
- `scripts/attenuation_budget.py`: exact nondispersive field-attenuation plausibility budget.
- `scripts/capture_gprmax_trace_contract.py`: preserve per-trace positions, attributes, shapes, and hashes before merge removal.
- `scripts/fingerprint_gprmax.py`: version/source/manual fingerprint for maintenance.
