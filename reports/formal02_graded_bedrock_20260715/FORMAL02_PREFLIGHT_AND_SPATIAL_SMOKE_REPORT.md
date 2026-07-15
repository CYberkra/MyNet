# FORMAL02 Graded-Bedrock Preflight and Spatial Smoke

Date: 2026-07-15

Case: `FORMAL02_GRADED_BEDROCK_G0_BASELINE`

## Decision

FORMAL02 G0 passes the source-deck, gprMax geometry, one-trace causal-pair,
and distributed full-span smoke gates. It may proceed to one native 256-trace
full/control validation run. It is not trainable or promotion-eligible.

FORMAL01 F0-F3 remain permanently archived as causal regression controls.
Their repeated transition horizons and regular wavelets are not accepted as
measured-line morphology.

## Construction

- 2-D TMz, 55 MHz Ricker controlled source proxy.
- Grid: 0.045 m; domain: 188.55 x 49.95 m.
- Acquisition: 256 traces, 0.09 m spacing, 22.95 m span, 0.18 m Tx/Rx offset.
- Flat ground and fixed 8.01 m flight height for this isolated baseline.
- Generic seeded, non-periodic basal profile; no measured line or label is read.
- One continuous 0.63-1.08 m weathered transition represented by 12 material
  levels, bounded by the thinnest transition in cells.
- One shared indexed HDF5 geometry for full and no-basal models. The control
  changes only transition/bedrock constitutive values back to cover values.
- No discrete anomalies, topsoil interface, periodic sinusoidal component, or
  pre-solver visible-phase label.

## Static and Geometry Gates

- Static audit: full and control both pass with zero errors and zero warnings.
- Minimum wavelength resolution at the conservative 2.8 fc bound: 12.24 cells.
- gprMax geometry-only: both models pass under gprMax 3.1.7.
- gprMax numerical dispersion estimate: approximately -0.55% worst case.
- gprMax reported memory: approximately 602 MB for geometry construction.
- Lateral physical guards: 80.01 m on both sides.
- Conservative earliest lateral boundary round trip: 533.77 ns.
- Supervision-protected interval: 0-500 ns; 500-650 ns is diagnostic only.
- Cover attenuation estimate at 14 m depth: approximately 31.1 dB two-way field
  loss before geometric spreading and interface response.
- Geometry-only VTI products were inspected through the successful build and
  removed; VTI is not retained as dataset content.

## One-Trace Causal Pair

Run: `formal02_smoke1_20260715`

- Full/control alignment and finite-value checks: pass.
- Geometric basal reference: 367.06 ns.
- Signed-difference envelope peak: 394.10 ns.
- Peak offset: +27.04 ns, inside the declared +/-35 ns search window.
- Target/pre-target difference contrast: 56.53 dB.
- GPU memory: approximately 1.72 GB.
- Air reference was intentionally omitted and explicitly recorded as absent.

The offset is compatible with a finite graded transition and broadband wavelet;
the geometric material interface and visible signed phase are intentionally not
treated as the same label.

## Distributed 32-Trace Full-Span Pair

Run: `formal02_distributed32_stride8_20260715`

- Selected canonical indices: 0, 8, ..., 248.
- Covered span: 22.32/22.95 m (97.25%).
- Full and control per-trace position contracts: pass.
- Maximum source/receiver position error: 3.66e-6 m.
- Target/background signed-difference RMS ratio: 116.60.
- Visible/geometric path correlation: 0.884.
- Visible dynamic-range retention: 0.695.
- Maximum sparse visible-path step: 3.90 ns.
- Target amplitude coefficient of variation: 0.243.
- Target dropout below 25% of the median: 0.
- No-basal control spatial residual in the target window: 0.
- 600-650 ns difference RMS / target RMS: 0.00177.
- Actual HDF5 analysis window: 650.10 ns; no 700 ns extrapolation was used.

The initially suspicious line in the control preview was an audit overlay, not
a simulated response. The audit renderer now leaves the control panel free of
target-path overlays and reports the measured control residual numerically.

After the run, the generator was regenerated once to replace direct source-deck
commands with the shared disposable runner. Geometry, full-material, and
control-material hashes remain identical to the executed run; only the
generator/manifest documentation hash changed.

## Remaining Limits

- Thirty-two traces are too sparse to validate native 0.09 m morphology or
  identify every side-lobe transition at full resolution.
- The clean homogeneous cover intentionally lacks measured-like clutter. This
  stage validates the basal mechanism, not full domain realism.
- The 55 MHz Ricker source remains a broadband proxy, not an instrument-faithful
  SFCW synthesis.
- Air/source diagnostics and terrain variability remain deferred.
- No visible-phase output from this stage is approved as a training label.

## Next Run

Run exactly one 256-trace full/control pair from the unchanged source deck.
Expected RTX 5070 time from the measured 32-trace run is roughly 65-75 minutes
for both groups combined. Do not run air yet.

The full-resolution gate must require:

- complete 256-trace full/control capture contracts;
- exact time/grid alignment and finite arrays;
- target/background difference ratio >= 3;
- target dropout <= 5%;
- visible/geometric path correlation >= 0.80;
- visible dynamic-range retention >= 0.60;
- control spatial residual / target difference <= 0.10;
- late diagnostic difference / target difference <= 0.05;
- no unexplained coherent branch in common-gain and difference-only views;
- human morphology acceptance at native trace spacing.

If the full run passes, add realism one factor at a time: first continuous
cover heterogeneity, then transition-thickness variability, then source/band
ablation. Keep discrete anomaly bodies and measured-Line9 conditioning out of
the baseline family. If it fails, revise FORMAL02 rather than tuning the path
extractor around a failed physical response.

## Repository Residuals

This delivery intentionally does not absorb unrelated dirty-worktree content.
In particular, the untracked `MACRO01_GENTLE_LONG_LINE_DIAGNOSTIC` and
`MACRO02_MULTISCALE_LONG_LINE_DIAGNOSTIC` directories still contain large VTI
diagnostics and runner documentation that does not match their current
manifest grid. They are not release evidence, are not referenced by FORMAL02,
and must not be promoted or copied into a training dataset.

The next repository-governance pass should either regenerate those diagnostic
families from one authoritative manifest or archive them, then remove retained
VTI after preserving small audited previews and hashes. That cleanup is kept
separate from this causal-baseline commit so that no user or historical work is
silently deleted.

## Full-Resolution Status Update

The planned full-resolution run was started on 2026-07-15. All 256 full-scene
traces completed and passed the capture contract. The no-basal control was
stopped after 26 traces once the full-only morphology was sufficient to reject
FORMAL02 as a measured-like training candidate. This is an intentional
development early stop, not a completed causal-pair gate.

The superseding audit is
`full256_full_only_audit/FORMAL02_FULL256_FULL_ONLY_MORPHOLOGY_AUDIT.md`.
FORMAL02 remains development-only and its promotion flags remain false.
