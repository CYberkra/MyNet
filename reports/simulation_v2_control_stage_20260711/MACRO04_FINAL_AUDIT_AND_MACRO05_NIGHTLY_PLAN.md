# MACRO04 Final Audit and MACRO05 Nightly Release

Date: 2026-07-12

## Decision

MACRO04 is a valid paired diagnostic, but not a released training case. Its signed full-minus-control response proves that the basal interface is causal and continuously recoverable. The solved scene is nevertheless smoother and less target-dominant than measured Line9, so the next batch varies depth, contrast, weathering, clutter, terrain, curvature, and dropout instead of cloning one attractive geometry.

MACRO05 is released only as a portable pre-solver/GPU-run package. All ten cases remain `formal_training_allowed=false` until solved-pair, visible-phase, endpoint/late-time, and human review gates pass.

## MACRO04 Evidence

- Full and no-basal outputs: 128/128 traces each, 5,946 CFL samples, complete per-trace source/receiver contracts.
- Canonical pair arrays: 501 x 128; geometry and material hashes match the declared manifest.
- Median visible-phase minus geometric reference: 5.32 ns.
- Maximum absolute local residual: 14.39 ns; maximum visible path step: 2.8 ns.
- Signed contrast target/background RMS: 7.116 under the broad pair audit.
- Signed contrast/full target RMS: 0.904.
- Late contrast/target RMS (600-700 ns): 0.0101.
- Endpoint/interior contrast RMS: 1.260; no decisive endpoint artifact is visible.
- Same-window full target/background: 1.380 versus Line9 2.589.
- Target envelope CV: 0.392 versus Line9 0.485.
- Curve extrema: 0 versus Line9 53 over the compared 128-trace span.

Visual conclusion: the interface is present and physically caused by the changed basal materials, but the model remains cleaner and geometrically smoother than the measured test line. Brightness is not compared across datasets because their amplitude units are not calibrated.

## Domain Decision

The previous 480.1 m width is conservative for a 215.9 m scan. Domain sizing is now tied to the protected signal window rather than a fixed ratio.

| Option | Physical guard per side | Earliest free-space side return | Decision |
|---|---:|---:|---|
| 442.1 m | 110 m | 733.8 ns | Required when the complete 0-700 ns window must be boundary-isolated |
| 382.1 m | 80 m | 533.7 ns | Selected for the 0-500 ns target/search window |
| About 350 m | Below 75 m | Inside 500 ns | Rejected |

The selected domain keeps the 60-cell/3 m PML in addition to the 80 m physical guards. It is 79.6% of the old width and is expected to reduce this 2D workload by about 20.4%. MACRO04 took about 61 minutes per pair, so ten reduced-domain pairs are estimated at about 8.1 hours on the reviewed RTX 5070 system.

F01 is an exact cropped/shifted MACRO04 geometry. Its solved output must be compared with MACRO04 before any reduced-domain family is promoted. The reduced width protects 0-500 ns, not the entire 700 ns output.

## MACRO05 Families

1. `F01_DOMAIN_EQUIVALENCE`: exact domain A/B contract.
2. `F02_SHALLOW_DRY_GENTLE`: shallow, dry, restrained relief.
3. `F03_DEEP_WEAK_CONTRAST`: deep weak target near detectability.
4. `F04_THICK_WEATHERING_DROPOUT`: broad transition-driven dropout.
5. `F05_MULTISCALE_FOLDED`: larger but continuous curvature.
6. `F06_CLUTTER_RICH_LENSES`: finite non-target coherent lenses.
7. `F07_TERRAIN_COUPLED_HEIGHT`: stronger terrain/air-path variation.
8. `F08_LOW_CONTRAST_BROAD_DROPOUT`: low contrast and one broad weak segment.
9. `F09_THIN_TRANSITION_SHARP`: thinner weathering and stronger contrast.
10. `F10_NEAR_FLAT_LOCAL_NOTCH`: near-flat regional interface with a local weak segment.

Common contract: 382.1 x 45 m, 0.05 m grid, 128 traces at 1.7 m, 55 MHz z-polarised Ricker, shared HDF5 full/no-basal pair, material indices 10-29 changed only, no Line9 input, and geometric references below 500 ns.

## Preflight Results

- Batch validator: 10/10 cases pass, zero errors.
- Source-aware static audit: 20/20 inputs pass, zero warnings and zero errors.
- F01 HDF5 and reference labels equal the exact cropped/shifted MACRO04 contract.
- Per-case and batch-level SHA256 manifests are present.
- Portable runner supports direct single-case execution and all-case resume.
- Existing merged full output resumes at the control; complete pairs are skipped by marker.
- Air inputs are prepared but intentionally excluded from the overnight runner.

## AeroPath Network Repair

The dataset already emitted explicit `trace_state`, `valid_trace_mask`, and `chainage_m`, and the model used chainage during soft-DP inference. The AeroPath loss branch in the trainer did not forward those fields. This would have silently fallen back to legacy weak-label inference during real training.

The repaired path now:

- forwards explicit trace semantics and validity into all structured losses;
- forwards measured chainage into the loss as well as the model;
- scales path smoothness by measured adjacent-trace spacing;
- skips window no-pick supervision when any trace is weak, ignored, or invalid padding;
- excludes unknown neighbours from path-start/path-end supervision.

The formal configuration remains disabled. Official Mamba2 could not be CUDA-smoked in this environment because `mamba_ssm` is not installed; PyTorch 2.11.0+cu128 and CUDA are available. No fallback to SSM-lite is permitted for a formal run.

## Verification

- AeroPath targeted tests: 15 passed.
- Simulation/night-run/AeroPath combined targeted tests: 28 passed.
- Python compilation: passed.
- gprMax skill validator: passed.
- MACRO05 package validator: passed.

## Run Order on the Second Computer

1. Set `GPRMAX_PYTHON`, optional `GPRMAX_SOURCE`, optional `GPRMAX_VCVARS`, and `CUDA_DEVICE`.
2. Run `RUN_ONE_CASE_GPU.cmd MACRO05_F01_DOMAIN_EQUIVALENCE` first on a new environment.
3. If the environment starts correctly, run `RUN_NIGHTLY_GPU.cmd`; it is resumable.
4. Do not train from geometric reference arrays.
5. Return merged outputs plus `run_logs` and trace-contract JSON for solved-pair audit.
