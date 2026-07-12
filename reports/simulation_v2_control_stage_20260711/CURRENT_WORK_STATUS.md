# Simulation Work Status

Updated: 2026-07-12

## Active Work

`MACRO05` portable ten-family overnight batch.

Status: MACRO04 full/no-basal execution and pair audit are complete. Its target response is causal and continuous, but it remains smoother and less target-dominant than Line9. The next batch contains ten independent 382.1 x 45 m families, including an exact cropped/shifted MACRO04 domain-equivalence control. All 20 full/control inputs and package hashes pass preflight; GPU execution is intentionally delegated to the user's second computer. The 55 MHz Ricker source is retained, air references are deferred, and `formal_training_allowed` remains false for every case.

## Verified Completed

- `CTRL05_GENTLE_TERRAIN_WEAK_LAYER_POS`: full/no-basal/air completed and postprocessed.
- `CTRL06_LATERAL_VARIATION_POS`: completed but rejected for pilot use because discrete material-zone boundaries create internal scattering artifacts.

## Required Evidence Before Any Completion Claim

- Generator and contract changes listed below.
- Static validation report.
- Geometry-only report.
- GPU run plan and process identifier.
- Full/no-basal merged outputs; air only when a decomposition experiment explicitly requires it.
- Postprocess validation and review images.

## Change Log

| Time | State | Evidence |
|---|---|---|
| 2026-07-12 | Tracking initialized | This file |
| 2026-07-12 | Implementation started | Extending generator for MACRO01 |
| 2026-07-12 | MACRO01 geometry accepted | 480.1 m domain, 216 m scan, static and geometry-only validation passed |
| 2026-07-12 | MACRO01 GPU started | `run_macro01_gpu.cmd`, runner PID 367316 |
| 2026-07-12 | MACRO01 completed | 128-trace full/no-basal/air and postprocess passed |
| 2026-07-12 | MACRO02 geometry accepted | 480.1 x 45 m, 60-cell PML, 65 MHz, dispersion error about -0.70% |
| 2026-07-12 | MACRO02 GPU started | `run_macro02_gpu.cmd`, runner PID 387468 |
| 2026-07-12 | MACRO02 pair audit completed | Full/no-basal complete; air intentionally stopped; `pair_audit/pair_audit_validation.json` passed |
| 2026-07-12 | gprMax skill created | `C:\Users\Di Jianhao\.codex\skills\gprmax-physics-audit`; validator passed |
| 2026-07-12 | MACRO03 static gate passed | Shared 9602 x 900 x 1 HDF5, 33 material IDs, deterministic seeds, no Peplinski at 55 MHz |
| 2026-07-12 | MACRO03 geometry gate passed | gprMax 3.1.7: 10 cells/min wavelength, estimated phase-velocity error -0.80%, about 1.08 GiB host geometry memory |
| 2026-07-12 | MACRO03 GPU smoke passed | Full and control each 5946 samples, finite; repeat full is bit-identical |
| 2026-07-12 | MACRO03 pair run started | `run_macro03_gpu.cmd`, runner PID 411568; full/control only, air deferred |
| 2026-07-12 | MACRO03 V1 stopped and rejected | Stopped at 75/128 full traces; deep-cover sigma 0.0075 S/m gave 2.4e-5 approximate two-way amplitude and only 1.55% target-window contrast contribution |
| 2026-07-12 | MACRO03 V2 smoke passed | Same geometry hash; deep-cover sigma 0.0025 S/m; one-trace target-window contrast contribution improved to 50.0% |
| 2026-07-12 | MACRO03 V2 pair run started | `run_macro03_gpu.cmd`, runner PID 419976 |
| 2026-07-12 | MACRO03 V2 pair completed | Full 128/128 and no-basal 128/128; merged outputs and pair audit generated |
| 2026-07-12 | Control trace contract completed | 128/128 positions, grid, steps, dt, version, and six components; zero failures |
| 2026-07-12 | MACRO03 physics diagnostic passed | Contrast/full target RMS 0.898; comparison target/background 1.97 full vs 1.08 control; no late/endpoint warning |
| 2026-07-12 | Cross-domain audit completed | MACRO03 target CV 0.509 vs Line9 0.485; earlier timing and excessive curve range retained as next-family revisions |
| 2026-07-12 | Label release remains blocked | Continuous signed-phase candidate has up to 31.9 ns local residual from geometric reference; segment review required |
| 2026-07-12 | Guarded runner installed | Future full/control runs synchronously require complete per-trace contracts before destructive merge |
| 2026-07-12 | MACRO04 preview gate completed | 480.1 x 45 m shared voxel geometry; 14.86-15.66 m basal depth; 0.80 m relief; 55 MHz Ricker retained; full/control static audits pass; GPU not started |
| 2026-07-12 | MACRO04 pair completed | Full/control 128/128, complete pre-merge trace contracts, pair physics audit passed; about 61 minutes total |
| 2026-07-12 | MACRO04 cross-domain audit completed | Causal target is continuous; envelope CV 0.392, target/background 1.380, but morphology remains smoother than Line9 |
| 2026-07-12 | Reduced-domain contract fixed | 382.1 m width, 80 m physical guards plus 3 m PML, 533.7 ns earliest free-space side return; protects 0-500 ns and cuts estimated cost 20.4% |
| 2026-07-12 | MACRO05 nightly batch prepared | Ten full/no-basal families, 20/20 static audits pass with zero warnings; F01 exact domain-equivalence; portable resumable runner and hashes complete |
