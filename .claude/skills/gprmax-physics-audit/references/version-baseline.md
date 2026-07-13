# Version Baseline And Maintenance Log

## Reviewed Baseline

- Review date: 2026-07-12
- Local package version: gprMax 3.1.7
- Codename: Big Smoke
- The reviewed local source root was machine-specific. Resolve the active one
  through `environment/project_runtime.local.json` or `PGDA_GPRMAX_ROOT`; do
  not copy the old drive-letter path into commands.
- Source provenance note: local archive has no `.git` metadata; use file hashes for traceability.
- Official manual: https://docs.gprmax.com/en/latest/

### Reviewed SHA256

- `gprMax/input_cmds_geometry.py`: `BAB2FE9FD902E163AEE84DF286BEDEB4C8B01047335B13ADD906E65E6AB6941F`
- `gprMax/fractals.py`: `0D78200A3D1D9DE1F643C142E62222E3CA8E0D440E52DA2161EAA7909B47D6CD`
- `gprMax/materials.py`: `2B015CAEEF5DD231E11901F0FB83743F3A53001812FD077E632C59482B5FA4F8`
- `gprMax/grid.py`: `3B8591757C336B53C0FC84B5F5A9A0B2FEC0C19E2B61FA9BE1E4B09879C9E190`
- `gprMax/pml.py`: `C95D7CF633B5CDE7A326E5BA934FFCB396B027FC5CCFF2B32B4FD9268E6E3748`
- `gprMax/model_build_run.py`: `2587EE2412C1B0825660C37012A4D02EAFDC46CE07B3BA90490E83BC4011221E`
- `tools/outputfiles_merge.py`: `57DEA77DE03E728794100135BE13E499C4134083D4E4E674BA2E71143FBCBDC6`
- `docs/source/input.rst`: `5DC5EC42E96E7551D4564B0B1689E7468DC30E5FCC67D5385DFB5B4BF0F891CC`
- `docs/source/gprmodelling.rst`: `7639A7507BD14589F16E42E90F91703897D338EEF4E102F367AB4D7FADAC15CC`

## Important Baseline Differences

- Local `#geometry_objects_read` parser requires exactly five parameters after the command; do not pass the optional smoothing flag described in newer/other documentation.
- Local external HDF5 import uses plain voxel construction when gprMax rigid/ID arrays are absent; dielectric averaging is off.
- Local package identifies as 3.1.7 even though selected file copyrights include later years. Record both version and hashes rather than inferring provenance from copyright lines.

## Maintenance Log

### 2026-07-12

- Created skill from official manual, local documentation, and local source.
- Added Peplinski frequency guard for 50-100 MHz PGDA simulations.
- Added strict shared-geometry full/no-basal contract.
- Added deterministic external-HDF5 heterogeneity guidance and static audit script.
- First geometry-only feedback: a 65 MHz Ricker pulse on a 0.05 m grid with max epsilon 14.6 was reported by gprMax at about eight cells per significant minimum wavelength and -1.13% phase-velocity error. Static Ricker checks now default to `2.8 * fc` rather than `2 * fc`.
- Audited bundled `user_models/heterogeneous_soil.in`: it demonstrates Peplinski at 1.5 GHz even though the local manual/source state a 0.3-1.3 GHz validity band, and it leaves stochastic seeds unspecified. The skill follows the declared validity and reproducibility contracts rather than copying the example parameters.
- Added a pre-run conductive attenuation budget and an early long-run checkpoint after MACRO03 V1 passed geometry/dispersion checks but hid the deep interface. Preserved V1 as rejection evidence and retained the same indexed geometry for the lower-loss V2 pair.
- Added a source-level execution map covering input expansion, command ordering, geometry-fixed reuse, B-scan stepping, GPU update order, receiver sample timing, and HDF5 output contracts.
- Corrected the MyNet label contract after finding that continuity mode returned the Hilbert-envelope centre under a visible-phase name. The maintained workflow now requires a continuous envelope-support path followed by a continuous signed-lobe path and a bipolar-waveform regression test.
- Added merge-tool provenance after verifying that local 3.1.7 drops per-trace source/receiver positions and most root grid/step attributes. Per-trace contracts must now be captured before `--remove-files`.
- Hardened the per-trace capture tool after a Windows reader briefly locked its JSON report during atomic replacement. Capture now resumes a hash-compatible partial report and retries destination replacement. Future runners synchronously require complete capture before destructive merge; a live watcher is secondary evidence only.
- Reviewed local `waveforms.py` and the official waveform/excitation-file contract for MACRO04. Kept 55 MHz Ricker as the controlled baseline, rejected plain Gaussian as the default ideal-radiator pulse because it is unipolar/DC-rich, reserved normalized first-derivative Gaussian for a fixed-geometry ablation, and recorded a calibrated measured `#excitation_file` pulse as the preferred future hardware-matched source.
- Completed MACRO04 full/no-basal 128-trace execution with complete pre-merge contracts. The 480.1 m domain required about 61 minutes per pair; its causal target contrast passed, but cross-domain morphology remained smoother and weaker than Line9. Added a protected-window domain-sizing rule and a 382.1 m overnight-pilot option with 80 m physical side guards, 60-cell PML, and a mandatory exact domain-equivalence case.
- Generated and preflighted the MACRO05 ten-family overnight batch at 382.1 x 45 m. All 20 full/control inputs passed the source-aware static audit with zero errors and zero warnings; F01 is an exact cropped/shifted MACRO04 geometry contract, all families keep geometric references below 500 ns, and every case remains blocked from training pending solved-pair and human review.
- Standardised portable Windows runners around explicit environment variables, per-trace contract capture before destructive merge, direct single-case execution, pair markers, and resume from an existing merged full scene. Package-level validation now verifies all hashes, HDF5 contracts, strict material differences, domain guards, and F01 equivalence before transfer to another machine.

## Update Procedure

1. Read active `gprMax/_version.py`.
2. Run `scripts/fingerprint_gprmax.py GPRMAX_ROOT --json fingerprint.json` and compare the recorded files.
3. Compare command signatures and behavior with this baseline.
4. Review official pages used by the changed commands.
5. Update this log with date, version, hashes, semantic differences, and affected project cases.
6. Run `quick_validate.py` for the skill and `audit_gprmax_input.py` on one analytic and one imported-geometry model.
