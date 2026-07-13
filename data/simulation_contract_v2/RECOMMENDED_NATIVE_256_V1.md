# Recommended Native 256 Simulation Standard

`PGDA_NATIVE_256_RELEASE_STANDARD_V1` is the long-term recommended starting
point for new gprMax scene families. It replaces neither the V2 controls nor
the legacy quarantine catalog: those remain useful regression evidence.

## Non-negotiable contract

- Produce the network's native `501 x 256` tensor. Do not resize, crop-pad, or
  horizontally interpolate a 128-trace long-line B-scan into a training case.
- Keep the canonical time axis at `0..700 ns` with `501` samples. gprMax runs
  to `701 ns` and output is resampled only after HDF5 coverage validation.
- Use a 2-D `x-y` model with one `z` cell, `dl=0.0225 m`, 20-cell PML, and a
  20-cell source/receiver/top-air clearance. `0.09 m` trace spacing is exactly
  four FDTD cells.
- Use a `55 MHz` Ricker source and `0.18 m` Tx/Rx offset unless a separately
  audited hardware model supersedes it. A plain Gaussian source remains
  disallowed because of its DC-rich spectrum.
- Place the scan at least `109.5 m` from each inner side PML boundary. The
  recommended cases use a nominal `110 m` margin; their earliest free-space
  side-boundary round trip exceeds the complete 700 ns canonical window.
- A positive case comprises `full_scene`, matched `no_basal_contrast_control`,
  and `air_reference`. Full and control share all geometry and acquisition
  settings; only basal contrast changes. A negative is target-absent by design
  and still runs `full_scene` plus `air_reference`.

## Labels and promotion

Geometric travel time is an audit prior only. A positive training mask may be
created only from signed `full - no_basal` visible-phase extraction after the
three solver outputs, HDF5 sampling checks, pre-merge trace capture, and
paired-file hashes pass. A negative mask must be confirmed zero by a completed
target-absent run; an unusable positive is never a negative.

All generated cases start with `formal_training_allowed=false`. The release
gate requires an independent family-level split, non-Line9 provenance, runtime
evidence, visible-phase review, and an explicit human promotion. See
`recommended_native_256_v1.json` for the machine-readable policy.

## First pilot family set

`recommended_native_256_cases_v1.json` defines four positive and two
target-absent negative families. Generate and statically validate them with:

```powershell
F:\codex\envs\psgn-csnet\python.exe scripts\generate_native_256_release_pilot.py
```

The command writes only a pre-solver package. Execute a reviewed case with the
native runner after configuring the local gprMax runtime:

```powershell
F:\codex\envs\psgn-csnet\python.exe scripts\run_native_256_release_pilot.py `
  data\PGDA_SYNTH_DATASET_V2\01_native_256_release_pilot\N256_F01_GENTLE_DEEP_MODERATE_POS `
  --gprmax-python F:\codex\envs\gprmax-3.1.7\python.exe --gprmax-root F:\codex\PSGN-CSNet\gprMax-master --gpu 0
```
