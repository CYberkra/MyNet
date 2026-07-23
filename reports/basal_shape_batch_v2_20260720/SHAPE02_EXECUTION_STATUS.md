# SHAPE02 execution status

Updated: 2026-07-20

## Completed evidence

- Geometry bank: 12/12 profiles generated and visually reviewed.
- Static gprMax input audit: 24/24 full/control decks passed with zero errors and zero warnings.
- gprMax fingerprint: `3.1.7` (`Big Smoke`); the reviewed file hashes are in
  `gprmax_runtime_fingerprint.json`.
- Geometry-only gprMax builds: 12/12 causal full/control pairs passed. Transient
  `.vti` geometry views were hashed and deleted by the runner; no VTI is a
  retained solver or training artifact.
- The native runner now accepts `--trace-start`; a one-trace morphology probe
  can explicitly represent canonical trace 128 and records that selection in
  `run_manifest.json`.

## Solver propagation gate

No SHAPE02 FDTD propagation has been completed yet. The first CAL00 central
single-trace attempts stopped during CUDA compilation. This is a correct
environment gate, not a CPU fallback and not a failed electromagnetic model.

The machine has the reviewed project Python, gprMax source, Visual Studio host
compiler setup, RTX 5070/PyCUDA, and a MATLAB-bundled CUDA 12.2 toolkit. The
remaining compatibility issue is the toolkit/host/compiler/`sm_120` combination.

The machine-local profile now points to MATLAB's CUDA 12.2 toolkit. Its host
compiler requires `-allow-unsupported-compiler`; the local gprMax source has a
documented environment-only bridge for that flag. `compute_89` PTX is still an
experimental compatibility probe, not a formal architecture setting.

The compatibility sequence was exercised and archived in
`trace128_solver_logs/`:

1. Native `sm_120`: CUDA 12.2 rejected the unknown architecture.
2. `compute_89`: PyCUDA's cubin target rejected a virtual architecture.
3. `sm_89` with the reviewed host/STL bridges: compilation completed, but the
   RTX 5070 driver rejected the module with `no kernel image is available for
   execution on the device`.

The local profile therefore leaves `gprmax_force_cuda_arch` unset. It must not
silently use `sm_89` as a production substitute. A CUDA Toolkit with native
`sm_120` support, expected to be CUDA 13.x or newer, is required before the
central CAL00 propagation can begin.

## Next permitted action

After a compatible CUDA Toolkit is located or installed outside the system
drive, validate it with:

```powershell
F:\\codex\\envs\\psgn-csnet\\python.exe scripts\\validate_machine_runtime.py --require-gprmax --require-gpu-compiler
```

Then run the central `CAL00` causal pair first:

```powershell
F:\\codex\\envs\\psgn-csnet\\python.exe scripts\\run_native_256_release_pilot.py `
  data\\simulations\\v2\\00_controls\\SHAPE02_BASAL_CAUSAL_PROBES\\CAL00_FLAT_REFERENCE\\CAL00_FLAT_REFERENCE_POS `
  --trace-count 1 --trace-start 128 --skip-air-reference `
  --run-id SHAPE02_TRACE128_CAL00_FLAT_REFERENCE --execute
```

Only after that pair passes should the same central one-trace full/control
probe be extended to the remaining 11 morphology cases.
