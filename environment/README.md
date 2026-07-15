# Machine Runtime Profiles

The repository contains no authoritative drive letters. Each computer creates
`project_runtime.local.json`, which is Git-ignored, from
`project_runtime.example.json`.

```powershell
python scripts/init_machine_runtime.py
python scripts/validate_machine_runtime.py --require-training
python scripts/validate_machine_runtime.py --require-gprmax
```

The active resolver accepts `PGDA_RUNTIME_CONFIG` when the local profile should
live outside the repository. Individual values can be overridden with
`PGDA_PROJECT_PYTHON`, `PGDA_GPRMAX_PYTHON`, `PGDA_GPRMAX_ROOT`,
`PGDA_GPU_INDEX`, `PGDA_OUTPUT_ROOT`, and `PGDA_SCRATCH_ROOT`.

For Windows GPU gprMax runs, set `gprmax_vcvars` to the existing Visual Studio
`vcvars64.bat` on that machine. The native pilot runner loads it only for the
solver subprocess so `nvcc` can find `cl.exe`; no absolute path is committed.

Commit source code, source decks, compact validated canonical arrays, manifests,
and reports. Do not commit solver `.out`, `.h5`, `.vti`, logs, scratch data, or
machine-local profiles.

Handoff records name required capabilities such as `training`,
`official_mamba2`, or `gprmax_gpu`. They must never copy executable paths from
the local runtime profile. The receiving computer resolves those capabilities
through its own ignored profile and validates them before resuming.
