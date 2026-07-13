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

Commit source code, source decks, compact validated canonical arrays, manifests,
and reports. Do not commit solver `.out`, `.h5`, `.vti`, logs, scratch data, or
machine-local profiles.
