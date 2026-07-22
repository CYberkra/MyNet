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

## Dependency evidence

`requirements.txt` and `requirements-dev.txt` define compatible installation
ranges for development and CPU CI. They are intentionally not a CUDA or paper
release lock: PyTorch, CUDA, driver, and `mamba-ssm` compatibility must be
captured on the actual execution machine.

Before a formal training run or a release evaluation, create an immutable
environment record inside that run's versioned evidence directory:

```powershell
python --version
python -m pip freeze --all | Set-Content reports\<task_id>\environment-freeze.txt
python -m pip check | Set-Content reports\<task_id>\pip-check.txt
```

Record the Python executable, GPU model, driver/CUDA versions, current Git
commit, and the two generated files in the handoff or experiment record. Never
replace a prior freeze file; a later run receives a new task identifier.

Handoff records name required capabilities such as `training`,
`official_mamba2`, or `gprmax_gpu`. They must never copy executable paths from
the local runtime profile. The receiving computer resolves those capabilities
through its own ignored profile and validates them before resuming.
