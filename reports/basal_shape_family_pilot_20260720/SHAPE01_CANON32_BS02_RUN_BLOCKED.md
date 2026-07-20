# SHAPE01 canonical-spacing run status

Date: 2026-07-20

## Status

The BS02 32-trace canonical-spacing run is **incomplete** and is not promotion eligible.

- Full scene: 32 traces were completed and merged.
- No-basal control: an initial multi-model process produced a partial set, but the process timed out before completion.
- Attempts to continue with gprMax `-restart` exposed a local PyCUDA/gprMax cleanup failure on RTX 5070 (`sm_120`). A model can print an output path, but the process exits with a context-stack error and the output is not reliably retained.
- No BS04 canonical-spacing run was started after this failure.

## Failure signature

```text
pycuda.driver.CompileError: nvcc preprocessing ... -arch sm_120 ... failed
PyCUDA ERROR: The context stack was not empty upon module cleanup.
```

The failure is in the local GPU execution lifecycle, not a static geometry-audit failure. The source deck remains unchanged and passes the static audit.

## Decision

```text
canonical_bs02_complete = false
canonical_bs04_started = false
formal_training_allowed = false
promotion_allowed = false
```

The existing eight-trace short diagnostics remain valid only as sparse visual screening. A fresh process/toolchain reset or a gprMax/PyCUDA CUDA-context fix is required before rerunning canonical-spacing pairs.

