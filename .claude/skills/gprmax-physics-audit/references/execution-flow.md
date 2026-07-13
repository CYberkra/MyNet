# gprMax 3.1.7 Execution Flow

Use this map when a manual statement, input file, or output array needs source-level verification. It describes the reviewed local 3.1.7 tree, not an abstract FDTD implementation.

## Standard Run

1. `gprMax/gprMax.py::run_std_sim` selects model numbers for `-n`, task, or restart mode and calls `run_model` once per model.
2. `model_build_run.py::run_model` creates `FDTDGrid` unless `--geometry-fixed` has kept a prior global grid.
3. `input_cmds_file.py::process_python_include_code` expands Python and include content. `check_cmd_names` separates single-use, multi-use, and geometry commands and checks required commands.
4. Built-in `pec` and `free_space` materials are inserted before user materials.
5. `process_singlecmds` establishes domain, resolution, time window, and other one-off settings. `process_multicmds` establishes materials, waveforms, sources, receivers, steps, PML controls, snapshots, and outputs.
6. The grid allocates solid material IDs, rigid/averaging masks, edge IDs, and CPU fields when appropriate.
7. `process_geometrycmds` executes geometry in input order. Later objects can overwrite earlier material assignments.
8. PML boundaries are built inside the declared domain. Dispersive temporary arrays and snapshot budgets are added when required.
9. `materials.process_materials` creates update coefficients. `grid.dispersion_analysis` checks the waveform spectrum and the material with the shortest significant wavelength.
10. For each B-scan model number, source and receiver coordinates are recomputed from their original coordinates plus `(model_number - 1) * step`.
11. Geometry views or object exports are written before the solver. `--geometry-only` exits after this phase.
12. The CPU or GPU time loop runs, receiver data are collected, and `fields_outputs.write_hdf5_outputfile` writes one `.out` HDF5 file per model number.

## Geometry-Fixed Meaning

With `--geometry-fixed`, the global `FDTDGrid` is built once and reused for later model numbers. Field/PML state is cleared and sources/receivers are stepped, but input Python/includes and geometry commands are not reprocessed. This reduces repeated build work for a static B-scan. It does not establish equality between two separately launched full/control inputs.

Do not use geometry-fixed when the intended B-scan changes geometry or material properties by model number through Python code. Those changes will be frozen after model 1.

## GPU Update Order

The reviewed `solve_gpu` loop performs, in order:

1. store current receiver fields;
2. store requested snapshots;
3. update magnetic fields;
4. apply magnetic PML corrections;
5. update magnetic dipole sources;
6. update electric fields (or dispersive part A);
7. apply electric PML corrections;
8. update voltage sources;
9. update Hertzian dipoles last;
10. complete dispersive part B when present.

Receiver sample zero therefore stores the initialized field state before the first update. Use the output `dt` attribute and sample index consistently; do not invent a separate shifted time axis during merge or labeling.

## Output Contract

Each `.out` root records at least gprMax version, title, iterations, grid shape/resolution, `dt`, source/receiver counts, and source/receiver steps. Source and receiver positions are stored in physical units. GPU receiver arrays contain all six field components in the reviewed implementation.

For a B-scan:

- verify every per-trace file has identical `Iterations`, `dt`, grid resolution, and component set before merging;
- verify source/receiver positions follow the intended quantized step exactly;
- preserve raw per-trace outputs until merge integrity is confirmed;
- record the merge command/tool version and hash the merged result.

The reviewed local `tools/outputfiles_merge.py` copies only `Title`, gprMax version, `Iterations`, `dt`, `nrx`, and receiver component matrices. It does not preserve per-trace source/receiver `Position`, `srcsteps`, `rxsteps`, grid shape, or grid resolution attributes. Capture and validate those attributes from every individual output before invoking `--remove-files`.

Prefer running the capture tool synchronously after gprMax exits and before merge. On Windows, any live-progress watcher should be resumable, write its report through a temporary file, and retry atomic replacement when a status reader briefly holds the destination open. The merge step must wait for a complete capture report. Do not treat a background watcher PID alone as evidence that all traces were preserved.

## Source Files To Inspect By Symptom

| Symptom | First source files |
|---|---|
| command rejected or silently different | `input_cmds_file.py`, relevant `input_cmds_*.py` |
| geometry/material index wrong | `input_cmds_geometry.py`, `input_cmd_funcs.py`, `materials.py` |
| source/receiver position shifted | `model_build_run.py`, `utilities.py`, `sources.py`, `receivers.py` |
| dispersion warning or unexpected `dt` | `grid.py`, `waveforms.py`, `materials.py` |
| GPU/CPU mismatch | `model_build_run.py`, `fields_updates_gpu.py`, `source_updates_gpu.py`, PML update modules |
| output timing/component mismatch | `fields_outputs.py`, `receivers.py`, merge utility |
| stochastic case not reproducible | `fractals.py`, input seed commands, user Python code |
