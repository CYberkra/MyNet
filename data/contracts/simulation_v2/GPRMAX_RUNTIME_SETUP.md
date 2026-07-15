# gprMax runtime setup for PGDA Simulation Contract V2

## Audited target

- Stable gprMax target: **3.1.7**.
- Use a dedicated environment created from the official release repository's `conda_env.yml`; do not install into the project training environment.
- Record the exact gprMax, Python, NumPy, SciPy, h5py, compiler, CUDA/driver and GPU versions in every solver run.

## Installation

Run on a machine with Miniconda/Miniforge, Git, and a compiler with OpenMP:

```bash
bash data/contracts/simulation_v2/setup_gprmax_3_1_7.sh /path/to/runtime
```

The script checks out the audited `v3.1.7` tag, creates an isolated `gprmax317` environment from the official `conda_env.yml`, builds the Cython extensions, installs the package, and probes the CLI.

## Verification

From the MyNet root:

```bash
python scripts/check_gprmax_runtime_v2.py --require-exact-stable
python scripts/validate_physical_sim_v2.py
python scripts/run_physical_sim_v2_controls.py --geometry-only
```

The last command is a dry run unless `--execute` is supplied. Review the generated command plan first.

## Geometry-only stage

```bash
python scripts/run_physical_sim_v2_controls.py --geometry-only --execute
```

Inspect every generated geometry view before launching 256-run B-scans.

## Solver stage

CPU:

```bash
python scripts/run_physical_sim_v2_controls.py --execute
```

GPU 0:

```bash
python scripts/run_physical_sim_v2_controls.py --gpu 0 --execute
```

The runner uses `--geometry-fixed` because each control has static geometry and only a simple Hertzian source/receiver pair is stepped. It must not be reused unchanged for moving detailed antenna geometry.

## Current status

The generated controls and static preflight are not training-approved. Solver execution, HDF5 inspection, matched-control visible-phase extraction, and human review are still required.
