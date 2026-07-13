# PGDA-CSNet / MyNet

UAV-GPR basal-interface picking research. The active paper candidate is
AeroPath-SSD; A/S/G decomposition remains a frozen development baseline.

## Current experiment status

Formal paper training is intentionally **frozen**. The measured-data structure is now restored from the original YingShan CSV archive: six canonical full-line NPZ files, the 78-window index, GNSS coordinates, ground elevation, measured flight height, and source hashes are registered. Formal release is still blocked because no confirmed real negative windows exist and all currently registered synthetic cases are Line9-conditioned or otherwise not approved.

Do not remove `enabled: false` from experiment configs until:

- confirmed negative windows are added and independently audited;
- non-Line9-conditioned synthetic scene families pass human audit;
- V4 and all other labels use one declared visible-phase convention;
- Batch 3 receives case-wise human decisions;
- the formal-ready contract gate passes.

Run the contract gate before training:

```bash
python scripts/validate_project_contracts.py
```

A formal run must return no errors and the dataset manifest must explicitly permit training.

## Installation

Python 3.10 or newer is required. Install the appropriate official PyTorch build for the target CUDA runtime, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

`requirements-lock-cpu.txt` records the CPU test environment used for this repair. It is not a substitute for selecting the correct CUDA-enabled PyTorch wheel on the RTX 5070 host.

## Two-Computer Setup

The checked-in source contains no required machine-specific drive paths. On
each computer, after pulling `master`, create and validate a local profile:

```powershell
python scripts/init_machine_runtime.py
# Edit environment/project_runtime.local.json once with local Python/gprMax paths.
python scripts/validate_machine_runtime.py --require-training
python scripts/validate_machine_runtime.py --require-gprmax
```

The local profile is ignored by Git. Project-relative data, source decks,
compact validated arrays, manifests, skills, and source code are shared.
Raw gprMax `.out`, `.h5`, and `.vti` files, logs, scratch directories, and
ordinary `outputs/` are intentionally local. See
[`environment/README.md`](environment/README.md) and
[`data/simulation_contract_v2/SYNC_AND_RETENTION_POLICY.md`](data/simulation_contract_v2/SYNC_AND_RETENTION_POLICY.md).

## Current Standard

Read [`docs/PROJECT_STANDARD.md`](docs/PROJECT_STANDARD.md) before starting a
new simulation, training run, data promotion, or work session on another
computer. It consolidates the active AeroPath mainline, split policy,
portable-runtime procedure, simulation controls, Git retention rules, and
formal-training release gate.

## Tests

```bash
python -m pytest -q tests
python -m pytest -q data/PGDA_SYNTH_DATASET_V1/tests
```

The root `pytest.ini` prevents run scripts and generated data from being collected as tests.

## Training safeguards

`scripts/train_raw_only.py` and `scripts/resume_train.py` fail fast when:

- a requested simulation loader cannot be built;
- an experiment config is disabled, malformed, or violates the split policy;
- a formal validation/test line lacks `lines/<line>.npz`;
- rejection heads are enabled without confirmed negative labels;
- arrival-time prior is enabled without explicit valid AGL height handling;
- Line9-conditioned simulation would contaminate a formal Line9 holdout.

Curve distributions, segmentation masks, and DP paths are evaluated and saved as separate artifacts. A/S/G component quality is reported as full-reference quality only when paired component truth exists.

## Measured-data source

`scripts/import_yingshan_raw_csv.py` creates canonical full-line archives directly from the original CSV ZIP. The verified five-column schema is longitude, latitude, ground elevation, radar reflection amplitude, and flight height AGL. `raw_full_normalized` uses per-line P99 absolute-amplitude normalisation; labels remain linked to the audited window cache. See `docs/real_data/YINGSHAN_RAW_CSV_SCHEMA.md`.

Terrain metadata features are rebuilt directly from these canonical arrays with fixed physical scaling. They no longer use Line9 or other split-level statistics. Full-line evaluation plots and centerline CSV files use GNSS cumulative distance when available.

## Data governance

The authoritative governance layer is `data/dataset_contract_v2/`. Directory names and automatic QC grades are not training permissions. Promotion into the accepted synthetic dataset requires case-local labels, case-local geometry/design metadata, matching source hashes, and one authoritative human audit record.
