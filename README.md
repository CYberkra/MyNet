# PGDA-CSNet / MyNet

PGDA-CSNet is a UAV-GPR basal-interface picking research repository. The active
paper candidate is **AeroPath-SSD**, an acquisition-conditioned structured path
model. GprMambaSep and the ConvNeXt/curve family remain frozen comparison
baselines; their A/S/G outputs must not be presented as validated physical
decompositions.

## Repository map

| Path | Purpose |
|---|---|
| `pgdacsnet/` | Active AeroPath model, losses, interfaces, and frozen baselines |
| `scripts/` | Training, evaluation, data validation, simulation, and release tools |
| `configs/` | One smoke config, one debug config, one locked formal config, and the split |
| `data/measured/yingshan_v15/` | Immutable six-line V15 measured release and 78 windows |
| `data/contracts/` | Dataset and simulation governance contracts |
| `data/simulations/v2/` | Current gprMax V2 sources and released evidence |
| `reports/` | Retained current audit and release evidence |
| `docs/` | Architecture, operating standard, and repository policy |
| `environment/` | Portable runtime template; local machine profile is ignored |

See [`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md) for ownership and
retention rules.

All cross-person, cross-agent, and cross-computer continuation follows the
versioned [`work and handoff standard`](docs/HANDOFF_STANDARD.md). Chat history
is background only; commits, contracts, validation results, and accepted
handoff records carry project state.

## Current status

The V15 measured release and the formal line split are complete:

```text
Train:      LineL1, Line3, Line7
Validation: Line6
Test:       Line9
Review:     LineX1
```

Formal paper training remains intentionally disabled for two reasons:

1. no confirmed real true-negative windows exist;
2. no independent, non-Line9-conditioned V2 simulation family is approved.

FORMAL06C is retained as development evidence only. FORMAL07A is its pre-solver
continuous-stratigraphy successor. Neither is a training release. The
authoritative state is
[`data/contracts/dataset_v2/dataset_manifest.json`](data/contracts/dataset_v2/dataset_manifest.json).

## Setup

Install the correct CUDA-enabled PyTorch build for the machine, then install
the project dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python scripts\init_machine_runtime.py
# Edit environment/project_runtime.local.json on this machine only.
python scripts\validate_machine_runtime.py --require-training
python scripts\validate_machine_runtime.py --require-gprmax
```

No machine-specific absolute path belongs in Git.

## Validation

Run these from the repository root before training, simulation promotion, or a
cross-computer handoff:

```powershell
python scripts\check_configs.py
python scripts\check_dataset.py --data-root data\measured\yingshan_v15
python scripts\validate_yingshan_v15_final.py
python scripts\validate_project_contracts.py
python scripts\handoff_record.py --help
python -m pytest -q tests
```

The formal gate must remain blocked until the two data blockers are resolved:

```powershell
python scripts\validate_project_contracts.py --require-formal-ready
```

## Training

Use the one-step debug closure before any full run:

```powershell
python scripts\train_raw_only.py configs\aeropath_ssd_v15_data_closure_debug.json
```

The formal config is
`configs/aeropath_ssd_v15_formal_blocked.json`. Do not set `enabled=true`
until the formal contract gate passes and the release is reviewed.

## Simulation storage

Source decks and compact audited releases are versioned. Machine-local solver
runs, per-trace `.out`, VTI geometry views, logs, scratch data, and training
outputs are ignored. Follow
[`docs/SIMULATION_ASSET_POLICY.md`](docs/SIMULATION_ASSET_POLICY.md); promotion
permissions come from manifests and human audit records, never directory names.

Historical V1 datasets, obsolete experiment configs, and superseded audit
packages were removed from the active tree after being preserved on
`archive/pre-master-cleanup-20260715`.
