# Repository Layout and Ownership

This document defines the active `master` layout after the 2026-07-15 cleanup.
Historical material remains recoverable from
`archive/pre-master-cleanup-20260715` and is not duplicated in the working tree.

## Active top-level paths

| Path | Owner and retention rule |
|---|---|
| `pgdacsnet/` | Model and loss library. AeroPath is active; frozen baselines remain for paper comparison and checkpoint compatibility. |
| `scripts/` | Reusable operational tools only. One-off migration and superseded V1 scripts belong in archive branches. |
| `configs/` | Current smoke/debug/formal configs and the locked V15 split only. Generated experiment sweeps belong in reports or a dedicated experiment branch. |
| `data/measured/` | Immutable measured releases and their source evidence. Never rewrite a published release in place. |
| `data/contracts/` | Authoritative permissions, splits, semantics, and provenance. Directory names do not grant training eligibility. |
| `data/simulations/v2/` | Current simulation sources, release specs, and checksum-verified released evidence. Runtime caches are ignored. |
| `docs/` | Current architecture and operating contracts. Superseded plans belong only in archives. |
| `reports/` | Current decision evidence required to understand retained data and simulations. Routine previews and run logs stay local. |
| `environment/` | Portable templates. The machine-local runtime profile is never committed. |
| `uavgpr_simlab/` | Independently packaged simulation utility retained for reproducible tooling. |

## Data layout

```text
data/
  measured/
    yingshan_v15/          immutable real-data release
  contracts/
    dataset_v2/            real/simulation training governance
    simulation_v2/         simulation physics and retention contract
  simulations/
    v2/                    active sources and released evidence
```

## What was archived

- Line9-conditioned V1 synthetic datasets and governance exports;
- obsolete LOO and GprMambaSep experiment configurations;
- superseded V15 candidate/build artifacts;
- one-time handoff packages, duplicate evidence ZIPs, and stale reports;
- deprecated scripts whose only inputs were removed V1 layouts.

Do not restore archived files to `master` merely for reference. Link the
archive branch or extract only a proven reusable component with tests and an
updated contract.

## Promotion rule

An asset becomes active only when all of the following agree:

1. a source or canonical file exists at the registered relative path;
2. its manifest and hashes are valid;
3. the appropriate human-audit row exists;
4. the contract explicitly permits the intended use;
5. repository validators pass.

Old names such as `accepted`, `GREEN`, or `formal` are not permissions by
themselves.
