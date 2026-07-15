# PGDA-CSNet Current Project Standard

> Status: active from `master` at the AeroPath-SSD transition. This document
> is the operational entry point. Detailed contracts remain authoritative for
> their individual domains.

## 1. Research Mainline

- **Active paper candidate:** AeroPath-SSD, an acquisition-conditioned
  structured basal-interface path model.
- **Frozen comparison baselines:** ConvNeXt curve baseline and Route-2
  GprMambaSep. Do not claim that the latter provides a physical A/S/G
  decomposition.
- **Split policy:** Train = LineL1, Line3, Line7; validation = Line6; test =
  Line9; review-only = LineX1. The formal configuration remains disabled until
  the data-release gate passes.
- **Archives:** historical mainline work is retained in
  `archive/pre-aeropath-master-20260713` and earlier named archive branches.

## 2. Two-Computer Workflow

1. Start every session with `git pull --ff-only origin master`.
2. On each computer create the Git-ignored local profile once:

   ```powershell
   python scripts\init_machine_runtime.py
   python scripts\validate_machine_runtime.py --require-training
   python scripts\validate_machine_runtime.py --require-gprmax
   ```

3. Fill only machine-specific executable paths and GPU index in
   `environment/project_runtime.local.json`.
4. Never commit that local file, Python environments, CUDA paths, gprMax paths,
   scratch folders, or outputs.

The portable template and supported environment variables are documented in
[`environment/README.md`](../environment/README.md).

## 3. Simulation Standard

- New pilot scenes use the native `501x256` contract: no horizontal resize or
  padding; 0-700 ns / 501 samples; 0.09 m trace spacing.
- A **positive** simulation requires `full_scene`, strictly paired
  `no_basal_contrast_control`, and `air_reference`.
- A **negative** simulation requires target-absent `full_scene` plus
  `air_reference`; a failed positive is never a negative sample.
- Training masks come only from the signed, solver-validated
  `full - no_basal` visible-phase result. Geometric priors are audit evidence,
  not training labels.
- New source decks belong in
  `data/simulations/v2/01_native_256_release_pilot/`; only explicitly
  promoted compact canonical results belong in
  `data/simulations/v2/02_released_canonical/`.

Use [RECOMMENDED_NATIVE_256_V1.md](../data/contracts/simulation_v2/RECOMMENDED_NATIVE_256_V1.md)
for exact physical settings and
[SYNC_AND_RETENTION_POLICY.md](../data/contracts/simulation_v2/SYNC_AND_RETENTION_POLICY.md)
for promotion and retention rules.

## 4. Storage and Git Rules

| Keep in Git | Keep local only |
|---|---|
| Code, tests, contracts, source decks, compact promoted arrays, hashes, release reports, skills | `.out`, `.h5`, `.hdf5`, `.vti`, per-trace solver files, logs, `outputs/`, scratch folders, local runtime profile |

`*.vti` is a geometry-inspection artifact. It may be generated for a geometry
check and then deleted. The native runner removes it after `--geometry-only`.
The cleanup utility is dry-run by default:

```powershell
python scripts\clean_disposable_sim_artifacts.py
python scripts\clean_disposable_sim_artifacts.py --execute
```

It only scans approved local solver/scratch roots and never deletes labels,
source decks, canonical releases, manifests, or measured data.

## 5. Training Release Gate

Do not enable formal training until all are true:

- confirmed real negative windows exist;
- approved non-Line9-conditioned V2 simulations exist;
- no source-trace/split leakage exists;
- the formal validation configuration is enabled deliberately;
- `python scripts/validate_project_contracts.py --require-formal-ready` passes.

Current normal governance validation should pass even while the formal gate is
intentionally blocked.

## 6. Change Discipline

1. Run a smoke/static check before a costly simulation or training run.
2. Keep data, model, and report changes in separate focused commits.
3. Promote simulation results only after runtime evidence, postprocessing,
   trace contracts, hashes, and human decision are present.
4. Treat unknown or ambiguous labels as weak/ignore, never as invented truth.
5. Update the gprMax skill when source/manual behavior or the project
   measurement contract changes.

## 7. Handoff Discipline

Every milestone continuation must follow
[`HANDOFF_STANDARD.md`](HANDOFF_STANDARD.md). The implementation commit is
created first; the final handoff record then references that immutable commit.
Local drafts stay ignored under `reports/handoffs/_drafts/`, while reviewed
milestone records are committed under `reports/handoffs/<task_id>/`.

No handoff may rely on chat alone, claim a timeout as a pass, contain a
machine-specific absolute path, or omit blockers and next-entry conditions.

## Related Documents

- [AeroPath architecture contract](AEROPATH_SSD.md)
- [Current research state](current_state.md)
- [Repository lifecycle](REPOSITORY_OPERATIONS.md)
- [Research-line registry](RESEARCH_LINE_REGISTRY.md)
- [Work and handoff standard](HANDOFF_STANDARD.md)
- [gprMax physics-audit skill](../.claude/skills/gprmax-physics-audit/SKILL.md)
