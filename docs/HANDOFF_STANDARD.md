# Work and Handoff Standard

> Status: normative for all work on `master`. Technical domain contracts remain
> authoritative for their own data, simulation, model, and release details.

## 1. Purpose

A handoff must let another person or agent answer, without relying on chat:

1. What was the objective and what changed?
2. Which files and datasets are authoritative?
3. What was actually validated, with which command and result?
4. What remains uncertain, blocked, or deliberately excluded?
5. What exact action should be taken next?

Conversation history is useful background, but it is not a project record. A
claim becomes project state only after it is represented by versioned code,
contracts, manifests, tests, reports, or an accepted handoff record.

## 2. Authority Order

When sources disagree, use this order:

1. immutable measured releases and checksum-verified released evidence;
2. machine-readable contracts, manifests, split files, and configuration;
3. executable code and passing tests at the referenced commit;
4. current normative documentation under `docs/`;
5. retained audit reports and decision records;
6. task handoff records;
7. chat, screenshots without provenance, directory names, and recollection.

Lower-ranked evidence may identify a problem, but it cannot silently override a
higher-ranked contract. Resolve the conflict explicitly and record the decision.

## 3. Work Units

Every non-trivial task has one `task_id`:

```text
YYYYMMDD-<area>-<short-purpose>
```

Allowed areas are `code`, `model`, `data`, `simulation`, `training`,
`evaluation`, `governance`, `documentation`, and `mixed`.

Use a branch named `agent/<short-purpose>-YYYYMMDD`. Keep one objective per
branch. A task is not complete merely because a command started or an artifact
exists; all required gates and the handoff acceptance criteria must be met.

## 4. Required Records

| Work type | Required durable record |
|---|---|
| Code/model/loss | Focused commit, tests, changed behavior, compatibility note |
| Data or labels | Versioned release/manifest, source hashes, semantics, split impact, human decision |
| Simulation source | Source deck, geometry/material contract, preflight, provenance |
| Simulation result | Trace contract, runtime metadata, controls, postprocess audit, hashes, human decision |
| Training | Exact config, code commit, data/split hashes, seed, environment summary, checkpoints/metrics location |
| Evaluation | Checkpoint identity, evaluation config, valid traces, metric semantics, outputs and exclusions |
| Research decision | Decision record with alternatives, evidence, consequences, and reversibility |
| Repository/governance | Migration map, archive/recovery point, validators, cleanup scope |

Use the templates under `docs/templates/`. Milestone handoffs belong under
`reports/handoffs/<task_id>/`; local drafts belong under
`reports/handoffs/_drafts/` and are ignored by Git.

## 5. Task Lifecycle

### Start

1. Pull with `git pull --ff-only origin master`.
2. Read `README.md`, `docs/current_state.md`, and the relevant domain contract.
3. Validate the machine profile if training or simulation is involved.
4. Create a focused branch and state the objective, scope, and exclusions.
5. Check the worktree before editing. Never overwrite unrelated local work.

### During work

1. Keep source, generated runtime output, and promoted evidence in their
   designated tiers.
2. Record irreversible or research-significant choices as decision records.
3. Report what has been observed separately from what has been inferred.
4. Preserve failed or ambiguous results as evidence only when they explain a
   decision; never relabel a failed positive as a true negative.
5. Do not leave required terminal jobs running at handoff.

### Close

1. Run the validation matrix for the affected work type.
2. Confirm the worktree contains no accidental output or machine-local paths.
3. Commit the implementation first.
4. Create a handoff record referencing that implementation commit.
5. Validate the record in final mode, commit it, and push.
6. Verify the remote branch or `master` resolves to the intended commit.

## 6. Validation Matrix

| Change | Minimum gate |
|---|---|
| Documentation/governance | `python scripts/handoff_record.py validate ... --final`, relevant contract tests |
| Python code | `python -m compileall -q pgdacsnet scripts tests` plus focused tests |
| Model/loss/path logic | Focused unit tests, shape/gradient tests, one smoke train |
| Config | `python scripts/check_configs.py` |
| Measured data/labels | `check_dataset.py`, V15 validator, project-contract validator, hashes |
| Simulation source | static/preflight validator and geometry/material provenance |
| Released simulation evidence | release package `--verify-only`, controls, trace contract, human decision |
| Training closure | machine runtime validation, config guard, smoke train, run manifest |
| Formal experiment | all above plus `validate_project_contracts.py --require-formal-ready` |

Record the exact command, exit status, concise result, and timestamp. An
expected failure is valid only when the governing contract deliberately blocks
the operation and the reason is recorded. A timeout is not a pass.

## 7. Handoff Acceptance Criteria

A final handoff record must include:

- implementation commit and branch;
- objective, in-scope work, and explicit exclusions;
- user-visible or contract-visible changes;
- validation commands and real outcomes;
- artifacts using repository-relative paths and SHA256 where applicable;
- decisions and evidence;
- known risks, blockers, and residual uncertainty;
- prioritized next actions with `done_when` conditions;
- environment capabilities required to resume;
- an exact restart command or a reason that none is needed.

For `status=complete`, the referenced implementation worktree must have been
clean and all claimed gates must be `passed` or documented `expected_fail`.
For `status=blocked`, at least one concrete blocker and an unblock condition are
mandatory. Never use `complete` for a partially run simulation or training job.

## 8. Portability and Storage

- Store repository-relative paths only. Never record drive letters, usernames,
  tokens, or private environment locations.
- Refer to runtime capabilities (`training`, `official_mamba2`, `gprmax_gpu`),
  not a machine's executable path.
- Commit reproducible source and promoted evidence; keep environments, raw
  solver caches, checkpoints, routine previews, and scratch output local unless
  a domain release contract explicitly promotes them.
- Every immutable or promoted artifact must be checksum-verifiable after a
  fresh clone on the second computer.

## 9. Git and Review Rules

1. Do not force-push `master` or rewrite published release history.
2. Archive the old tip before a destructive repository-wide cleanup.
3. Keep code, data release, and report-only changes in separate commits when
   they can be reviewed independently.
4. A commit message describes the durable outcome, not the activity.
5. Another computer resumes only from a pushed commit, never from an untracked
   ZIP or a chat-only claim.
6. Frozen baselines may be changed only for correctness or compatibility and
   must retain a regression test and an explicit baseline note.

## 10. Commands

Create a local draft after the implementation commit:

```powershell
python scripts\handoff_record.py create `
  --task-id 20260715-governance-example `
  --title "Example handoff" `
  --work-type governance `
  --objective "Describe the durable objective"
```

Fill the generated JSON, then validate it before promotion:

```powershell
python scripts\handoff_record.py validate `
  reports\handoffs\_drafts\20260715-governance-example\HANDOFF.json `
  --final --verify-artifacts
```

Promote the validated file to `reports/handoffs/<task_id>/HANDOFF.json`, commit,
push, and verify the remote reference.

## Related Contracts

- [Project standard](PROJECT_STANDARD.md)
- [Repository lifecycle](REPOSITORY_OPERATIONS.md)
- [Repository layout](REPOSITORY_LAYOUT.md)
- [Simulation asset policy](SIMULATION_ASSET_POLICY.md)
- [Machine runtime profiles](../environment/README.md)
- [AeroPath architecture](AEROPATH_SSD.md)
