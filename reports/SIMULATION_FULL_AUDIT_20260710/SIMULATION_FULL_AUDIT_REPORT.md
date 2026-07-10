# PGDA-CSNet current simulation full audit

Audit date: 2026-07-10  
Repository branch: `integration/7a250a7-v15`  
Scope: all simulation case bodies currently present under `data/PGDA_SYNTH_DATASET_V1`, their labels, duplicate copies, metadata, prior QC results, and training-export semantics.

## Executive conclusion

The user's visual concern was valid, because several old overview images made the target event almost invisible and the old automatic QC marked many clear labels RED. After reviewing every unique case under a single display contract (background removal, time gain, AGC, envelope, and local peak comparison), the result is:

- 45 physical case copies exist on disk, but only 33 unique raw-plus-label pairs exist.
- 12 accepted copies are byte-identical duplicates of Batch 1 cases.
- All 33 unique visible-phase curves lie on a laterally coherent reflection event.
- No active unique case was classified as `NO_VISIBLE_SIGNAL`.
- 29 cases are normal visual passes.
- 2 cases are low-contrast but still have a coherent labeled event.
- 2 cases contain a local phase/artifact ambiguity that should be masked or reviewed.
- Visual support does not make any case eligible for formal Line9 holdout training: all 33 are Line9-conditioned and all have `train_allowed=false`.

The main newly discovered defect is not the red visible-phase curve itself. It is inconsistent semantics in `y_soft_501x128.npy`: Batch 3 centers this tensor on the visible-phase curve, while Batch 1 and `LINE9_STYLE_V1_001` use a geometry-to-visible wide/shifted soft target whose center is about 6 ns earlier. The current exporter selects `y_soft` first, and the curve loss interprets it as the picking distribution. Those 13 cases must have their soft label rebuilt before any development training, even outside a strict holdout experiment.

## Inventory and integrity

| Item | Result |
|---|---:|
| Physical case copies | 45 |
| Unique raw-plus-label pairs | 33 |
| Exact duplicate copies | 12 |
| Batch 1 copies | 12 |
| Batch 3 copies | 20 |
| Accepted copies | 13 |
| Readable NPY files | 1070 / 1070 |
| NPY files with NaN or Inf | 0 |
| Parseable JSON files | 26 / 26 |
| Parseable CSV files | 13 / 13 |

The unique set is:

- Batch 1: 12 unique cases.
- Batch 3: 20 unique cases.
- Accepted-only: `LINE9_STYLE_V1_001`.

The other 12 accepted cases are exact duplicates of Batch 1, including both raw and visible-phase label hashes. They must not be counted as independent samples.

Historical `V4_QUICK_TEST` and `PILOT_VALIDATION_000001` to `000005` are referenced by old manifests but no longer have case bodies in the active repository. They cannot be visually re-audited from the current data and are not included in the 33 unique cases.

## Visual label-to-signal audit

Every case was inspected in four views:

1. Full 0-700 ns B-scan with common background removal and time gain.
2. AGC target-zone zoom.
3. Hilbert-equivalent analytic envelope target-zone zoom.
4. Current visible-phase curve versus the strongest envelope peak within +/-35 ns.

Across all 33 unique cases:

- Median absolute nearest-envelope-peak offset ranges from 0.83 to 2.86 ns.
- P90 absolute offset ranges from 2.25 to 4.81 ns.
- Median local target contrast ranges from 3.86x to 8.14x.
- The visible-phase hard mask center agrees with the curve to within one 1.4 ns label sample in every case.

### Visual decisions

| Decision | Count | Meaning |
|---|---:|---|
| `VISUAL_PASS` | 29 | Curve follows a coherent event in common-gain, AGC, and envelope views. |
| `VISUAL_PASS_LOW_CONTRAST` | 2 | Event is weak in the common full view but coherent in AGC/envelope. |
| `VISUAL_REVIEW_LOCAL_ARTIFACT` | 2 | Event exists, but a short trace span has competing phase/artifact ambiguity. |
| `NO_VISIBLE_SIGNAL` | 0 | No active unique case met this condition. |

Low-contrast cases:

- `B003_SHALLOW_DISTRACTOR_011`
- `B003_SHALLOW_DISTRACTOR_012`

Local-review cases:

- `B003_SHALLOW_DISTRACTOR_006`, approximately traces 32-35.
- `LINE9_STYLE_V1_001`, approximately traces 47-52.

These trace ranges should be masked if the cases are used for development. The complete per-case decisions are in `SIMULATION_VISUAL_DECISIONS.csv`.

## Why the old RED grades were misleading

The old `after_run_qc.py` support metric sampled the absolute oscillatory waveform at one exact time sample and divided it by the maximum absolute amplitude in a +/-30 ns search window. This is sensitive to:

- phase zero crossings;
- 501-sample temporal resampling;
- choosing a neighboring lobe of the same wavelet;
- polarity and sub-sample phase shifts.

That metric marked several cases RED even though the labeled event is continuous and strong in the envelope view. It is suitable only as a triage signal, not as the final label-validity decision. Future QC should combine a narrow envelope support window, lateral continuity, local contrast, and geometry/component evidence.

## P0: label-array semantics are inconsistent

The red visible-phase curves are visually supported, and the visible-phase hard masks agree with them. However, the training soft masks do not have one common meaning:

| Family | Unique cases | `y_soft` center relative to visible curve | Curve-training contract |
|---|---:|---:|---|
| Batch 3 | 20 | approximately 0 ns | pass |
| Batch 1 | 12 | approximately -6.01 ns | blocked |
| `LINE9_STYLE_V1_001` | 1 | median -6.54 ns; P90 absolute 14.74 ns | blocked |

The accepted metadata explicitly describes Batch 1 `y_soft` as a wide label from geometric onset to visible phase. Batch 3 generation code creates a Gaussian centered on visible phase. Both are stored under the same filename.

This becomes a real training error because:

- `scripts/export_sim_training_npz.py` currently prefers `y_soft_501x128.npy` before explicit visible-phase masks.
- `scripts/losses_gprmambasep.py` normalizes `y_mask` over time and treats it as the curve-picking distribution.
- the exported policy calls the result `visible_phase`, even when the selected tensor is not centered on the visible phase.

Required repair before any simulation export:

1. Store distinct fields for `visible_phase_distribution`, `geometry_to_visible_band`, and `visible_phase_hard_mask`.
2. Make the exporter require an explicit target semantic instead of filename priority.
3. Use a visible-phase-centered distribution for the curve head.
4. Use the wider band only for the segmentation head when explicitly configured.
5. Add a fail-fast check requiring the curve-target center to agree with `target_visible_phase_time_ns` within a defined tolerance.

## P0: provenance and formal eligibility

All 33 unique cases are Line9-conditioned. Therefore:

- formal Line9 holdout eligibility remains 0;
- none may enter the paper's strict Line9 training set;
- visual correctness does not remove the leakage;
- existing cases are development/ablation data only.

Physical reproducibility is also blocked:

- The 32 Batch 1/3 run directories contain only NPY arrays; no case-local `raw.in`, scene model, or run metadata is present.
- The 13 accepted directories contain metadata, but all 13 `scene_world.json` files have one identical hash.
- All 13 `design_metrics.csv` files also have one identical hash.
- No case contains paired `background_only`, `basal_only`, `target_only`, or equivalent component arrays.

Consequently, this audit proves that a labeled event is visible, but cannot prove from current case-local evidence that the event is the intended physical basal interface or provide A/S/G component supervision.

## Diversity audit

| Family | Cases | Median full-raw correlation | Median aligned target-patch correlation | Median label-curve correlation |
|---|---:|---:|---:|---:|
| Batch 1 Line9 style | 12 | 0.9984 | 0.7646 | approximately 1.0000 |
| Batch 3 med-depth | 8 | 0.9965 | 0.4980 | 0.8090 |
| Batch 3 shallow distractor | 12 | 0.8407 | 0.4211 | 0.7803 |

Batch 1 is effectively one label/template family with small variants. Its 12 cases must not be presented as 12 independent geometries. Batch 3 has more target-shape diversity, but its full raw correlation is still strongly influenced by shared acquisition/direct-wave structure.

## Keep, isolate, rebuild, or remove

### Keep as development evidence

- Keep all 33 unique raw cases for audit and non-formal development.
- Keep Batch 3 visible-phase label arrays; their soft-target centers are internally consistent.
- Keep the red visible-phase curves for Batch 1; the curves themselves are visually supported.

### Rebuild before development training

- Rebuild `y_soft_501x128.npy` for the 12 Batch 1 cases around the explicit visible-phase curve.
- Rebuild the soft target for `LINE9_STYLE_V1_001` and mask/review traces 47-52.
- Mask/review traces 32-35 in `B003_SHALLOW_DISTRACTOR_006`.
- Treat `B003_SHALLOW_DISTRACTOR_011` and `012` as weak/low-contrast positives if used.

### Deduplicate

- The 12 accepted duplicates should eventually be replaced by manifest references to the canonical Batch 1 bodies or moved into a hash-preserving archive.
- Do not delete them until the canonical mapping and archive checksum are committed and verified.
- Never count both copies in sampling or dataset statistics.

### Exclude from formal training

- Exclude all 33 unique cases from strict Line9 holdout training.
- Do not reinterpret any old RED case as a true negative; every active unique case has a visible labeled event.
- Do not use these cases for A/S/G component supervision because paired component truth is absent.

## Final release decision

Current simulation status:

> Signal-label visibility is substantially better than the old QC report implied, but the simulation set is not formally releasable. The blocking issues are Line9 conditioning, duplicated/template-heavy samples, missing case-local provenance, absent component truth, and inconsistent `y_soft` semantics across batches.

Formal simulation training must remain frozen until a non-Line9-conditioned, case-local reproducible simulation release is generated and passes the same visual and semantic checks.

## Audit artifacts

- `SIMULATION_CASE_AUDIT_AUTO.csv`: per-case numeric, semantic, provenance, and visual-decision fields.
- `SIMULATION_VISUAL_DECISIONS.csv`: authoritative per-case visual decisions.
- `SIMULATION_DUPLICATE_MAP.csv`: all 45 copies mapped to 33 canonical pairs.
- `SIMULATION_FAMILY_DIVERSITY.csv`: family-level diversity metrics.
- `BATCH1_UNIQUE_LABEL_ZOOM.png`: Batch 1 contact sheet.
- `BATCH3_UNIQUE_LABEL_ZOOM.png`: Batch 3 contact sheet.
- `ACCEPTED_ONLY_UNIQUE_LABEL_ZOOM.png`: accepted-only unique case.
- `previews/*.png`: four-panel evidence for each unique case.
- `audit_simulations.py`: reproducible audit generator.
