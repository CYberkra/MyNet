# Legacy Simulation Revalidation

## Scope
- Physical copies on disk: 45
- Unique raw-plus-visible-label pairs: 33
- Signal-supported unique cases: 33/33
- Cases with visible-phase-centred curve distributions: 20/33
- Formal-training-approved cases: 0/33

## Decision
All currently materialised V1 cases remain excluded from formal Line9-holdout training. The evidence checks whether a visible-phase label follows a signal event; it does not remove Line9 conditioning, duplicate/template dependence, missing gprMax inputs, or absent component truth.

## Signal And Label Findings
- `17` cases have signal-supported labels and internally consistent visible-phase distributions; they remain development-only.
- `13` cases have a visually supported hard visible-phase curve but a shifted/wide `y_soft` tensor. Rebuild their curve distribution before any development export.
- `2` case has a visible event with a local competing phase and requires an ignore span.
- `2` cases retain a coherent but low-contrast event and may only be used as weak positives in development.
- Every unique case is Line9-conditioned, and every one lacks case-local gprMax input plus paired component truth. These are hard formal-release blockers independent of visual appearance.

## Interpretation Of Old QC
The old point-sampled waveform QC is not used as a release decision. It can report RED at a wavelet zero crossing even when the visible-phase curve follows a continuous envelope-supported event. The new decision combines envelope support, curve-target semantics, duplication, provenance, and the immutable holdout policy.

## Required Action By Category
- `REBUILD_LABEL_BEFORE_DEVELOPMENT`: rebuild the curve distribution from the explicit visible-phase curve before any development export.
- `REBUILD_LABEL_AND_LOCAL_IGNORE`: perform that rebuild and retain the stated local trace span as ignore.
- `DEVELOPMENT_ONLY_AFTER_LOCAL_IGNORE`: preserve the stated local trace span as ignore during any non-formal experiment.
- `DEVELOPMENT_ONLY_WEAK_POSITIVE`: retain reduced supervision weight; never reinterpret it as a negative.
- `DEVELOPMENT_ONLY_SIGNAL_SUPPORTED`: signal evidence is adequate for development only, not paper training.

The authoritative per-case evidence is `SIMULATION_REAUDIT_CASES.csv`.

## V2 Control Status
V2 contains 4 non-Line9 control definitions but 0 official FDTD output files and 0 materialised B-scan arrays. It is therefore an engineering control stage, not a visually auditable or training-releasable dataset. The required next gate is official gprMax runtime plus postprocessing.
