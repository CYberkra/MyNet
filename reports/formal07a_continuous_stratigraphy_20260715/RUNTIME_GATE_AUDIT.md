# FORMAL07A Runtime Gate Audit

> Subsequent decision: a strict common-trace blind comparison with FORMAL06C
> rejected FORMAL07A as its visual successor. The causal/runtime passes below
> remain valid, but the proposed 32-trace next gate is cancelled. See
> `FORMAL06C_VS_FORMAL07A_DECISION.md`.

Date: 2026-07-15

Case: `FORMAL07A_CONTINUOUS_STRATIGRAPHY_DEVELOPMENT`

## Decision

FORMAL07A passes the one-trace strict causal gate and the scan-wide sparse
full-scene morphology gate. It remains development-only, Line9-conditioned,
and forbidden from formal training. No solver result is promoted or released
by this audit.

## Runtime Environment

- Existing environment: `F:/codex/envs/psgn-csnet/python.exe`
- Existing gprMax source: `F:/codex/PSGN-CSNet/gprMax-master`
- gprMax: 3.1.7
- GPU: NVIDIA RTX 5070 12 GB
- No environment was installed or downloaded for this run.

## One-Trace Strict Pair

Run: `formal07a_smoke1_20260715`

The run solved `full_scene` and `no_basal_contrast_control`; air reference was
explicitly omitted at this gate. Static audits and per-trace source/receiver
contracts passed.

| Metric | Result |
|---|---:|
| Material/source reference | 410.756 ns |
| Signed-pair visible phase | 410.843 ns |
| Visible minus reference | +0.087 ns |
| Early full/control relative difference | 3.495e-7 |
| Signed target/background RMS | 77.166 |
| Causal contrast/full target RMS | 0.984 |

Decision: causal attribution passed for one trace. Horizontal morphology was
not evaluated and cannot be inferred from this result.

## Blind Full-Scene Morphology

An initial eight contiguous traces covered only 0.63 m. It was intentionally
treated as non-diagnostic for scan-scale morphology: the basal reference varied
too little and horizontal median suppression became ill-conditioned.

A second blind full-scene-only run sampled indices
`0,32,64,96,128,160,192,224`, spanning 20.16 m without loading a label overlay.
It used no control or air run and therefore makes no causal or training-label
claim.

| Metric | Result |
|---|---:|
| Output/trace contract | pass |
| Path/geometric correlation | 0.999974 |
| Dynamic-range retention | 0.998092 |
| Median path minus reference | -0.105 ns |
| Target/adjacent-background RMS | 15.305 |
| Dropout below 25% median | 0.000 |
| Aligned-template correlation median | 0.947964 |
| Significant alternating lobes | 9 |
| Aligned peak frequency | 79.365 MHz |

Decision: the sparse scan-wide morphology gate passed. The result shows a
continuous, gently varying, multi-lobe interface response. Eight sparse
positions are insufficient for final visual acceptance or release.

## Code Contract Fix

The runner correctly emits unnumbered `stem.out` files for a single trace,
while two audit/preview readers previously required numbered or merged names.
Both readers now accept the runner's bare single-trace output; multi-trace
audits continue to prefer `stem_merged.out`. Regression tests cover both cases.

## Claim Limits And Next Gate

- `formal_training_allowed=false`
- `strict_line9_holdout_allowed=false`
- `solver_evidence_released=false`
- no visible-phase training label is released
- no 256-trace run is authorized by this audit

Historical proposed gate: a 32-position distributed matched pair. This gate
was cancelled after the common-trace visual decision; no further FORMAL07A
solver run is authorized.
