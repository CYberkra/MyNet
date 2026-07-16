# FORMAL08A Runtime Visual Decision

Date: 2026-07-16

## Decision

FORMAL08A is retained as a solved Line9-conditioned background ablation, not as
FORMAL06C's successor. FORMAL06C remains the sole visual mother model.

The decision follows the corrected project-owner ranking:

```text
FORMAL06C > Independent Family 02 > Independent Family 03
```

The realism-calibration track is allowed to use Line9 explicitly. Every case
selected this way must remain `line9_conditioned=true` and cannot support an
unseen-Line9 claim. A separate independent or fold-specific calibration track
is required for strict holdout claims.

## Solver Evidence

- Eight consecutive full-scene traces completed without dropout.
- Thirty-two full-span traces at stride 8 completed and passed trace capture.
- Static input audit: no errors or warnings.
- Path/geometric correlation: `0.9998898`.
- Significant signed lobes: `7`.
- Aligned peak frequency: `79.365 MHz`.
- Target dropout below 25% median: `0.0`.
- Full-span target/adjacent-background RMS: `14.774`.
- FORMAL06C comparison value: about `17.29`.
- Line9 diagnostic value: about `2.35`.

The lower ratio confirms that the added middle-cover texture has an effect. It
does not establish measured realism because the simulated and measured domains
have different clutter and processing contracts.

## Blind Visual Review

At exact common canonical traces and a shared display scale:

- the continuous, gently varying multi-cycle basal packet is preserved;
- no chain of isolated hyperbolas or F03-like sharp fragmented packet appears;
- the source/wavelet character remains essentially the FORMAL06C character;
- middle-time background is only slightly richer;
- the synthetic section remains much cleaner and more regular than Line9;
- the visual difference from FORMAL06C is too small to justify a new mother
  model.

Therefore FORMAL08A is not rejected as physically broken. It is stopped because
its one-factor change does not create enough visual gain to warrant a matched
control or native-256 run.

## Claim Limits

- `line9_conditioned=true`
- `formal_training_allowed=false`
- `strict_line9_holdout_allowed=false`
- `causal_pair_complete=false`
- `visible_phase_training_label_allowed=false`
- `solver_evidence_released=false`

## Evidence

- `blind8/FORMAL06C_vs_FORMAL08A_blind_common8.png`
- `distributed32/FORMAL06C_vs_FORMAL08A_blind_common32.png`
- `distributed32/FORMAL06C_vs_FORMAL08A_explained_common32.png`
- `distributed32/FORMAL08A_distributed32_raw_tpower16.png`
- `distributed32/FORMAL08A_vs_Line9_equal_width_tpower16.png`
- `distributed32/morphology_audit/full_only_morphology_audit.json`

## Next Design Rule

The next realism candidate must still inherit FORMAL06C directly. It must make
a visibly meaningful change to continuous non-target geology without changing
the accepted source, basal packet, or weak-interface materials in the same
experiment. Background design should be calibrated against Line9 on the
development track, while strict paper claims remain protected by a separate
independent or leave-one-line-out contract.
