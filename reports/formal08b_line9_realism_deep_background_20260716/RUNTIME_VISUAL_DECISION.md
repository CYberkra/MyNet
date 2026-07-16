# FORMAL08B Runtime Visual Decision

Date: 2026-07-16

## Decision

FORMAL08B is retained as a completed failed stronger-deep-background ablation,
not as FORMAL06C's successor. FORMAL06C remains the visual mother model.

Line9 was used openly as a measured-realism calibration reference. FORMAL08B
is therefore `line9_conditioned=true`, development-only, and unable to support
an unseen-Line9 claim.

## Solver Evidence

- Eight consecutive full-scene traces completed and were reviewed blind.
- Thirty-two full-span traces at stride 8 completed in 11 minutes 49 seconds.
- GPU memory was about 2.21 GB; host memory was about 0.89 GB.
- Trace capture completed for 8/8 and 32/32 before merging.
- Full-span path/geometric correlation: `0.9999482`.
- Significant signed lobes: `7`.
- Aligned peak frequency: `79.365 MHz`.
- Target dropout below 25% median: `0.0`.
- Full-span target/adjacent-background RMS: `21.330`.
- FORMAL06C comparison value: about `17.29`.
- Line9 diagnostic value: about `2.35`.

The eight-trace extractor temporarily reported 17 lobes and weak template
correlation because the local span was too short and the stronger field
contaminated path support. The full-span run recovered the expected seven-lobe
packet and geometry tracking. This confirms why the local checkpoint is a
gross visual stop, not a final morphology statistic.

## Blind Visual Review

At exact common canonical traces and a shared 0-500 ns display contract:

- the FORMAL06C continuous multi-cycle basal packet is preserved;
- no isolated-hyperbola chain, point-target response, or packet dropout appears;
- the source and wavelet character remain essentially unchanged;
- the added deep-cover material field does not produce a useful measured-like
  background increase in the solved section;
- the basal response is slightly more dominant, not less;
- the synthetic section remains much cleaner and more regular than Line9.

FORMAL08B is not physically broken. It is rejected as a successor because its
changed factor acts partly as interface conditioning and worsens the intended
target/background difficulty.

## Claim Limits And Stop Decision

- `formal_training_allowed=false`
- `strict_line9_holdout_allowed=false`
- `causal_pair_complete=false`
- `visible_phase_training_label_allowed=false`
- `solver_evidence_released=false`
- matched control: stopped
- native 256: stopped

## Evidence

- `blind8/FORMAL06C_vs_FORMAL08B_blind_common8.png`
- `distributed32/FORMAL06C_vs_FORMAL08B_blind_common32.png`
- `distributed32/FORMAL06C_vs_FORMAL08B_explained_common32.png`
- `distributed32/FORMAL08B_distributed32_raw_tpower16.png`
- `distributed32/FORMAL08B_vs_Line9_equal_width_tpower16.png`
- `distributed32/morphology_audit/full_only_morphology_audit.json`

## Next Design Rule

Do not continue by scaling the same transition-following texture. The next
factor must place broad non-target structure away from the protected basal
corridor and vary its orientation or local continuity without using isolated
bodies. Keep source, materials, grid, acquisition, basal packet, and transition
locked for interpretability.
