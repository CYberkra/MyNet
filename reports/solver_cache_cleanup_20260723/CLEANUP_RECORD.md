# Solver Cache Cleanup Record - 2026-07-23

## Rationale

Per `SIMULATION_ASSET_POLICY.md`, rejected and not-promoted solver runs should be cleaned from `01_solver_runs/` after their status is recorded in `simulation_cases.csv` or decision reports.

## Cases Removed

| Case | Size | Status | Reason |
|---|---|---|---|
| FORMAL07A_CONTINUOUS_STRATIGRAPHY_DEVELOPMENT | 7.6M | rejected_over_stratified_background | Blind review rejected regular stratified background; status already in simulation_cases.csv |
| FORMAL06E_NONLAYERED_COVER_DEVELOPMENT | 17M | rejected_blind_morphology | Native-64 blind review rejected; parallel lobe train persists; decision in FORMAL06E_NATIVE64_DECISION.md |
| FORMAL06F_SINGLE_CAP_TRANSITION_DEVELOPMENT | 16M | rejected_blind_morphology | Native-64 blind review rejected; single cap transition does not fix the artifact; decision in FORMAL06F_NATIVE64_DECISION.md |
| FORMAL08A_LINE9_REALISM_BACKGROUND_DEVELOPMENT | 12M | not_promoted | Runtime passed but visual gain insufficient; retained as background ablation record only; status in simulation_cases.csv |
| FORMAL08B_LINE9_REALISM_DEEP_BACKGROUND_DEVELOPMENT | 13M | not_promoted | Runtime passed but target dominance worsened; retained as failed ablation record only; status in simulation_cases.csv |
| FORMAL06G_TERRAIN_ACQUISITION_DEVELOPMENT | 16M | rejected_blind_morphology_native64 | Native-64 blind review rejected; terrain+acquisition coupling does not fix artifact; decision in FORMAL06G_NATIVE64_DECISION.md |
| FORMAL06H_SOURCE_TEMPORAL_DEVELOPMENT | 18M | rejected_blind_morphology_native64 | Native-64 blind review rejected; source temporal variation does not fix artifact; decision in FORMAL06H_NATIVE64_DECISION.md |

## Total Space Reclaimed

99.6 MB

## Remaining Solver Runs (after cleanup)

| Case | Size | Status |
|---|---|---|
| FORMAL06C_SUBTLE_INTERFACE_DEVELOPMENT | 15M | development_evidence (has release_spec) |
| FORMAL06D_INDEPENDENT_MECHANISM_DEVELOPMENT | 53M | development_evidence (pair32 passed; native64 rejected; causal mechanism evidence only) |
| FORMAL07B_WEAK_APERIODIC_BACKGROUND_DEVELOPMENT | 15M | development_successor (not a training release; status in simulation_cases.csv) |
| FORMAL09C_P1_DENSE_PHYSICAL_FINITE_LAMINAE | 19M | source_only (smoke only) |
| FORMAL09C_P2_SPARSE_IRREGULAR_FINITE_LAMINAE | 18M | source_only (smoke only) |
| IV2_F01_GENTLE_APERIODIC_POS | 19M | training_candidate (needs release_spec) |
| IV2_F01_MATCHED_BACKGROUND_NEG | 3.4M | training_candidate (needs release_spec) |
| IV2_F02_FORMAL06C_MECHANISM_POS | 32M | development_evidence |
| IV2_F02_MATCHED_BACKGROUND_NEG | 2.1M | development_evidence |
| IV2_F03_INSTRUMENT_BAND_POS | 15M | training_candidate (needs release_spec) |
| IV2_F03_MATCHED_BACKGROUND_NEG | 2.2M | training_candidate (needs release_spec) |
| SHAPE02 BS01-04, CAL00-01 | 3.7-47M | candidate / calibration |
| SHAPE02 GEO01-15 | 2.3-22M | candidate |
| SHAPE03-05 (probes/banks) | various | candidate |

## Note

The source definitions (`.in`, `.json`, `.npy` labels) in `00_controls/` are preserved for all cases. Only per-trace solver outputs, logs, and temporary merge products were deleted.
