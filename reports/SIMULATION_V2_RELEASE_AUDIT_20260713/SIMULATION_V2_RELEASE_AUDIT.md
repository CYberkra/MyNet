# Simulation V2 Release Audit

All current independent simulations remain blocked from formal paper training. The audit separates useful development evidence from formal eligibility; it never treats directory names or successful solver completion as promotion.

| Case | State | Decision | Key reason |
|---|---|---|---|
| CTRL01_FLAT_SHALLOW_LOWLOSS_POS | full_control_air_complete | diagnostic_reference_only | Flat low-loss reference is visually idealised and dominated by endpoint artifacts. |
| CTRL02_FLAT_DEEP_MODERATE_POS | full_control_air_complete | diagnostic_reference_only | Flat deep reference is useful for timing regression, not morphology training. |
| CTRL03_SMOOTH_INTERFACE_POS | full_control_air_complete | development_positive_candidate | Clean matched contrast, but the interface is too smooth and old per-trace provenance is incomplete. |
| CTRL04_MATCHED_BACKGROUND_NEG | negative_full_air_complete | development_true_negative_candidate | The scene is explicitly basal-free and its zero mask is validated; old metadata remains untrusted. |
| CTRL05_GENTLE_TERRAIN_WEAK_LAYER_POS | full_control_air_complete | development_positive_candidate | Clean weak-layer response, but morphology and acquisition remain idealised. |
| CTRL06_LATERAL_VARIATION_POS | full_control_air_complete | artifact_diagnostic_only | A strong crossing artifact intersects the target and can confound positive supervision. |
| MACRO01_GENTLE_LONG_LINE_DIAGNOSTIC | full_control_air_complete | development_positive_candidate | Long-line response is clean but overly smooth and only 128 traces wide. |
| MACRO02_MULTISCALE_LONG_LINE_DIAGNOSTIC | full_control_complete | reprocess_required | The solved pair exists, but the formal postprocess lifecycle record is missing. |
| MACRO03_CORRELATED_VOXEL_LONG_LINE_DIAGNOSTIC | full_control_complete | repair_and_relabel_required | Missing height metadata, incomplete full-run provenance, and 31.9 ns phase-path residual. |
| MACRO04_DEEPER_GENTLE_DROPOUT_DIAGNOSTIC | full_control_complete | best_pair_pilot_candidate | Strict hashes and per-trace provenance pass; still 128 traces, no air run, and manual review is pending. |
| MACRO05_F01_DOMAIN_EQUIVALENCE | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F02_SHALLOW_DRY_GENTLE | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F03_DEEP_WEAK_CONTRAST | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F04_THICK_WEATHERING_DROPOUT | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F05_MULTISCALE_FOLDED | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F06_CLUTTER_RICH_LENSES | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F07_TERRAIN_COUPLED_HEIGHT | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F08_LOW_CONTRAST_BROAD_DROPOUT | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F09_THIN_TRANSITION_SHARP | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |
| MACRO05_F10_NEAR_FLAT_LOCAL_NOTCH | preflight_only_not_run | solver_run_required | Static design exists, but no completed solver pair or visual result is present. |

## Release Decision

- CTRL04 is a valid development true-negative candidate, not a formal negative release.
- MACRO04 is the strongest completed positive pair and the reference design for the next 256-trace pilot.
- CTRL01/02 remain timing and boundary-regression controls only.
- CTRL06 and MACRO03 must not provide positive curve supervision in their current state.
- MACRO05 requires solver execution before any morphology or label decision.
