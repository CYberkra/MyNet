# SHAPE02 geometry visual review

Date: 2026-07-20

## Result

All 12 geometry-bank profiles pass the final automatic gate after broadening
four initially over-steep candidates. The review used the complete 22.95 m
native acquisition aperture, not a contiguous 32-trace crop.

## Visual decision

- `CAL00_FLAT_REFERENCE` and `CAL01_GENTLE_DIP` remain calibration cases.
- `GEO01` and `GEO02` are the symmetric broad high/trough pair.
- `GEO03` and `GEO10` cover asymmetric flexure/compound-shoulder behaviour.
- `GEO04` provides two unequal broad relief features without sharp curvature.
- `GEO05` and `GEO06` provide low and medium aperiodic multiscale relief.
- `GEO07` and `GEO08` intentionally form a related but distinct terrace versus
  asymmetric-incision subgroup.
- `GEO09` is accepted only as a minority stress family; it must not dominate
  later training-family counts.

No profile contains an abrupt vertical wall, sharp V notch, isolated body, or
periodic sine train. Every profile is bounded and approaches a stable depth in
the large lateral guard domain.

```text
geometry_bank_pass = 12/12
solver_executed = false
formal_training_allowed = false
next_gate = one_trace_full_control_causal_probe
```

The preview is
`data/simulations/v2/00_controls/SHAPE02_BASAL_GEOMETRY_BANK/SHAPE02_GEOMETRY_CONTACT_SHEET.png`.

